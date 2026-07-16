"""Planning Scene World Model Server — a READ-ONLY observer of the MoveIt
planning scene.

The model never writes the scene. Perception and MTC write it directly
(perception: world objects, fire-and-forget; MTC: attachments + ACM as part of
executing picks). This node:
  * fetches the full scene once at startup via /get_planning_scene (waiting
    until move_group is available),
  * then subscribes to /monitored_planning_scene and folds every diff into the
    authoritative in-memory view (FREE/GRASPED status, last-observed stamps),
  * guards GRASPED objects: a world update for an object the scene holds
    attached is refused and FLAGGED — the camera seeing a "grasped" object out
    in the world is phantom-grasp evidence, never noise,
  * exposes the view: latched ~/world_state topic and a ~/get_objects Trigger
    returning JSON (id, pose, status, held_by, last_observed, stale, conflict)
    for the orchestrator, which dispatches error hunting elsewhere.

Staleness is information, not a trigger: the scene retains objects the camera
stopped seeing (silence is not removal), so the model marks them stale and
keeps mirroring them. Removing ghosts from the scene belongs to a future
repair skill, never to this node.

CONCURRENCY CONTRACT:
  * _model_cb_group (mutually exclusive) — EVERY callback that touches
    self._wm lives here. The executor never runs two of its members
    concurrently, so this group IS the model's lock.
  * _io_cb_group — holds the /get_planning_scene client, and nothing that
    touches self._wm.
Currently no callback blocks on a service call: the one fetch happens in
initialize(), BEFORE executor.spin(), via call_async +
spin_until_future_complete (a plain client.call() there would deadlock —
nothing is pumping the node yet). The two-group split and the second executor
thread are kept so a future blocking call inside a callback (e.g. a ~/resync
service) drops in without redesigning this.
"""
from __future__ import annotations

import json

import rclpy
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents, PlanningSceneWorld
from moveit_msgs.srv import GetPlanningScene
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import Executor, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger

from fer_world_model.planning_scene_adapter import PlanningSceneAdapter
from fer_world_model.core.planning_scene_object import ObjectStatus, PlanningSceneObject
from fer_world_model.core.planning_scene_world_model import PlanningSceneWorldModel

# Which parts of the scene we observe: world objects WITH geometry (names alone
# cannot rebuild the model) and the robot's attached objects.
SCENE_COMPONENTS = (PlanningSceneComponents.WORLD_OBJECT_NAMES
                    | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                    | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)


class PlanningSceneWorldModelServer:
    def __init__(self, node: Node) -> None:
        self._node: Node = node

        # -- parameters -------------------------------------------------------
        self._node.declare_parameter("stale_ttl", 2.0)      # s without a diff before an object is reported stale
        self._node.declare_parameter("fetch_timeout", 5.0)  # s to wait for a /get_planning_scene answer

        self.stale_ttl = float(self._node.get_parameter("stale_ttl").value)
        self.fetch_timeout = float(self._node.get_parameter("fetch_timeout").value)

        # -- state ------------------------------------------------------------
        self._wm = PlanningSceneWorldModel()
        self._adapter = PlanningSceneAdapter()
        
        # Closed until initialize() has rebuilt from the full scene. Needed
        # because spin_until_future_complete also pumps our subscription:
        # without the gate, diffs would fold into an empty model mid-fetch.
        self._initialized = False

        # -- concurrency (see module docstring) --------------------------------
        self._model_cb_group = MutuallyExclusiveCallbackGroup()
        self._io_cb_group = MutuallyExclusiveCallbackGroup()

        # -- ROS interfaces ---------------------------------------------------
        self._get_planning_scene_client = self._node.create_client(
            GetPlanningScene,
            "/get_planning_scene",
            callback_group=self._io_cb_group
        )

        # move_group publishes every scene change here (usually diffs, but
        # is_diff=False full scenes are legal and mean "rebuild").
        self._scene_sub = self._node.create_subscription(
            PlanningScene,
            "/monitored_planning_scene",
            self._on_scene_update,
            10,
            callback_group=self._model_cb_group
        )

        # Latched view of the model for introspection: keeps the last published
        # state available to late subscribers.
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._world_state_pub = self._node.create_publisher(
            PlanningSceneWorld,
            "~/world_state",
            latched
        )

        # Orchestrator front door (read-only, model-touching -> model group).
        self._node.create_service(Trigger, "~/get_objects", self._on_get_objects,
                                  callback_group=self._model_cb_group)

        self._node.get_logger().info(
            f"world_model_server constructed (stale_ttl={self.stale_ttl}s)")

    # -- startup ----------------------------------------------------------------
    def initialize(self, executor: Executor) -> None:
        """Fetch the full planning scene once and open the gate. MUST be called
        after the node is added to `executor` and before executor.spin().

        A blocking client.call() here would deadlock: nothing is spinning the
        node yet, so the response could never be delivered. call_async +
        spin_until_future_complete is the bounded spin that pumps the node just
        long enough to complete this one future. Retries until move_group
        answers — the model is useless without a first snapshot."""
        while rclpy.ok():
            while not self._get_planning_scene_client.wait_for_service(timeout_sec=5.0):
                if not rclpy.ok():
                    return
                self._node.get_logger().info(
                    "waiting for /get_planning_scene (is move_group up?) ...")

            future = self._get_planning_scene_client.call_async(
                GetPlanningScene.Request(
                    components=PlanningSceneComponents(components=SCENE_COMPONENTS)))
            executor.spin_until_future_complete(future, timeout_sec=self.fetch_timeout)
            if future.done() and future.exception() is None:
                self._rebuild_from_scene(future.result().scene)
                self._initialized = True
                self._publish_world_state()
                self._node.get_logger().info(
                    f"initialized from full scene: {len(self._wm)} object(s)")
                return
            future.cancel()
            self._node.get_logger().warn(
                f"/get_planning_scene gave no answer within "
                f"{self.fetch_timeout}s — retrying")

    # -- helpers --------------------------------------------------------------
    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _publish_world_state(self) -> None:
        world = PlanningSceneWorld()
        world.collision_objects = [
            self._adapter.to_collision_object(o) for o in self._wm.snapshot()]
        self._world_state_pub.publish(world)

    def _rebuild_from_scene(self, scene: PlanningScene) -> None:
        """Replace the model's view with a full-scene snapshot."""
        now = self._now()
        objects: list[PlanningSceneObject] = []
        for co in scene.world.collision_objects:
            try:
                objects.append(self._adapter.to_scene_object(co, stamp=now))
            except (ValueError, NotImplementedError) as exc:
                self._node.get_logger().warn(f"rebuild: skipping world object: {exc}")
        for aco in scene.robot_state.attached_collision_objects:
            try:
                objects.append(self._adapter.to_scene_object(
                    aco.object, stamp=now,
                    status=ObjectStatus.GRASPED, held_by=aco.link_name))
            except (ValueError, NotImplementedError) as exc:
                self._node.get_logger().warn(f"rebuild: skipping attached object: {exc}")
        self._wm.rebuild(objects)

    def _fold_diff(self, msg: PlanningScene) -> bool:
        """Fold one diff message into the model. Returns True if the model
        changed. Conflicts (GRASPED guard) are logged, never applied."""
        changed = False
        now = self._now()

        for co in msg.world.collision_objects:
            if co.operation in (CollisionObject.ADD, CollisionObject.APPEND):
                try:
                    obj = self._adapter.to_scene_object(co, stamp=now)
                except (ValueError, NotImplementedError) as exc:
                    self._node.get_logger().warn(f"diff: skipping world object: {exc}")
                    continue
                changed |= self._log_result(self._wm.observe_object(obj))
            elif co.operation == CollisionObject.MOVE:
                changed |= self._log_result(self._wm.observe_moved(
                    co.id, self._adapter.from_ros_pose(co.pose), now))
            elif co.operation == CollisionObject.REMOVE:
                if co.id:
                    changed |= self._log_result(self._wm.observe_removed(co.id))
                else:  # empty id = MoveIt's "clear all world objects"
                    cleared = self._wm.observe_world_cleared()
                    if cleared:
                        self._node.get_logger().info(f"world cleared: {cleared}")
                        changed = True

        for aco in msg.robot_state.attached_collision_objects:
            if aco.object.operation == CollisionObject.ADD:
                # Attach diffs may carry only the id (object moved from the
                # world); build a fallback only if geometry came along.
                fallback = None
                if aco.object.primitives:
                    try:
                        fallback = self._adapter.to_scene_object(
                            aco.object, stamp=now,
                            status=ObjectStatus.GRASPED, held_by=aco.link_name)
                    except (ValueError, NotImplementedError) as exc:
                        self._node.get_logger().warn(f"diff: attached object: {exc}")
                changed |= self._log_result(self._wm.observe_attached(
                    aco.object.id, aco.link_name, now, fallback=fallback))
            elif aco.object.operation == CollisionObject.REMOVE:
                if aco.object.id:
                    changed |= self._log_result(
                        self._wm.observe_detached(aco.object.id, now))
                else:  # empty id = detach everything (from aco.link_name, or all)
                    for obj in self._wm.by_status(ObjectStatus.GRASPED).values():
                        if not aco.link_name or obj.held_by == aco.link_name:
                            changed |= self._log_result(
                                self._wm.observe_detached(obj.id, now))

        return changed

    def _log_result(self, res) -> bool:
        """Log a WMResult; returns True if the observation was applied."""
        if res.ok:
            self._node.get_logger().debug(res.message)
            return True
        self._node.get_logger().warn(res.message)
        return False

    # -- callbacks ---------------------------------------------------------------
    def _on_scene_update(self, msg: PlanningScene) -> None:
        if not self._initialized:
            return  # the startup snapshot will contain this anyway
        if msg.is_diff:
            if self._fold_diff(msg):
                self._publish_world_state()
        else:
            self._rebuild_from_scene(msg)
            self._publish_world_state()

    def _on_get_objects(self, _req, resp) -> Trigger.Response:
        """Orchestrator query: the model's full view as JSON in resp.message.
        `conflict` non-null means the GRASPED guard refused a world update for
        that object — phantom-grasp evidence the orchestrator should chase."""
        now = self._now()
        conflicts = self._wm.conflicts()
        objects = []
        for o in self._wm.snapshot():
            objects.append({
                "id": o.id,
                "frame": o.frame,
                "pose": {
                    "position": [o.pose.position.x, o.pose.position.y, o.pose.position.z],
                    "orientation": [o.pose.orientation.x, o.pose.orientation.y,
                                    o.pose.orientation.z, o.pose.orientation.w],
                },
                "status": o.status.value,
                "held_by": o.held_by,
                "last_observed": o.stamp,
                "stale": o.is_stale(now, self.stale_ttl),
                "conflict": conflicts.get(o.id),
            })
        resp.success = self._initialized
        resp.message = json.dumps({"count": len(objects), "objects": objects})
        return resp


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Node(node_name="fer_planning_scene_world_model")
    world_model = PlanningSceneWorldModelServer(node=node)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        world_model.initialize(executor)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
