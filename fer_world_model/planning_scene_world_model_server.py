"""Planning Scene Server that owns the world model and projects it into the MoveIt planning
scene.

Responsibilities:
  * hold the single authoritative WorldModel,
  * project the model into the planning scene via MoveIt's /apply_planning_scene
    service. This is blocking until move_group confirms, so a success reply to the
    caller (the BT, at skill boundaries) means the scene is ready to plan against,
  * age out perishable (perceived) objects,
  * expose Trigger services to reconcile / purge.

CONCURRENCY CONTRACT — read this before adding any callback:
A blocking service call inside a service callback deadlocks a single-threaded
executor: the one thread waits for a response that only it could deliver. This
node therefore spins a MultiThreadedExecutor (2 threads) with two callback
groups:
  * _model_cb_group (mutually exclusive) — EVERY callback that touches
    self._wm lives here. The executor never runs two of its members
    concurrently, so this group IS the model's lock. Registering a
    model-touching callback outside it reintroduces a silent data race.
  * _apply_planning_scene_cb_group — holds the MoveIt service clients
    (/apply_planning_scene and /get_planning_scene) and nothing that touches
    self._wm, so their responses can be processed on the second thread while a
    model callback blocks waiting for one of them. Both clients may share this
    one group because a model callback only ever has a single such call
    outstanding at a time (get, then apply — never concurrently).

NOTE: typed Add/Update/SetStatus request-response services (with rich WMResult
replies and scoped purge) want custom .srv files, which need a separate
ament_cmake interfaces package. That is also planned; the internal API here
(self._wm, self._sync) is already shaped for it. For now, control is Trigger
services. There is deliberately NO ingestion topic: mutation goes through this
node's services, never past the model into the scene.
"""
from __future__ import annotations

from typing import List

import rclpy
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents, PlanningSceneWorld
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger

from fer_world_model.planning_scene_adapter import PlanningSceneAdapter, SceneFacts
from fer_world_model.core.planning_scene_object import ObjectStatus, PlanningSceneObject
from fer_world_model.core.planning_scene_world_model import PlanningSceneWorldModel


class PlanningSceneWorldModelServer:
    def __init__(self, node: Node) -> None:
        self._node: Node = node
        # -- parameters -------------------------------------------------------
        self._node.declare_parameter("planning_frame", "base")
        self._node.declare_parameter("perceived_ttl", 2.0)      # s before a detection is stale
        self._node.declare_parameter("prune_period", 1.0)       # s between stale sweeps
        self._node.declare_parameter("apply_timeout", 2.0)      # s to wait for move_group's ack

        self.planning_frame = self._node.get_parameter("planning_frame").value
        self.perceived_ttl = float(self._node.get_parameter("perceived_ttl").value)
        self.apply_timeout = float(self._node.get_parameter("apply_timeout").value)

        # -- state ------------------------------------------------------------
        self._wm = PlanningSceneWorldModel()
        self._adapter = PlanningSceneAdapter()

        # -- concurrency (see module docstring) --------------------------------
        self._model_cb_group = MutuallyExclusiveCallbackGroup()
        self._apply_planning_scene_cb_group = MutuallyExclusiveCallbackGroup()

        # -- ROS interfaces ---------------------------------------------------
        # Scene mutation goes through MoveIt's apply service to get a real ack.
        self._apply_planning_scene_client = self._node.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
            callback_group=self._apply_planning_scene_cb_group
        )

        self._get_planning_scene_client = self._node.create_client(
            GetPlanningScene,
            "/get_planning_scene",
            callback_group=self._apply_planning_scene_cb_group
        )   
        # Latched view of the world for introspection, keeps the last published state 
        # of the world model available
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._world_state_pub = self._node.create_publisher(
            PlanningSceneWorld, 
            "~/world_state", 
            latched
        )

        # Control services (model-touching -> model group).
        self._node.create_service(Trigger, "~/reconcile", self._on_reconcile,
                                  callback_group=self._model_cb_group)
        self._node.create_service(Trigger, "~/purge", self._on_purge,
                                  callback_group=self._model_cb_group)

        # Aging sweep (model-touching -> model group).
        prune_period = float(self._node.get_parameter("prune_period").value)
        self._node.create_timer(prune_period, self._on_prune,
                                callback_group=self._model_cb_group)

        self._node.get_logger().info(
            f"world_model_server up (frame='{self.planning_frame}', "
            f"ttl={self.perceived_ttl}s)")

    # -- helpers --------------------------------------------------------------
    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9


    def _publish_world_state(self) -> None:
        world = PlanningSceneWorld()
        world.collision_objects = [
            self._adapter.to_collision_object(o) for o in self._wm.snapshot()]
        self._world_state_pub.publish(world)


    def _sync(self, add: List[PlanningSceneObject] = (), remove_ids: List[str] = ()) -> bool:
        """Project a change into the planning scene. Returns True only once move_group 
        has confirmed the diff (also True when there was nothing to apply)."""
        applied = True
        if add or remove_ids:
            applied = self._apply_scene(self._adapter.build_diff(add, remove_ids))
        
        return applied


    def _apply_scene(self, diff: PlanningScene) -> bool:
        """Blocking call to /apply_planning_scene. Always runs inside a
        model-group callback; the response is processed by the second executor
        thread via the IO group (see module docstring), which is what makes
        blocking here safe."""
        if not self._apply_planning_scene_client.service_is_ready():
            self._node.get_logger().warn(
                "/apply_planning_scene unavailable — is move_group running? "
                "Scene NOT updated; the model is unchanged and can be "
                "re-asserted with ~/reconcile.")
            return False
        resp = self._apply_planning_scene_client.call(
            ApplyPlanningScene.Request(scene=diff), timeout_sec=self.apply_timeout)
        if resp is None:
            self._node.get_logger().warn(
                f"/apply_planning_scene gave no answer within "
                f"{self.apply_timeout}s — scene state unknown; re-assert with "
                f"~/reconcile.")
            return False
        if not resp.success:
            self._node.get_logger().warn("/apply_planning_scene rejected the diff")
        return resp.success


    def _fetch_scene(self) -> SceneFacts | None:
        """Blocking read of /get_planning_scene, reduced to SceneFacts (world
        object ids + attachments). Returns None if move_group did not answer.

        The request carries a component bitmask selecting WHICH parts of the
        scene to return — we ask only for world object names and the robot's
        attached objects. GetPlanningScene.Response has no success flag (unlike
        ApplyPlanningScene); a returned scene IS the answer, a None is failure.

        Runs inside a model-group callback; the response is processed on the
        second executor thread via the IO group (see module docstring), which is
        what makes blocking here safe."""
        if not self._get_planning_scene_client.service_is_ready():
            self._node.get_logger().warn(
                "/get_planning_scene unavailable — is move_group running? "
                "Skipping divergence check.")
            return None
        components = PlanningSceneComponents(
            components=(PlanningSceneComponents.WORLD_OBJECT_NAMES
                        | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS))
        resp = self._get_planning_scene_client.call(
            GetPlanningScene.Request(components=components),
            timeout_sec=self.apply_timeout)
        if resp is None:
            self._node.get_logger().warn(
                f"/get_planning_scene gave no answer within "
                f"{self.apply_timeout}s — skipping divergence check.")
            return None
        return self._adapter.scene_facts(resp.scene)

    def _divergences(self, facts: SceneFacts) -> List[str]:
        """Compare the model (source of truth) against the scene and return a
        list of human-readable mismatches. DETECTION ONLY, never mutates the
        model or the scene.

        Iterates the MODEL's objects, not the scene's: the scene also holds the
        static environment (table, fixtures) the model does not own, so walking
        the scene would mis-flag all of that as 'unexpected'. Walking the model
        asks the only question that has a defined answer: is each thing I own
        where I believe it is?"""
        out: List[str] = []
        for obj in self._wm.snapshot():
            in_world = obj.id in facts.world_ids
            attached_link = facts.attached.get(obj.id)  # None if not attached

            if obj.status is ObjectStatus.FREE:
                if attached_link is not None:
                    out.append(
                        f"{obj.id}: model FREE but scene reports it attached to "
                        f"'{attached_link}'")
                elif not in_world:
                    out.append(
                        f"{obj.id}: model holds it but scene has neither a world "
                        f"object nor an attachment for it")
            else:  # GRASPED
                if attached_link is None:
                    where = "in the world (free)" if in_world else "nowhere"
                    out.append(
                        f"{obj.id}: model GRASPED by '{obj.held_by}' but scene "
                        f"has it {where}, not attached")
                elif attached_link != obj.held_by:
                    out.append(
                        f"{obj.id}: scene attaches it to '{attached_link}' but "
                        f"model records holder '{obj.held_by}'")

        return out

    # -- services -------------------------------------------------------------
    def _on_reconcile(self, _req, resp) -> Trigger.Response:
        """Bidirectional reconcile at a skill boundary: read the scene and report
        any divergence from the model, then re-assert the model into the scene.
        success=True means move_group applied the projection — the caller may
        plan immediately.

        The read is best-effort and read-only: a failed read or a reported
        divergence does not block the push (repair is a later step; for now the
        model stays the source of truth and simply logs what disagrees)."""
        facts = self._fetch_scene()
        if facts is not None:
            for msg in self._divergences(facts):
                self._node.get_logger().warn(f"divergence: {msg}")

        applied = self._sync(add=self._wm.snapshot())
        self._publish_world_state()
        resp.success = applied
        resp.message = (f"reconciled {len(self._wm)} object(s)" if applied
                        else "apply failed — scene not updated, model unchanged")
        return resp

    def _on_purge(self, _req, resp) -> Trigger.Response:
        ids = [o.id for o in self._wm.snapshot()]
        self._wm.purge_scene()
        applied = self._sync(remove_ids=ids)
        self._publish_world_state()
        resp.success = applied
        resp.message = (f"purged {len(ids)} object(s)" if applied
                        else f"purged {len(ids)} object(s) from the model, but "
                             f"the scene apply failed — the scene may still "
                             f"hold them")
        return resp

    # -- aging ----------------------------------------------------------------
    def _on_prune(self) -> None:
        removed = self._wm.prune_stale(self._now(), self.perceived_ttl)
        if removed:
            self._node.get_logger().info(f"pruned stale: {removed}")
            if not self._sync(remove_ids=removed):
                self._node.get_logger().warn(
                    f"prune: scene apply failed; {removed} remain in the scene "
                    f"until the next reconcile")
            self._publish_world_state()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Node(node_name="fer_planning_scene_world_model")
    world_model = PlanningSceneWorldModelServer(node=node)

    # Two threads: one may block inside a model-group callback waiting on
    # /apply_planning_scene; the other processes that client's response.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
