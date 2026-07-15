"""Planning Scene Server that owns the world model and projects it into the MoveIt planning
scene.

Responsibilities:
  * hold the single authoritative WorldModel,
  * ingest object updates (perception publishes moveit_msgs/CollisionObject on
    the `~/collision_object` topic),
  * project every change into the planning scene as a diff on `/planning_scene`,
  * age out perishable (perceived) objects,
  * expose Trigger services to reconcile / purge.

Why publish diffs to /planning_scene instead of calling /apply_planning_scene:
the service would have to be called from inside a subscription callback, which
deadlocks a single-threaded executor; a topic publish does not. move_group's
PlanningSceneMonitor consumes /planning_scene diffs.

NOTE: typed Add/Update/SetStatus request-response services (with rich WMResult
replies) want custom .srv files, which need a separate ament_cmake interfaces
package. That is the intended next step; the internal API here (self._wm,
self._sync) is already shaped for it. For now, ingestion is the CollisionObject
topic and control is Trigger services.
"""
from __future__ import annotations

from typing import List

import rclpy
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger

from fer_world_model.planning_scene_adapter import PlanningSceneAdapter
from fer_world_model.core.planning_scene_object import (
    ObjectSource,
    Point,
    Pose,
    Quaternion,
    PlanningSceneObject,
)
from fer_world_model.core.planning_scene_world_model import PlanningSceneWorldModel


class PlanningSceneWorldModelServer:
    def __init__(self, node: Node) -> None:
        self._node: Node = node
        # -- parameters -------------------------------------------------------
        self._node.declare_parameter("planning_frame", "base")
        self._node.declare_parameter("perceived_ttl", 2.0)      # s before a detection is stale
        self._node.declare_parameter("prune_period", 10.0)       # s between stale sweeps

        self._node.declare_parameter("seed_demo", False)        # spawn cylinder_1 on startup

        self.planning_frame = self._node.get_parameter("planning_frame").value
        self.perceived_ttl = float(self._node.get_parameter("perceived_ttl").value)
        touch_links = list(self._node.get_parameter("gripper_touch_links").value)

        # -- state ------------------------------------------------------------
        self._wm = PlanningSceneWorldModel()
        self._adapter = PlanningSceneAdapter()

        # -- ROS interfaces ---------------------------------------------------
        # Reliable publisher for scene diffs.
        self._scene_pub = self._node.create_publisher(msg_type=PlanningScene, 
                                                      topic="/planning_scene", 
                                                      qos_profile=10)

        # Latched view of the world for introspection (`ros2 topic echo`).
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._world_pub = self._node.create_publisher(
            PlanningSceneWorld, 
            "~/world", 
            latched)

        # Perception / external ingest.
        self._node.create_subscription(
            CollisionObject, 
            "~/collision_object", 
            self._on_collision_object, 
            10)

        # Control services.
        self._node.create_service(Trigger, "~/reconcile", self._on_reconcile)
        self._node.create_service(Trigger, "~/purge", self._on_purge)

        # Aging sweep.
        prune_period = float(self._node.get_parameter("prune_period").value)
        self._node.create_timer(prune_period, self._on_prune)

        if bool(self._node.get_parameter("seed_demo").value):
            self._seed_demo()

        self._node.get_logger().info(
            f"world_model_server up (frame='{self.planning_frame}', "
            f"ttl={self.perceived_ttl}s)")

    # -- helpers --------------------------------------------------------------
    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _sync(self, add: List[PlanningSceneObject] = (), remove_ids: List[str] = ()) -> None:
        """Push a diff to the planning scene and republish the latched world."""
        if add or remove_ids:
            self._scene_pub.publish(self._adapter.build_diff(add, remove_ids))
        world = PlanningSceneWorld()
        world.collision_objects = [
            self._adapter.to_collision_object(o) for o in self._wm.snapshot()]
        self._world_pub.publish(world)

    # -- ingest ---------------------------------------------------------------
    def _on_collision_object(self, msg: CollisionObject) -> None:
        """Perception (or any client) adds/moves/removes an object."""
        if msg.operation == CollisionObject.REMOVE:
            res = self._wm.rm_scene_object(msg.id)
            if res.ok:
                self._sync(remove_ids=[msg.id])
            self._log_result(res)
            return

        obj = self._from_collision_object(msg)
        if obj is None:
            self._node.get_logger().warn(f"ignoring '{msg.id}': no usable primitive")
            return

        if msg.operation == CollisionObject.MOVE:
            res = self._wm.update_scene_object(
                obj.id, pose=obj.pose, frame=obj.frame, stamp=obj.stamp)
        else:  # ADD / APPEND -> upsert
            res = self._wm.add_scene_object(obj, overwrite=True)

        if res.ok:
            self._sync(add=[self._wm.get_scene_object(obj.id)])
        self._log_result(res)

    def _from_collision_object(self, msg: CollisionObject) -> PlanningSceneObject | None:
        if not msg.primitives:
            return None
        primitive = msg.primitives[0]
        p = msg.pose.position
        q = msg.pose.orientation
        return PlanningSceneObject(
            id=msg.id,
            shape=Primitive(primitive.type),
            dimensions=list(primitive.dimensions),
            pose=Pose(Point(p.x, p.y, p.z), Quaternion(q.x, q.y, q.z, q.w)),
            frame=msg.header.frame_id or self.planning_frame,
            stamp=self._now(),
            source=ObjectSource.PERCEIVED,   # anything arriving live is perishable
        )

    # -- services -------------------------------------------------------------
    def _on_reconcile(self, _req, resp) -> Trigger.Response:
        """Re-assert the full model into the scene (idempotent)."""
        self._sync(add=self._wm.snapshot())
        resp.success = True
        resp.message = f"reconciled {len(self._wm)} object(s)"
        return resp

    def _on_purge(self, _req, resp) -> Trigger.Response:
        ids = [o.id for o in self._wm.snapshot()]
        self._wm.purge_scene()
        self._sync(remove_ids=ids)
        resp.success = True
        resp.message = f"purged {len(ids)} object(s)"
        return resp

    # -- aging ----------------------------------------------------------------
    def _on_prune(self) -> None:
        removed = self._wm.prune_stale(self._now(), self.perceived_ttl)
        if removed:
            self._node.get_logger().info(f"pruned stale: {removed}")
            self._sync(remove_ids=removed)

    # -- misc -----------------------------------------------------------------
    def _log_result(self, res) -> None:
        if not res.ok:
            self._node.get_logger().warn(f"[{res.status.value}] {res.message}")

    def _seed_demo(self) -> None:
        """Spawn cylinder_1 so the chained Pick&Place BT can run immediately."""
        from fer_world_model.core.planning_scene_object import Cylinder
        cyl = PlanningSceneObject(
            id="cylinder_1",
            shape=Primitive.CYLINDER,
            dimensions=[0.10, 0.02],   # [height, radius]
            pose=Pose(Point(0.50, 0.0, 0.05)),
            frame=self.planning_frame,
            source=ObjectSource.SEEDED,
        )
        res = self._wm.add_scene_object(cyl)
        if res.ok:
            self._sync(add=[cyl])
            self._node.get_logger().info("seeded demo object 'cylinder_1'")
        else:
            self._log_result(res)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Node(node_name="fer_planning_scene_world_model_server")
    world_model = PlanningSceneWorldModelServer(node=node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
