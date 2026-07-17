"""Mock camera: publishes the objects from config/scene_objects.yaml to the
planning scene ONCE, then idles.

Stands in for the future perception node so the pick-and-place pipeline (and
the world model observing it) can run before a real camera exists. It writes
the way perception does — fire-and-forget diffs on the /planning_scene topic,
no strings attached — and is deliberately self-contained: no imports from the
world model's core, because a camera is an external writer, not part of the
model.

One-shot by design: this mock has nothing new to say after the initial ADD (a
real camera tracks poses and publishes when they change). Consequence for the
world model: with nobody re-asserting, every object trips `stale` after
stale_ttl — expected, ignore it while running this mock.

The single publish waits until move_group is actually subscribed to
/planning_scene; publishing into a topic nobody listens to yet would silently
seed nothing.
"""
from __future__ import annotations

import yaml

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

# yaml `primitive.type` -> (SolidPrimitive constant, expected dimension count)
PRIMITIVE_TYPES = {
    "box": (SolidPrimitive.BOX, 3),
    "cylinder": (SolidPrimitive.CYLINDER, 2),
    "sphere": (SolidPrimitive.SPHERE, 1),
    "cone": (SolidPrimitive.CONE, 2),
}


class MockCamera(Node):
    def __init__(self) -> None:
        super().__init__("fer_mock_camera")
        default_config = (get_package_share_directory("fer_world_model")
                          + "/config/scene_objects.yaml")
        self.declare_parameter("scene_config", default_config)
        config_path = self.get_parameter("scene_config").value

        self._objects = self._load_objects(config_path)
        self._scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        # Poll until move_group is listening, publish once, stop.
        self._timer = self.create_timer(0.5, self._try_publish)

        self.get_logger().info(
            f"mock camera loaded {len(self._objects)} object(s) from {config_path}")

    def _load_objects(self, path: str) -> list[CollisionObject]:
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        objects: list[CollisionObject] = []
        for entry in config.get("objects", []):
            try:
                objects.append(self._to_collision_object(entry))
            except (KeyError, ValueError) as exc:
                self.get_logger().error(
                    f"skipping malformed entry {entry.get('id', '<no id>')!r}: {exc}")
        return objects

    @staticmethod
    def _to_collision_object(entry: dict) -> CollisionObject:
        prim_type, expected_dims = PRIMITIVE_TYPES[entry["primitive"]["type"]]
        dims = [float(d) for d in entry["primitive"]["dimensions"]]
        if len(dims) != expected_dims:
            raise ValueError(
                f"{entry['primitive']['type']} needs {expected_dims} "
                f"dimension(s), got {len(dims)}")

        pos = entry["pose"]["position"]
        ori = entry["pose"]["orientation"]
        co = CollisionObject()
        co.id = entry["id"]
        co.header.frame_id = entry["frame_id"]
        co.operation = CollisionObject.ADD
        co.pose = Pose(
            position=Point(x=float(pos["x"]), y=float(pos["y"]), z=float(pos["z"])),
            orientation=Quaternion(x=float(ori["x"]), y=float(ori["y"]),
                                   z=float(ori["z"]), w=float(ori["w"])))
        co.primitives = [SolidPrimitive(type=prim_type, dimensions=dims)]
        co.primitive_poses = [Pose()]  # primitive at object origin
        return co

    def _try_publish(self) -> None:
        if self._scene_pub.get_subscription_count() == 0:
            self.get_logger().info(
                "waiting for a /planning_scene subscriber (is move_group up?) ...",
                throttle_duration_sec=5.0)
            return
        diff = PlanningScene()
        diff.is_diff = True
        diff.robot_state.is_diff = True
        diff.world.collision_objects = self._objects
        self._scene_pub.publish(diff)
        self._timer.cancel()
        self.get_logger().info(
            f"published {len(self._objects)} object(s) to /planning_scene — done, idling")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
