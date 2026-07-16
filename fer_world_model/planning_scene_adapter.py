"""Translation layer between the pure world model and MoveIt messages.

Kept separate from both the pure model (which stays ROS-free) and the node
(which owns the ROS graph). Nothing here touches a clock or a topic — it only
converts data, so it is still straightforward to test.

Direction of travel: the world model is a READ-ONLY observer of the planning
scene, so the main road here is ROS -> pure-model (scene messages reduced to
PlanningSceneObjects the model can fold in). The one pure-model -> ROS export
that remains is to_collision_object, used to publish the model's own view on
~/world_state — a plain introspection topic, never a scene write.

Ownership of attachment:
    - Only MTC writes attachment (and the ACM entries that let the fingers
      touch the object). Perception writes world objects. The world model
      writes NOTHING to the scene — it observes attach/detach diffs and tracks
      FREE/GRASPED accordingly.
    - Repairing a phantom grasp (scene says attached, gripper provably empty)
      therefore belongs to a dedicated repair skill dispatched by the
      orchestrator: detach the ghost (which removes ACM entries, so no
      touch_links needed) and re-add the object at the model's remembered
      pose. That skill, not this package, owns that write.
"""

from __future__ import annotations

from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Pose as RosPose
from geometry_msgs.msg import Quaternion as RosQuaternion
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

from fer_world_model.core.planning_scene_object import (
    ObjectStatus,
    Point,
    Pose,
    Quaternion,
    PlanningSceneObject,
    Box,
    Cylinder,
    Sphere,
    Cone,
    Mesh
)


class PlanningSceneAdapter:
    def __init__(self) -> None:
        pass

    # -- pure-model -> ROS ----------------------------------------------------
    @staticmethod
    def to_ros_pose(pose: Pose) -> RosPose:
        return RosPose(
            position=RosPoint(x=pose.position.x, y=pose.position.y, z=pose.position.z),
            orientation=RosQuaternion(
                x=pose.orientation.x, y=pose.orientation.y,
                z=pose.orientation.z, w=pose.orientation.w),
        )

    def to_moveit_shape(self, shape: Box | Sphere | Cylinder | Cone | Mesh) -> SolidPrimitive:
        match shape:
            case Box(x, y, z):
                return SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[x, y, z])
            case Sphere(r):
                return SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[r])
            case Cylinder(h, r):
                return SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[h, r])
            case Cone(h, r):
                return SolidPrimitive(type=SolidPrimitive.CONE, dimensions=[h, r])
            case Mesh():
                raise NotImplementedError("Meshes are not currently supported. WIP.")
            case _:
                raise ValueError(f"Unsupported Shape for: {type(shape).__name__}")

    def to_collision_object(
            self,
            obj: PlanningSceneObject,
            operation: int = CollisionObject.ADD
    ) -> CollisionObject:

        co = CollisionObject()
        co.header.frame_id = obj.frame
        co.id = obj.id
        co.operation = operation

        if operation != CollisionObject.REMOVE:
            co.pose = self.to_ros_pose(obj.pose)          # object origin in obj.frame
            co.primitives = [self.to_moveit_shape(shape=obj.shape)]
            co.primitive_poses = [RosPose()]              # primitive at object origin

        return co

    # -- ROS -> pure-model ----------------------------------------------------
    @staticmethod
    def from_ros_pose(pose: RosPose) -> Pose:
        return Pose(
            position=Point(x=pose.position.x, y=pose.position.y, z=pose.position.z),
            orientation=Quaternion(
                x=pose.orientation.x, y=pose.orientation.y,
                z=pose.orientation.z, w=pose.orientation.w),
        )

    def from_moveit_to_primitive(self, primitive: SolidPrimitive) -> Box | Sphere | Cylinder | Cone:
        """SolidPrimitive -> core shape.

        This is the boundary: incoming messages are untrusted, so they are checked here.
        Once it is a Box/Cylinder/... it is correct by
        construction and the core never has to re-validate.
        """
        d = primitive.dimensions

        def need(n: int) -> None:
            if len(d) != n:
                raise ValueError(
                    f"SolidPrimitive type {primitive.type} needs {n} dimension(s), "
                    f"got {len(d)}: {list(d)}")

        match primitive.type:
            case SolidPrimitive.BOX:
                need(3)
                return Box(x=d[SolidPrimitive.BOX_X],
                           y=d[SolidPrimitive.BOX_Y],
                           z=d[SolidPrimitive.BOX_Z])
            case SolidPrimitive.SPHERE:
                need(1)
                return Sphere(radius=d[SolidPrimitive.SPHERE_RADIUS])
            case SolidPrimitive.CYLINDER:
                need(2)
                return Cylinder(height=d[SolidPrimitive.CYLINDER_HEIGHT],
                                radius=d[SolidPrimitive.CYLINDER_RADIUS])
            case SolidPrimitive.CONE:
                need(2)
                return Cone(height=d[SolidPrimitive.CONE_HEIGHT],
                            radius=d[SolidPrimitive.CONE_RADIUS])
            case _:
                raise ValueError(f"unsupported SolidPrimitive type: {primitive.type}")

    def to_scene_object(
        self,
        co: CollisionObject,
        stamp: float,
        status: ObjectStatus = ObjectStatus.FREE,
        held_by: str | None = None,
    ) -> PlanningSceneObject:
        """CollisionObject -> PlanningSceneObject, for folding observed scene
        content into the model.

        Raises ValueError on messages the model cannot represent (no/multiple
        primitives, bad dimensions) — the caller decides whether to skip or
        complain. MOVE diffs never come through here (they carry no geometry);
        see PlanningSceneWorldModel.observe_moved.
        """
        if not co.primitives:
            raise ValueError(f"'{co.id}': collision object carries no primitives")
        if len(co.primitives) > 1:
            raise ValueError(
                f"'{co.id}': {len(co.primitives)} primitives — the model "
                f"represents single-primitive objects only")
        return PlanningSceneObject(
            id=co.id,
            shape=self.from_moveit_to_primitive(co.primitives[0]),
            frame=co.header.frame_id,
            pose=self.from_ros_pose(co.pose),
            stamp=stamp,
            status=status,
            held_by=held_by,
        )
