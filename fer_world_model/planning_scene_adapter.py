"""Translation layer between the pure world model and MoveIt messages.

Kept separate from both the pure model (which stays ROS-free) and the node
(which owns the ROS graph). Nothing here touches a clock or a topic — it only
converts data, so it is still straightforward to test.

Design note: we only ever emit **diffs** (PlanningScene.is_diff = True) with
explicit ADD/REMOVE ops. We never send a full (is_diff = False) scene, because
that would wipe the pre-registered static environment (table, fixtures) that the
world model does NOT own.

Ownership of attachment:
    - Only MTC attaches an object to the gripper aka it modifies the planning scene. It does so as 
    part of executing a pick, together with the ACM entries (allowCollisions) that let the fingers
    touch the object without it counting as a collision. 
    - The world model only reads attachment. It adopts the scene's view into ObjectStatus.GRASPED and 
    the single write it may perform is a detach (see detach_diff), which removes ACM entries rather 
    than creating them. 
    - The model can never re-create an attachment.
"""

from __future__ import annotations

from typing import Iterable

from geometry_msgs.msg import Point as RosPoint
from geometry_msgs.msg import Pose as RosPose
from geometry_msgs.msg import Quaternion as RosQuaternion
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive

from fer_world_model.core.planning_scene_object import (
    ObjectStatus,
    Pose,
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


    # -- ROS -> pure-model ----------------------------------------------------
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

    # -- pure-model -> ROS ----------------------------------------------------
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

    @staticmethod
    def remove_object(object_id: str) -> CollisionObject:
        co = CollisionObject()
        co.id = object_id
        co.operation = CollisionObject.REMOVE
        return co

    # -- scene diffs ----------------------------------------------------------
    def build_diff(
        self,
        add_objects: Iterable[PlanningSceneObject] = (),
        remove_ids: Iterable[str] = (),
    ) -> PlanningScene:
        """One atomic diff: FREE objects are asserted as world collision objects,
        remove_ids are deleted from the world.

        GRASPED objects are deliberately SKIPPED. MTC attaches/detaches as part of
        executing a pick/place and owns both the attachment and its ACM entries
        (allowCollisions), so re-asserting a grasped object here would fight it:
        we would re-add the object to the *world* at its stale pose while MTC holds
        it attached to the hand. The model still tracks GRASPED — it just does not
        project it. See detach_diff() for the one case we do write attachment.
        """
        scene = PlanningScene()
        scene.is_diff = True
        # Even though we no longer touch robot_state, this must stay True: an
        # empty robot_state with is_diff=False reads as a full, empty robot state
        # rather than "no change".
        scene.robot_state.is_diff = True

        for obj in add_objects:
            if obj.status is ObjectStatus.GRASPED:
                continue  # MTC owns this object's scene representation
            scene.world.collision_objects.append(
                self.to_collision_object(obj, CollisionObject.ADD))

        for oid in remove_ids:
            scene.world.collision_objects.append(self.remove_object(oid))

        return scene

    def detach_diff(self, obj: PlanningSceneObject, link: str) -> PlanningScene:
        """Detach a previously-grasped object and re-add it to the world at its
        current pose. This is the REPAIR path (GRASPED -> FREE), used when the
        scene says the object is attached but grasp verification says the gripper
        is empty (the phantom grasp).

        This is the only place the model writes attachment state, and it only ever
        *removes* one: detaching drops ACM entries rather than creating them, so it
        needs no touch_links. The model can never re-create an attachment.
        """
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True

        detach = AttachedCollisionObject()
        detach.link_name = link
        detach.object = self.remove_object(obj.id)
        scene.robot_state.attached_collision_objects.append(detach)

        scene.world.collision_objects.append(
            self.to_collision_object(obj, CollisionObject.ADD))
        return scene
