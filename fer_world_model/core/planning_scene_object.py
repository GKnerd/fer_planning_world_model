"""
Pure-Python PlanningScene model data types. 
This module is the source-of-truth schema for a manipulable object.`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    z: float

@dataclass(frozen=True)
class Cylinder:
    height: float
    radius: float

@dataclass(frozen=True)
class Cone:
    height: float
    radius: float

@dataclass(frozen=True)
class Sphere:
    radius: float

@dataclass(frozen=True)
class Mesh:
    resource: str # package:// URI


class ObjectStatus(Enum):
    """
    Where the object sits in its manipulation lifecycle.
    This is the anchor for grasp-verification reconciliation.
    """
    FREE = "free"        # resting in the scene, graspable
    GRASPED = "grasped"  # attached to a gripper (held_by set)


""" 
Minimal geometry. 
Mirrors geometry_msgs field-for-field so the # adapter is a trivial copy, but carries no rclpy dependency.
"""
@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PlanningSceneObject:
    """A single manipulable object owned by the world model."""

    id: str                        # == the MoveIt collision-object id. One identity.
    shape: Box | Sphere | Cylinder | Cone | Mesh
    frame: str                     # reference frame of `pose`; required — the caller must know it
    pose: Pose = field(default_factory=Pose)
    stamp: float | None = None     # seconds; when the last scene diff about it was observed
    status: ObjectStatus = ObjectStatus.FREE
    held_by: str | None = None     # gripper link id when GRASPED, else None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.status is ObjectStatus.GRASPED and not self.held_by:
            raise ValueError(f"{self.id}: GRASPED object must have held_by set")

    def age(self, now: float) -> float | None:
        """Seconds since last update, or None if never stamped."""
        return None if self.stamp is None else now - self.stamp

    def is_stale(self, now: float, ttl: float) -> bool:
        """True if no scene diff has mentioned this object for longer than ttl.

        Staleness is INFORMATION, never a deletion trigger: the object is still
        in the planning scene (the scene is retained state — silence is not
        removal), so the model keeps mirroring it and merely reports it stale.
        Only meaningful if the detector re-asserts what it sees periodically.

        GRASPED objects are exempt: they are occluded inside the gripper by
        construction, so 'not re-observed' is expected, not suspicious."""
        if self.status is ObjectStatus.GRASPED:
            return False
        age = self.age(now)
        return age is not None and age > ttl
