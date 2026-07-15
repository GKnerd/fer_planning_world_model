"""Pure-Python world model: owns the set of PlanningSceneObjects and the rules for
mutating them. No ROS, no MoveIt — fully unit-testable.

The MoveIt planning scene is a *projection* of this model (see
planning_scene_adapter / world_model_server); this class is the source of truth.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from fer_world_model.core.result import WMResult
from fer_world_model.core.planning_scene_object import (
    ObjectSource,
    ObjectStatus,
    Pose,
    PlanningSceneObject,
)

# Sentinel so update_scene_object can tell "argument omitted" from "set to None"
# (held_by is legitimately cleared to None).
_UNSET = object()


class PlanningSceneWorldModel:
    def __init__(self) -> None:
        self._scene: Dict[str, PlanningSceneObject] = {}

    # -- dunder conveniences --------------------------------------------------
    def __len__(self) -> int:
        return len(self._scene)

    def __contains__(self, key: str) -> bool:
        return key in self._scene

    # -- mutation -------------------------------------------------------------
    def add_scene_object(self, obj: PlanningSceneObject, *, overwrite: bool = False) -> WMResult:
        """Insert a new object. Fails if the id already exists unless overwrite."""
        if obj.id in self._scene and not overwrite:
            return WMResult.already_exists(obj.id)
        try:
            obj.validate()
        except ValueError as exc:
            return WMResult.invalid(str(exc), obj.id)
        self._scene[obj.id] = obj
        return WMResult.success(obj.id, f"Object: {obj.id} added to scene.")

    def update_scene_object(
        self,
        key: str,
        *,
        pose: Optional[Pose] = None,
        frame: Optional[str] = None,
        dimensions: Optional[List[float]] = None,
        status: Optional[ObjectStatus] = None,
        held_by=_UNSET,
        stamp: Optional[float] = None,
    ) -> WMResult:
        """Patch selected fields of an existing object. Only non-None args are
        applied (held_by uses a sentinel so it can be cleared to None).
        Enforces the status transition rules that make reconciliation sane."""
        obj = self._scene.get(key)
        if obj is None:
            return WMResult.not_found(key)

        new_held_by = obj.held_by if held_by is _UNSET else held_by
        new_status = status if status is not None else obj.status

        # Transition guards.
        if new_status is ObjectStatus.GRASPED:
            if not new_held_by:
                return WMResult.invalid("GRASPED requires held_by", key)
            if obj.status is ObjectStatus.GRASPED and obj.held_by != new_held_by:
                return WMResult.conflict(
                    f"'{key}' already grasped by '{obj.held_by}'", key)
        else:
            # Leaving/!GRASPED clears the holder unless caller set one explicitly.
            if held_by is _UNSET:
                new_held_by = None

        if pose is not None:
            obj.pose = pose
        if frame is not None:
            obj.frame = frame
        if dimensions is not None:
            obj.dimensions = dimensions
        obj.status = new_status
        obj.held_by = new_held_by
        if stamp is not None:
            obj.stamp = stamp

        try:
            obj.validate()
        except ValueError as exc:
            return WMResult.invalid(str(exc), key)
        return WMResult.success(key, f"Object: {obj.id} updated.")


    def rm_scene_object(self, key: str) -> WMResult:
        if key not in self._scene:
            return WMResult.not_found(key)
        obj_to_rm = self._scene[key].id
        del self._scene[key]
        return WMResult.success(key,  f"Object: {obj_to_rm} removed")

    def purge_scene(self) -> None:
        self._scene.clear()

    def prune_stale(self, now: float, ttl: float) -> List[str]:
        """Drop perceived objects older than ttl. Returns removed ids so the
        caller can emit matching REMOVE ops to the planning scene."""
        stale = [k for k, o in self._scene.items() if o.is_stale(now, ttl)]
        for k in stale:
            del self._scene[k]
        return stale

    # -- read-only views ------------------------------------------------------
    def get_scene_object(self, key: str) -> Optional[PlanningSceneObject]:
        return self._scene.get(key)

    def filter_scene_objects(
        self, 
        predicate: Optional[Callable[[PlanningSceneObject], bool]] = None
    ) -> Dict[str, PlanningSceneObject]:
        """Return the subset matching predicate (all objects if predicate None)."""
        if predicate is None:
            return dict(self._scene)
        return {k: o for k, o in self._scene.items() if predicate(o)}

    def by_status(self, status: ObjectStatus) -> Dict[str, PlanningSceneObject]:
        return self.filter_scene_objects(lambda o: o.status is status)

    def by_source(self, source: ObjectSource) -> Dict[str, PlanningSceneObject]:
        return self.filter_scene_objects(lambda o: o.source is source)

    def snapshot(self) -> List[PlanningSceneObject]:
        """Flat list of all objects — the input to a full scene reconcile."""
        return list(self._scene.values())
