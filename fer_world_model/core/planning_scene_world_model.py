"""Pure-Python world model: a read-only observer's view of the MoveIt planning
scene, enriched with what the scene structurally cannot hold — object lifecycle
status (FREE/GRASPED), last-observed stamps, and grasp-conflict flags.

The model NEVER writes the planning scene. Perception and MTC write the scene
directly; this class is fed exclusively by *observations* of scene diffs (see
the server's /monitored_planning_scene subscription). Hence the API is
observe_*: each method folds one observed scene event into the model's view.

No ROS, no MoveIt — fully unit-testable.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from fer_world_model.core.result import WMResult
from fer_world_model.core.planning_scene_object import (
    ObjectStatus,
    Pose,
    PlanningSceneObject,
)


class PlanningSceneWorldModel:
    def __init__(self) -> None:
        self._scene: Dict[str, PlanningSceneObject] = {}
        # id -> description of the last observation the GRASPED guard refused.
        # A world update arriving for a grasped object means the camera saw it
        # somewhere in the world while the scene claims it is in the gripper —
        # phantom-grasp evidence, kept for the orchestrator, never dropped.
        self._conflicts: Dict[str, str] = {}

    # -- dunder conveniences --------------------------------------------------
    def __len__(self) -> int:
        return len(self._scene)

    def __contains__(self, key: str) -> bool:
        return key in self._scene

    # -- observations (the only mutation path) --------------------------------
    def observe_object(self, obj: PlanningSceneObject) -> WMResult:
        """Fold an observed world ADD/APPEND: upsert the object as FREE.

        GRASPED guard: if the model holds this id as GRASPED, the update is
        refused and recorded as a conflict — the scene's attachment claim wins
        until a detach is observed."""
        try:
            obj.validate()
        except ValueError as exc:
            return WMResult.invalid(str(exc), obj.id)

        existing = self._scene.get(obj.id)
        if existing is not None and existing.status is ObjectStatus.GRASPED:
            return self._flag_conflict(
                existing, f"world object update at t={obj.stamp}")

        self._scene[obj.id] = obj
        return WMResult.success(obj.id, f"Object: {obj.id} observed.")

    def observe_moved(self, key: str, pose: Pose, stamp: float) -> WMResult:
        """Fold an observed world MOVE (pose only — MOVE diffs carry no
        geometry). Same GRASPED guard as observe_object."""
        obj = self._scene.get(key)
        if obj is None:
            return WMResult.not_found(key)
        if obj.status is ObjectStatus.GRASPED:
            return self._flag_conflict(obj, f"world MOVE at t={stamp}")
        obj.pose = pose
        obj.stamp = stamp
        return WMResult.success(key, f"Object: {key} moved.")

    def observe_removed(self, key: str) -> WMResult:
        """Fold an observed world REMOVE."""
        if key not in self._scene:
            return WMResult.not_found(key)
        del self._scene[key]
        self._conflicts.pop(key, None)
        return WMResult.success(key, f"Object: {key} removed.")

    def observe_world_cleared(self) -> List[str]:
        """Fold an observed REMOVE with an empty id — MoveIt's 'clear all world
        objects'. Drops FREE objects only: GRASPED objects live in the robot
        state, not the world, so a world clear does not touch them."""
        removed = [k for k, o in self._scene.items()
                   if o.status is ObjectStatus.FREE]
        for k in removed:
            del self._scene[k]
            self._conflicts.pop(k, None)
        return removed

    def observe_attached(
        self,
        key: str,
        link: str,
        stamp: float,
        fallback: Optional[PlanningSceneObject] = None,
    ) -> WMResult:
        """Fold an observed attach: transition to GRASPED, held by `link`.

        Attach diffs may carry only the object id (MoveIt moves an existing
        world object into the robot state); if the id is unknown to the model
        and the diff did carry geometry, the caller passes it as `fallback` so
        the object can be adopted."""
        obj = self._scene.get(key)
        if obj is None:
            if fallback is None:
                return WMResult.not_found(key)
            obj = fallback
            self._scene[key] = obj
        if obj.status is ObjectStatus.GRASPED and obj.held_by != link:
            return WMResult.conflict(
                f"'{key}' already grasped by '{obj.held_by}', "
                f"scene now attaches it to '{link}'", key)
        obj.status = ObjectStatus.GRASPED
        obj.held_by = link
        obj.stamp = stamp
        return WMResult.success(key, f"Object: {key} grasped by '{link}'.")

    def observe_detached(self, key: str, stamp: float) -> WMResult:
        """Fold an observed detach: transition to FREE. The object's world pose
        arrives separately as a world diff (MoveIt re-adds detached objects to
        the world), so only the status changes here."""
        obj = self._scene.get(key)
        if obj is None:
            return WMResult.not_found(key)
        obj.status = ObjectStatus.FREE
        obj.held_by = None
        obj.stamp = stamp
        self._conflicts.pop(key, None)
        return WMResult.success(key, f"Object: {key} detached.")

    def rebuild(self, objects: List[PlanningSceneObject]) -> None:
        """Replace the whole view with a full-scene snapshot (startup fetch or
        an is_diff=False message). Conflicts are cleared: they described a
        history the snapshot supersedes."""
        self._scene = {o.id: o for o in objects}
        self._conflicts.clear()

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

    def snapshot(self) -> List[PlanningSceneObject]:
        """Flat list of all objects the model currently mirrors."""
        return list(self._scene.values())

    def conflicts(self) -> Dict[str, str]:
        """id -> description of the last guarded (refused) observation."""
        return dict(self._conflicts)

    def stale_ids(self, now: float, ttl: float) -> List[str]:
        """Objects no diff has mentioned within ttl (GRASPED exempt). Pure
        information for the caller — nothing is removed."""
        return [k for k, o in self._scene.items() if o.is_stale(now, ttl)]

    # -- internals -------------------------------------------------------------
    def _flag_conflict(self, obj: PlanningSceneObject, event: str) -> WMResult:
        msg = (f"{event} while model holds '{obj.id}' GRASPED by "
               f"'{obj.held_by}' — possible phantom grasp; update refused")
        self._conflicts[obj.id] = msg
        return WMResult.conflict(msg, obj.id)
