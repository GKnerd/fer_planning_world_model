"""Return types for PlanningSceneWorldModel operations.

Every mutating PlanningSceneWorldModel method returns a WMResult instead of raising or
returning a bare bool, so callers get a machine-readable 
status + a human message. 
WMStatus maps cleanly onto a service response: `success = result.ok`, `message = result.message`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WMStatus(Enum):
    OK = "ok"
    NOT_FOUND = "not_found"            # no object with that id
    ALREADY_EXISTS = "already_exists"  # add() on an id already present
    INVALID = "invalid"               # malformed request (bad dims, missing held_by, ...)
    CONFLICT = "conflict"             # op contradicts current state (e.g. re-grasp)


@dataclass(frozen=True)
class WMResult:
    status: WMStatus
    message: str = ""
    key: str | None = None  # object id the operation concerned, when relevant

    @property
    def ok(self) -> bool:
        return self.status is WMStatus.OK

    def __bool__(self) -> bool:
        return self.ok

    # --- convenience constructors -------------------------------------------
    @classmethod
    def success(cls, key: str | None = None, message: str = "") -> "WMResult":
        return cls(WMStatus.OK, message or "ok", key)

    @classmethod
    def not_found(cls, key: str) -> "WMResult":
        return cls(WMStatus.NOT_FOUND, f"no object with id '{key}'", key)

    @classmethod
    def already_exists(cls, key: str) -> "WMResult":
        return cls(WMStatus.ALREADY_EXISTS, f"object '{key}' already exists", key)

    @classmethod
    def invalid(cls, message: str, key: str | None = None) -> "WMResult":
        return cls(WMStatus.INVALID, message, key)

    @classmethod
    def conflict(cls, message: str, key: str | None = None) -> "WMResult":
        return cls(WMStatus.CONFLICT, message, key)
