from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any


HUMAN_HISTORY_SEC = 15.0
HUMAN_NO_MOTION_DANGER_SEC = 4.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ZoneSurvivorState:
    first_human_at: float | None = None
    last_human_at: float | None = None
    motion_started_at: float | None = None
    no_motion_started_at: float | None = None


class SurvivorStateTracker:
    """센서 영역별 사람 감지와 호흡 모션 지속 시간을 독립 관리합니다."""

    def __init__(
        self,
        *,
        human_history_sec: float = HUMAN_HISTORY_SEC,
        human_risk_after_sec: float = HUMAN_NO_MOTION_DANGER_SEC,
    ) -> None:
        if human_history_sec <= 0:
            raise ValueError("사람 감지 이력 유지 시간은 0보다 커야 합니다.")
        if human_risk_after_sec <= 0:
            raise ValueError("사람 위험 전환 시간은 0보다 커야 합니다.")
        self.human_history_sec = human_history_sec
        self.human_risk_after_sec = human_risk_after_sec
        self.states: dict[str, ZoneSurvivorState] = {}

    @staticmethod
    def sensor_key(data: dict[str, Any]) -> str:
        return (
            f"{data.get('sensor', 'unknown')}:"
            f"{data.get('room')}:{data.get('location')}"
        )

    @staticmethod
    def _motion_active(data: dict[str, Any]) -> bool:
        """명시적인 motion을 우선하고, 없는 센서는 status를 대체값으로 씁니다."""
        motion = data.get("motion")
        if isinstance(motion, bool):
            return motion
        if isinstance(motion, str):
            normalized = motion.strip().lower()
            if normalized in {"true", "1", "yes", "motion"}:
                return True
            if normalized in {"false", "0", "no", "no_motion"}:
                return False
        return data.get("status") is True

    def enrich(
        self,
        data: dict[str, Any],
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        if data.get("type") != "radar_data":
            return data

        current_time = monotonic() if now is None else now
        key = self.sensor_key(data)
        state = self.states.setdefault(key, ZoneSurvivorState())
        ai = data.get("ai") if isinstance(data.get("ai"), dict) else {}
        target = str(ai.get("target") or "").strip().lower()
        human_now = bool(ai.get("ready")) and target == "human"

        if human_now:
            if state.first_human_at is None:
                state.first_human_at = current_time
            state.last_human_at = current_time

        human_recent = (
            state.last_human_at is not None
            and current_time - state.last_human_at <= self.human_history_sec
        )
        motion_active = self._motion_active(data) and (human_now or human_recent)

        if motion_active:
            if state.motion_started_at is None:
                state.motion_started_at = current_time
            state.no_motion_started_at = None
        else:
            state.motion_started_at = None
            if human_recent and state.no_motion_started_at is None:
                state.no_motion_started_at = current_time
            if not human_recent:
                state.no_motion_started_at = None

        if not human_recent and target in {"empty", "dog", "pet"}:
            state.first_human_at = None

        human_observed_sec = (
            max(0.0, current_time - state.first_human_at)
            if human_recent and state.first_human_at is not None
            else 0.0
        )
        motion_duration_sec = (
            max(0.0, current_time - state.motion_started_at)
            if motion_active and state.motion_started_at is not None
            else 0.0
        )
        no_motion_duration_sec = (
            max(0.0, current_time - state.no_motion_started_at)
            if human_recent and state.no_motion_started_at is not None
            else 0.0
        )

        human_risk: dict[str, Any] | None = None
        if human_recent:
            danger = no_motion_duration_sec >= self.human_risk_after_sec
            human_risk = {
                "level": "danger" if danger else "normal",
                "is_danger": danger,
                "score": round(
                    min(no_motion_duration_sec / self.human_risk_after_sec, 1.0),
                    4,
                ),
                "no_motion_duration_sec": round(no_motion_duration_sec, 2),
                "threshold_sec": self.human_risk_after_sec,
                "reason": "no_motion_timeout" if danger else "motion_observed",
            }

        enriched = dict(data)
        enriched["survivor_state"] = {
            "sensor_key": key,
            "human_recent": human_recent,
            "motion_active": motion_active,
            "human_observed_sec": round(human_observed_sec, 2),
            "motion_duration_sec": round(motion_duration_sec, 2),
            "no_motion_duration_sec": round(no_motion_duration_sec, 2),
            "updated_at": data.get("timestamp") or _utc_now(),
        }
        if human_risk is not None:
            enriched["human_risk"] = human_risk
        else:
            enriched.pop("human_risk", None)
        return enriched

    def clear(self) -> None:
        self.states.clear()
