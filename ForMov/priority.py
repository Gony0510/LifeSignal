from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from ForMov.config import DANGER_HOLD_SEC, SLEEPING_DANGER_AFTER_SEC
except ModuleNotFoundError:
    from config import DANGER_HOLD_SEC, SLEEPING_DANGER_AFTER_SEC  # type: ignore


@dataclass
class PriorityState:
    sleeping_since: float | None = None
    danger_until: float = 0.0
    danger_reason: str | None = None


class RescuePriorityEngine:
    """영상 시나리오 판정 결과를 구조 우선순위로 변환합니다."""

    def __init__(
        self,
        *,
        sleeping_danger_after: float = SLEEPING_DANGER_AFTER_SEC,
        danger_hold: float = DANGER_HOLD_SEC,
    ) -> None:
        self.sleeping_danger_after = sleeping_danger_after
        self.danger_hold = danger_hold
        self.states: dict[str, PriorityState] = {}

    def evaluate(
        self,
        key: str,
        scenario: str,
        *,
        now: float,
        observed_since: float | None = None,
    ) -> dict[str, object]:
        state = self.states.setdefault(key, PriorityState())

        if scenario == "fallen":
            state.sleeping_since = None
            state.danger_until = max(state.danger_until, now + self.danger_hold)
            state.danger_reason = "fallen_detected"
        elif scenario == "sleeping":
            if state.sleeping_since is None:
                state.sleeping_since = observed_since if observed_since is not None else now
            sleeping_elapsed = max(0.0, now - state.sleeping_since)
            if sleeping_elapsed >= self.sleeping_danger_after:
                state.danger_until = max(state.danger_until, now + self.danger_hold)
                state.danger_reason = "sleeping_over_threshold"
        else:
            state.sleeping_since = None

        if now < state.danger_until:
            if state.danger_reason == "fallen_detected":
                label = "위험 · 쓰러짐 추정"
            else:
                threshold = f"{self.sleeping_danger_after:g}"
                label = f"위험 · {threshold}초 이상 대피 움직임 없음"
            return self._result(
                level="danger",
                rank=1,
                label=label,
                reason=state.danger_reason or "danger_hold",
                hold_remaining=max(0.0, state.danger_until - now),
            )

        state.danger_reason = None
        if scenario == "sleeping":
            sleeping_since = state.sleeping_since
            elapsed = max(
                0.0,
                now - sleeping_since if sleeping_since is not None else 0.0,
            )
            return self._result(
                level="normal",
                rank=2,
                label="보통 · 누운 상태 관찰 중",
                reason="sleeping_observed",
                sleeping_elapsed=elapsed,
            )
        if scenario == "evacuating":
            return self._result(
                level="normal",
                rank=2,
                label="보통 · 자력 대피 중",
                reason="evacuating",
            )
        if scenario == "no_signal":
            return self._result(
                level="none",
                rank=3,
                label="구조 대상 없음",
                reason="no_signal",
            )
        return self.pending("unknown_scenario")

    @staticmethod
    def pending(reason: str = "ai_pending") -> dict[str, object]:
        return {
            "level": "pending",
            "rank": None,
            "label_ko": "AI 분석 대기",
            "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _result(
        *,
        level: str,
        rank: int,
        label: str,
        reason: str,
        **details: float,
    ) -> dict[str, object]:
        return {
            "level": level,
            "rank": rank,
            "label_ko": label,
            "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **{name: round(value, 2) for name, value in details.items()},
        }
