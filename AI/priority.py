from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


DANGER_LEVELS = {"danger", "critical", "high", "urgent", "위험", "긴급"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _human_risk(human_risk: object) -> tuple[bool, float | None]:
    if isinstance(human_risk, str):
        return human_risk.strip().lower() in DANGER_LEVELS, None
    if isinstance(human_risk, bool):
        return human_risk, None
    if not isinstance(human_risk, dict):
        return False, None

    level = str(human_risk.get("level", "")).strip().lower()
    is_danger = bool(human_risk.get("is_danger")) or level in DANGER_LEVELS
    score_value = human_risk.get("score")
    try:
        score = float(score_value)
    except (TypeError, ValueError):
        score = None
    if score is not None and not 0.0 <= score <= 1.0:
        score = None
    if score is not None and score >= 0.7:
        is_danger = True
    return is_danger, score


def build_rescue_priority(
    ai_result: dict[str, Any],
    *,
    human_risk: object = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    timestamp = updated_at or ai_result.get("updated_at") or _utc_now()
    if not ai_result.get("ready"):
        if ai_result.get("reason") == "no_target":
            return {
                "level": "none",
                "rank": 0,
                "label_ko": "구조 대상 없음",
                "target_type": "empty",
                "reason_codes": ["no_sensor_target"],
                "updated_at": timestamp,
            }
        return {
            "level": "pending",
            "rank": None,
            "label_ko": "AI 분석 대기",
            "target_type": None,
            "reason_codes": [str(ai_result.get("reason", "ai_pending"))],
            "updated_at": timestamp,
        }

    target = str(ai_result.get("target", "")).lower()
    if target == "empty":
        return {
            "level": "none",
            "rank": 0,
            "label_ko": "구조 대상 없음",
            "target_type": "empty",
            "reason_codes": ["empty_detected"],
            "updated_at": timestamp,
        }
    if target in {"dog", "pet"}:
        return {
            "level": "low",
            "rank": 1,
            "label_ko": "낮음 · 반려동물",
            "target_type": "dog",
            "reason_codes": ["pet_detected"],
            "updated_at": timestamp,
        }
    if target == "human":
        danger, risk_score = _human_risk(human_risk)
        result = {
            "level": "danger" if danger else "normal",
            "rank": 3 if danger else 2,
            "label_ko": "위험 · 사람" if danger else "보통 · 사람",
            "target_type": "human",
            "reason_codes": [
                "human_detected",
                "human_risk_detected" if danger else "human_default_priority",
            ],
            "updated_at": timestamp,
        }
        if risk_score is not None:
            result["risk_score"] = round(risk_score, 4)
        return result

    return {
        "level": "pending",
        "rank": None,
        "label_ko": "AI 판정 대기",
        "target_type": target or None,
        "reason_codes": ["unknown_ai_target"],
        "updated_at": timestamp,
    }
