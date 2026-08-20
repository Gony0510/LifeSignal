from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from aiohttp import WSMsgType, web

from AI.inference import SensorAIEngine, load_classifier
from AI.priority import build_rescue_priority
from AI.survivor_state import SurvivorStateTracker


connected_clients: set[web.WebSocketResponse] = set()
dashboard_clients: set[web.WebSocketResponse] = set()
latest_sensor_states: dict[str, str] = {}
ai_engine = SensorAIEngine(None, update_interval=5.0)
sensor_ai_engines: dict[str, SensorAIEngine] = {}
survivor_state_tracker = SurvivorStateTracker()
DETECTION_HOLD_SECONDS = 1.0
HTML_PATH = Path(__file__).with_name("LifeSignal.html")
ARTIFACT_DIR = Path(__file__).with_name("AI") / "artifacts"
DEFAULT_MODEL_CANDIDATES = {
    "vpr100": (
        ARTIFACT_DIR / "vpr100-svm.joblib",
        ARTIFACT_DIR / "vpr100-cnn.keras",
    ),
    "c4001": (
        ARTIFACT_DIR / "c4001-svm.joblib",
        ARTIFACT_DIR / "c4001-cnn.keras",
    ),
}


def sensor_key(data: dict) -> str | None:
    if data.get("type") != "radar_data":
        return None
    return (
        f"{data.get('sensor', 'unknown')}:"
        f"{data.get('room')}:{data.get('location')}"
    )


def sensor_family(data: dict) -> str | None:
    sensor_name = str(data.get("sensor", "")).strip().lower()
    if "c4001" in sensor_name:
        return "c4001"
    compact_name = sensor_name.replace("-", "").replace("_", "")
    if "vpr100" in compact_name:
        return "vpr100"
    return None


def select_ai_engine(data: dict) -> SensorAIEngine:
    family = sensor_family(data)
    if family is not None and family in sensor_ai_engines:
        return sensor_ai_engines[family]
    return ai_engine


class DetectionHoldFilter:
    """순간적인 감지 해제를 센서·위치별로 짧게 유예합니다."""

    def __init__(self, hold_seconds: float = DETECTION_HOLD_SECONDS) -> None:
        self.hold_seconds = hold_seconds
        self.last_detected_at: dict[str, float] = {}

    def apply(self, data: dict, *, now: float | None = None) -> dict:
        key = sensor_key(data)
        if (
            key is None
            or sensor_family(data) not in {"vpr100", "c4001"}
            or not isinstance(data.get("status"), bool)
        ):
            return data

        current_time = monotonic() if now is None else now
        stabilized = dict(data)
        if data["status"]:
            self.last_detected_at[key] = current_time
            return stabilized

        last_detected = self.last_detected_at.get(key)
        if (
            last_detected is not None
            and current_time - last_detected < self.hold_seconds
        ):
            stabilized["status"] = True
        else:
            self.last_detected_at.pop(key, None)
        return stabilized


detection_hold_filter = DetectionHoldFilter()


def discover_default_models(
    candidates: dict[str, tuple[Path, ...]] | None = None,
) -> dict[str, Path]:
    """학습 명령의 기본 출력 위치에서 센서별 모델을 찾습니다."""
    discovered: dict[str, Path] = {}
    for family, paths in (candidates or DEFAULT_MODEL_CANDIDATES).items():
        for path in paths:
            if path.exists():
                discovered[family] = path
                break
    return discovered


def process_message(message: str) -> tuple[str, str | None]:
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return message, None
    if not isinstance(data, dict):
        return message, None

    key = sensor_key(data)
    if key is None:
        return message, None

    stabilized = detection_hold_filter.apply(data)
    enriched = select_ai_engine(stabilized).enrich(stabilized)
    enriched = survivor_state_tracker.enrich(enriched)
    ai_data = enriched.get("ai")
    enriched["rescue_priority"] = build_rescue_priority(
        ai_data if isinstance(ai_data, dict) else {},
        human_risk=enriched.get("human_risk"),
    )
    return json.dumps(enriched, ensure_ascii=False), key


def is_dashboard_subscription(message: str) -> bool:
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(data, dict)
        and data.get("type") == "subscribe"
        and data.get("role") == "dashboard"
    )


async def subscribe_dashboard(websocket: web.WebSocketResponse) -> None:
    dashboard_clients.add(websocket)
    for latest_message in latest_sensor_states.values():
        await websocket.send_str(latest_message)


async def broadcast_to_dashboards(message: str) -> None:
    stale_clients: list[web.WebSocketResponse] = []
    for client in dashboard_clients:
        if client.closed:
            stale_clients.append(client)
            continue
        try:
            await client.send_str(message)
        except (ConnectionError, RuntimeError):
            stale_clients.append(client)
    for client in stale_clients:
        dashboard_clients.discard(client)
        connected_clients.discard(client)


def print_sensor_summary(message: str) -> None:
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        print(f"수신한 데이터: {message}")
        return

    ai_data = data.get("ai") or {}
    if ai_data.get("ready"):
        ai_text = (
            f"{ai_data.get('target_ko')} "
            f"{float(ai_data.get('confidence', 0)) * 100:.1f}%"
        )
    else:
        ai_text = ai_data.get("reason", "AI 대기")
    print(
        "센서 수신/AI 처리: "
        f"{data.get('room')}호 {data.get('location')} "
        f"감지={data.get('status')} AI={ai_text}"
    )


async def websocket_handler(
    request: web.Request,
    websocket: web.WebSocketResponse,
) -> web.WebSocketResponse:
    await websocket.prepare(request)
    connected_clients.add(websocket)
    print(f"새로운 기기 연결됨 (현재 {len(connected_clients)}대)")

    try:
        async for incoming in websocket:
            if incoming.type == WSMsgType.TEXT:
                if is_dashboard_subscription(incoming.data):
                    await subscribe_dashboard(websocket)
                    print(
                        "대시보드 연결됨 "
                        f"(현재 {len(dashboard_clients)}대)"
                    )
                    continue

                outgoing_message, key = process_message(incoming.data)
                if key is not None:
                    latest_sensor_states[key] = outgoing_message
                    print_sensor_summary(outgoing_message)
                    await broadcast_to_dashboards(outgoing_message)
                else:
                    print(f"수신한 데이터: {incoming.data}")
            elif incoming.type == WSMsgType.ERROR:
                print(f"WebSocket 오류: {websocket.exception()}")
    finally:
        dashboard_clients.discard(websocket)
        connected_clients.discard(websocket)
        print(f"기기 해제됨 (현재 {len(connected_clients)}대)")
    return websocket


async def http_or_websocket(request: web.Request) -> web.StreamResponse:
    websocket = web.WebSocketResponse(
        protocols=("arduino",),
        heartbeat=30,
    )
    if websocket.can_prepare(request).ok:
        return await websocket_handler(request, websocket)

    if request.path in {"/", "/LifeSignal.html"}:
        return web.FileResponse(
            HTML_PATH,
            headers={"Cache-Control": "no-store"},
        )
    if request.path == "/favicon.ico":
        return web.Response(status=204)
    raise web.HTTPNotFound(text="LifeSignal 페이지를 찾을 수 없습니다.")


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/{path:.*}", http_or_websocket)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LifeSignal HTML·WebSocket 중앙 관제 및 AI 추론 서버"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8881)
    parser.add_argument(
        "--ai-model",
        help="학습된 SVM(.joblib) 또는 1D CNN(.keras) 모델 경로",
    )
    parser.add_argument("--vpr100-model", help="V-PR100 전용 AI 모델 경로")
    parser.add_argument("--c4001-model", help="C4001 전용 AI 모델 경로")
    parser.add_argument(
        "--no-auto-models",
        action="store_true",
        help="AI/artifacts의 기본 센서 모델 자동 탐색을 비활성화",
    )
    parser.add_argument(
        "--ai-model-type",
        choices=["auto", "svm", "cnn", "1d-cnn"],
        default="auto",
    )
    parser.add_argument(
        "--ai-update-interval",
        type=float,
        default=5.0,
        help="사람/빈 공간/개 AI 판정 갱신 주기(초)",
    )
    parser.add_argument(
        "--human-risk-after",
        type=float,
        default=4.0,
        help="사람의 움직임 중단 후 위험으로 전환하는 시간(초)",
    )
    return parser.parse_args()


def configure_ai(args: argparse.Namespace) -> None:
    global ai_engine, sensor_ai_engines, survivor_state_tracker
    if args.ai_update_interval <= 0:
        raise SystemExit("AI 갱신 주기는 0보다 커야 합니다.")
    human_risk_after = getattr(args, "human_risk_after", 4.0)
    if human_risk_after <= 0:
        raise SystemExit("사람 위험 전환 시간은 0보다 커야 합니다.")

    survivor_state_tracker = SurvivorStateTracker(
        human_risk_after_sec=human_risk_after,
    )

    vpr100_model = getattr(args, "vpr100_model", None)
    c4001_model = getattr(args, "c4001_model", None)

    if args.ai_model and (vpr100_model or c4001_model):
        raise SystemExit(
            "--ai-model은 --vpr100-model/--c4001-model과 함께 사용할 수 없습니다."
        )

    if not args.ai_model and not getattr(args, "no_auto_models", False):
        discovered = discover_default_models()
        auto_selected: dict[str, Path] = {}
        if not vpr100_model and discovered.get("vpr100"):
            vpr100_model = discovered["vpr100"]
            auto_selected["vpr100"] = discovered["vpr100"]
        if not c4001_model and discovered.get("c4001"):
            c4001_model = discovered["c4001"]
            auto_selected["c4001"] = discovered["c4001"]
        for family, model_path in auto_selected.items():
            print(f"{family} 기본 AI 모델 자동 발견: {model_path}")

    sensor_ai_engines = {}
    classifier = None
    if args.ai_model:
        model_path = Path(args.ai_model)
        try:
            classifier = load_classifier(model_path, args.ai_model_type)
        except Exception as exc:
            raise SystemExit(f"AI 모델 로드 실패: {exc}") from exc
        print(
            f"AI 모델 로드 완료: {model_path} "
            f"(종류={classifier.model_name}, "
            f"윈도우={classifier.window_size}개)"
        )
    elif not vpr100_model and not c4001_model:
        print(
            "AI 모델이 지정되지 않았습니다. 센서 중계는 계속하며 "
            "대시보드에는 '모델 없음'으로 표시합니다."
        )

    ai_engine = SensorAIEngine(
        classifier,
        update_interval=args.ai_update_interval,
    )

    for family, model_value in (
        ("vpr100", vpr100_model),
        ("c4001", c4001_model),
    ):
        if not model_value:
            continue
        model_path = Path(model_value)
        try:
            sensor_classifier = load_classifier(model_path, args.ai_model_type)
        except Exception as exc:
            raise SystemExit(f"{family} AI 모델 로드 실패: {exc}") from exc
        model_sensor_type = getattr(sensor_classifier, "sensor_type", None)
        if model_sensor_type != family:
            raise SystemExit(
                f"{family} 모델 슬롯에 사용할 수 없는 모델입니다: "
                f"sensor_type={model_sensor_type!r}"
            )
        sensor_ai_engines[family] = SensorAIEngine(
            sensor_classifier,
            update_interval=args.ai_update_interval,
        )
        print(
            f"{family} AI 모델 로드 완료: {model_path} "
            f"(종류={sensor_classifier.model_name}, "
            f"윈도우={sensor_classifier.window_size}개)"
        )
    print(f"AI 판정 갱신 주기: {args.ai_update_interval:.1f}초")
    print(f"사람 위험 전환 시간: {human_risk_after:.1f}초")


def main() -> None:
    args = parse_args()
    configure_ai(args)
    print("🚨 LifeSignal 서버 실행 완료")
    print(f"포트 {args.port}에서 센서 및 대시보드 연결 대기 중")
    web.run_app(
        create_app(),
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    main()
