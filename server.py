from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

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
C4001_DEMO_UPDATE_INTERVAL_SECONDS = 5.0
DEFAULT_CAMERA_STREAM_URL = "http://192.168.0.5:8890/stream.mjpg"
CAMERA_STREAM_URL = os.getenv(
    "LIFESIGNAL_CAMERA_STREAM_URL",
    DEFAULT_CAMERA_STREAM_URL,
)
HTML_PATH = Path(__file__).with_name("LifeSignal.html")
ARTIFACT_DIR = Path(__file__).with_name("AI") / "artifacts"
DEFAULT_MODEL_CANDIDATES = {
    "vpr100": (
        ARTIFACT_DIR / "vpr100-svm.joblib",
        ARTIFACT_DIR / "vpr100-cnn.keras",
    ),
}
C4001_DEMO_RULES = {
    (402, "거실B"): {
        "target": "human",
        "target_ko": "사람",
        "confidence_min": 830,
        "confidence_max": 980,
    },
    (402, "방B"): {
        "target": "dog",
        "target_ko": "개",
        "confidence_min": 900,
        "confidence_max": 980,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_location(value: object) -> str:
    return "".join(str(value or "").split())


def c4001_demo_rule(data: dict) -> dict | None:
    try:
        room = int(data.get("room"))
    except (TypeError, ValueError):
        return None
    return C4001_DEMO_RULES.get(
        (room, normalize_location(data.get("location")))
    )


class C4001DemoEngine:
    """402호 C4001 두 위치에만 적용하는 시연용 고정 분류기입니다."""

    def __init__(
        self,
        *,
        update_interval: float = C4001_DEMO_UPDATE_INTERVAL_SECONDS,
    ) -> None:
        if update_interval <= 0:
            raise ValueError("C4001 시연 결과 갱신 주기는 0보다 커야 합니다.")
        self.update_interval = update_interval
        self.latest_results: dict[str, dict] = {}
        self.last_updated_at: dict[str, float] = {}

    @staticmethod
    def sensor_key(data: dict) -> str:
        return (
            f"{data.get('sensor', 'C4001')}:"
            f"{data.get('room')}:"
            f"{normalize_location(data.get('location'))}"
        )

    @staticmethod
    def _disabled_result() -> dict:
        return {
            "ready": False,
            "disabled": True,
            "target": None,
            "target_ko": None,
            "confidence": None,
        }

    def enrich(self, data: dict, *, now: float | None = None) -> dict:
        enriched = dict(data)
        rule = c4001_demo_rule(data)
        if rule is None:
            enriched["ai"] = self._disabled_result()
            return enriched

        current_time = monotonic() if now is None else now
        key = self.sensor_key(data)
        if data.get("status") is not True:
            self.latest_results.pop(key, None)
            self.last_updated_at.pop(key, None)
            enriched["ai"] = {
                "ready": True,
                "simulated": True,
                "model": "rule_based_demo",
                "target": "empty",
                "target_ko": "감지 대상 없음",
                "confidence": None,
                "update_interval_sec": self.update_interval,
                "updated_at": _utc_now(),
            }
            return enriched

        result = self.latest_results.get(key)
        last_updated = self.last_updated_at.get(key)
        if result is None:
            confidence = random.randint(
                int(rule["confidence_min"]),
                int(rule["confidence_max"]),
            ) / 1000.0
            result = {
                "ready": True,
                "simulated": True,
                "model": "rule_based_demo",
                "target": rule["target"],
                "target_ko": rule["target_ko"],
                "confidence": confidence,
                "update_interval_sec": self.update_interval,
                "updated_at": _utc_now(),
            }
            self.latest_results[key] = result
            self.last_updated_at[key] = current_time
        elif (
            last_updated is None
            or current_time - last_updated >= self.update_interval
        ):
            result = {**result, "updated_at": _utc_now()}
            self.latest_results[key] = result
            self.last_updated_at[key] = current_time

        enriched["ai"] = dict(result)
        return enriched

    def clear(self) -> None:
        self.latest_results.clear()
        self.last_updated_at.clear()


c4001_demo_engine = C4001DemoEngine()


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


def select_ai_engine(data: dict) -> SensorAIEngine | None:
    family = sensor_family(data)
    # C4001은 현재 AI 모델을 사용하지 않습니다. V-PR100 모델이
    # C4001 데이터에 잘못 적용되지 않도록 명시적으로 제외합니다.
    if family == "c4001":
        return None
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
    family = sensor_family(stabilized)
    engine = select_ai_engine(stabilized)

    if family == "c4001":
        enriched = c4001_demo_engine.enrich(stabilized)
        enriched = survivor_state_tracker.enrich(enriched)
        ai_data = enriched.get("ai")
        if isinstance(ai_data, dict) and not ai_data.get("disabled"):
            # C4001 시연 분류는 위치별 고정 우선순위만 사용합니다.
            enriched["rescue_priority"] = build_rescue_priority(ai_data)
        else:
            enriched.pop("rescue_priority", None)
    else:
        if engine is None:
            engine = ai_engine
        enriched = engine.enrich(stabilized)
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

    if sensor_family(data) == "c4001":
        ai_data = data.get("ai") or {}
        if ai_data.get("ready") and ai_data.get("target") not in {None, "empty"}:
            ai_text = (
                f"{ai_data.get('target_ko')} "
                f"{float(ai_data.get('confidence', 0)) * 100:.1f}%"
            )
        elif ai_data.get("target") == "empty":
            ai_text = "감지 대상 없음"
        else:
            print(
                "센서 수신: "
                f"{data.get('room')}호 {data.get('location')} "
                f"감지={data.get('status')}"
            )
            return
        print(
            "센서 수신/시연 처리: "
            f"{data.get('room')}호 {data.get('location')} "
            f"감지={data.get('status')} 대상={ai_text}"
        )
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
    if request.path == "/camera/stream.mjpg":
        return await camera_stream_proxy(request)

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


async def camera_stream_proxy(request: web.Request) -> web.StreamResponse:
    """카메라 MJPEG를 같은 서버 주소로 중계해 브라우저 차단을 피합니다."""
    timeout = ClientTimeout(total=None, connect=5.0)
    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.get(
                CAMERA_STREAM_URL,
                headers={"Accept": "multipart/x-mixed-replace"},
            ) as upstream:
                if upstream.status != 200:
                    return web.Response(
                        status=502,
                        text=(
                            "카메라 서버가 스트림을 제공하지 않습니다. "
                            f"HTTP {upstream.status}"
                        ),
                    )

                response = web.StreamResponse(status=200)
                response.headers["Content-Type"] = upstream.headers.get(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                response.headers["Cache-Control"] = "no-store"
                response.headers["Access-Control-Allow-Origin"] = "*"
                await response.prepare(request)
                try:
                    async for chunk in upstream.content.iter_chunked(64 * 1024):
                        await response.write(chunk)
                except (ConnectionResetError, BrokenPipeError):
                    return response
                except asyncio.CancelledError:
                    raise
                finally:
                    with contextlib.suppress(ConnectionResetError, BrokenPipeError):
                        await response.write_eof()
                return response
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return web.Response(
            status=502,
            text=f"카메라 스트림 연결 실패: {exc}",
        )


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
    # 기존 실행 명령과의 호환성을 위해 옵션만 남기고 실제 모델은 로드하지 않습니다.
    parser.add_argument("--c4001-model", help=argparse.SUPPRESS)
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
    global ai_engine, c4001_demo_engine, sensor_ai_engines, survivor_state_tracker
    if args.ai_update_interval <= 0:
        raise SystemExit("AI 갱신 주기는 0보다 커야 합니다.")
    human_risk_after = getattr(args, "human_risk_after", 4.0)
    if human_risk_after <= 0:
        raise SystemExit("사람 위험 전환 시간은 0보다 커야 합니다.")

    survivor_state_tracker = SurvivorStateTracker(
        human_risk_after_sec=human_risk_after,
    )
    c4001_demo_engine = C4001DemoEngine(
        update_interval=args.ai_update_interval,
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
        for family, model_path in auto_selected.items():
            print(f"{family} 기본 AI 모델 자동 발견: {model_path}")

    if c4001_model:
        print("C4001 AI 모델 옵션은 현재 비활성화되어 모델을 로드하지 않습니다.")

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
    elif not vpr100_model:
        print(
            "AI 모델이 지정되지 않았습니다. 센서 중계는 계속하며 "
            "대시보드에는 '모델 없음'으로 표시합니다."
        )

    ai_engine = SensorAIEngine(
        classifier,
        update_interval=args.ai_update_interval,
    )

    for family, model_value in (("vpr100", vpr100_model),):
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
