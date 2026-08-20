from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiohttp import WSMsgType, web

try:
    from ForMov.config import (
        DEFAULT_HTML_PATH,
        DEFAULT_SVM_PATH,
        VPR100_DEFAULT_SVM_PATH,
    )
    from ForMov.inference import SensorAIEngine, load_classifier
except ModuleNotFoundError:
    from config import (  # type: ignore
        DEFAULT_HTML_PATH,
        DEFAULT_SVM_PATH,
        VPR100_DEFAULT_SVM_PATH,
    )
    from inference import SensorAIEngine, load_classifier  # type: ignore


connected_clients: set[web.WebSocketResponse] = set()
latest_sensor_states: dict[str, str] = {}
last_summaries: dict[str, tuple[object, ...]] = {}
ai_engine = SensorAIEngine(None)


def sensor_key(data: dict[str, object]) -> str | None:
    if data.get("type") != "radar_data":
        return None
    return f"{data.get('sensor', 'C4001')}:{data.get('room')}:{data.get('location')}"


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
    enriched = ai_engine.enrich(data)
    return json.dumps(enriched, ensure_ascii=False), key


async def broadcast(message: str, sender: web.WebSocketResponse) -> None:
    stale: list[web.WebSocketResponse] = []
    for client in connected_clients:
        if client is sender or client.closed:
            continue
        try:
            await client.send_str(message)
        except (ConnectionError, RuntimeError):
            stale.append(client)
    for client in stale:
        connected_clients.discard(client)


def print_sensor_summary(message: str, key: str) -> None:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return
    ai = data.get("ai") or {}
    priority = data.get("rescue_priority") or {}
    summary = (
        data.get("status"),
        ai.get("ready"),
        ai.get("scenario"),
        ai.get("confidence"),
        priority.get("level"),
        priority.get("label_ko"),
    )
    if last_summaries.get(key) == summary:
        return
    last_summaries[key] = summary
    if ai.get("ready"):
        ai_text = f"{ai.get('scenario_ko')} {float(ai.get('confidence', 0)) * 100:.1f}%"
    else:
        ai_text = str(ai.get("reason", "AI 대기"))
    print(
        "영상용 센서/AI 처리: "
        f"{data.get('room')}호 {data.get('location')} 감지={data.get('status')} "
        f"AI={ai_text} 우선순위={priority.get('label_ko', '대기')}"
    )


async def websocket_handler(
    request: web.Request, websocket: web.WebSocketResponse
) -> web.WebSocketResponse:
    await websocket.prepare(request)
    connected_clients.add(websocket)
    print(f"새 기기 연결 (현재 {len(connected_clients)}대)")
    try:
        for latest in latest_sensor_states.values():
            await websocket.send_str(latest)
        async for incoming in websocket:
            if incoming.type == WSMsgType.TEXT:
                outgoing, key = process_message(incoming.data)
                if key is not None:
                    latest_sensor_states[key] = outgoing
                    print_sensor_summary(outgoing, key)
                await broadcast(outgoing, websocket)
            elif incoming.type == WSMsgType.ERROR:
                print(f"WebSocket 오류: {websocket.exception()}")
    finally:
        connected_clients.discard(websocket)
        print(f"기기 연결 해제 (현재 {len(connected_clients)}대)")
    return websocket


async def http_or_websocket(request: web.Request) -> web.StreamResponse:
    websocket = web.WebSocketResponse(heartbeat=30)
    if websocket.can_prepare(request).ok:
        return await websocket_handler(request, websocket)
    if request.path in {"/", "/LifeSignal_ForMov.html"}:
        return web.FileResponse(
            DEFAULT_HTML_PATH, headers={"Cache-Control": "no-store"}
        )
    if request.path == "/favicon.ico":
        return web.Response(status=204)
    raise web.HTTPNotFound(text="영상용 LifeSignal 페이지를 찾을 수 없습니다.")


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/{path:.*}", http_or_websocket)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LifeSignal 영상용 HTML·WebSocket·구조 우선순위 AI 서버"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8881)
    parser.add_argument(
        "--sensor",
        choices=["c4001", "vpr100"],
        default="c4001",
        help="--model을 생략했을 때 불러올 기본 센서 모델",
    )
    parser.add_argument(
        "--model",
        help=(
            "영상용 SVM(.joblib) 또는 CNN(.keras) 경로. 생략 시 기본 SVM이 "
            "있으면 자동으로 사용합니다."
        ),
    )
    parser.add_argument(
        "--model-type", choices=["auto", "svm", "cnn", "1d-cnn"], default="auto"
    )
    return parser.parse_args()


def configure_ai(args: argparse.Namespace) -> None:
    global ai_engine
    default_model = (
        VPR100_DEFAULT_SVM_PATH if args.sensor == "vpr100" else DEFAULT_SVM_PATH
    )
    requested = Path(args.model) if args.model else default_model
    if not requested.exists():
        ai_engine = SensorAIEngine(None)
        print("영상용 AI 모델이 아직 없습니다. 서버와 대시보드는 모델 대기 상태로 실행합니다.")
        return
    try:
        classifier = load_classifier(requested, args.model_type)
    except Exception as exc:
        raise SystemExit(f"영상용 AI 모델 로드 실패: {exc}") from exc
    ai_engine = SensorAIEngine(classifier)
    print(
        f"영상용 AI 모델 로드 완료: {requested} "
        f"(종류={classifier.model_name}, 윈도우={classifier.window_size}개, "
        f"갱신={classifier.step_size}개마다)"
    )


def main() -> None:
    args = parse_args()
    configure_ai(args)
    print("🚨 LifeSignal 영상용 서버 실행 완료")
    print(f"포트 {args.port}에서 센서 브리지와 대시보드 연결 대기 중")
    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
