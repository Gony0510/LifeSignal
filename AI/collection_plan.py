"""실제 시연 배치에 맞춘 센서 데이터 수집 프리셋."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CollectionPreset:
    """한 번의 독립적인 센서 데이터 수집을 설명합니다."""

    name: str
    sensor: str
    session_id: str
    label: str
    scenario: str
    output: str
    duration: float = 60.0
    include_inactive: bool = False
    expected_room: int | None = None
    expected_location: str | None = None
    instruction: str = ""

    @property
    def output_path(self) -> Path:
        return PROJECT_ROOT / self.output


def _build_presets() -> tuple[CollectionPreset, ...]:
    presets: list[CollectionPreset] = []

    for sensor, output, room, location, placement in (
        (
            "vpr100",
            "AI/data/vpr100_samples.csv",
            401,
            "거실A",
            "V-PR100을 401호 거실A에 설치하고 한 번에 한 인형만 측정합니다.",
        ),
        (
            "c4001",
            "AI/data/c4001_samples.csv",
            None,
            None,
            "C4001 펌웨어의 room/location 설정이 실제 설치 위치와 일치해야 합니다.",
        ),
    ):
        for index in (1, 2):
            session_id = f"empty_{index:02d}"
            presets.append(
                CollectionPreset(
                    name=f"{sensor}_empty_{index:02d}",
                    sensor=sensor,
                    session_id=session_id,
                    label="empty",
                    scenario="empty",
                    output=output,
                    include_inactive=True,
                    expected_room=room,
                    expected_location=location,
                    instruction=(
                        f"{placement} 대상 없이 빈 공간을 측정합니다."
                    ),
                )
            )

        if sensor == "vpr100":
            vibration_instructions = (
                (
                    "센서 앞을 비운 상태에서 사람이 주변을 지나가며 생기는 "
                    "현실적인 약한 책상 진동을 재현합니다."
                ),
                (
                    "센서 앞을 비운 상태에서 책상을 가볍게 건드리거나 "
                    "주변 인형 펌프가 작동할 때의 진동을 재현합니다."
                ),
            )
            for index, vibration_instruction in enumerate(
                vibration_instructions,
                start=1,
            ):
                session_id = f"empty_vibration_{index:02d}"
                presets.append(
                    CollectionPreset(
                        name=f"vpr100_{session_id}",
                        sensor="vpr100",
                        session_id=session_id,
                        label="empty",
                        scenario="empty_vibration",
                        output=output,
                        duration=30.0,
                        include_inactive=True,
                        expected_room=room,
                        expected_location=location,
                        instruction=(
                            f"{placement} {vibration_instruction} "
                            "센서 앞에는 사람이나 개 인형을 두지 않습니다."
                        ),
                    )
                )

        for index in (1, 2):
            session_id = f"dog_{index:02d}"
            expected_room = 402 if sensor == "c4001" else room
            expected_location = "방B" if sensor == "c4001" else location
            presets.append(
                CollectionPreset(
                    name=f"{sensor}_dog_{index:02d}",
                    sensor=sensor,
                    session_id=session_id,
                    label="dog",
                    scenario="dog",
                    output=output,
                    expected_room=expected_room,
                    expected_location=expected_location,
                    instruction=(
                        f"{placement} 개 인형만 배치하고 계획한 개 인형 모션을 줍니다."
                    ),
                )
            )

        human_scenario = "human_danger" if sensor == "vpr100" else "human_normal"
        human_name = "위험 사람" if sensor == "vpr100" else "보통 사람"
        for index in (1, 2):
            session_id = f"{human_scenario}_{index:02d}"
            expected_room = 402 if sensor == "c4001" else room
            expected_location = "거실B" if sensor == "c4001" else location
            instruction = (
                f"{placement} {human_name} 인형만 배치합니다. "
                "호흡 모션을 주면서 시작하고 이후 호흡 모션을 중단합니다."
                if sensor == "vpr100"
                else (
                    f"{placement} {human_name} 인형만 배치하고 "
                    "수집 시간 동안 계속 호흡 모션을 줍니다."
                )
            )
            presets.append(
                CollectionPreset(
                    name=f"{sensor}_{human_scenario}_{index:02d}",
                    sensor=sensor,
                    session_id=session_id,
                    label="human",
                    scenario=human_scenario,
                    output=output,
                    include_inactive=(sensor == "vpr100"),
                    expected_room=expected_room,
                    expected_location=expected_location,
                    instruction=instruction,
                )
            )

    return tuple(presets)


PRESETS: tuple[CollectionPreset, ...] = _build_presets()
PRESETS_BY_NAME = {preset.name: preset for preset in PRESETS}


def get_preset(name: str) -> CollectionPreset:
    try:
        return PRESETS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(PRESETS_BY_NAME)
        raise ValueError(f"알 수 없는 수집 프리셋입니다: {name}\n사용 가능: {available}") from exc


def preset_names() -> tuple[str, ...]:
    return tuple(PRESETS_BY_NAME)
