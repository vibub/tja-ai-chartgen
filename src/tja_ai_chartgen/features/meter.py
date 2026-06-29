from dataclasses import dataclass


@dataclass(frozen=True)
class MeterSpec:
    time_signature: str
    beats_per_bar: float
    grids_per_bar: int
    measure_ratio: str
    accent_grids: set[int]


METER_SPECS = {
    "4/4": MeterSpec(
        time_signature="4/4",
        beats_per_bar=4.0,
        grids_per_bar=16,
        measure_ratio="1/1",
        accent_grids={0, 4, 8, 12},
    ),
    "3/4": MeterSpec(
        time_signature="3/4",
        beats_per_bar=3.0,
        grids_per_bar=12,
        measure_ratio="3/4",
        accent_grids={0, 4, 8},
    ),
    "6/8": MeterSpec(
        time_signature="6/8",
        beats_per_bar=3.0,
        grids_per_bar=12,
        measure_ratio="3/4",
        accent_grids={0, 4, 8},
    ),
}

SUPPORTED_TIME_SIGNATURES = tuple(METER_SPECS)


def get_meter_spec(time_signature: str) -> MeterSpec:
    try:
        return METER_SPECS[time_signature]
    except KeyError as error:
        allowed = ", ".join(SUPPORTED_TIME_SIGNATURES)
        raise ValueError(f"Invalid time signature: {time_signature}. Expected one of: {allowed}") from error


def validate_time_signature(time_signature: str) -> None:
    get_meter_spec(time_signature)
