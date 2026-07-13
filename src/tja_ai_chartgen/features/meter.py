from dataclasses import dataclass


CANONICAL_SUBDIVISIONS_PER_QUARTER = 12
LEGACY_SUBDIVISIONS_PER_QUARTER = 4


@dataclass(frozen=True)
class MeterSpec:
    time_signature: str
    beats_per_bar: float
    measure_ratio: str
    beat_positions_quarters: tuple[float, ...]

    @property
    def grids_per_bar(self) -> int:
        return self.grids_for_subdivisions(CANONICAL_SUBDIVISIONS_PER_QUARTER)

    @property
    def legacy_grids_per_bar(self) -> int:
        return self.grids_for_subdivisions(LEGACY_SUBDIVISIONS_PER_QUARTER)

    @property
    def beat_grids(self) -> tuple[int, ...]:
        return self.beat_grids_for_resolution(self.grids_per_bar)

    @property
    def accent_grids(self) -> set[int]:
        return set(self.beat_grids)

    def grids_for_subdivisions(self, subdivisions_per_quarter: int) -> int:
        if subdivisions_per_quarter <= 0:
            raise ValueError("subdivisions_per_quarter must be positive")
        return round(self.beats_per_bar * subdivisions_per_quarter)

    def beat_grids_for_resolution(self, resolution: int) -> tuple[int, ...]:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        return tuple(
            round(position / self.beats_per_bar * resolution)
            for position in self.beat_positions_quarters
        )

    def accent_grids_for_resolution(self, resolution: int) -> set[int]:
        return set(self.beat_grids_for_resolution(resolution))


METER_SPECS = {
    "4/4": MeterSpec(
        time_signature="4/4",
        beats_per_bar=4.0,
        measure_ratio="1/1",
        beat_positions_quarters=(0.0, 1.0, 2.0, 3.0),
    ),
    "3/4": MeterSpec(
        time_signature="3/4",
        beats_per_bar=3.0,
        measure_ratio="3/4",
        beat_positions_quarters=(0.0, 1.0, 2.0),
    ),
    "6/8": MeterSpec(
        time_signature="6/8",
        beats_per_bar=3.0,
        measure_ratio="3/4",
        beat_positions_quarters=(0.0, 1.5),
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
