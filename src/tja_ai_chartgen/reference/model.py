from pydantic import BaseModel, Field


class ReferenceTempoChange(BaseModel):
    position: int = Field(ge=0)
    bpm: float = Field(gt=0)


class ReferenceBar(BaseModel):
    index: int = Field(ge=0)
    notes: str
    resolution: int = Field(gt=0)
    start_time: float
    end_time: float
    bpm: float = Field(gt=0)
    tempo_changes: list[ReferenceTempoChange] = Field(default_factory=list)
    measure_ratio: str = "1/1"
    gogo: bool = False
    barline_visible: bool = True
    balloon_counts: list[int] = Field(default_factory=list)


class ReferenceCourse(BaseModel):
    course: str
    level: int = Field(ge=0)
    bars: list[ReferenceBar]


class ReferenceTja(BaseModel):
    source_id: str
    title: str | None = None
    artist: str | None = None
    wave: str | None = None
    bpm: float = Field(gt=0)
    internal_offset: float
    courses: list[ReferenceCourse]
