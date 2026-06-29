from pydantic import BaseModel, Field


class BarFeature(BaseModel):
    index: int
    start_time: float
    end_time: float
    energy: float = Field(ge=0.0, le=1.0)
    time_signature: str = "4/4"
    grids_per_bar: int = 16
    onset_16: list[int] = Field(default_factory=list)
    accent_16: list[int] = Field(default_factory=list)
    section: str = "unknown"


class SongAnalysis(BaseModel):
    title: str
    artist: str | None = None
    audio_file: str
    ogg_file: str
    bpm: float
    offset: float
    time_signature: str = "4/4"
    bars: list[BarFeature]


class ChartMetadata(BaseModel):
    title: str
    artist: str | None = None
    wave: str
    bpm: float
    offset: float
    course: str = "Oni"
    level: int = 10
    maker: str = "tja-ai-chartgen"


class ChartBar(BaseModel):
    index: int
    notes: str
    time_signature: str = "4/4"


class TjaChart(BaseModel):
    metadata: ChartMetadata
    bars: list[ChartBar]
