from pydantic import BaseModel, Field


class GridFeature(BaseModel):
    grid: int
    onset: bool = False
    accent: bool = False
    beat: int | None = None
    downbeat: bool = False
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    activity: float = Field(default=0.0, ge=0.0, le=1.0)


class BarFeature(BaseModel):
    index: int
    start_time: float
    end_time: float
    energy: float = Field(ge=0.0, le=1.0)
    time_signature: str = "4/4"
    grids_per_bar: int = 16
    onset_16: list[int] = Field(default_factory=list)
    accent_16: list[int] = Field(default_factory=list)
    activity_16: list[float] = Field(default_factory=list)
    grid_features: list[GridFeature] = Field(default_factory=list)
    beat_grids: list[int] = Field(default_factory=list)
    downbeat_grid: int | None = None
    phrase_position: str = "unknown"
    fill_candidate: bool = False
    section: str = "unknown"


class TempoAnalysisDecision(BaseModel):
    fallback_source: str
    selected_source: str
    estimated_bpm: float
    estimated_offset: float
    normalized_support: float = Field(ge=0.0, le=1.0)
    onset_count: int = Field(ge=0)
    time_coverage: float = Field(ge=0.0, le=1.0)
    runner_up_bpm: float | None = None
    runner_up_support: float | None = Field(default=None, ge=0.0, le=1.0)
    accepted: bool
    reason: str


class SongAnalysis(BaseModel):
    title: str
    artist: str | None = None
    audio_file: str
    ogg_file: str
    bpm: float
    offset: float
    time_signature: str = "4/4"
    analyzer: str = "unknown"
    tempo_analysis: TempoAnalysisDecision | None = None
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
    balloon_counts: list[int] = Field(default_factory=list)


class TjaChart(BaseModel):
    metadata: ChartMetadata
    bars: list[ChartBar]
