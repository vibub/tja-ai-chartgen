from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class GridFeature(BaseModel):
    grid: int
    onset: bool = False
    accent: bool = False
    beat: int | None = None
    downbeat: bool = False
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    activity: float = Field(default=0.0, ge=0.0, le=1.0)


class SpectralGridFeature(BaseModel):
    grid: int = Field(ge=0)
    low_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    mid_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    high_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    spectral_flux: float = Field(default=0.0, ge=0.0, le=1.0)


class InstrumentGridFeature(BaseModel):
    grid: int = Field(ge=0)
    vocal_onset: float = Field(default=0.0, ge=0.0, le=1.0)
    drum_onset: float = Field(default=0.0, ge=0.0, le=1.0)
    bass_onset: float = Field(default=0.0, ge=0.0, le=1.0)
    accompaniment_onset: float = Field(default=0.0, ge=0.0, le=1.0)


class RhythmicSaliencePoint(BaseModel):
    grid: int = Field(ge=0)
    hit: float = Field(default=0.0, ge=0.0, le=1.0)
    accent: float = Field(default=0.0, ge=0.0, le=1.0)
    don_preference: float = Field(default=0.0, ge=0.0, le=1.0)
    ka_preference: float = Field(default=0.0, ge=0.0, le=1.0)
    sustained_activity: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class BarRhythmicSalience(BaseModel):
    points: list[RhythmicSaliencePoint] = Field(default_factory=list)
    active_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    onset_evidence_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fallback_reason: str | None = None
    burst_score: float = Field(default=0.0, ge=0.0, le=1.0)
    burst_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    burst_start_grid: int | None = Field(default=None, ge=0)
    burst_end_grid: int | None = Field(default=None, ge=0)
    burst_reasons: list[str] = Field(default_factory=list)


class InstrumentBarFeature(BaseModel):
    vocal_activity: float = Field(default=0.0, ge=0.0, le=1.0)
    vocal_presence_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    drum_activity: float = Field(default=0.0, ge=0.0, le=1.0)
    bass_activity: float = Field(default=0.0, ge=0.0, le=1.0)
    other_activity: float = Field(default=0.0, ge=0.0, le=1.0)
    guitar: float = Field(default=0.0, ge=0.0, le=1.0)
    piano_keyboard: float = Field(default=0.0, ge=0.0, le=1.0)
    strings: float = Field(default=0.0, ge=0.0, le=1.0)
    brass: float = Field(default=0.0, ge=0.0, le=1.0)
    woodwind: float = Field(default=0.0, ge=0.0, le=1.0)
    synth: float = Field(default=0.0, ge=0.0, le=1.0)
    organ: float = Field(default=0.0, ge=0.0, le=1.0)
    other_instrument: float = Field(default=0.0, ge=0.0, le=1.0)
    dominant_source: str | None = None
    dominant_instrument: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class BarFeature(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    index: int
    start_time: float
    end_time: float
    energy: float = Field(ge=0.0, le=1.0)
    rms_dbfs: float | None = None
    peak_rms_dbfs: float | None = None
    relative_rms_db: float | None = None
    sustained_activity_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    time_signature: str = "4/4"
    grids_per_bar: int = 16
    onset_grids: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("onset_grids", "onset_16"),
    )
    accent_grids: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("accent_grids", "accent_16"),
    )
    activity_grids: list[float] = Field(
        default_factory=list,
        validation_alias=AliasChoices("activity_grids", "activity_16"),
    )
    grid_features: list[GridFeature] = Field(default_factory=list)
    spectral_grid_features: list[SpectralGridFeature] = Field(default_factory=list)
    instrument_grid_features: list[InstrumentGridFeature] = Field(default_factory=list)
    instrument: InstrumentBarFeature = Field(default_factory=InstrumentBarFeature)
    low_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    mid_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    high_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    spectral_flux: float = Field(default=0.0, ge=0.0, le=1.0)
    brightness: float = Field(default=0.0, ge=0.0, le=1.0)
    harmonic_novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    texture_novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    percussive_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    beat_grids: list[int] = Field(default_factory=list)
    downbeat_grid: int | None = None
    phrase_position: str = "unknown"
    fill_candidate: bool = False
    section: str = "unknown"
    energy_percentile: float = Field(default=0.0, ge=0.0, le=1.0)
    energy_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    phrase_id: int | None = Field(default=None, ge=0)
    phrase_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    transition_role: str = "stable"
    transition_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    section_id: str | None = None
    section_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fill_candidate_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def onset_16(self) -> list[int]:
        return self.onset_grids

    @property
    def accent_16(self) -> list[int]:
        return self.accent_grids

    @property
    def activity_16(self) -> list[float]:
        return self.activity_grids


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


class BarStructureFeature(BaseModel):
    index: int = Field(ge=0)
    energy_percentile: float = Field(default=0.0, ge=0.0, le=1.0)
    energy_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    onset_density: float = Field(default=0.0, ge=0.0, le=1.0)
    onset_strength_mean: float = Field(default=0.0, ge=0.0, le=1.0)
    onset_strength_max: float = Field(default=0.0, ge=0.0, le=1.0)
    activity_mean: float = Field(default=0.0, ge=0.0, le=1.0)
    activity_peak: float = Field(default=0.0, ge=0.0, le=1.0)
    active_grid_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    accent_density: float = Field(default=0.0, ge=0.0, le=1.0)
    low_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    mid_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    high_onset_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    spectral_flux: float = Field(default=0.0, ge=0.0, le=1.0)
    brightness: float = Field(default=0.0, ge=0.0, le=1.0)
    harmonic_novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    texture_novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    percussive_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    instrument: InstrumentBarFeature = Field(default_factory=InstrumentBarFeature)
    rhythm_profile: list[float] = Field(default_factory=list)
    activity_profile: list[float] = Field(default_factory=list)
    edge_silent: bool = False
    boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    phrase_id: int = Field(default=0, ge=0)
    phrase_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    phrase_position: str = "middle"
    transition_role: str = "stable"
    transition_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    section_id: str = "section-1"
    section: str = "unknown"
    section_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fill_candidate_score: float = Field(default=0.0, ge=0.0, le=1.0)


class PhraseFeature(BaseModel):
    phrase_id: int = Field(ge=0)
    start_bar: int = Field(ge=0)
    end_bar: int = Field(ge=0)
    section_id: str = "section-1"
    section: str = "unknown"
    mean_energy: float = Field(default=0.0, ge=0.0, le=1.0)
    peak_energy: float = Field(default=0.0, ge=0.0, le=1.0)
    energy_trend: float = Field(default=0.0, ge=-1.0, le=1.0)
    primary_role: str = "stable"
    ending_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    previous_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    next_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    resolution: int | None = Field(default=None, gt=0)


class SongAnalysis(BaseModel):
    analysis_schema_version: int = 1
    beatnet_analysis_status: str = "unavailable"
    beatnet_analysis_reason: str | None = None
    spectral_feature_version: str | None = None
    spectral_analysis_status: str = "unavailable"
    spectral_analysis_reason: str | None = None
    instrument_feature_version: str | None = None
    instrument_analysis_status: str = "unavailable"
    instrument_analysis_reason: str | None = None
    instrument_demucs_model: str | None = None
    instrument_classifier_model: str | None = None
    instrument_analysis_device: str | None = None
    structure_feature_version: str | None = None
    structure_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bar_structures: list[BarStructureFeature] = Field(default_factory=list)
    phrase_plan: list[PhraseFeature] = Field(default_factory=list)
    resolution_policy_version: str | None = None
    resolution_plan: ResolutionPlan | None = None
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


class ChartHitEvent(BaseModel):
    tick: int
    note: str


class ChartLongNoteEvent(BaseModel):
    start_tick: int
    end_tick: int
    kind: str
    balloon_count: int | None = None


class ChartBarEvents(BaseModel):
    index: int
    hits: list[ChartHitEvent] = Field(default_factory=list)
    long_notes: list[ChartLongNoteEvent] = Field(default_factory=list)


class ResolutionDecision(BaseModel):
    selected_resolution: int = Field(gt=0)
    candidate_errors: dict[str, float] = Field(default_factory=dict)
    evidence_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class ResolutionPlan(BaseModel):
    canonical_grids_per_bar: int = Field(gt=0)
    base_resolution: int = Field(gt=0)
    bar_resolutions: list[int] = Field(default_factory=list)
    change_points: list[int] = Field(default_factory=list)
    policy_version: str = "song-global-v1"
    decision: ResolutionDecision | None = None


class ChartBar(BaseModel):
    index: int
    notes: str
    time_signature: str = "4/4"
    balloon_counts: list[int] = Field(default_factory=list)


class TjaChart(BaseModel):
    metadata: ChartMetadata
    bars: list[ChartBar]
