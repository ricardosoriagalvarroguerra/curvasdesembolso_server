from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic import ConfigDict


class FiltersRequest(BaseModel):
	macrosectors: List[int] = Field(default_factory=lambda: [11, 22, 33, 44, 55, 66])
	modalities: List[int] = Field(default_factory=lambda: [111, 222, 333, 444])
	countries: List[str] = Field(default_factory=list)
	mdbs: List[str] = Field(default_factory=list)
	ticketMin: float = 0.0
	ticketMax: float = 1_000_000_000.0
	yearFrom: int = 2010
	yearTo: int = 2024
	onlyExited: bool = True


class CurvePoint(BaseModel):
	iatiidentifier: str
	k: int
	d: float
	hd: float
	y: float
	class_: str = Field(alias="class")
	# Optional grouping metadata for variance-by-group analysis
	country_id: Optional[str] = None
	macrosector_id: Optional[int] = None
	modality_id: Optional[int] = None
	sector_id: Optional[int] = None
	approved_amount: Optional[float] = None
	approval_year: Optional[int] = None

	# Pydantic v2 config
	model_config = ConfigDict(populate_by_name=True)


class CurveBand(BaseModel):
	k: int
	hd: float
	hd_up: float
	hd_dn: float


class CurveParams(BaseModel):
	b0: float
	b1: float
	b2: float
	sigma: float
	# Factor z para bandas fijas (p.ej., ~1.28155 para P10–P90)
	band_z: float | None = None
	n_points: int
	n_projects: int
	mean_y: float | None = None
	median_y: float | None = None
	r2: float | None = None
	k30: float | None = None
	k50: float | None = None
	k80: float | None = None
	var_y: float | None = None
	# Intervalos de confianza bootstrap (p.ej., [p2.5, p97.5])
	k30_ci: list[float] | None = None
	k50_ci: list[float] | None = None
	k80_ci: list[float] | None = None
	# Ámbito del ajuste de parámetros base: "global" o "country"
	fit_scope: str | None = None
	# Conteos de muestra usados para el baseline seleccionado
	baseline_n_points: int | None = None
	baseline_n_projects: int | None = None
	# Indicadores de cartera por serie
	approved_sum: float | None = None
	approved_avg: float | None = None
	disb_count: int | None = None
	portfolio_total_approved: float | None = None
	portfolio_share: float | None = None


class CurveFitResponse(BaseModel):
	params: CurveParams
	points: List[CurvePoint]
	bands: List[CurveBand]
	kDomain: Tuple[int, int]
	activePoints: List[CurvePoint] | None = None


class FiltersResponse(BaseModel):
	macrosectors: List[dict]
	modalities: List[dict]
	countries: List[str]
	mdbs: List[dict]
	ticketMin: float
	ticketMax: float
	yearMin: int
	yearMax: int


class ProjectTimeseriesPoint(BaseModel):
	ym: str
	disb_month: float
	disb_cum_usd: float
	k: int
	d: float


class ProjectInfo(BaseModel):
	iatiidentifier: str
	country_id: Optional[str]
	macrosector_id: Optional[int]
	modality_id: Optional[int]
	approved_amount: Optional[float]


class ProjectTimeseriesResponse(BaseModel):
        project: ProjectInfo
        series: List[ProjectTimeseriesPoint]


class PredictionMeta(BaseModel):
        method: str
        level: float
        smooth: bool
        num_points: int
        notes: str = ""
        low_sample: bool | None = None


class EtaMetrics(BaseModel):
    median: Optional[float] = None
    p10: Optional[float] = None
    p90: Optional[float] = None


class PredictionBandPoint(BaseModel):
    """Single prediction band point across quantiles."""

    k: int
    p50: float
    p10: float
    p90: float
    p2_5: float
    p97_5: float


class PredictionBandsResponse(BaseModel):
    """Prediction bands for the historical disbursement curve."""

    k: List[int]
    p50: List[float]
    p10: List[float]
    p90: List[float]
    p2_5: List[float]
    p97_5: List[float]
    meta: PredictionMeta
    bands: List[PredictionBandPoint] | None = None


