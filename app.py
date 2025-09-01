import json
import math
import os
import random
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
from cachetools import TTLCache
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session

from db import get_db, get_engine
from models import (
        CurveBand,
        CurveFitResponse,
        CurveParams,
        CurvePoint,
        FiltersRequest,
        FiltersResponse,
        PredictionBandsResponse,
        ProjectInfo,
        ProjectTimeseriesPoint,
        ProjectTimeseriesResponse,
)
from utils_curve import logistic3, fit_logistic3


app = FastAPI(title="Curvas de Desembolso API", version="0.1.0")


# CORS
# Permit localhost by default and optionally allow additional origins via env vars.
# ``CORS_ORIGINS`` may contain a comma separated list of explicit origins and
# ``CORS_ORIGIN_REGEX`` can be used for pattern based matching (e.g. for Railway).
# If neither variable is provided we fall back to allowing any origin which keeps
# the previous behaviour useful for development.
env_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
allow_origins = env_origins if env_origins else ["*"]

# Allow overriding the regex via env; by default permit localhost and any
# ``*.railway.app`` domains so the hosted client can reach the API without
# requiring extra configuration.
default_origin_regex = r"https?://(localhost|127\.0\.0\.1):\d+$|https://.*\.railway\.app$"
allow_origin_regex = os.getenv("CORS_ORIGIN_REGEX") or default_origin_regex

app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"]
)


# Simple in-memory caches with TTL
cache = TTLCache(maxsize=128, ttl=int(os.getenv("CACHE_TTL_SECONDS", "180")))
filters_cache = TTLCache(maxsize=8, ttl=int(os.getenv("FILTERS_CACHE_TTL_SECONDS", "600")))
ts_cache = TTLCache(maxsize=512, ttl=int(os.getenv("TS_CACHE_TTL_SECONDS", "600")))
pred_cache = TTLCache(maxsize=256, ttl=int(os.getenv("PRED_CACHE_TTL_SECONDS", "300")))
base_query_cache = TTLCache(maxsize=128, ttl=int(os.getenv("BASE_QUERY_CACHE_TTL_SECONDS", "300")))


def _filters_dep(
    macrosectors: List[int] | None = Query(None),
    modalities: List[int] | None = Query(None),
    countries: List[str] | None = Query(None),
    mdbs: List[str] | None = Query(None),
    ticketMin: float = Query(0.0),
    ticketMax: float = Query(1_000_000_000.0),
    yearFrom: int = Query(2010),
    yearTo: int = Query(2024),
    onlyExited: bool = Query(True),
) -> Tuple[FiltersRequest, Set[str]]:
    fields_set: Set[str] = set()
    if macrosectors is not None:
        fields_set.add("macrosectors")
    if modalities is not None:
        fields_set.add("modalities")
    if countries is not None:
        fields_set.add("countries")
    if mdbs is not None:
        fields_set.add("mdbs")
    filters = FiltersRequest(
        macrosectors=macrosectors or [11, 22, 33, 44, 55, 66],
        modalities=modalities or [111, 222, 333, 444],
        countries=countries or [],
        mdbs=mdbs or [],
        ticketMin=ticketMin,
        ticketMax=ticketMax,
        yearFrom=yearFrom,
        yearTo=yearTo,
        onlyExited=onlyExited,
    )
    return filters, fields_set


MACROSECTOR_LABELS = {
        11: "Infraestructura",
        22: "Productivo",
        33: "Social",
        44: "Ambiental",
        55: "Gobernanza – Público",
        66: "Multisectorial – Otros",
}

MODALITY_LABELS = {
        111: "Investment",
        222: "Results",
        333: "Emergency",
        444: "Policy-Based",
}


def _has_trans_type_table(db: Session) -> bool:
        # Detect if public.trans_type exists using current session
        try:
                res = db.execute(
                        text(
                                """
                                SELECT 1 FROM information_schema.tables
                                WHERE table_schema='public' AND table_name='trans_type'
                                """
                        )
                )
                return res.scalar() is not None
        except Exception:
                return False


def _cte_sql(has_tmap: bool) -> str:
        """Build the SQL CTE, conditionally including ``trans_type`` if available."""
        tmap_cte = (
                "tmap AS ( SELECT trans_id, trans_name FROM public.trans_type ),"
                if has_tmap
                else ""
        )
        join_tmap = (
                "LEFT JOIN tmap ON tmap.trans_id::text = t.trans_id::text"
                if has_tmap
                else ""
        )
        trans_name_expr = (
                "COALESCE(tmap.trans_name, CASE WHEN t.trans_id::text='100' THEN 'Commitment' WHEN t.trans_id::text='200' THEN 'Disbursement' ELSE 'Other' END)"
                if has_tmap
                else "CASE WHEN t.trans_id::text='100' THEN 'Commitment' WHEN t.trans_id::text='200' THEN 'Disbursement' ELSE 'Other' END"
        )
        return f"""
WITH exited AS (
  SELECT iatiidentifier, country_id, sector_id, macrosector_id, modality_id,
         approval_date, status
  FROM public.activities
  WHERE ((:status_target = 'ALL') OR (UPPER(status) = :status_target))
    AND ((:macrosectors_is_empty) OR (macrosector_id::text = ANY(:macrosectors_txt)))
    AND ((:modalities_is_empty) OR (modality_id::text = ANY(:modalities_txt)))
    AND ((:countries_is_empty) OR (country_id::text = ANY(:countries_txt)))
),
{tmap_cte}
t AS (
  SELECT t.iatiidentifier,
         {trans_name_expr} AS trans_name,
         t.value_usd,
         t.iso_date
  FROM public.trans t
  JOIN exited a USING(iatiidentifier)
  {join_tmap}
  WHERE t.value_usd > 0
),
approved AS (
  SELECT iatiidentifier, SUM(value_usd) AS approved_amount
  FROM t
  WHERE LOWER(trans_name) = 'commitment'
  GROUP BY 1
),
approval_date_fallback AS (
  SELECT
    e.iatiidentifier,
    COALESCE(e.approval_date,
             MIN(t.iso_date) FILTER (WHERE LOWER(t.trans_name) = 'commitment'),
             MIN(t.iso_date)) AS approval_date
  FROM exited e
  LEFT JOIN t ON t.iatiidentifier = e.iatiidentifier
  GROUP BY 1, e.approval_date
),
disb AS (
  SELECT iatiidentifier, date_trunc('month', iso_date)::date AS ym, SUM(value_usd) AS disb_month
  FROM t
  WHERE LOWER(trans_name) = 'disbursement'
    AND iso_date >= make_date(:year_from,1,1)
    AND iso_date <= make_date(:year_to,12,31)
  GROUP BY 1,2
),
disb_cum AS (
  SELECT d.iatiidentifier, d.ym,
         SUM(d.disb_month) OVER (PARTITION BY d.iatiidentifier ORDER BY d.ym) AS disb_cum_usd
  FROM disb d
),
base AS (
  SELECT
    dc.iatiidentifier, dc.ym, dc.disb_cum_usd,
    a.approved_amount, ad.approval_date,
    ex.country_id, ex.sector_id, ex.macrosector_id, ex.modality_id
  FROM disb_cum dc
  JOIN approved a USING (iatiidentifier)
  JOIN approval_date_fallback ad USING (iatiidentifier)
  JOIN exited ex USING (iatiidentifier)
  WHERE a.approved_amount > 0 AND ad.approval_date IS NOT NULL
),
final AS (
  SELECT
    iatiidentifier, ym,
    (EXTRACT(YEAR FROM age(ym, approval_date))::int)*12
      + (EXTRACT(MONTH FROM age(ym, approval_date))::int) AS k,
    LEAST(1.0, (disb_cum_usd/approved_amount))::float8 AS d,
    approved_amount, country_id, sector_id, macrosector_id, modality_id
  FROM base
  WHERE ym >= approval_date
)
SELECT * FROM final
WHERE (
  (:macrosectors_is_empty) OR (macrosector_id::text = ANY(:macrosectors_txt))
) AND (
  (:modalities_is_empty) OR (modality_id::text = ANY(:modalities_txt))
) AND (
  (:countries_is_empty) OR (country_id::text = ANY(:countries_txt))
) AND approved_amount BETWEEN :ticket_min AND :ticket_max
ORDER BY iatiidentifier, ym
"""


def _cte_sql_v2(only_last: bool = False, select_meta: bool = True) -> str:
        # Optimized: no trans_type join, no string LOWER/COALESCE; use trans_id codes
        if select_meta:
                projection = "iatiidentifier, ym, k, d, approved_amount, country_id, sector_id, macrosector_id, modality_id, approval_year"
        else:
                projection = "iatiidentifier, ym, k, d"
        select_clause = (
                f"SELECT DISTINCT ON (iatiidentifier) {projection} FROM final ORDER BY iatiidentifier, ym DESC"
                if only_last
                else f"SELECT {projection} FROM final ORDER BY iatiidentifier, ym"
        )
        return f"""
WITH exited AS (
  SELECT iatiidentifier, country_id, sector_id, macrosector_id, modality_id,
         approval_date, status
  FROM public.activities
  WHERE (
    (:status_target = 'ALL' AND UPPER(status) IN ('EXITED','ACTIVE'))
    OR (UPPER(status) = :status_target)
  )
    AND approval_date IS NOT NULL
    AND approval_date >= make_date(:year_from,1,1)
    AND approval_date <= make_date(:year_to,12,31)
    AND ((:macrosectors_is_empty) OR (macrosector_id::text = ANY(:macrosectors_txt)))
    AND ((:modalities_is_empty) OR (modality_id::text = ANY(:modalities_txt)))
    AND ((:countries_is_empty) OR (country_id::text = ANY(:countries_txt)))
    AND ((:mdbs_is_empty) OR (prefix_id::text = ANY(:mdbs_txt)))
),
approved AS (
  SELECT t.iatiidentifier, SUM(t.value_usd) AS approved_amount
  FROM public.trans t
  JOIN exited a USING(iatiidentifier)
  WHERE t.value_usd > 0 AND t.trans_id::text = '100'
    AND t.iso_date >= make_date(:year_from,1,1)
    AND t.iso_date <= make_date(:year_to,12,31)
  GROUP BY 1
),
approval_date_fallback AS (
  SELECT
    e.iatiidentifier,
    COALESCE(
      e.approval_date,
      MIN(CASE WHEN t.trans_id::text = '100' THEN t.iso_date END),
      MIN(t.iso_date)
    ) AS approval_date
  FROM exited e
  LEFT JOIN public.trans t ON t.iatiidentifier = e.iatiidentifier AND t.value_usd > 0
  GROUP BY 1, e.approval_date
),
disb AS (
  SELECT t.iatiidentifier, date_trunc('month', t.iso_date)::date AS ym, SUM(t.value_usd) AS disb_month
  FROM public.trans t
  JOIN exited a USING(iatiidentifier)
  WHERE t.value_usd > 0 AND t.trans_id::text = '200'
    AND t.iso_date >= make_date(:year_from,1,1)
    AND t.iso_date <= make_date(:year_to,12,31)
  GROUP BY 1,2
),
disb_cum AS (
  SELECT d.iatiidentifier, d.ym,
         SUM(d.disb_month) OVER (PARTITION BY d.iatiidentifier ORDER BY d.ym) AS disb_cum_usd
  FROM disb d
),
base AS (
  SELECT
    dc.iatiidentifier, dc.ym, dc.disb_cum_usd,
    a.approved_amount, ad.approval_date,
    ex.country_id, ex.sector_id, ex.macrosector_id, ex.modality_id
  FROM disb_cum dc
  JOIN approved a USING (iatiidentifier)
  JOIN approval_date_fallback ad USING (iatiidentifier)
  JOIN exited ex USING (iatiidentifier)
  WHERE a.approved_amount > 0 AND ad.approval_date IS NOT NULL
),
final AS (
  SELECT
    iatiidentifier, ym,
    (EXTRACT(YEAR FROM age(ym, approval_date))::int)*12
      + (EXTRACT(MONTH FROM age(ym, approval_date))::int) AS k,
    LEAST(1.0, (disb_cum_usd/approved_amount))::float8 AS d,
    approved_amount, country_id, sector_id, macrosector_id, modality_id,
    EXTRACT(YEAR FROM approval_date)::int AS approval_year
  FROM base
  WHERE ym >= approval_date AND approved_amount BETWEEN :ticket_min AND :ticket_max
)
{select_clause}
"""


def _filters_sql(has_tmap: bool) -> str:
        # Optimized: no trans_type join or string functions; rely on trans_id codes
        return """
WITH approved_sum AS (
  SELECT t.iatiidentifier, SUM(t.value_usd) AS approved_amount
  FROM public.trans t
  WHERE t.value_usd > 0 AND t.trans_id::text = '100'
  GROUP BY 1
),
years AS (
  SELECT EXTRACT(YEAR FROM a.approval_date)::int AS y
  FROM public.activities a
  WHERE a.approval_date IS NOT NULL
  GROUP BY 1
)
SELECT
  (SELECT COALESCE(MIN(approved_amount), 0) FROM approved_sum) AS ticket_min,
  (SELECT COALESCE(MAX(approved_amount), 0) FROM approved_sum) AS ticket_max,
  (SELECT COALESCE(MIN(y), 2010) FROM years) AS year_min,
  (SELECT COALESCE(MAX(y), 2024) FROM years) AS year_max
"""


@app.get("/", include_in_schema=False)
def root():
        """Simple root endpoint to avoid 404 at the base URL."""
        return {"message": "Curvas de Desembolso API"}

@app.get("/api/health")
def health():
        return {"status": "ok"}


@app.options("/api/filters")
def options_filters():
        return Response(status_code=200)


@app.get("/api/filters", response_model=FiltersResponse)
def get_filters(db: Session = Depends(get_db)):
        # Macro and modality label maps are static; countries are from activities
        # Cache full response since it changes rarely
        if "__filters__" in filters_cache:
                cached = filters_cache["__filters__"]
                # Avoid returning a cached response without MDBs if not yet populated
                try:
                        if getattr(cached, "mdbs", None) and len(cached.mdbs) > 0:
                                return cached
                except Exception:
                        return cached
        has_tmap = _has_trans_type_table(db)
        res = db.execute(
                text(
                        "SELECT DISTINCT country_id FROM public.activities WHERE country_id IS NOT NULL ORDER BY country_id"
                )
        )
        countries = [r[0] for r in res.fetchall()]

        # MDBs list (prefixes). Try to read name if available, else fallback to prefix_id
        mdbs_list: list[dict] = []
        try:
                res_m = db.execute(
                        text("SELECT prefix_id, COALESCE(mdb_name, prefix_id) AS name FROM public.mdbs ORDER BY 1")
                )
                mdbs_list = [{"id": row[0], "name": row[1]} for row in res_m.fetchall()]
        except Exception:
                try:
                        res_m2 = db.execute(text("SELECT prefix_id FROM public.mdbs ORDER BY 1"))
                        mdbs_list = [{"id": row[0], "name": row[0]} for row in res_m2.fetchall()]
                except Exception:
                        mdbs_list = []

        # If still empty, derive prefixes from activities (split by '-')
        if not mdbs_list:
                try:
                        res_p = db.execute(
                                text(
                                        "SELECT DISTINCT split_part(iatiidentifier, '-', 1) AS prefix FROM public.activities WHERE iatiidentifier IS NOT NULL AND iatiidentifier <> '' ORDER BY 1"
                                )
                        )
                        mdbs_list = [
                                {"id": row[0], "name": row[0]} for row in res_p.fetchall() if row[0]
                        ]
                except Exception:
                        mdbs_list = []

        # Ranges
        try:
                res2 = db.execute(text(_filters_sql(has_tmap)))
                row = res2.fetchone()
        except Exception:
                # Fallback: if joining trans_type failed (permissions/absence), retry without it
                res2 = db.execute(text(_filters_sql(False)))
                row = res2.fetchone()
        if row is None:
                # Default ranges
                ticket_min, ticket_max, year_min, year_max = 0.0, 0.0, 2010, 2024
        else:
                ticket_min = float(row[0] or 0.0)
                ticket_max = float(row[1] or 0.0)
                year_min = int(row[2] or 2010)
                year_max = int(row[3] or 2024)

        resp = FiltersResponse(
                macrosectors=[{"id": k, "name": v} for k, v in MACROSECTOR_LABELS.items()],
                modalities=[{"id": k, "name": v} for k, v in MODALITY_LABELS.items()],
                countries=countries,
                mdbs=mdbs_list,
                ticketMin=ticket_min,
                ticketMax=ticket_max,
                yearMin=year_min,
                yearMax=year_max,
        )
        filters_cache["__filters__"] = resp
        return resp


def _serialize_point(row: Row) -> Tuple[str, int, float]:
        return row[0], int(row[2]), float(row[3])


def _class_from_residual(residual: float, half_sigma: float) -> str:
        if residual > half_sigma:
                return "above"
        if residual < -half_sigma:
                return "below"
        return "average"


def _bands_from_params(b0: float, b1: float, b2: float, sigma: float, k_max: int) -> List[CurveBand]:
        bands: List[CurveBand] = []
        # Use configurable z*sigma band (default z≈1.28155 → ~P10–P90)
        try:
                z = float(os.getenv("BAND_SIGMA_Z", "1.2815515655446004"))
        except Exception:
                z = 1.2815515655446004
        delta = z * sigma
        for k in range(0, k_max + 1):
                hd = float(logistic3(k, b0, b1, b2))
                hd_up = min(1.0, max(0.0, hd + delta))
                hd_dn = min(1.0, max(0.0, hd - delta))
                bands.append(CurveBand(k=k, hd=hd, hd_up=hd_up, hd_dn=hd_dn))
        return bands


def _run_base_query(
        filters: FiltersRequest,
        db: Session,
        status_target: str = 'ALL',
        *,
        select_meta: bool = True,
        start_from_first_disb: bool = False,
) -> List[Row]:
        cache_key = (
                json.dumps(filters.model_dump(), sort_keys=True),
                str(status_target).upper(),
                bool(select_meta),
                bool(start_from_first_disb),
        )
        if cache_key in base_query_cache:
                return base_query_cache[cache_key]
        # Use optimized CTE; when targeting ACTIVE, return only last snapshot per project
        only_last = (str(status_target).upper() == 'ACTIVE')
        sql = _cte_sql_v2(only_last=only_last, select_meta=select_meta)
        # Prepare MDB LIKE filters (prefix%)
        mdbs_val = getattr(filters, 'mdbs', [])
        mdbs_txt = [str(mdbs_val)] if isinstance(mdbs_val, str) else [str(v) for v in (mdbs_val or [])]
        params = {
                "status_target": str(status_target).upper(),
                "year_from": int(filters.yearFrom),
                "year_to": int(filters.yearTo),
                "macrosectors_is_empty": len(filters.macrosectors) == 0,
                "macrosectors_txt": [str(x) for x in filters.macrosectors],
                "modalities_is_empty": len(filters.modalities) == 0,
                "modalities_txt": [str(x) for x in filters.modalities],
                "countries_is_empty": len(filters.countries) == 0,
                "countries_txt": [str(x) for x in filters.countries],
                "mdbs_is_empty": len(mdbs_txt) == 0,
                "mdbs_txt": mdbs_txt,
                "ticket_min": float(filters.ticketMin),
                "ticket_max": float(filters.ticketMax),
        }
        try:
                db.execute(text("SET LOCAL statement_timeout = 60000"))
        except Exception:
                pass
        res = db.execute(text(sql), params)
        rows = res.fetchall()
        if start_from_first_disb:
                adj_rows: List[tuple] = []
                first_k: dict[str, int] = {}
                for r in rows:
                        pid = r[0]
                        k = int(r[2])
                        d = float(r[3])
                        if d <= 0:
                                continue
                        if pid not in first_k:
                                first_k[pid] = k
                        new_k = k - first_k[pid]
                        if select_meta:
                                adj_rows.append((r[0], r[1], new_k, d, r[4], r[5], r[6], r[7], r[8], r[9]))
                        else:
                                adj_rows.append((r[0], r[1], new_k, d))
                rows = adj_rows
        base_query_cache[cache_key] = rows
        return rows


def _sample_indices(n: int, frac: float) -> List[int]:
        count = max(1, int(n * frac))
        return sorted(random.sample(range(n), count)) if count < n else list(range(n))


@app.post("/api/curves/fit", response_model=CurveFitResponse)
def fit_curve(
        payload: FiltersRequest,
        start_from_first_disb: bool = Query(False, alias="fromFirstDisbursement"),
        db: Session = Depends(get_db),
):
        key = json.dumps(
                {"payload": payload.model_dump(), "start_from_first_disb": start_from_first_disb},
                sort_keys=True,
        )
        if key in cache:
                return cache[key]

        # Rows for current filter context (used for domain/KPIs). Use both ACTIVE & EXITED
        rows = _run_base_query(
                payload,
                db,
                status_target='ALL',
                start_from_first_disb=start_from_first_disb,
        )
        if not rows:
                raise HTTPException(status_code=400, detail="No hay suficientes observaciones para ajustar la curva; afloja filtros")

        # Build arrays of (k, d) and compute n_projects
        points_simple: List[Tuple[str, int, float]] = [(r[0], int(r[2]), float(r[3])) for r in rows]
        unique_projects = sorted({pid for pid, _, _ in points_simple})
        n_projects = len(unique_projects)

        # Filter invalid values
        points_simple = [(pid, k, max(0.0, min(1.0, d))) for pid, k, d in points_simple if math.isfinite(k) and math.isfinite(d)]
        n_points_total = len(points_simple)
        if n_points_total < 30:
                raise HTTPException(status_code=400, detail="No hay suficientes observaciones para ajustar la curva; afloja filtros")

        # Build baseline dataset for parameter fitting with optional per-country scope
        min_country_projects = int(os.getenv("MIN_COUNTRY_PROJECTS", "40"))
        min_country_points = int(os.getenv("MIN_COUNTRY_POINTS", "500"))
        min_sector_projects = int(os.getenv("MIN_SECTOR_PROJECTS", "60"))
        min_sector_points = int(os.getenv("MIN_SECTOR_POINTS", "800"))
        fit_scope = "global"
        baseline_n_projects: int | None = None
        baseline_n_points: int | None = None

        # Always in Investment for baseline shape comparability
        global_baseline_filters = FiltersRequest(
                macrosectors=[],
                modalities=[111],
                countries=[],
                ticketMin=0.0,
                ticketMax=1_000_000_000_000.0,
                yearFrom=payload.yearFrom,
                yearTo=payload.yearTo,
                onlyExited=True,
        )

        # Try country-specific fit if exactly one country is selected
        rows_fit = None
        if len(payload.countries) == 1:
                country_id = payload.countries[0]
                country_baseline_filters = FiltersRequest(
                        macrosectors=[],
                        modalities=[111],
                        countries=[country_id],
                        ticketMin=0.0,
                        ticketMax=1_000_000_000_000.0,
                        yearFrom=payload.yearFrom,
                        yearTo=payload.yearTo,
                        onlyExited=True,
                )
                rows_country = _run_base_query(country_baseline_filters, db, status_target='ALL', select_meta=False)
                # Count projects and points
                if rows_country:
                        projects_country = len({r[0] for r in rows_country})
                        points_country = len(rows_country)
                        if (projects_country >= min_country_projects) and (points_country >= min_country_points):
                                rows_fit = rows_country
                                fit_scope = "country"
                                baseline_n_projects = projects_country
                                baseline_n_points = points_country

        # If not country-level and exactly one macrosector selected (multi-country sector analysis), try sector baseline
        if (rows_fit is None) and (len(payload.countries) != 1) and (len(payload.macrosectors) == 1):
                sector_id = payload.macrosectors[0]
                sector_baseline_filters = FiltersRequest(
                        macrosectors=[sector_id],
                        modalities=[111],
                        countries=[],
                        ticketMin=0.0,
                        ticketMax=1_000_000_000_000.0,
                        yearFrom=payload.yearFrom,
                        yearTo=payload.yearTo,
                        onlyExited=True,
                )
                rows_sector = _run_base_query(sector_baseline_filters, db, status_target='ALL', select_meta=False)
                if rows_sector:
                        projects_sector = len({r[0] for r in rows_sector})
                        points_sector = len(rows_sector)
                        if (projects_sector >= min_sector_projects) and (points_sector >= min_sector_points):
                                rows_fit = rows_sector
                                fit_scope = "sector"
                                baseline_n_projects = projects_sector
                                baseline_n_points = points_sector

        # Fallback to global Investment baseline
        if rows_fit is None:
                rows_fit = _run_base_query(global_baseline_filters, db, status_target='ALL', select_meta=False)
                fit_scope = "global"
                baseline_n_projects = len({r[0] for r in rows_fit}) if rows_fit else None
                baseline_n_points = len(rows_fit) if rows_fit else None

        if not rows_fit or len(rows_fit) < 30:
                raise HTTPException(status_code=400, detail="No hay suficientes observaciones para ajustar (Investment baseline).")

        fit_points_all: List[Tuple[str, int, float]] = [(r[0], int(r[2]), float(r[3])) for r in rows_fit]
        fit_points_all = [(pid, k, max(0.0, min(1.0, d))) for pid, k, d in fit_points_all if math.isfinite(k) and math.isfinite(d)]
        fit_points = fit_points_all
        fit_sample_frac = float(os.getenv("FIT_SAMPLE_FRAC", "0.25"))
        if len(fit_points_all) > 20000:
                idx = _sample_indices(len(fit_points_all), fit_sample_frac)
                fit_points = [fit_points_all[i] for i in idx]

        k_arr = [k for _, k, _ in fit_points]
        d_arr = [d for _, _, d in fit_points]

        try:
                b0, b1, b2, sigma = fit_logistic3(k_arr, d_arr)
        except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Adjust only b0 for the ACTIVE filter to allow visual differentiation (keep b1,b2 global)
        def find_b0_shift(points: List[Tuple[str,int,float]], b0_base: float, b1_fix: float, b2_fix: float) -> float:
                if not points:
                        return 0.0
                ks = np.array([k for _, k, _ in points], dtype=float)
                ds = np.array([d for _, _, d in points], dtype=float)
                # Coarse grid search for stability
                # Higher resolution grid configurable via env (reduced default for speed)
                requested_gp = max(101, int(os.getenv("B0_GRID_POINTS", "401")))
                n_obs = len(points)
                if n_obs > 80000:
                        grid_points = min(requested_gp, 201)
                elif n_obs > 30000:
                        grid_points = min(requested_gp, 401)
                else:
                        grid_points = requested_gp
                grid_range = float(os.getenv("B0_GRID_RANGE", "6.0"))
                grid = np.linspace(-grid_range, grid_range, grid_points)
                best = 0.0
                best_loss = float('inf')
                for delta in grid:
                        z = b0_base + delta + b1_fix * ks + b2_fix * (ks ** 2)
                        hd = 1.0 / (1.0 + np.exp(-z))
                        loss = float(np.mean((ds - hd) ** 2))
                        if loss < best_loss:
                                best_loss = loss
                                best = float(delta)
                return best

        b0_shift = find_b0_shift(points_simple, b0, b1, b2)
        b0_display = float(b0 + b0_shift)

        # Build per-project metadata dict from rows
        meta_by_pid: dict[str, tuple[str|None,int|None,int|None,int|None,float|None,int|None]] = {}
        for r in rows:
                try:
                        pid = r[0]
                        approved_amount = float(r[4]) if r[4] is not None else None
                        country_id = r[5]
                        sector_id = r[6]
                        macrosector_id = r[7]
                        modality_id = r[8]
                        approval_year = int(r[9]) if len(r) > 9 and r[9] is not None else None
                        if pid not in meta_by_pid:
                                meta_by_pid[pid] = (country_id, sector_id, macrosector_id, modality_id, approved_amount, approval_year)
                except Exception:
                        continue

        # Compute hd, residuals, and classes for EXITED visualization points
        vis_points = points_simple
        max_vis_points = int(os.getenv("MAX_VIS_POINTS", "8000"))
        if n_points_total > max_vis_points:
                frac = max(0.05, min(0.5, max_vis_points / n_points_total))
                idx2 = _sample_indices(n_points_total, frac)
                vis_points = [points_simple[i] for i in idx2]

        points: List[CurvePoint] = []
        half = 0.5 * sigma
        for pid, k, d in vis_points:
                hd = float(logistic3(k, b0_display, b1, b2))
                y = float(d - hd)
                meta = meta_by_pid.get(pid, (None, None, None, None, None, None))
                points.append(
                        CurvePoint(
                                iatiidentifier=pid,
                                k=int(k),
                                d=float(d),
                                hd=hd,
                                y=y,
                                class_=_class_from_residual(y, half),
                                country_id=meta[0],
                                sector_id=meta[1],
                                macrosector_id=meta[2],
                                modality_id=meta[3],
                                approved_amount=meta[4],
                                approval_year=meta[5],
                        )
                )

        k_max = max((k for _, k, _ in points_simple), default=0)
        bands = _bands_from_params(b0_display, b1, b2, sigma, k_max)

        # KPIs for current filter: mean/median residual using all points_simple
        residuals_all = []
        for _, k, d in points_simple:
                hd = float(logistic3(k, b0_display, b1, b2))
                residuals_all.append(float(d - hd))
        mean_y = float(np.mean(residuals_all)) if residuals_all else None
        median_y = float(np.median(residuals_all)) if residuals_all else None
        var_y = float(np.var(residuals_all, ddof=1)) if len(residuals_all) > 1 else None

        # R-squared using all available (k,d) points for this context
        try:
                if points_simple:
                        ds = np.array([d for _, _, d in points_simple], dtype=float)
                        hds = np.array([float(logistic3(k, b0_display, b1, b2)) for _, k, _ in points_simple], dtype=float)
                        d_mean = float(np.mean(ds)) if ds.size > 0 else 0.0
                        sst = float(np.sum((ds - d_mean) ** 2)) if ds.size > 1 else 0.0
                        ssr = float(np.sum((ds - hds) ** 2)) if ds.size > 0 else 0.0
                        r2 = (1.0 - ssr / sst) if sst > 0 else None
                else:
                        r2 = None
        except Exception:
                r2 = None

        # Portfolio/series indicators
        # Number of operations (projects) already computed as n_projects
        # Count of disbursements (monthly aggregated rows)
        disb_count_series = int(len(rows))
        # Sum and average of approved amounts for the series (unique projects)
        approved_sum_series = 0.0
        for pid in unique_projects:
                try:
                        amt = meta_by_pid.get(pid, (None,None,None,None,None,None))[4]
                        if amt is not None:
                                approved_sum_series += float(amt)
                except Exception:
                        continue
        approved_avg_series = (approved_sum_series / n_projects) if n_projects > 0 else None

        # Total portfolio approved amount for current context (global across countries/macros with same modality/year/ticket)
        portfolio_total_approved = None
        try:
                engine = get_engine()
                with engine.connect() as conn:
                        conn.execute(text("SET LOCAL statement_timeout = 60000"))
                        # Prepare MDB LIKE filters (prefix%) for portfolio total
                        mdbs_val = getattr(payload, 'mdbs', [])
                        mdbs_txt = [str(mdbs_val)] if isinstance(mdbs_val, str) else [str(v) for v in (mdbs_val or [])]
                        params_total = {
                                "yf": int(payload.yearFrom),
                                "yt": int(payload.yearTo),
                                "modalities_is_empty": len(payload.modalities) == 0,
                                "modalities_txt": [str(x) for x in payload.modalities],
                                "mdbs_is_empty": len(mdbs_txt) == 0,
                                "mdbs_txt": mdbs_txt,
                                "ticket_min": float(payload.ticketMin),
                                "ticket_max": float(payload.ticketMax),
                        }
                        sql_total = text(
                                """
                                WITH exited AS (
                                  SELECT iatiidentifier
                                  FROM public.activities
                                  WHERE approval_date IS NOT NULL
                                    AND approval_date >= make_date(:yf,1,1)
                                    AND approval_date <= make_date(:yt,12,31)
                                    AND ((:modalities_is_empty) OR (modality_id::text = ANY(:modalities_txt)))
                                    AND ((:mdbs_is_empty) OR (prefix_id::text = ANY(:mdbs_txt)))
                                ),
                                approved AS (
                                  SELECT t.iatiidentifier, SUM(t.value_usd) AS approved_amount
                                  FROM public.trans t
                                  JOIN exited a USING(iatiidentifier)
                                  WHERE t.value_usd > 0 AND t.trans_id::text = '100'
                                  GROUP BY 1
                                )
                                SELECT COALESCE(SUM(approved_amount), 0)
                                FROM approved
                                WHERE approved_amount BETWEEN :ticket_min AND :ticket_max
                                """
                        )
                        row_total = conn.execute(sql_total, params_total).fetchone()
                        portfolio_total_approved = float(row_total[0] or 0.0) if row_total else 0.0
        except Exception:
                portfolio_total_approved = None

        portfolio_share = None
        if portfolio_total_approved and portfolio_total_approved > 0:
                portfolio_share = float(approved_sum_series / portfolio_total_approved)

        # Invert hd to compute k at target p using scalar search (robust for b2 != 0)
        def solve_k_for_p(p: float, k_upper: int = int(k_max) or 120) -> float | None:
                try:
                        for k in range(0, max(1, k_upper)+1):
                                # Use adjusted intercept for the ACTIVE filter so thresholds vary by series
                                if logistic3(k, b0_display, b1, b2) >= p:
                                        return float(k)
                        return None
                except Exception:
                        return None

        k30 = solve_k_for_p(0.3)
        k50 = solve_k_for_p(0.5)
        k80 = solve_k_for_p(0.8)

        # Bootstrap confidence intervals for k30/k50/k80
        boot_enabled = os.getenv("BOOTSTRAP_ENABLE", "0") not in ("0", "false", "False")
        boot_n = max(50, min(400, int(os.getenv("BOOTSTRAP_N", "120"))))
        # Adapt bootstrap effort to dataset size
        if n_points_total > 30000:
                boot_enabled = False
        elif n_points_total > 15000:
                boot_n = min(boot_n, 80)
        k30_ci = None
        k50_ci = None
        k80_ci = None
        if boot_enabled and points_simple:
                ks_np = np.array([k for _, k, _ in points_simple], dtype=float)
                ds_np = np.array([d for _, _, d in points_simple], dtype=float)
                # Precompute for speed; reuse grid from b0 search with adaptive resolution
                # Indices for bootstrap
                n_obs = len(points_simple)
                requested_gp = max(101, int(os.getenv("B0_GRID_POINTS", "401")))
                if n_obs > 80000:
                        grid_points = min(requested_gp, 201)
                elif n_obs > 30000:
                        grid_points = min(requested_gp, 401)
                else:
                        grid_points = requested_gp
                grid_range = float(os.getenv("B0_GRID_RANGE", "6.0"))
                grid_boot = np.linspace(-grid_range, grid_range, grid_points)
                k30_samples: list[float] = []
                k50_samples: list[float] = []
                k80_samples: list[float] = []
                for _ in range(boot_n):
                        idx = np.random.randint(0, n_obs, size=n_obs)
                        ks_b = ks_np[idx]
                        ds_b = ds_np[idx]
                        # Find b0 shift for bootstrap sample
                        best = 0.0
                        best_loss = float('inf')
                        for delta in grid_boot:
                                z = b0 + delta + b1 * ks_b + b2 * (ks_b ** 2)
                                hd = 1.0 / (1.0 + np.exp(-z))
                                loss = float(np.mean((ds_b - hd) ** 2))
                                if loss < best_loss:
                                        best_loss = loss
                                        best = float(delta)
                        b0_b = float(b0 + best)
                        # Compute ks
                        def solve_k_local(p: float, k_upper: int = int(k_max) or 120) -> float | None:
                                for kk in range(0, max(1, k_upper)+1):
                                        if logistic3(kk, b0_b, b1, b2) >= p:
                                                return float(kk)
                                return None
                        k30_b = solve_k_local(0.3)
                        k50_b = solve_k_local(0.5)
                        k80_b = solve_k_local(0.8)
                        if k30_b is not None:
                                k30_samples.append(k30_b)
                        if k50_b is not None:
                                k50_samples.append(k50_b)
                        if k80_b is not None:
                                k80_samples.append(k80_b)
                # Percentiles
                def pct(arr: list[float], p_low: float = 2.5, p_hi: float = 97.5):
                        if not arr:
                                return None
                        vals = np.percentile(np.array(arr, dtype=float), [p_low, p_hi])
                        return [float(vals[0]), float(vals[1])]
                k30_ci = pct(k30_samples)
                k50_ci = pct(k50_samples)
                k80_ci = pct(k80_samples)

        # ACTIVE points for scatter (sampled). SQL already returns only last row per project
        rows_active = _run_base_query(payload, db, status_target='ACTIVE', select_meta=False)
        active_points_simple: List[Tuple[str, int, float]] = [
                (r[0], int(r[2]), max(0.0, min(1.0, float(r[3]))))
                for r in rows_active
                if math.isfinite(int(r[2])) and math.isfinite(float(r[3]))
        ]
        max_active = int(os.getenv("MAX_ACTIVE_POINTS", "3000"))
        if len(active_points_simple) > max_active:
                frac_a = max(0.02, min(0.5, max_active / max(1, len(active_points_simple))))
                idx_a = _sample_indices(len(active_points_simple), frac_a)
                active_vis_points = [active_points_simple[i] for i in idx_a]
        else:
                active_vis_points = active_points_simple

        active_points: List[CurvePoint] = []
        half_a = 0.5 * sigma
        for pid, k, d in active_vis_points:
                hd = float(logistic3(k, b0_display, b1, b2))
                y = float(d - hd)
                meta = meta_by_pid.get(pid, (None, None, None, None, None, None))
                active_points.append(
                        CurvePoint(
                                iatiidentifier=pid,
                                k=int(k),
                                d=float(d),
                                hd=hd,
                                y=y,
                                class_=_class_from_residual(y, half_a),
                                country_id=meta[0],
                                sector_id=meta[1],
                                macrosector_id=meta[2],
                                modality_id=meta[3],
                                approved_amount=meta[4],
                                approval_year=meta[5],
                        )
                )

        # Band z used for fixed bands (exposed to client for Sigma(k) scaling)
        try:
                band_z = float(os.getenv("BAND_SIGMA_Z", "1.2815515655446004"))
        except Exception:
                band_z = 1.2815515655446004

        resp = CurveFitResponse(
                params=CurveParams(
                        b0=b0_display,
                        b1=b1,
                        b2=b2,
                        sigma=sigma,
                        band_z=band_z,
                        n_points=n_points_total,
                        n_projects=n_projects,
                        mean_y=mean_y,
                        median_y=median_y,
                        r2=(float(r2) if r2 is not None else None),
                        var_y=var_y,
                        k30=k30,
                        k50=k50,
                        k80=k80,
                        k30_ci=k30_ci,
                        k50_ci=k50_ci,
                        k80_ci=k80_ci,
                        fit_scope=fit_scope,
                        baseline_n_points=baseline_n_points,
                        baseline_n_projects=baseline_n_projects,
                        approved_sum=approved_sum_series,
                        approved_avg=(float(approved_avg_series) if approved_avg_series is not None else None),
                        disb_count=disb_count_series,
                        portfolio_total_approved=(float(portfolio_total_approved) if portfolio_total_approved is not None else None),
                        portfolio_share=(float(portfolio_share) if portfolio_share is not None else None),
                ),
                points=points,
                bands=bands,
                kDomain=(0, int(k_max)),
                activePoints=active_points,
        )
        cache[key] = resp
        return resp


@app.options("/api/curves/fit")
def options_curves_fit():
        return Response(status_code=200)


@app.get("/api/projects/{iatiidentifier}/timeseries", response_model=ProjectTimeseriesResponse)
def project_timeseries(
        iatiidentifier: str,
        yearFrom: int = Query(2010),
        yearTo: int = Query(2024),
        start_from_first_disb: bool = Query(False, alias="fromFirstDisbursement"),
        db: Session = Depends(get_db),
):
        # Build approved amount and approval date using trans_id codes (no joins)
        approved_sql = text(
                """
                SELECT SUM(t.value_usd) AS approved
                FROM public.trans t
                WHERE t.iatiidentifier=:pid AND t.value_usd>0 AND t.trans_id::text='100'
                """
        )
        row = db.execute(approved_sql, {"pid": iatiidentifier}).fetchone()
        approved_amount = float(row[0] or 0.0)

        # approval_date fallback
        appr_sql = text(
                """
                SELECT COALESCE(a.approval_date,
                       MIN(CASE WHEN t.trans_id::text='100' THEN t.iso_date END),
                       MIN(t.iso_date)) AS approval_date
                FROM public.activities a
                LEFT JOIN public.trans t ON t.iatiidentifier=a.iatiidentifier
                WHERE a.iatiidentifier=:pid
                GROUP BY a.approval_date
                """
        )
        row2 = db.execute(appr_sql, {"pid": iatiidentifier}).fetchone()
        approval_date = row2[0] if row2 else None

        # project info
        info_row = db.execute(
                text(
                        "SELECT iatiidentifier, country_id, macrosector_id, modality_id FROM public.activities WHERE iatiidentifier=:pid LIMIT 1"
                ),
                {"pid": iatiidentifier},
        ).fetchone()
        country_id = info_row[1] if info_row else None
        macrosector_id = info_row[2] if info_row else None
        modality_id = info_row[3] if info_row else None

        # disbursements monthly within range
        disb_sql = text(
                """
                SELECT date_trunc('month', iso_date)::date AS ym, SUM(value_usd) AS disb_month
                FROM public.trans t
                WHERE t.iatiidentifier=:pid AND t.value_usd>0 AND t.trans_id::text='200'
                  AND iso_date >= make_date(:yf,1,1) AND iso_date <= make_date(:yt,12,31)
                GROUP BY 1
                ORDER BY 1
                """
        )
        rows = db.execute(disb_sql, {"pid": iatiidentifier, "yf": yearFrom, "yt": yearTo}).fetchall()

        series: List[ProjectTimeseriesPoint] = []
        # accumulate
        cum = 0.0
        first_k: int | None = None
        for (ym, disb_month) in rows:
                mval = float(disb_month or 0.0)
                cum += mval
                # compute k and d
                if approval_date is None:
                        k = None
                        d = 0.0
                else:
                        # months between approval_date and ym (SQL's age-based approach)
                        age_years = (ym.year - approval_date.year)
                        age_months = (ym.month - approval_date.month)
                        k = max(0, age_years * 12 + age_months)
                        d = float(min(1.0, cum / approved_amount)) if approved_amount > 0 else 0.0
                if start_from_first_disb:
                        if d <= 0:
                                continue
                        if first_k is None:
                                first_k = int(k or 0)
                        k = (int(k or 0) - first_k)
                series.append(
                        ProjectTimeseriesPoint(
                                ym=ym.isoformat(), disb_month=mval, disb_cum_usd=float(cum), k=int(k or 0), d=float(d)
                        )
                )

        return ProjectTimeseriesResponse(
                project=ProjectInfo(
                        iatiidentifier=iatiidentifier,
                        country_id=country_id,
                        macrosector_id=macrosector_id,
                        modality_id=modality_id,
                        approved_amount=approved_amount if approved_amount > 0 else None,
                ),
                series=series,
        )



@app.options("/api/curves/prediction-bands")
def prediction_bands_options() -> Response:
    return Response(status_code=204)


@app.get("/api/curves/prediction-bands", response_model=PredictionBandsResponse)
def prediction_bands(
    min_points: int = Query(30),
    min_n_k: int = Query(30),
    iatiidentifier: str | None = Query(None),
    debug: bool = Query(False),
    start_from_first_disb: bool = Query(False, alias="fromFirstDisbursement"),
    filters_data: Tuple[FiltersRequest, Set[str]] = Depends(_filters_dep),
    db: Session = Depends(get_db),
):
    filters, _ = filters_data
    key = (
        min_points,
        min_n_k,
        iatiidentifier,
        debug,
        json.dumps(filters.model_dump(), sort_keys=True),
        bool(start_from_first_disb),
    )
    if key in pred_cache:
        return pred_cache[key]

    rows = _run_base_query(
        filters,
        db,
        status_target="ALL",
        select_meta=False,
        start_from_first_disb=start_from_first_disb,
    )
    if len(rows) < min_points:
        raise HTTPException(status_code=400, detail=f"not enough data points ({len(rows)} < {min_points})")

    df_hist = pd.DataFrame(
        [
            (
                str(r[0]),
                int(r[2]),
                float(r[3]),
                str(r[5]),
                str(r[7]),
                str(r[8]),
            )
            for r in rows
        ],
        columns=["pid", "k", "d", "country", "macrosector", "modality"],
    ).dropna(subset=["k", "d"])
    df_hist["d"] = df_hist["d"].clip(0.0, 1.0)
    if iatiidentifier:
        df_hist = df_hist[df_hist["pid"] != iatiidentifier]
    if df_hist.empty:
        raise HTTPException(status_code=400, detail="not enough valid data points")
    if len(df_hist) < min_points:
        raise HTTPException(
            status_code=400,
            detail=f"not enough data points ({len(df_hist)} < {min_points})",
        )

    grouped_comb = (
        df_hist.groupby(["country", "macrosector", "modality", "k"])["d"]
        .mean()
        .reset_index()
    )
    pivot = grouped_comb.pivot_table(
        index="k", columns=["country", "macrosector", "modality"], values="d"
    )
    pivot = pivot.dropna(how="all")
    counts = pivot.count(axis=1)
    try:
        q = pivot.quantile([0.025, 0.10, 0.5, 0.90, 0.975], axis=1).dropna(axis=1)
    except TypeError as e:
        raise HTTPException(status_code=400, detail="unable to compute quantiles") from e

    k_vals = q.columns.astype(int).tolist()
    counts = counts.loc[k_vals].astype(int).to_numpy()
    p50 = q.loc[0.5].to_numpy(dtype=float)
    p10 = q.loc[0.10].to_numpy(dtype=float)
    p90 = q.loc[0.90].to_numpy(dtype=float)
    p2_5 = q.loc[0.025].to_numpy(dtype=float)
    p97_5 = q.loc[0.975].to_numpy(dtype=float)

    # Enforce non-decreasing by k
    def _fix_series(arr: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(arr)

    p50 = _fix_series(p50)
    p10 = _fix_series(p10)
    p90 = _fix_series(p90)
    p2_5 = _fix_series(p2_5)
    p97_5 = _fix_series(p97_5)

    # Ensure quantile ordering at each k
    for i in range(len(k_vals)):
        p10[i] = max(p10[i], p2_5[i])
        p50[i] = max(p50[i], p10[i])
        p90[i] = max(p90[i], p50[i])
        p97_5[i] = max(p97_5[i], p90[i])

    low_sample_flags = counts < min_n_k
    bands = [
        {
            "k": k,
            "p50": float(m),
            "p10": float(lo),
            "p90": float(hi),
            "p2_5": float(lo2),
            "p97_5": float(hi2),
            "n": int(n),
            "low_sample": bool(ls),
        }
        for k, m, lo, hi, lo2, hi2, n, ls in zip(
            k_vals, p50, p10, p90, p2_5, p97_5, counts, low_sample_flags
        )
    ]

    n_points = int(len(df_hist))
    resp = {
        "k": k_vals,
        "p50": p50.tolist(),
        "p10": p10.tolist(),
        "p90": p90.tolist(),
        "p2_5": p2_5.tolist(),
        "p97_5": p97_5.tolist(),
        "bands": bands,
        "meta": {
            "method": "historical_quantiles",
            "coverage": {"outer": 0.95, "inner": 0.80},
            "smooth": False,
            "num_points": n_points,
            "notes": "n decreases at high k due to cohort truncation",
        },
        "n": counts.tolist(),
    }

    if iatiidentifier:
        proj = project_timeseries(
            iatiidentifier=iatiidentifier,
            yearFrom=filters.yearFrom,
            yearTo=filters.yearTo,
            start_from_first_disb=start_from_first_disb,
            db=db,
        )
        resp["project"] = proj.model_dump()

    if debug:
        p2_dict = dict(zip(k_vals, p2_5))
        p97_dict = dict(zip(k_vals, p97_5))
        p10_dict = dict(zip(k_vals, p10))
        p90_dict = dict(zip(k_vals, p90))
        outer_hits = 0
        inner_hits = 0
        for _, row in df_hist.iterrows():
            k = int(row["k"])
            d = float(row["d"])
            if k in p2_dict and p2_dict[k] <= d <= p97_dict[k]:
                outer_hits += 1
            if k in p10_dict and p10_dict[k] <= d <= p90_dict[k]:
                inner_hits += 1
        total = len(df_hist)
        resp["meta"]["coverage_empirical"] = {
            "outer": outer_hits / total if total else None,
            "inner": inner_hits / total if total else None,
        }

    pred_cache[key] = resp
    return resp


@app.get("/api/curves/{iatiidentifier}/prediction-bands", response_model=PredictionBandsResponse)
def prediction_bands_alias(
    iatiidentifier: str,
    min_points: int = Query(30),
    min_n_k: int = Query(30),
    debug: bool = Query(False),
    start_from_first_disb: bool = Query(False, alias="fromFirstDisbursement"),
    filters_data: Tuple[FiltersRequest, Set[str]] = Depends(_filters_dep),
    db: Session = Depends(get_db),
):
    return prediction_bands(
        min_points=min_points,
        min_n_k=min_n_k,
        iatiidentifier=iatiidentifier,
        debug=debug,
        start_from_first_disb=start_from_first_disb,
        filters_data=filters_data,
        db=db,
    )
