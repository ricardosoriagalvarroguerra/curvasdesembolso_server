import os
import sys
import numpy as np
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(__file__))
import app as app_module
from app import app
from models import (
    ProjectInfo,
    ProjectTimeseriesPoint,
    ProjectTimeseriesResponse,
)

client = TestClient(app)


def _synthetic_series(n=20, noise=0.02, zeros=False):
    series = []
    for k in range(n):
        if zeros:
            d = 0.0
        else:
            base = 1.0 / (1.0 + np.exp(-0.2 * (k - 10)))
            d = float(max(0.0, min(1.0, base + np.random.uniform(-noise, noise))))
        series.append(
            ProjectTimeseriesPoint(
                ym=f"2023-{k+1:02d}-01",
                disb_month=0.0,
                disb_cum_usd=0.0,
                k=k,
                d=d,
            )
        )
    return ProjectTimeseriesResponse(
        project=ProjectInfo(
            iatiidentifier="P1",
            country_id=None,
            macrosector_id=None,
            modality_id=None,
            approved_amount=None,
        ),
        series=series,
    )


def _synthetic_rows(n_projects=5, n_k=20, noise=0.02, zeros=False):
    rows = []
    for pid in range(n_projects):
        for k in range(n_k):
            if zeros:
                d = 0.0
            else:
                base = 1.0 / (1.0 + np.exp(-0.2 * (k - 10)))
                d = float(max(0.0, min(1.0, base + np.random.uniform(-noise, noise))))
            rows.append((f"P{pid}", None, k, d, 1_000_000.0, "XX", 0, 11, 111, 2020))
    return rows


def test_prediction_bands_quantiles(monkeypatch):
    def fake_ts(project_id, db=None, yearFrom=2010, yearTo=2024):
        return _synthetic_series()

    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False):
        return _synthetic_rows()

    monkeypatch.setattr("app.project_timeseries", fake_ts)
    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/P1/prediction-bands")
    assert r.status_code == 200
    j = r.json()
    assert j["meta"]["method"] == "historical_quantiles"
    assert len(j["k"]) == len(j["p50"]) == len(j["p10"]) == len(j["p90"]) == len(j["p2_5"]) == len(j["p97_5"])
    assert len(j["project_k"]) == len(j["project_y"])


def test_prediction_bands_min_points(monkeypatch):
    def fake_ts(project_id, db=None, yearFrom=2010, yearTo=2024):
        return _synthetic_series()

    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False):
        return _synthetic_rows(n_projects=1, n_k=2)

    monkeypatch.setattr("app.project_timeseries", fake_ts)
    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/P1/prediction-bands")
    assert r.status_code == 400


def test_prediction_bands_zero_series(monkeypatch):
    def fake_ts(project_id, db=None, yearFrom=2010, yearTo=2024):
        return _synthetic_series(n=10, zeros=True)

    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False):
        return _synthetic_rows(n_projects=5, n_k=10, zeros=True)

    monkeypatch.setattr("app.project_timeseries", fake_ts)
    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/P1/prediction-bands")
    assert r.status_code == 200
    j = r.json()
    assert all(abs(v) < 1e-8 for v in j["p50"])
    assert all(abs(v) < 1e-8 for v in j["p10"]) and all(abs(v) < 1e-8 for v in j["p90"])
    assert all(abs(v) < 1e-8 for v in j["p2_5"]) and all(abs(v) < 1e-8 for v in j["p97_5"])
