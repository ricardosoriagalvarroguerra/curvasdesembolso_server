import os
import sys
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(__file__))
import app as app_module
from app import app
from models import FiltersRequest

client = TestClient(app)


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
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        return _synthetic_rows(n_k=40)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands")
    assert r.status_code == 200
    j = r.json()
    assert j["meta"]["method"] == "historical_quantiles"
    assert j["meta"]["coverage"]["outer"] == 0.95
    assert len(j["k"]) == len(j["p50"]) == len(j["p10"]) == len(j["p90"]) == len(j["p2_5"]) == len(j["p97_5"]) == len(j["n"])
    assert len(j["bands"]) == len(j["k"])
    for bp, k, p50, p10, p90, p2_5, p97_5, n in zip(
        j["bands"], j["k"], j["p50"], j["p10"], j["p90"], j["p2_5"], j["p97_5"], j["n"]
    ):
        assert bp["k"] == k
        assert bp["p50"] == p50
        assert bp["p10"] == p10
        assert bp["p90"] == p90
        assert bp["p2_5"] == p2_5
        assert bp["p97_5"] == p97_5
        assert bp["n"] == n
        assert "low_sample" in bp


def test_prediction_bands_respects_filters(monkeypatch):
    captured = {}

    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        captured["filters"] = filters
        return _synthetic_rows(n_k=40)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands?macrosectors=22&countries=XX")
    assert r.status_code == 200
    assert captured["filters"].macrosectors == [22]
    assert captured["filters"].countries == ["XX"]


def test_prediction_bands_min_points(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        return _synthetic_rows(n_projects=1, n_k=2)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands")
    assert r.status_code == 400


def test_prediction_bands_zero_series(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        return _synthetic_rows(n_projects=5, n_k=10, zeros=True)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands")
    assert r.status_code == 200
    j = r.json()
    assert all(abs(v) < 1e-8 for v in j["p50"])
    assert all(abs(v) < 1e-8 for v in j["p10"]) and all(abs(v) < 1e-8 for v in j["p90"])
    assert all(abs(v) < 1e-8 for v in j["p2_5"]) and all(abs(v) < 1e-8 for v in j["p97_5"])
    assert len(j["n"]) == len(j["k"])


def test_prediction_bands_drop_nan(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        return _synthetic_rows(n_projects=5, n_k=10)

    def fake_quantile(self, qs, axis=0):
        ks = list(self.index)
        rows = []
        for q in qs:
            row = []
            for k in ks:
                if k == 1 and q == 0.10:
                    row.append(np.nan)
                else:
                    row.append(0.5)
            rows.append(row)
        return pd.DataFrame(rows, index=qs, columns=ks)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    monkeypatch.setattr(pd.DataFrame, "quantile", fake_quantile)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands")
    assert r.status_code == 200
    j = r.json()
    for arr in (j["p50"], j["p10"], j["p90"], j["p2_5"], j["p97_5"]):
        assert np.isfinite(arr).all()
    assert 1 not in j["k"]
    assert len(j["n"]) == len(j["k"])


def test_prediction_bands_empirical_coverage(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        return _synthetic_rows(n_k=20)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands?debug=true")
    assert r.status_code == 200
    j = r.json()
    assert "coverage_empirical" in j["meta"]
    ce = j["meta"]["coverage_empirical"]
    assert 0 <= ce["outer"] <= 1
    assert 0 <= ce["inner"] <= 1


def test_prediction_bands_per_combination(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        rows = []
        for k in range(10):
            d_fast = min(1.0, 0.1 * k + 0.1)
            rows.append(("F0", None, k, d_fast, 1_000_000.0, "F", 0, 11, 111, 2020))
        for pid in range(9):
            for k in range(10):
                d_slow = min(1.0, 0.05 * k)
                rows.append((f"S{pid}", None, k, d_slow, 1_000_000.0, "S", 0, 11, 111, 2020))
        return rows

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands")
    assert r.status_code == 200
    j = r.json()
    idx = j["k"].index(5)
    assert j["n"][idx] == 2
    assert j["p50"][idx] > 0.3


def test_prediction_bands_from_first_disbursement(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        rows = []
        for pid in range(5):
            for k in range(3, 33):
                d = min(1.0, 0.05 * (k - 2))
                rows.append((f"P{pid}", None, k, d, 1_000_000.0, "XX", 0, 11, 111, 2020))
        if start_from_first_disb:
            rows = [(pid, ym, k-3, d, amt, c, s, m, mod, yr) for (pid, ym, k, d, amt, c, s, m, mod, yr) in rows]
        return rows

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands?fromFirstDisbursement=true")
    assert r.status_code == 200
    j = r.json()
    assert j["k"][0] == 0


def test_run_base_query_without_db():
    filters = FiltersRequest(
        macrosectors=[],
        modalities=[111],
        countries=[],
        mdbs=[],
        ticketMin=0.0,
        ticketMax=1_000_000_000.0,
        yearFrom=2010,
        yearTo=2024,
        onlyExited=True,
    )
    with pytest.raises(HTTPException) as exc:
        app_module._run_base_query(filters, None)
    assert exc.value.status_code == 503
