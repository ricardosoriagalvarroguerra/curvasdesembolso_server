import numpy as np
from fastapi.testclient import TestClient

import app as app_module
from app import app
from models import FiltersRequest

client = TestClient(app)


def _dummy_get_db():
    yield None


def _synthetic_rows(n_projects=5, n_k=20, noise=0.02):
    rows = []
    for pid in range(n_projects):
        for k in range(n_k):
            base = 1.0 / (1.0 + np.exp(-0.2 * (k - 10)))
            d = float(max(0.0, min(1.0, base + np.random.uniform(-noise, noise))))
            rows.append((f"P{pid}", None, k, d, 1_000_000.0, "XX", 0, 11, 111, 2020))
    return rows


def _maybe_trim_meta(rows, select_meta):
    if select_meta:
        return rows
    return [r[:4] for r in rows]


def test_prediction_bands_basic(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        rows = _synthetic_rows(n_k=40)
        return _maybe_trim_meta(rows, select_meta)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    monkeypatch.setattr(app_module, "get_db", _dummy_get_db)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands")
    assert r.status_code == 200
    j = r.json()
    assert j["meta"]["method"] == "historical"
    assert j["meta"]["level"] == 0.8
    assert len(j["k"]) == len(j["p50"]) == len(j["p_low"]) == len(j["p_high"]) == len(j["n"])


def test_prediction_bands_respects_filters(monkeypatch):
    captured = {}

    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        captured["filters"] = filters
        rows = _synthetic_rows(n_k=10)
        return _maybe_trim_meta(rows, select_meta)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    monkeypatch.setattr(app_module, "get_db", _dummy_get_db)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands?macrosectors=22&countries=XX")
    assert r.status_code == 200
    assert captured["filters"].macrosectors == [22]
    assert captured["filters"].countries == ["XX"]


def test_prediction_bands_min_n_mask(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        rows = _synthetic_rows(n_projects=10, n_k=5)
        return _maybe_trim_meta(rows, select_meta)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    monkeypatch.setattr(app_module, "get_db", _dummy_get_db)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands?min_n=11")
    assert r.status_code == 200
    j = r.json()
    assert j["k"] == []
    assert "no k with sufficient coverage" in j["meta"]["warning"]


def test_prediction_bands_level_adjustment(monkeypatch):
    def fake_run_base_query(filters, db, status_target="ALL", select_meta=False, start_from_first_disb=False):
        rows = _synthetic_rows(n_projects=2, n_k=20)
        return _maybe_trim_meta(rows, select_meta)

    monkeypatch.setattr("app._run_base_query", fake_run_base_query)
    monkeypatch.setattr(app_module, "get_db", _dummy_get_db)
    app_module.pred_cache.clear()

    r = client.get("/api/curves/prediction-bands?level=0.95")
    assert r.status_code == 200
    j = r.json()
    assert j["meta"]["level"] == 0.8  # adjusted
    assert "level adjusted" in j["meta"]["warning"]
