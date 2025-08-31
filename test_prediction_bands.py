import numpy as np
from fastapi.testclient import TestClient

from app import app
from models import ProjectInfo, ProjectTimeseriesPoint, ProjectTimeseriesResponse


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
    return ProjectTimeseriesResponse(project=ProjectInfo(iatiidentifier="P1", country_id=None, macrosector_id=None, modality_id=None, approved_amount=None), series=series)


def test_prediction_bands_methods(monkeypatch):
    def fake_ts(project_id, db=None, yearFrom=2010, yearTo=2024):
        return _synthetic_series()

    monkeypatch.setattr("app.project_timeseries", fake_ts)

    # rolling_std
    r = client.get("/api/curves/P1/prediction-bands?method=rolling_std")
    assert r.status_code == 200
    j = r.json()
    assert j["meta"]["method"] == "rolling_std"
    assert len(j["t"]) == 20

    # bootstrap
    monkeypatch.setenv("BOOTSTRAP_B", "100")
    r = client.get("/api/curves/P1/prediction-bands?method=bootstrap")
    assert r.status_code == 200
    j = r.json()
    assert j["meta"]["method"] == "bootstrap"

    # quantile_reg
    r = client.get("/api/curves/P1/prediction-bands?method=quantile_reg&smooth=false")
    assert r.status_code == 200
    j = r.json()
    assert j["meta"]["method"] == "quantile_reg"


def test_prediction_bands_min_points(monkeypatch):
    def fake_ts(project_id, db=None, yearFrom=2010, yearTo=2024):
        return _synthetic_series(n=5)

    monkeypatch.setattr("app.project_timeseries", fake_ts)

    r = client.get("/api/curves/P1/prediction-bands")
    assert r.status_code == 400


def test_prediction_bands_zero_series(monkeypatch):
    def fake_ts(project_id, db=None, yearFrom=2010, yearTo=2024):
        return _synthetic_series(n=10, zeros=True)

    monkeypatch.setattr("app.project_timeseries", fake_ts)

    r = client.get("/api/curves/P1/prediction-bands?method=rolling_std")
    assert r.status_code == 200
    j = r.json()
    assert all(abs(v) < 1e-8 for v in j["y_hat"])  # all zeros
    assert all(abs(v) < 1e-8 for v in j["lower"]) and all(abs(v) < 1e-8 for v in j["upper"])

