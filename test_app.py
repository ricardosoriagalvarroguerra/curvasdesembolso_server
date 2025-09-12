import os
import random
import sys
from collections import namedtuple
from datetime import date

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(__file__))
from app import app, project_timeseries


client = TestClient(app)


def test_health():
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_curve_fit_monkeypatch(monkeypatch):
    # Monkeypatch the base query to avoid DB

    def fake_run_base_query(filters, db, status_target='ALL', *, select_meta=True, start_from_first_disb=False):
        # Generate synthetic logistic-ish data for multiple projects
        rows = []
        for pid in range(10):
            for k in range(0, 60, 3):
                # hd ~ logistic
                z = -1.0 + 0.12*k + (-0.0005)*(k**2)
                hd = 1.0/(1.0 + pow(2.718281828, -z))
                d = max(0.0, min(1.0, hd + random.uniform(-0.05, 0.05)))
                # emulate row: (iatiidentifier, ym, k, d, approved_amount, country, sector, macro, modality)
                rows.append((f"P{pid}", None, k, d, 1_000_000.0, 'XX', 0, 11, 111))
        return rows

    monkeypatch.setattr('app._run_base_query', fake_run_base_query)

    payload = {
        "macrosectors": [11,22,33,44,55,66],
        "modalities": [111,222,333,444],
        "countries": [],
        "ticketMin": 0,
        "ticketMax": 1_000_000_000,
        "yearFrom": 2015,
        "yearTo": 2024,
        "onlyExited": True,
    }
    r = client.post('/api/curves/fit', json=payload)
    assert r.status_code == 200
    j = r.json()
    assert 'params' in j and 'points' in j and 'bands' in j and 'kDomain' in j
    assert j['params']['n_points'] > 30
    assert j['params']['n_projects'] >= 1
    assert 'bandsQuantile' not in j


def test_curve_fit_from_first_disbursement(monkeypatch):
    def fake_run_base_query(filters, db, status_target='ALL', *, select_meta=True, start_from_first_disb=False):
        rows = []
        for pid in range(10):
            for k in range(3, 60, 3):
                d = min(1.0, 0.02 * (k - 2))
                rows.append((f"P{pid}", None, k, d, 1_000_000.0, 'XX', 0, 11, 111))
        if start_from_first_disb:
            rows = [(pid, ym, k-3, d, amt, c, s, m, mod) for (pid, ym, k, d, amt, c, s, m, mod) in rows]
        return rows

    monkeypatch.setattr('app._run_base_query', fake_run_base_query)

    payload = {
        "macrosectors": [11,22,33,44,55,66],
        "modalities": [111,222,333,444],
        "countries": [],
        "ticketMin": 0,
        "ticketMax": 1_000_000_000,
        "yearFrom": 2015,
        "yearTo": 2024,
        "onlyExited": True,
    }
    r = client.post('/api/curves/fit?fromFirstDisbursement=true', json=payload)
    assert r.status_code == 200
    j = r.json()
    ks = [p['k'] for p in j['points']]
    assert 0 in ks


def test_curve_fit_from_first_disbursement_body(monkeypatch):
    def fake_run_base_query(filters, db, status_target='ALL', *, select_meta=True, start_from_first_disb=False):
        rows = []
        for pid in range(10):
            for k in range(3, 60, 3):
                d = min(1.0, 0.02 * (k - 2))
                rows.append((f"P{pid}", None, k, d, 1_000_000.0, 'XX', 0, 11, 111))
        if start_from_first_disb:
            rows = [(pid, ym, k-3, d, amt, c, s, m, mod) for (pid, ym, k, d, amt, c, s, m, mod) in rows]
        return rows

    monkeypatch.setattr('app._run_base_query', fake_run_base_query)

    payload = {
        "macrosectors": [11,22,33,44,55,66],
        "modalities": [111,222,333,444],
        "countries": [],
        "ticketMin": 0,
        "ticketMax": 1_000_000_000,
        "yearFrom": 2015,
        "yearTo": 2024,
        "onlyExited": True,
        "fromFirstDisbursement": True,
    }
    r = client.post('/api/curves/fit', json=payload)
    assert r.status_code == 200
    j = r.json()
    ks = [p['k'] for p in j['points']]
    assert 0 in ks


def test_curve_fit_band_coverage(monkeypatch):
    def fake_run_base_query(filters, db, status_target='ALL', *, select_meta=True, start_from_first_disb=False):
        rows = []
        for pid in range(10):
            for k in range(0, 60, 3):
                z = -1.0 + 0.12*k + (-0.0005)*(k**2)
                hd = 1.0/(1.0 + pow(2.718281828, -z))
                d = max(0.0, min(1.0, hd + random.uniform(-0.05, 0.05)))
                rows.append((f"P{pid}", None, k, d, 1_000_000.0, 'XX', 0, 11, 111))
        return rows

    monkeypatch.setattr('app._run_base_query', fake_run_base_query)

    payload = {
        "macrosectors": [11,22,33,44,55,66],
        "modalities": [111,222,333,444],
        "countries": [],
        "ticketMin": 0,
        "ticketMax": 1_000_000_000,
        "yearFrom": 2015,
        "yearTo": 2024,
        "onlyExited": True,
        "bandCoverage": 0.9,
    }
    r = client.post('/api/curves/fit', json=payload)
    assert r.status_code == 200
    j = r.json()
    assert 'bandsQuantile' in j and j['bandsQuantile']
    for b in j['bandsQuantile'][:5]:
        assert 0 <= b['hd_dn'] <= b['hd'] <= b['hd_up'] <= 1


def test_curve_fit_band_coverage_query_overrides_body(monkeypatch):
    def fake_run_base_query(filters, db, status_target='ALL', *, select_meta=True, start_from_first_disb=False):
        rows = []
        for pid in range(10):
            for k in range(0, 60, 3):
                z = -1.0 + 0.12*k + (-0.0005)*(k**2)
                hd = 1.0/(1.0 + pow(2.718281828, -z))
                d = max(0.0, min(1.0, hd + random.uniform(-0.05, 0.05)))
                rows.append((f"P{pid}", None, k, d, 1_000_000.0, 'XX', 0, 11, 111))
        return rows

    monkeypatch.setattr('app._run_base_query', fake_run_base_query)

    payload = {
        "macrosectors": [11,22,33,44,55,66],
        "modalities": [111,222,333,444],
        "countries": [],
        "ticketMin": 0,
        "ticketMax": 1_000_000_000,
        "yearFrom": 2015,
        "yearTo": 2024,
        "onlyExited": True,
        "bandCoverage": 0.7,
    }
    r = client.post('/api/curves/fit?bandCoverage=0.9', json=payload)
    assert r.status_code == 200
    j = r.json()
    assert 'bandsQuantile' in j and j['bandsQuantile']
    for b in j['bandsQuantile'][:5]:
        assert 0 <= b['hd_dn'] <= b['hd'] <= b['hd_up'] <= 1


def test_cors_preflight():
    r = client.options(
        '/api/health',
        headers={
            'Origin': 'https://clientcurvasdesembolso-production.up.railway.app',
            'Access-Control-Request-Method': 'GET',
        },
    )
    assert r.status_code == 204
    assert r.headers['access-control-allow-origin'] == 'https://clientcurvasdesembolso-production.up.railway.app'


def test_project_timeseries_single_query():
    Row = namedtuple(
        'Row',
        'ym disb_month approved approval_date country_id macrosector_id modality_id',
    )
    rows = [
        Row(date(2020, 2, 1), 50.0, 300.0, date(2020, 1, 15), 'AA', 11, 111),
        Row(date(2020, 3, 1), 250.0, 300.0, date(2020, 1, 15), 'AA', 11, 111),
        Row(date(2021, 2, 1), 200.0, 300.0, date(2020, 1, 15), 'AA', 11, 111),
    ]

    class DummyResult:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

    class DummyDB:
        def __init__(self, rows):
            self.rows = rows
            self.call_count = 0

        def execute(self, sql, params):
            self.call_count += 1
            assert 'WITH t AS' in str(sql)
            return DummyResult(self.rows)

    db = DummyDB(rows)
    resp = project_timeseries('P1', 2020, 2021, False, db)

    assert db.call_count == 1
    assert resp.project.approved_amount == 300.0
    assert resp.project.country_id == 'AA'
    ks = [p.k for p in resp.series]
    assert ks == [1, 2, 13]
    disb_months = [p.disb_month for p in resp.series]
    assert disb_months == [50.0, 250.0, 200.0]
    cum = [p.disb_cum_usd for p in resp.series]
    assert cum == [50.0, 300.0, 500.0]
    ds = [p.d for p in resp.series]
    assert ds[0] == pytest.approx(1 / 6)
    assert ds[1] == 1.0
    assert ds[2] == 1.0


