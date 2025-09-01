import os
import random
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(__file__))
from app import app


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


