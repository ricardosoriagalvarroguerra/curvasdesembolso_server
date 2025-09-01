import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.optimize import curve_fit


def months_between(d1: date | None, d2: date | None) -> int | None:
        """Return the number of whole months between ``d1`` and ``d2``.

        If either date is ``None`` this function returns ``None``. When the end
        date precedes the start date, ``0`` is returned.
        """

        if d1 is None or d2 is None:
                return None
        if d2 < d1:
                return 0
        rd = relativedelta(d2, d1)
        return rd.years * 12 + rd.months


def logistic3(k, b0, b1, b2):
	"""3-parameter logistic functional form used for historical disbursement curve."""
	z = b0 + b1 * k + b2 * (k ** 2)
	return 1.0 / (1.0 + np.exp(-z))


def fit_logistic3(
	k_array,
	d_array,
	p0=(-1.0, 0.10, -0.0005),
	bounds=((-20.0, -2.0, -1.0), (20.0, 2.0, 1.0)),
):
	"""Fit 3-parameter logistic hd(k) to points (k, d) via NLS and return (b0, b1, b2, sigma).

	- x: months since approval (k)
	- y: cumulative proportion (d) clipped to [0,1]
	- sigma: std. dev. of residuals with ddof=1
	"""
	x = np.asarray(k_array, dtype=float)
	y = np.asarray(d_array, dtype=float)
	ok = np.isfinite(x) & np.isfinite(y)
	x, y = x[ok], y[ok]
	if x.size < 30:
		raise ValueError("Insuficientes puntos para ajustar la curva (min 30).")
	params, _ = curve_fit(
		logistic3, x, y, p0=p0, bounds=bounds, maxfev=400_000
	)
	b0, b1, b2 = [float(p) for p in params]
	hd = logistic3(x, b0, b1, b2)
	sigma = float(np.std(y - hd, ddof=1))
	return b0, b1, b2, sigma


