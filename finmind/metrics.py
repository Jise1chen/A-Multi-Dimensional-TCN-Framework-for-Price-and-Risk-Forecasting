import numpy as np


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require aligned, finite arrays")
    residual = predicted - actual
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {"mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r2": float(1 - np.sum(residual ** 2) / denominator) if denominator > 1e-12 else None}


def sharpe(pnl, annual_rf=0.0):
    x = np.asarray(pnl, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or annual_rf <= -1 or not np.isfinite(annual_rf):
        raise ValueError("Invalid returns/risk-free rate")
    if len(x) < 2 or x.std(ddof=1) < 1e-12:
        return None
    return float(np.sqrt(252) * (x.mean() - ((1 + annual_rf) ** (1 / 252) - 1)) / x.std(ddof=1))


def max_drawdown(pnl):
    x = np.asarray(pnl, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < -1).any():
        raise ValueError("Simple returns must be finite and >= -1")
    # Include initial capital so a first-day loss is counted.
    wealth = np.r_[1.0, np.cumprod(1 + x)]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
