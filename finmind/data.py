"""Time-aligned data: row i is observed at EOD i; label i uses input < i."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_NAMES = ("open", "close", "volatility", "sharpe")


@dataclass(frozen=True)
class MarketData:
    dates: np.ndarray
    ohlcv: np.ndarray
    source: str


@dataclass(frozen=True)
class PreparedData:
    dates: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple
    source: str


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("Scaler requires at least two finite rows")
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std > 1e-8, std, 1.0))

    def transform(self, values):
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values):
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean

    def as_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def validate_market(dates, values, source):
    dates = np.asarray(dates, dtype="datetime64[D]")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(dates), 5) or len(dates) < 3:
        raise ValueError("Expected at least three OHLCV rows")
    if np.isnat(dates).any() or not np.isfinite(values).all():
        raise ValueError("Missing dates or non-finite OHLCV are not silently filled")
    order = np.argsort(dates, kind="stable")
    dates, values = dates[order], values[order]
    if (np.diff(dates) <= np.timedelta64(0, "D")).any():
        raise ValueError("Duplicate trading dates; use one asset per CSV")
    o, h, low, c, v = values.T
    if (values[:, :4] <= 0).any() or (v < 0).any():
        raise ValueError("OHLC must be positive; volume must be nonnegative")
    if (h < np.maximum(o, c) - 1e-6).any() or (low > np.minimum(o, c) + 1e-6).any():
        raise ValueError("Invalid OHLC bounds; verify consistent adjustment")
    return MarketData(dates, values, source)


def load_csv(path, price_mode, start="2015-01-01", end="2024-06-30"):
    """adj-close adjusts every OHLC by Adj Close / Close; volume is as supplied."""
    if price_mode not in {"already-adjusted", "adj-close"}:
        raise ValueError("Explicit price_mode is required for a real CSV")
    rows, dates = [], []
    normalize = lambda k: k.strip().lower().replace("_", "").replace(" ", "")
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for line, original in enumerate(reader, 2):
            row = {normalize(k): v for k, v in original.items() if k is not None}
            try:
                date = np.datetime64(row["date"][:10], "D")
                if np.isnat(date):
                    raise ValueError("NaT date")
                if date < np.datetime64(start) or date > np.datetime64(end):
                    continue
                values = [float(row[k]) for k in ("open", "high", "low", "close", "volume")]
                if price_mode == "adj-close":
                    factor = float(row["adjclose"]) / values[3]
                    values[:4] = [x * factor for x in values[:4]]
                dates.append(date)
                rows.append(values)
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                raise ValueError(f"Invalid CSV row {line}: {exc}") from exc
    return validate_market(dates, rows, f"{Path(path).resolve()} ({price_mode})")


def synthetic_market(n=1700, seed=9416042):
    """Causal synthetic OHLCV with clustered volatility; never a paper dataset."""
    if n < 100:
        raise ValueError("Synthetic demonstration requires >=100 rows")
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    var = np.full(n, 0.0001)
    for t in range(1, n):
        var[t] = 0.000002 + 0.10 * r[t - 1] ** 2 + 0.88 * var[t - 1]
        r[t] = 0.0002 + 0.12 * r[t - 1] + np.sqrt(var[t]) * rng.normal()
    close = 100 * np.exp(np.cumsum(r))
    previous = np.r_[100.0, close[:-1]]
    opening = previous * np.exp(rng.normal(0, 0.003, n))
    spread = np.abs(rng.normal(0, 0.006, n)) + 0.001
    high = np.maximum(opening, close) * (1 + spread)
    low = np.minimum(opening, close) / (1 + spread)
    volume = np.exp(rng.normal(14, 0.4, n))
    dates = np.busday_offset("2015-01-01", np.arange(n))
    return validate_market(dates, np.column_stack([opening, high, low, close, volume]),
                           f"synthetic(seed={seed}, n={n}); not market data")


def rolling_moments(x, window):
    """O(N) trailing moments, ddof=1. No centered windows or future filling."""
    x = np.asarray(x, dtype=np.float64)
    if window < 2 or window > len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid rolling window/input")
    # Center first to reduce cancellation in the cumulative second moment.
    origin = x[0]
    z = x - origin
    c1, c2 = np.r_[0.0, np.cumsum(z)], np.r_[0.0, np.cumsum(z * z)]
    sums, squares = c1[window:] - c1[:-window], c2[window:] - c2[:-window]
    mean, std = np.full(len(x), np.nan), np.full(len(x), np.nan)
    mean[window - 1:] = sums / window + origin
    std[window - 1:] = np.sqrt(np.maximum((squares - sums ** 2 / window) / (window - 1), 0))
    return mean, std


def prepare(market, risk_window=20, annual_rf=0.0, volatility="rolling"):
    """Implementation assumption: trailing W-day Sharpe at label date, annualized.

    Predicting label i from input <=i-1 forecasts the updated rolling statistic.
    It is not a Sharpe ratio calculated from one return, nor a future W-day label.
    """
    if not np.isfinite(annual_rf) or annual_rf <= -1:
        raise ValueError("annual_rf must be finite and > -1")
    if volatility not in {"rolling", "parkinson"}:
        raise ValueError("Unknown volatility estimator")
    o, h, low, c, v = market.ohlcv.T
    returns = c[1:] / c[:-1] - 1
    mean, std = rolling_moments(returns, risk_window)
    mean, std = np.r_[np.nan, mean], np.r_[np.nan, std]
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    rolling_vol = np.sqrt(252) * std
    sharpe = np.sqrt(252) * (mean - daily_rf) / np.maximum(std, 1e-8)
    log_range = np.log(h / low)
    sigma = rolling_vol if volatility == "rolling" else np.sqrt(252 / (4 * np.log(2))) * log_range
    ret = np.r_[np.nan, returns]
    gap = np.r_[np.nan, o[1:] / c[:-1] - 1]
    momenta = []
    for lag in (5, 20):
        result = np.full(len(c), np.nan)
        result[lag:] = c[lag:] / c[:-lag] - 1
        momenta.append(result)
    features = np.column_stack([o, h, low, c, np.log1p(v), ret, log_range,
                               gap, c / o - 1, *momenta, rolling_vol, sharpe])
    targets = np.column_stack([o, c, sigma, sharpe])
    warmup = max(risk_window, 20)
    features, targets = features[warmup:], targets[warmup:]
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite features/targets after warmup")
    names = ("open", "high", "low", "close", "log_volume", "return", "log_range",
             "overnight_gap", "intraday_return", "momentum_5", "momentum_20",
             "rolling_volatility", "rolling_sharpe")
    return PreparedData(market.dates[warmup:], features, targets, names, market.source)


class WindowDataset(Dataset):
    """Keep O(N F) storage; slice overlapping [L,F] views only when fetched."""
    def __init__(self, features, targets, indices, lookback):
        self.x = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
        self.y = torch.as_tensor(np.ascontiguousarray(targets), dtype=torch.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.lookback = lookback
        if lookback < 1 or len(self.indices) == 0 or self.indices.min() < lookback or self.indices.max() >= len(self.x):
            raise ValueError("Invalid window indices / insufficient history")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        return self.x[i - self.lookback:i], self.y[i]
