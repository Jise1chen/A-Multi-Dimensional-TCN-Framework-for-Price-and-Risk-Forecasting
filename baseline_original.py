"""User-provided baseline, restored from pasted Markdown escaping.

This is a synthetic fixed-filter demonstration, not FIN-MIND.
"""
import csv
import json
from pathlib import Path

import numpy as np


def sharpe(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(252) * x.mean() / (x.std(ddof=1) + 1e-12))


def max_drawdown(pnl):
    wealth = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def save_summary(summary):
    Path("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


def save_csv(name, rows, header):
    with open(name, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    rng = np.random.default_rng(9416042)
    n = 1700
    x = rng.normal(0, 0.01, n)
    y = 0.5 * np.roll(x, 1) + 0.25 * np.roll(x, 5) + rng.normal(0, 0.012, n)
    y[:5] = y[5]
    forecast = 0.4 * np.roll(y, 1) + 0.4 * np.roll(y, 3) + 0.2 * np.roll(y, 8)
    forecast[:8] = forecast[8]
    risk = np.abs(np.diff(y, prepend=y[0]))
    joint_loss = float(np.mean((forecast[300:] - y[300:]) ** 2) + 0.5 * np.mean((np.roll(forecast, 1)[300:] - risk[300:]) ** 2))
    summary = {
        "price_rmse": float(np.sqrt(np.mean((forecast[300:] - y[300:]) ** 2))),
        "risk_rmse": float(np.sqrt(np.mean((np.roll(forecast, 1)[300:] - risk[300:]) ** 2))),
        "joint_loss": joint_loss,
        "signal_to_risk_ratio": float(np.mean(np.abs(forecast[300:])) / (np.mean(risk[300:]) + 1e-12)),
        "verdict": "Partial",
    }
    save_summary(summary)
    print("FIN_MIND_SUMMARY " + json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
