"""Run an explicitly labeled synthetic demonstration or an OHLCV experiment."""
import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from finmind.data import TARGET_NAMES, load_csv, prepare, synthetic_market
from finmind.metrics import regression_metrics
from finmind.train import Config, run_walk_forward, split_indices


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows, header):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--csv", type=Path)
    parser.add_argument("--symbol", default="SYNTHETIC")
    parser.add_argument("--price-mode", choices=["already-adjusted", "adj-close"])
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--n", type=int, default=1700)
    parser.add_argument("--data-seed", type=int, default=9416042)
    parser.add_argument("--seeds", nargs="+", type=int, default=[9416042])
    parser.add_argument("--variants", nargs="+", choices=["decoupled", "shared", "none", "price_only"], default=["decoupled"])
    parser.add_argument("--lookback", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--refit-every", type=int, default=63, help="0 = one fixed model; >0 = expanding-window test blocks")
    parser.add_argument("--risk-window", type=int, default=20)
    parser.add_argument("--annual-rf", type=float, default=0.0)
    parser.add_argument("--volatility", choices=["rolling", "parkinson"], default="rolling")
    parser.add_argument("--val-start", default="2020-01-01")
    parser.add_argument("--test-start", default="2021-01-01")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--price-target", choices=["level", "residual"], default="level",
                        help="residual is an explicit extension, not the paper's level-target setup")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    for name in ("lookback", "hidden", "levels", "kernel", "batch_size", "epochs", "patience", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.refit_every < 0 or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid optimizer/dropout/refit configuration")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("Seeds and variants must be unique")
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if args.csv and not args.price_mode:
        parser.error("CSV input requires explicit --price-mode")
    if args.csv and args.symbol == "SYNTHETIC":
        parser.error("CSV input requires --symbol")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    market = synthetic_market(args.n, args.data_seed) if args.demo else load_csv(args.csv, args.price_mode, args.start, args.end)
    data = prepare(market, args.risk_window, args.annual_rf, args.volatility)
    split = split_indices(data, cfg, args.demo, args.val_start, args.test_start)
    train, val, test = split
    args.output.mkdir(parents=True, exist_ok=True)
    target = data.targets[test]
    # A meaningful price baseline: tomorrow's open AND close equal today's close.
    naive = data.targets[test - 1].copy()
    naive[:, 0] = data.targets[test - 1, 1]
    metrics_rows, prediction_rows, records = [], [], []
    def evaluate(name, seed, pred, standardized_error=None):
        result = {target_name: regression_metrics(target[:, j], pred[:, j])
                  for j, target_name in enumerate(TARGET_NAMES[:pred.shape[1]])}
        for target_name, metrics in result.items():
            metrics_rows.append([args.symbol, name, seed, target_name, *metrics.values()])
        for row, i in enumerate(test):
            for j, target_name in enumerate(TARGET_NAMES[:pred.shape[1]]):
                prediction_rows.append([args.symbol, name, seed, str(data.dates[i - 1]), str(data.dates[i]),
                                        target_name, float(target[row, j]), float(pred[row, j])])
        entry = {"variant": name, "seed": seed, "metrics": result,
                 "price_rmse": float(np.sqrt(np.mean((pred[:, :2] - target[:, :2]) ** 2)))}
        if pred.shape[1] == 4:
            entry["negative_volatility_predictions"] = int((pred[:, 2] < 0).sum())
        if standardized_error is not None:
            entry["standardized_joint_mse"] = float(np.mean(standardized_error ** 2))
        return entry
    baseline = evaluate("persistence", None, naive)
    for variant in args.variants:
        for seed in args.seeds:
            pred, err, audit = run_walk_forward(data, cfg, split, seed, variant, args.output / "checkpoints")
            record = evaluate(variant, seed, pred, err)
            record["training_audit"] = audit
            records.append(record)
    aggregate = {}
    for variant in args.variants:
        runs = [r for r in records if r["variant"] == variant]
        aggregate[variant] = {}
        for target_name in runs[0]["metrics"]:
            aggregate[variant][target_name] = {}
            for metric in ("mae", "rmse", "r2"):
                values = [r["metrics"][target_name][metric] for r in runs]
                aggregate[variant][target_name][metric] = {
                    "mean": float(np.mean(values)) if all(v is not None for v in values) else None,
                    "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None,
                    "seeds": len(values)}
    reasons = ["Risk-label definitions and model/training hyperparameters are explicit reconstruction assumptions.",
               "Original author code, exact seed list, and rolling refit schedule are not supplied by the paper.",
               "ARIMA/LSTM/Transformer comparisons and statistical significance are not reproduced."]
    if args.demo:
        reasons.insert(0, "Synthetic OHLCV only: these are software validation results, not paper results.")
    elif market.dates[-1] < np.datetime64("2024-06-28") or market.dates[0] > np.datetime64("2015-01-02"):
        reasons.insert(0, "Input data does not cover the paper's full 2015-2024-06 period.")
    if args.price_target == "residual":
        reasons.append("Price residual targets are an algorithmic extension, not a literal reproduction of the paper.")
    summary = {
        "paper": "https://doi.org/10.20944/preprints202510.2049.v1", "verdict": "Partial",
        "status": "synthetic_smoke_test" if args.demo else "method_reconstruction_on_csv",
        "limitations": reasons, "symbol": args.symbol, "source": data.source,
        "input_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest() if args.csv else None,
        "raw_rows": len(market.dates), "data_start": str(market.dates[0]), "data_end": str(market.dates[-1]),
        "feature_names": list(data.feature_names), "target_names": list(TARGET_NAMES),
        "risk_definition": {"volatility": args.volatility, "window": args.risk_window, "annual_rf": args.annual_rf,
                            "annualization": 252, "sharpe": "trailing-window statistic ending on target day, ddof=1"},
        "config": asdict(cfg), "seeds": args.seeds, "data_seed": args.data_seed if args.demo else None,
        "split": {name: {"n": len(ids), "start": str(data.dates[ids[0]]), "end": str(data.dates[ids[-1]])}
                  for name, ids in zip(("train", "validation", "test"), split)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "baseline": baseline, "aggregate": aggregate, "runs": records,
        "metric_notes": "Price metrics are USD only for USD input. Risk metrics are separate annualized units. "
                        "Standardized joint MSE is a diagnostic, not the learned uncertainty training objective. "
                        "No trading performance is inferred from forecast metrics."
    }
    save_csv(args.output / "predictions.csv", prediction_rows,
             ["symbol", "variant", "seed", "feature_end", "target_date", "target", "actual", "predicted"])
    save_csv(args.output / "metrics.csv", metrics_rows,
             ["symbol", "variant", "seed", "target", "mae", "rmse", "r2"])
    save_json(args.output / "summary.json", summary)
    print("FIN_MIND_SUMMARY " + json.dumps({"verdict": summary["verdict"], "status": summary["status"],
          "output": str(args.output.resolve()), "aggregate": aggregate}, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
