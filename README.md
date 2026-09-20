# A Multi-Dimensional TCN Framework for Price and Risk Forecasting

Research code for reconstructing [FIN-MIND v1](https://doi.org/10.20944/preprints202510.2049.v1): joint next-day stock price and risk forecasting with a shared causal temporal convolutional network and task-specific attention.

The pipeline:

1. Validate daily OHLCV data and apply a consistent price-adjustment convention.
2. Construct causal features, historical windows, and four aligned prediction targets.
3. Train a shared TCN with separate price and risk attention modules.
4. Select the training duration on a chronological validation set and refit on expanding historical samples.
5. Evaluate predictions against persistence and architecture ablations, retaining daily outputs and training metadata.

**Reproduction status: Partial.** This is an independent method reconstruction, not the authors' official implementation or a numerical reproduction of their results. Risk-label definitions and several training settings are explicit implementation choices. The currently downloaded reference datasets do not cover the paper's full evaluation period.

This repository contains source code, tests, dependency specifications, downloaded reference data, development checkpoints, and experiment outputs. The virtual environment and Python caches are excluded. Machine-specific absolute paths have been normalized in the publication copy; input files are under `data/raw/`. `PROJECT_COPY_MANIFEST.json` records the original local copy operation, while `PUBLICATION_MANIFEST.json` records the files in this publication snapshot. The included datasets are incomplete reference snapshots and the checkpoints are development artifacts, not the authors' models.

## Method overview

Given information available at the close of trading day `t`, the model predicts four targets for the next observed trading day:

- Opening price.
- Closing price.
- Volatility scale.
- Sharpe ratio.

A shared residual TCN extracts temporal features using left-padded dilated convolutions. Two independent attention modules then aggregate the shared sequence for the price and risk tasks. Each task has its own output head. All four training targets are standardized separately, and learned uncertainty weights balance their losses.

The implementation uses 13 input features: open, high, low, close, log volume, daily return, log high-low range, overnight gap, intraday return, 5-day momentum, 20-day momentum, trailing volatility, and trailing Sharpe. This feature set is a local reconstruction choice, not a confirmed author-provided specification.

```text
Daily OHLCV observations
          |
          v
Validated data and causal features [N, 13]
          |
          v
Chronological splits and training-only standardization
          |
          v
Historical windows [B, L, 13]
          |
          v
Shared causal TCN representations [B, L, D]
          |
          +--> Price attention --> Price head --> Open, close
          |
          +--> Risk attention  --> Risk head  --> Volatility, Sharpe
          |
          v
Expanding-window evaluation and inverse scaling
          |
          v
Daily predictions, metrics, checkpoints, and audit metadata
```

Here `B` is batch size, `L` is lookback length, and `D` is hidden width. The model produces point forecasts, not generated market scenarios or a complete predictive distribution. Independent attention parameters do not eliminate all possible task conflicts: both tasks still update the shared backbone.

## Reference configurations

The following values describe this implementation and the existing development run. They are not presented as the paper's exact hyperparameters.

| Setting | CLI default | Existing TSLA development run |
|---|---|---|
| Lookback length | 192 trading days | 64 trading days |
| Input features | 13 | 13 |
| Hidden width | 32 | 16 |
| Residual TCN blocks | 6 | 4 |
| Convolutions per block / kernel size | 2 / 3 | 2 / 3 |
| Batch size | 64 | 64 |
| Maximum training epochs | 50 | 15 |
| Early-stopping patience | 8 | 4 |
| Optimizer / learning rate | AdamW / 0.001 | Same |
| Risk window / annualization factor | 20 / 252 | Same |
| Test-block refit interval | 63 trading days | Same; four test blocks |
| Default model seed | 9416042 | One seed: 9416042 |
| Price target | Price level | Level and residual runs stored separately |
| Test coverage | Depends on input data | 2021-01-04 through 2021-10-14; 198 observations |

CPU execution is the default and deterministic algorithms are enabled. `--device cuda` and `--device mps` are also supported where the required operations are available. Bitwise equivalence across hardware and software versions is not guaranteed; unsupported deterministic operations may raise an error.

## Current empirical interpretation

These figures describe a limited development experiment, not a fully tuned study or a statistical conclusion. Closing-price errors use the USD adjustment convention of the downloaded TSLA file.

| Method | Closing-price MAE | Closing-price RMSE | Closing-price R-squared |
|---|---:|---:|---:|
| Persistence: tomorrow's price equals today's close | 15.9651 | 22.3637 | 0.9163 |
| Decoupled-attention TCN with direct price targets | 144.9833 | 187.8096 | -4.8998 |
| Extension: price residuals around today's close | 16.0684 | 22.4973 | 0.9153 |

The residual extension substantially reduces the error of direct price regression in this run, but it does not outperform persistence. The current risk forecasts also do not outperform their persistence baseline. No confidence intervals or hypothesis tests establish general superiority.

The residual extension was introduced after inspecting the initial development results. These comparisons are exploratory; further confirmation requires a fixed protocol and an untouched evaluation period. The two-seed synthetic ablation experiments validate the software pipeline and are not evidence of market forecasting performance.

## Repository map

| Path | Purpose |
|---|---|
| `baseline_original.py` | Original user-provided synthetic fixed-filter example, with pasted formatting restored |
| `run_finmind.py` | Training, ablations, evaluation, and CSV/JSON export |
| `finmind/data.py` | OHLCV validation, causal features, four targets, window slicing, and scaling |
| `finmind/model.py` | Residual causal TCN, task-specific attention, output heads, and uncertainty-weighted loss |
| `finmind/train.py` | Validation-based epoch selection, expanding-window refits, and checkpoint saving |
| `finmind/metrics.py` | Regression metrics and corrected Sharpe and maximum-drawdown utilities |
| `tests/test_finmind.py` | Data alignment, leakage prevention, attention, loss, and metric checks |
| `data/provenance.json` | Download sources, date coverage, and file hashes |
| `requirements.txt` | Supported dependency ranges |
| `requirements-lock.txt` | Versions installed in the verified development environment |
| `pyproject.toml` | Package metadata and the optional installed `finmind` command |
| `PROJECT_VALIDATION.json` | Recorded project verification and documentation updates |
| `outputs/` | Local development predictions, metrics, summaries, and checkpoints |

Additional interpretive documents are currently written in Chinese: the [research workflow guide](%E7%A0%94%E7%A9%B6%E6%B5%81%E7%A8%8B%E8%A7%A3%E8%AF%BB.md), [full logical interpretation](%E5%85%A8%E6%96%87%E9%80%BB%E8%BE%91%E8%A7%A3%E8%AF%BB.md), [original-code audit](paper_notes_zh.md), and [development results report](RESULTS.md). This README is self-contained in English for setup, method assumptions, execution, and reproduction boundaries.

## Installation and verification

Run all commands from this project directory. Clone the repository, then create a local virtual environment:

```bash
git clone https://github.com/Jise1chen/A-Multi-Dimensional-TCN-Framework-for-Price-and-Risk-Forecasting.git
cd A-Multi-Dimensional-TCN-Framework-for-Price-and-Risk-Forecasting
```

Install dependencies and run the data-free checks:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

The project requires Python 3.10 or later and has been exercised with Python 3.12. `requirements-lock.txt` records the actual development package versions; use it instead of `requirements.txt` when those versions are available for your platform. The commands above use a macOS/Linux virtual-environment layout.

The existing 18-test suite passes without market data or pretrained checkpoints. It checks chronological alignment, invariance to future-data changes, training-only scaling, padding and causal masks, independent attention parameters and gradients, the loss formula, residual reconstruction, and metric edge cases. Test success establishes software behavior, not forecasting superiority.

## Input data

Each CSV represents one asset and must include:

```text
Date,Open,High,Low,Close,Volume
```

Dates use ISO format, and all prices must use the same currency. Header matching normalizes case, spaces, and underscores.

- `--price-mode adj-close` additionally requires `Adj Close` or `Adj_Close`. Every OHLC price is multiplied by `Adj Close / Close`; volume retains the source convention.
- `--price-mode already-adjusted` explicitly declares that the supplied OHLC columns are consistently adjusted, avoiding a second adjustment.
- After sorting by date, the loader rejects duplicate dates, missing or non-finite values, nonpositive prices, negative volume, and invalid OHLC bounds.
- The pipeline does not fabricate holiday observations, backward-fill missing data, or concatenate different assets into a single sequence. Suspensions and unexpected missing trading days require source-level review.
- The default input date range is `2015-01-01` through `2024-06-30`; change it with `--start` and `--end`.
- Adjusted historical data can incorporate later corporate-action adjustments. The implementation checks sample and scaler boundaries but does not certify that an external dataset is a point-in-time historical vintage.

The downloaded AAPL reference file ends on **2017-12-28**, and the TSLA reference file ends on **2021-10-14**. Neither provides the full stated study period. This observation concerns the currently downloaded files and does not establish which versions the authors used. See `data/provenance.json` for the source URLs and SHA-256 hashes.

## Temporal alignment and risk targets

After feature warmup, let `i` denote a row in the prepared arrays:

```text
features[i-L : i] -> [open[i], close[i], volatility[i], sharpe[i]]
Last input date = dates[i-1]
Target date     = dates[i]
```

The default risk window is `W=20`. Daily simple returns are `r[i] = close[i] / close[i-1] - 1`, sample standard deviations use `ddof=1`, and the annualization factor is 252:

```text
volatility[i] = sqrt(252) * std(r[i-W+1:i+1])
rf_daily = (1 + annual_rf) ** (1/252) - 1
sharpe[i] = sqrt(252) * (mean(r[i-W+1:i+1]) - rf_daily)
            / max(std(r[i-W+1:i+1]), 1e-8)
```

Sharpe is a trailing-window statistic ending on the target date. The model forecasts its next-day update; it does not compute Sharpe from a single return or forecast a forward W-day holding-period statistic.

The annual risk-free rate defaults to zero and can be changed with `--annual-rf`. `--volatility parkinson` instead uses an annualized daily high-low log-range proxy for volatility; Sharpe continues to use the trailing return window. Treat these as separate, explicitly specified experiments, rather than choosing a definition after inspecting test performance.

The first `max(W,20)` raw rows provide feature warmup, and targets without L preceding prepared rows are skipped. The dataset therefore trains on complete windows. The model separately supports valid-length masks for padded sequences, and that behavior is tested.

## Training and evaluation protocol

Each residual block contains two left-padded convolutions. Dilation rates increase as 1, 2, 4, and so on. The implemented backbone has receptive field:

```text
1 + 2 * (kernel - 1) * (2**levels - 1)
```

Available real history is still limited by the input window L. The backbone avoids normalization across time that could mix later states into earlier representations.

Four architecture variants are available:

| Variant | Behavior |
|---|---|
| `decoupled` | Separate price and risk Q/K/V projections and MLP heads |
| `shared` | Shared attention with separate output heads |
| `none` | Mean pooling and linear output heads |
| `price_only` | Price attention and two price outputs only |

With separately standardized targets and `s_m = log(uncertainty_m**2)`, training minimizes:

```text
L = 0.5 * sum(exp(-s_m) * MSE_m + s_m)
```

AdamW updates the model and loss parameters, with gradient-norm clipping at 1. The loss parameters have no weight decay, and `s_m` is clamped to `[-10,10]` as a numerical guard. These learned task weights are distinct from the volatility values predicted by the risk head. Outputs are not clipped after regression; the summary records any negative volatility predictions.

The default real-data protocol is:

1. Train on 2015-2019 and validate on 2020. Select the training duration using validation standardized MSE only.
2. Discard the selection-stage weights and refit from scratch on the available 2015-2020 samples.
3. Refit from scratch after each 63-day test block, expanding the training history while retaining the selected epoch count. Use `--refit-every 0` for a single fixed training interval.
4. Earlier test observations may enter subsequent training only after their labels become available. The current test block's labels remain excluded.
5. Refit feature and target scalers using each fold's training data only. Feature statistics use unique dates rather than repeatedly counting overlapping windows.
6. Compare variants using the same features, target dates, splits, and available history. Training batches may be shuffled without changing within-window order or chronological split boundaries.

Synthetic demonstrations use a chronological 60%/20%/20% split. The refit interval is a reconstruction assumption; the paper does not provide a fully specified rolling schedule that can be copied exactly.

## Experiment commands

### Synthetic pipeline demonstration

This command checks the end-to-end workflow and runs four variants over two model seeds:

```bash
.venv/bin/python run_finmind.py --demo --n 700 \
  --lookback 64 --hidden 16 --levels 4 --epochs 8 --patience 3 \
  --refit-every 80 --variants decoupled shared none price_only \
  --seeds 9416042 9416043 --output outputs/demo
```

### Partial TSLA experiment

Use the downloaded reference data for the available test interval:

```bash
.venv/bin/python run_finmind.py --csv data/raw/TSLA.csv --symbol TSLA \
  --price-mode adj-close --lookback 64 --hidden 16 --levels 4 \
  --epochs 15 --patience 4 --refit-every 63 --output outputs/tsla_partial
```

### Optional price-residual extension

Add `--price-target residual` to the TSLA command and change the destination to `--output outputs/tsla_residual`.

The price targets become `open[t+1] - close[t]` and `close[t+1] - close[t]`. Predictions are converted back to price levels by adding the known `close[t]`; risk targets remain unchanged. This supplies an observed price anchor and reduces the need to extrapolate price levels. The default `level` mode retains direct price regression.

Residual targets are an explicit algorithmic extension, not a confirmed part of the original method. Because target scaling differs between modes, compare errors after inverse scaling rather than directly comparing their standardized joint losses.

### Experiment with complete data

After obtaining a suitable complete dataset, use the paper's calendar split with a disclosed local configuration:

```bash
.venv/bin/python run_finmind.py --csv data/AAPL_2015_2024.csv --symbol AAPL \
  --price-mode adj-close --lookback 192 --hidden 32 --levels 6 \
  --epochs 50 --patience 8 --refit-every 63 \
  --variants decoupled shared none price_only \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --output outputs/aapl
```

`data/AAPL_2015_2024.csv` is a user-supplied input path, not a bundled complete dataset. The listed seeds, window length, network width, training duration, and refit frequency are implementation choices, not an asserted reconstruction of the authors' exact configuration.

## Outputs and metric interpretation

| Output | Contents |
|---|---|
| `summary.json` | Source, input SHA-256, software versions, risk definitions, settings, per-seed metrics, cross-seed mean and sample standard deviation, fold dates, scaler parameters, and training histories |
| `predictions.csv` | One row per target prediction, including `feature_end`, `target_date`, observed value, and predicted value |
| `metrics.csv` | MAE, RMSE, and R-squared for each model, seed, and target |
| `checkpoints/*.pt` | Per-fold model and loss weights, feature and target scalers, and model configuration |

Price errors can be pooled across opening and closing targets when their units match. Volatility and Sharpe errors have different units and should not be collapsed into a raw `risk_rmse`.

`standardized_joint_mse` is an evaluation diagnostic, distinct from the uncertainty-weighted training objective. R-squared for a constant target and Sharpe for a zero-variance return sequence are reported as `None` rather than fabricated extreme values.

The `persistence` baseline predicts both next-day prices using today's close and carries forward today's risk indicators. More complex architecture alone is not evidence of improvement over this baseline.

Forecast metrics do not imply trading returns. Sharpe and maximum-drawdown utilities remain available independently; maximum drawdown includes the initial wealth of 1 so that a first-day loss is counted. No trading strategy performance is inferred from the forecasting outputs.

## Computational considerations

The implementation stores features as `[N,F]`, targets as `[N,4]`, and sample target indices. Windows are sliced on demand and assembled into batches by the data loader. Relative to pre-materializing `[N,L,F]` windows, resident dataset storage falls from `O(NLF)` to `O(NF + N)`; a batch still requires `O(BLF)` storage.

Trailing means and standard deviations use cumulative sums in `O(N)` time for a fixed window. Centering around the first input value reduces cancellation in the second-moment calculation.

Attention uses PyTorch scaled dot-product attention, which may use a fused kernel depending on the device. Arithmetic complexity remains `O(BL^2D)`, and masks require `O(BL^2)` storage. Longer windows, wider networks, and additional seeds increase training cost. This project does not claim to run faster than the original fixed-filter example.

## Reproducibility boundaries

- The original author's code, checkpoints, exact seed list, and complete rolling schedule have not been supplied.
- Risk-label formulas and model/training settings are disclosed reconstruction assumptions.
- The downloaded reference files do not cover the full stated experiment period.
- Complete ARIMA, LSTM, and Transformer comparisons, the authors' statistical evidence, and their figures have not been reproduced.
- Local development outputs are retained for inspection; they are not presented as an archival reproduction of the paper.
- Current test and experiment results establish a working implementation and limited empirical observations, not general forecasting superiority or trading value.

For subsequent studies, freeze the data and label definitions, record the exact source revision and environment, preserve training metadata and raw predictions, and evaluate on an untouched test period. The output verdict remains `Partial` until the unresolved reproduction requirements are addressed.
