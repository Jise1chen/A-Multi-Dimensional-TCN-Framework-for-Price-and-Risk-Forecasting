import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from finmind.data import (MarketData, Standardizer, WindowDataset, load_csv, prepare,
                          rolling_moments, synthetic_market, validate_market)
from finmind.metrics import max_drawdown, regression_metrics, sharpe
from finmind.model import CausalAttention, CausalConv, FINMIND, UncertaintyLoss
from finmind.train import Config, fit_scalers, modeling_targets, predict, run_walk_forward, split_indices


class DataTests(unittest.TestCase):
    def setUp(self):
        self.market = synthetic_market(240)

    def test_rolling_matches_explicit_trailing_windows(self):
        x = np.random.default_rng(8).normal(size=100)
        mean, std = rolling_moments(x, 20)
        windows = np.lib.stride_tricks.sliding_window_view(x, 20)
        np.testing.assert_allclose(mean[19:], windows.mean(1), atol=1e-12)
        np.testing.assert_allclose(std[19:], windows.std(1, ddof=1), atol=1e-12)
        self.assertTrue(np.isnan(mean[:19]).all())

    def test_future_mutations_do_not_change_past_features_or_targets(self):
        values = self.market.ohlcv.copy()
        values[150:, :4] *= 3
        changed = MarketData(self.market.dates, values, "mutated future")
        a, b = prepare(self.market), prepare(changed)
        np.testing.assert_allclose(a.features[:130], b.features[:130])
        np.testing.assert_allclose(a.targets[:130], b.targets[:130])

    def test_next_day_window_alignment(self):
        data = prepare(self.market)
        ds = WindowDataset(data.features, data.targets, [40, 41], 32)
        x, y = ds[0]
        np.testing.assert_allclose(x[-1], data.features[39], rtol=1e-6)
        np.testing.assert_allclose(y, data.targets[40], rtol=1e-6)
        self.assertEqual(tuple(x.shape), (32, 13))
        self.assertLess(data.dates[39], data.dates[40])

    def test_scaler_never_fits_validation_or_test(self):
        data = prepare(self.market)
        ids = np.arange(32, 100)
        xs, ys = fit_scalers(data, ids, 32)
        data.features[99:] = 1e6
        data.targets[100:] = 1e6
        xs2, ys2 = fit_scalers(data, ids, 32)
        np.testing.assert_array_equal(xs.mean, xs2.mean)
        np.testing.assert_array_equal(ys.mean, ys2.mean)

    def test_risk_label_formula(self):
        data = prepare(self.market, risk_window=20)
        close = self.market.ohlcv[:, 3]
        r = close[1:21] / close[:20] - 1
        self.assertAlmostEqual(data.targets[0, 2], np.sqrt(252) * r.std(ddof=1))
        self.assertAlmostEqual(data.targets[0, 3], np.sqrt(252) * r.mean() / r.std(ddof=1))

    def test_scaler_round_trip_constant_feature(self):
        x = np.array([[2, 4], [2, 6], [2, 8]], dtype=float)
        scaler = Standardizer.fit(x)
        np.testing.assert_allclose(scaler.inverse(scaler.transform(x)), x, atol=1e-6)

    def test_csv_adjusts_all_ohlc_and_sorts_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
                for day in [3, 1, 2]:
                    writer.writerow([f"2021-01-0{day}", 100, 120, 90, 110, 55, 1000])
            market = load_csv(path, "adj-close")
            np.testing.assert_allclose(market.ohlcv[0], [50, 60, 45, 55, 1000])
            self.assertEqual(str(market.dates[0]), "2021-01-01")

    def test_duplicate_dates_rejected(self):
        dates = self.market.dates.copy()
        dates[5] = dates[4]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_market(dates, self.market.ohlcv, "test")

    def test_unavailable_paper_test_period_fails(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            split_indices(prepare(self.market), Config(lookback=32), demo=False)

    def test_residual_targets_use_previous_close_and_reconstruct_levels(self):
        data = prepare(self.market)
        original = data.targets.copy()
        residual = modeling_targets(data, "residual")
        np.testing.assert_allclose(residual[1:, :2] + original[:-1, 1, None], original[1:, :2])
        np.testing.assert_array_equal(data.targets, original)
        cfg = Config(lookback=16, hidden=4, levels=1, price_target="residual")
        xs, ys = fit_scalers(data, np.arange(16, 100), 16, "residual")
        class ZeroModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(len(x), 4)
        indices = np.array([100, 101])
        prediction, error = predict(ZeroModel(), data, indices, xs, ys, cfg)
        np.testing.assert_allclose(prediction[:, :2], ys.mean[:2] + data.targets[indices - 1, 1, None])
        np.testing.assert_allclose(error, (prediction - original[indices]) / ys.scale)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_convolution_is_causal(self):
        conv = CausalConv(3, 4, 3, dilation=4)
        x = torch.randn(2, 3, 40)
        altered = x.clone()
        altered[:, :, 20:] += 100
        torch.testing.assert_close(conv(x)[:, :, :20], conv(altered)[:, :, :20])

    def test_attention_causal_normalized_and_fused_equivalent(self):
        module = CausalAttention(8)
        h = torch.randn(2, 12, 8)
        valid = torch.ones(2, 12, dtype=torch.bool)
        valid[0, :3] = False
        context, weights = module(h, valid, True)
        fused, _ = module(h, valid, False)
        torch.testing.assert_close(context, fused, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(weights.sum(-1), valid.float())
        self.assertEqual(weights.triu(1).abs().sum().item(), 0)
        self.assertEqual(weights[0, :, :3].abs().sum().item(), 0)

    def test_padding_does_not_change_predictions(self):
        model = FINMIND(3, hidden=8, levels=2, dropout=0).eval()
        x = torch.randn(2, 20, 3)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[:, :5] = False
        changed = x.clone()
        changed[:, :5] = 1e8
        torch.testing.assert_close(model(x, mask), model(changed, mask))
        with self.assertRaisesRegex(ValueError, "observed"):
            model(x, torch.zeros_like(mask))

    def test_heads_are_independent_and_both_receive_gradients(self):
        model = FINMIND(3, hidden=8, levels=2, dropout=0)
        pids = {id(p) for p in model.price_attention.parameters()}
        rids = {id(p) for p in model.risk_attention.parameters()}
        self.assertFalse(pids & rids)
        pred = model(torch.randn(4, 20, 3))
        objective = UncertaintyLoss()
        objective(pred, torch.randn(4, 4)).backward()
        self.assertGreater(model.price_attention.q.weight.grad.norm().item(), 0)
        self.assertGreater(model.risk_attention.q.weight.grad.norm().item(), 0)
        self.assertTrue(torch.isfinite(objective.log_variance.grad).all())

    def test_uncertainty_loss_formula(self):
        objective = UncertaintyLoss()
        with torch.no_grad():
            objective.log_variance.fill_(np.log(4))
        pred, actual = torch.ones(3, 4), torch.zeros(3, 4)
        expected = 4 * (0.5 / 4 + np.log(2))
        self.assertAlmostEqual(objective(pred, actual).item(), expected, places=6)

    def test_expanding_fold_labels_precede_test_and_test_not_dropped(self):
        data = prepare(synthetic_market(180))
        cfg = Config(lookback=16, hidden=4, levels=1, epochs=1, batch_size=32, refit_every=17)
        split = split_indices(data, cfg, demo=True)
        with tempfile.TemporaryDirectory() as tmp:
            pred, errors, audit = run_walk_forward(data, cfg, split, 123, "decoupled", tmp)
        self.assertEqual(pred.shape, (len(split[2]), 4))
        self.assertEqual(errors.shape, pred.shape)
        self.assertTrue(np.isfinite(pred).all())
        folds = audit["folds"]
        self.assertEqual(sum(f["test_count"] for f in folds), len(split[2]))
        for fold in folds:
            self.assertLess(fold["train_target_end"], fold["test_start"])
            self.assertLess(fold["scaler_feature_end"], fold["train_target_end"])
        self.assertGreater(folds[-1]["train_count"], folds[0]["train_count"])


class MetricTests(unittest.TestCase):
    def test_first_day_drawdown_is_counted(self):
        self.assertAlmostEqual(max_drawdown([-0.1, 0.05]), -0.1)
        self.assertEqual(max_drawdown([]), 0)
        self.assertEqual(max_drawdown([-1.0]), -1)

    def test_degenerate_metrics_are_explicit(self):
        self.assertIsNone(sharpe([0, 0]))
        self.assertIsNone(regression_metrics([1, 1], [1, 2])["r2"])
        with self.assertRaises(ValueError):
            regression_metrics([1], [float("nan")])


if __name__ == "__main__":
    unittest.main()
