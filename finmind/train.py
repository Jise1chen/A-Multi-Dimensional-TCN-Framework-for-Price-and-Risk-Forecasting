"""Validation-only epoch selection followed by expanding-window refitting."""
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Standardizer, WindowDataset
from .model import FINMIND, UncertaintyLoss


@dataclass(frozen=True)
class Config:
    lookback: int = 192
    hidden: int = 32
    levels: int = 6
    kernel: int = 3
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 50
    patience: int = 8
    lr: float = 0.001
    weight_decay: float = 0.0001
    refit_every: int = 63
    device: str = "cpu"
    price_target: str = "level"


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def modeling_targets(data, price_target):
    """Optional extension: predict price innovation around last known close.

    The original observed labels remain unchanged for reporting. No future
    close is used as an anchor. Row zero is never a target for lookback >= 1.
    """
    if price_target == "level":
        return data.targets
    if price_target != "residual":
        raise ValueError("Unknown price target mode")
    targets = data.targets.copy()
    targets[1:, :2] -= data.targets[:-1, 1, None]
    targets[0, :2] = 0
    return targets


def fit_scalers(data, indices, lookback, price_target="level"):
    """Fit once per chronological training fold, without repeated window rows."""
    indices = np.asarray(indices)
    if len(indices) < 2 or not np.all(np.diff(indices) == 1):
        raise ValueError("Expected contiguous chronological training indices")
    xscale = Standardizer.fit(data.features[indices[0] - lookback:indices[-1]])
    yscale = Standardizer.fit(modeling_targets(data, price_target)[indices])
    return xscale, yscale


def create_model(data, cfg, variant):
    return FINMIND(len(data.feature_names), cfg.hidden, cfg.levels, cfg.kernel,
                   cfg.dropout, variant).to(cfg.device)


def train_model(data, train_indices, val_indices, cfg, seed, variant, epochs=None):
    seed_all(seed)
    xscale, yscale = fit_scalers(data, train_indices, cfg.lookback, cfg.price_target)
    features = xscale.transform(data.features)
    targets = yscale.transform(modeling_targets(data, cfg.price_target))
    train = WindowDataset(features, targets, train_indices, cfg.lookback)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = None
    if val_indices is not None:
        val_loader = DataLoader(WindowDataset(features, targets, val_indices, cfg.lookback),
                                batch_size=cfg.batch_size)
    model = create_model(data, cfg, variant)
    objective = UncertaintyLoss(2 if variant == "price_only" else 4).to(cfg.device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "weight_decay": cfg.weight_decay},
        {"params": objective.parameters(), "weight_decay": 0.0},
    ], lr=cfg.lr)
    best_score, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, (epochs or cfg.epochs) + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = objective(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(objective.parameters()), 1.0)
            optimizer.step()
            # Practical numerical guard; not a paper-specified hyperparameter.
            with torch.no_grad():
                objective.log_variance.clamp_(-10, 10)
            total += loss.item() * len(x)
            count += len(x)
        score = None
        if val_loader is not None:
            model.eval()
            squared, elements = 0.0, 0
            with torch.inference_mode():
                for x, y in val_loader:
                    pred = model(x.to(cfg.device))
                    err = (pred - y[:, :pred.shape[1]].to(cfg.device)).square()
                    squared += err.sum().item()
                    elements += err.numel()
            score = squared / elements
            if score < best_score - 1e-8:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(objective.state_dict()))
            else:
                stale += 1
        history.append({"epoch": epoch, "train_uncertainty_loss": total / count,
                        "validation_standardized_mse": score})
        if val_loader is not None and stale >= cfg.patience:
            break
    if val_loader is not None:
        if best_state is None:
            raise FloatingPointError("No finite validation result")
        model.load_state_dict(best_state[0])
        objective.load_state_dict(best_state[1])
    else:
        best_epoch = len(history)
    return model, objective, xscale, yscale, best_epoch, history


def predict(model, data, indices, xscale, yscale, cfg):
    ds = WindowDataset(xscale.transform(data.features), yscale.transform(modeling_targets(data, cfg.price_target)), indices, cfg.lookback)
    loader = DataLoader(ds, batch_size=cfg.batch_size)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for x, _ in loader:
            chunks.append(model(x.to(cfg.device)).cpu().numpy())
    standardized = np.concatenate(chunks)
    d = standardized.shape[1]
    pred = standardized.astype(np.float64) * yscale.scale[:d] + yscale.mean[:d]
    if cfg.price_target == "residual":
        pred[:, :2] += data.targets[np.asarray(indices) - 1, 1, None]
    error = (pred - data.targets[indices, :d]) / yscale.scale[:d]
    return pred, error


def split_indices(data, cfg, demo=False, val_start="2020-01-01", test_start="2021-01-01"):
    indices = np.arange(cfg.lookback, len(data.dates))
    if demo:
        a, b = int(len(indices) * 0.6), int(len(indices) * 0.8)
        train, val, test = indices[:a], indices[a:b], indices[b:]
    else:
        if np.datetime64(val_start) >= np.datetime64(test_start):
            raise ValueError("val_start must be earlier than test_start")
        dates = data.dates[indices]
        train = indices[dates < np.datetime64(val_start)]
        val = indices[(dates >= np.datetime64(val_start)) & (dates < np.datetime64(test_start))]
        test = indices[dates >= np.datetime64(test_start)]
    if min(len(train), len(val), len(test)) < 2:
        raise ValueError("Insufficient train/validation/test data for the requested chronological split")
    return train, val, test


def run_walk_forward(data, cfg, split, seed, variant, output_dir):
    train, val, test = split
    print(f"[{variant} seed={seed}] selecting epochs on validation only", flush=True)
    *_, selected_epochs, selection_history = train_model(data, train, val, cfg, seed, variant)
    width = cfg.refit_every or len(test)
    predictions, errors, fold_info = [], [], []
    for fold, begin in enumerate(range(0, len(test), width)):
        block = test[begin:begin + width]
        known = np.arange(cfg.lookback, block[0])
        print(f"[{variant} seed={seed}] fold={fold} train={len(known)} test={len(block)} epochs={selected_epochs}", flush=True)
        model, objective, xs, ys, _, history = train_model(data, known, None, cfg, seed, variant,
                                                         epochs=selected_epochs)
        pred, err = predict(model, data, block, xs, ys, cfg)
        predictions.append(pred)
        errors.append(err)
        checkpoint = Path(output_dir) / f"{variant}_seed{seed}_fold{fold}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "loss": objective.state_dict(),
                    "config": asdict(cfg), "variant": variant, "seed": seed,
                    "feature_names": list(data.feature_names), "x_scaler": xs.as_dict(),
                    "y_scaler": ys.as_dict(), "train_end": str(data.dates[known[-1]]),
                    "test_start": str(data.dates[block[0]])}, checkpoint)
        fold_info.append({"fold": fold, "train_count": len(known), "test_count": len(block),
                          "train_target_end": str(data.dates[known[-1]]),
                          "scaler_feature_end": str(data.dates[known[-1] - 1]),
                          "test_start": str(data.dates[block[0]]), "test_end": str(data.dates[block[-1]]),
                          "x_scaler": xs.as_dict(), "y_scaler": ys.as_dict(),
                          "log_variance": objective.log_variance.detach().cpu().tolist(),
                          "checkpoint": str(checkpoint), "history": history})
    return np.concatenate(predictions), np.concatenate(errors), {
        "selected_epochs": selected_epochs, "selection_history": selection_history, "folds": fold_info}
