"""
src/valuation.py
================
Regression models for market value prediction on log1p(market_value).

Models
------
- Ridge (linear baseline)
- LightGBM (gradient boosting, handles NaNs natively)
- XGBoost

All models train on log_market_value and predict log_market_value.
Predictions are inverted via expm1 for EUR-scale metrics.

Evaluation: MAE, RMSE, R2 (on log scale + EUR scale) and Directional Accuracy.
CV: expanding-window (rolling-origin) temporal split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def directional_accuracy(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                         y_prev_log: Optional[np.ndarray] = None) -> float:
    """
    Fraction of observations where sign(y_pred - y_prev) == sign(y_true - y_prev).
    If y_prev is None, uses median(y_true) as baseline.
    """
    if y_prev_log is None:
        y_prev_log = np.full_like(y_true_log, np.median(y_true_log))
    # Remove NaN prev
    mask = ~np.isnan(y_prev_log) & ~np.isnan(y_true_log) & ~np.isnan(y_pred_log)
    if mask.sum() == 0:
        return float("nan")
    true_dir = np.sign(y_true_log[mask] - y_prev_log[mask])
    pred_dir = np.sign(y_pred_log[mask] - y_prev_log[mask])
    # Count zero as correct if both zero, else mismatch
    return float((true_dir == pred_dir).mean())


def compute_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_prev_log: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Metrics on both log scale and EUR scale.
    """
    # Log scale
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_log = float(r2_score(y_true_log, y_pred_log)) if len(np.unique(y_true_log)) > 1 else float("nan")

    # EUR scale
    y_true_eur = np.expm1(y_true_log)
    y_pred_eur = np.expm1(y_pred_log)
    mae_eur = float(mean_absolute_error(y_true_eur, y_pred_eur))
    rmse_eur = float(np.sqrt(mean_squared_error(y_true_eur, y_pred_eur)))
    r2_eur = float(r2_score(y_true_eur, y_pred_eur)) if len(np.unique(y_true_eur)) > 1 else float("nan")

    da = directional_accuracy(y_true_log, y_pred_log, y_prev_log)

    return {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "mae_eur": mae_eur,
        "rmse_eur": rmse_eur,
        "r2_eur": r2_eur,
        "directional_accuracy": da,
        "n": int(len(y_true_log)),
    }

# ---------------------------------------------------------------------------
# Temporal CV
# ---------------------------------------------------------------------------

def expanding_window_splits(
    df: pd.DataFrame,
    date_col: str = "valuation_date",
    n_splits: int = 3,
    min_train_frac: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window splits sorted by date.
    The earliest min_train_frac of samples is the initial train.
    Each split expands train and shifts test forward.
    """
    if df.empty or len(df) < 10:
        return []

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    min_train = int(n * min_train_frac)
    if min_train < 5:
        min_train = 5

    # Test size per split
    remaining = n - min_train
    if remaining < n_splits:
        n_splits = max(1, remaining // 2)
    test_size = max(1, remaining // n_splits)

    splits = []
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_end = min(train_end + test_size, n)
        if test_end <= train_end:
            break
        # Map back to original index
        train_idx = df_sorted.index[:train_end].values
        test_idx = df_sorted.index[train_end:test_end].values
        # Convert to positions in the original df ( valuations sort may differ )
        # We use the sorted-df index directly; caller should sort beforehand.
        splits.append((train_idx, test_idx))
        if test_end == n:
            break

    return splits


def _map_splits_to_original(
    df_sorted: pd.DataFrame,
    splits_sorted_idx: List[Tuple[np.ndarray, np.ndarray]],
    original_index: pd.Index,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Translate splits from sorted-df positional indices to original-df index labels.
    """
    # Our splits are positional in the sorted frame. Map via the sorted frame's
    # 'original_idx' column if present, else assume df_sorted index IS the original.
    mapped = []
    for tr_pos, te_pos in splits_sorted_idx:
        tr_labels = df_sorted.index[tr_pos]
        te_labels = df_sorted.index[te_pos]
        # Convert labels to positional indices in original df
        tr = np.array([original_index.get_loc(lab) for lab in tr_labels])
        te = np.array([original_index.get_loc(lab) for lab in te_labels])
        mapped.append((tr, te))
    return mapped

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_ridge(alpha: float = 10.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def build_lgbm(**kwargs) -> "lgb.LGBMRegressor":
    if not HAS_LGBM:
        raise ImportError("lightgbm not installed")
    defaults = dict(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return lgb.LGBMRegressor(**defaults)


def build_xgboost(**kwargs) -> "xgb.XGBRegressor":
    if not HAS_XGB:
        raise ImportError("xgboost not installed")
    defaults = dict(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return xgb.XGBRegressor(**defaults)

# ---------------------------------------------------------------------------
# Training + CV loop
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    model_name: str
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    oof_pred: Optional[np.ndarray] = None
    oof_true: Optional[np.ndarray] = None
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    fitted_model: object = None  # last fold's fitted model (or refit on full)


def _prepare_xy(
    fm: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "log_market_value",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract X/y, handling categoricals and ensuring numeric dtypes.
    """
    X = fm[feature_cols].copy()
    # Encode categoricals
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(fm[target_col], errors="coerce").values.astype(float)
    # Drop rows where target is NaN
    mask = ~np.isnan(y)
    # Keep all rows; caller will handle train/test split masking
    cols = list(X.columns)
    Xv = X.values.astype(float)
    return Xv, y, cols


def cross_validate(
    fm: pd.DataFrame,
    feature_cols: List[str],
    model_name: str = "ridge",
    n_splits: int = 3,
    target_col: str = "log_market_value",
    model_kwargs: Optional[Dict] = None,
) -> CVResult:
    """
    Expanding-window CV for a single model type.
    """
    model_kwargs = model_kwargs or {}

    # Sort chronologically for temporal split
    fm_sorted = fm.sort_values("valuation_date").reset_index(drop=True)
    splits = expanding_window_splits(fm_sorted, n_splits=n_splits)

    if not splits:
        raise ValueError("Not enough data for cross-validation")

    X_all, y_all, feat_names = _prepare_xy(fm_sorted, feature_cols, target_col)

    result = CVResult(model_name=model_name)
    oof_pred = np.full(len(fm_sorted), np.nan)
    oof_true = y_all.copy()

    for fold, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        # Remove rows with NaN target in train/test
        tr_mask = ~np.isnan(y_tr)
        te_mask = ~np.isnan(y_te)
        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_te, y_te = X_te[te_mask], y_te[te_mask]

        if len(y_tr) < 5 or len(y_te) < 1:
            continue

        # Build model
        if model_name == "ridge":
            model = build_ridge(**model_kwargs)
        elif model_name == "lgbm":
            model = build_lgbm(**model_kwargs)
        elif model_name == "xgboost":
            model = build_xgboost(**model_kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        # For directional accuracy, need previous log value for test rows
        # Use y_tr's last value or fm's lag feature
        y_prev = None
        if "log_mv_lag1" in fm_sorted.columns:
            y_prev = fm_sorted.iloc[te_idx]["log_mv_lag1"].values.astype(float)
        else:
            # Use rolling mean of train as naive baseline for direction
            y_prev = np.full_like(y_te, np.median(y_tr))

        m = compute_metrics(y_te, y_pred, y_prev)
        m["fold"] = fold
        result.fold_metrics.append(m)

        # Record OOF
        # Map te_idx (in sorted fm) back to oof array
        valid_te = np.where(te_mask)[0]
        for j, pos in enumerate(te_idx[valid_te]):
            oof_pred[pos] = y_pred[j]

        print(f"    Fold {fold} [{model_name}]: R2_log={m['r2_log']:.3f}  MAE_log={m['mae_log']:.3f}  DA={m['directional_accuracy']:.2%}")

        result.fitted_model = model  # keep last

    # Aggregate
    if result.fold_metrics:
        for k in ["mae_log", "rmse_log", "r2_log", "mae_eur", "rmse_eur", "r2_eur", "directional_accuracy"]:
            vals = [fm[k] for fm in result.fold_metrics if not np.isnan(fm.get(k, np.nan))]
            result.mean_metrics[k] = float(np.nanmean(vals)) if vals else float("nan")
        result.mean_metrics["n_folds"] = len(result.fold_metrics)

        # OOF overall
        oof_mask = ~np.isnan(oof_pred) & ~np.isnan(oof_true)
        if oof_mask.sum() > 1:
            oof_m = compute_metrics(oof_true[oof_mask], oof_pred[oof_mask])
            result.mean_metrics["oof_r2_log"] = oof_m["r2_log"]
            result.mean_metrics["oof_mae_log"] = oof_m["mae_log"]

        result.oof_pred = oof_pred
        result.oof_true = oof_true

    return result


def train_all_models(
    fm: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 3,
) -> Dict[str, CVResult]:
    """
    Train Ridge, LightGBM, XGBoost with expanding-window CV.
    Returns dict model_name -> CVResult.
    """
    results: Dict[str, CVResult] = {}

    # Ridge baseline (always available) -- filter ultra-sparse features for stability
    # Ridge is sensitive to high-missing / high-variance columns; drop those with >60% null
    fm_null = fm[feature_cols].isna().mean()
    ridge_cols = [c for c in feature_cols if fm_null.get(c, 1.0) < 0.60]
    if len(ridge_cols) < 5:
        ridge_cols = feature_cols  # fallback
    print(f"\n  [Ridge] using {len(ridge_cols)}/{len(feature_cols)} features (<60% null)")
    results["ridge"] = cross_validate(fm, ridge_cols, model_name="ridge", n_splits=n_splits,
                                      model_kwargs={"alpha": 10.0})

    # LightGBM
    if HAS_LGBM:
        print("\n  [LightGBM]")
        try:
            results["lgbm"] = cross_validate(fm, feature_cols, model_name="lgbm", n_splits=n_splits)
        except Exception as exc:
            print(f"    LightGBM failed: {exc}")
    else:
        print("\n  [LightGBM] skipped (not installed)")

    # XGBoost
    if HAS_XGB:
        print("\n  [XGBoost]")
        try:
            results["xgboost"] = cross_validate(fm, feature_cols, model_name="xgboost", n_splits=n_splits)
        except Exception as exc:
            print(f"    XGBoost failed: {exc}")
    else:
        print("\n  [XGBoost] skipped (not installed)")

    return results


def save_metrics(
    results: Dict[str, CVResult],
    out_path: Path,
) -> None:
    """Write a JSON summary of all model results."""
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "mean_metrics": res.mean_metrics,
            "fold_metrics": res.fold_metrics,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Metrics saved to {out_path}")
