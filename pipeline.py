"""
src/pipeline.py
===============
Full pipeline orchestration: ingest -> Elo -> features -> train/evaluate/predict.

CLI
---
    python -m src.pipeline --train          Train models with CV
    python -m src.pipeline --evaluate       Evaluate on hold-out (same as train in CV mode)
    python -m src.pipeline --predict        Train on full data and predict next valuations
    python -m src.pipeline --with-elo       Include Elo features (requires SQL parsing; slow)
    python -m src.pipeline --no-cache       Ignore parquet cache
    python -m src.pipeline --output DIR     Custom output directory

Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Allow running as `python -m src.pipeline` and `python src/pipeline.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import load_all, load_valuations, load_transfers, load_dossier
from src.features import build_feature_matrix, get_feature_columns
from src.valuation import train_all_models, save_metrics

try:
    from src.elo import run_full_elo
    HAS_ELO = True
except ImportError:
    HAS_ELO = False

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_MODELS = ROOT / "models"

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_ingest(use_cache: bool = True, with_elo: bool = False) -> Dict:
    """
    Load all data sources.
    If with_elo, also parse the 6.2 GB SQL (slow, cached to parquet after first run).
    """
    print("=" * 60)
    print("STEP 1: Ingest")
    print("=" * 60)

    if with_elo:
        print("  Full ingest (CSV + SQL with parquet cache)...")
        data = load_all(use_cache=use_cache)
    else:
        print("  Fast ingest (CSV only, Elo neutral)...")
        data = {
            "valuations": load_valuations(),
            "transfers": load_transfers(),
            "dossier": load_dossier(safe_only=False),
            "sql": {},
        }
        # Still try to load players for age features if cache exists
        try:
            players_pq = ROOT / "Data" / "processed" / "players.parquet"
            if players_pq.exists():
                data["sql"]["sports.players"] = pd.read_parquet(players_pq)
                print(f"  Loaded players from cache ({len(data['sql']['sports.players']):,} rows)")
        except Exception:
            pass

    print(f"  Valuations: {len(data['valuations']):,} rows")
    print(f"  Transfers:  {len(data['transfers']):,} rows")
    print(f"  Dossier:    {len(data['dossier']):,} rows")
    if data.get("sql"):
        for k, v in data["sql"].items():
            if not v.empty:
                print(f"  SQL {k}: {len(v):,} rows")
    return data


def step_elo(data: Dict) -> Optional[pd.DataFrame]:
    """
    Compute Team + Player Elo if SQL data is available.
    Returns player_history DataFrame or None.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Elo")
    print("=" * 60)

    sql = data.get("sql", {})
    matches = sql.get("sports.matches", pd.DataFrame())
    stats = sql.get("sports.player_match_stats_total", pd.DataFrame())
    players = sql.get("sports.players", pd.DataFrame())
    match_teams = sql.get("sports.match_teams", pd.DataFrame())

    if matches.empty or stats.empty:
        print("  No match data available -- Elo will be neutral (1500 for all).")
        return None

    if not HAS_ELO:
        print("  src.elo not available -- skipping.")
        return None

    print(f"  Running Elo over {len(matches):,} matches, {len(stats):,} player-match rows ...")
    t0 = time.time()
    result = run_full_elo(matches, match_teams if not match_teams.empty else None,
                          stats, players if not players.empty else None)
    elapsed = time.time() - t0
    ph = result["player_history"]
    print(f"  Done in {elapsed:.1f}s. Player history: {len(ph):,} rows, {ph['player_id'].nunique():,} players")
    return ph


def step_features(
    data: Dict,
    player_history: Optional[pd.DataFrame],
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3: Feature Matrix")
    print("=" * 60)

    sql = data.get("sql", {})
    players = sql.get("sports.players", None)

    fm = build_feature_matrix(
        valuations=data["valuations"],
        dossier=data["dossier"],
        transfers=data["transfers"],
        players_sql=players,
        player_history=player_history,
    )
    feat_cols = get_feature_columns(fm)
    print(f"  Matrix: {fm.shape[0]:,} rows x {fm.shape[1]} cols")
    print(f"  Feature columns ({len(feat_cols)}): {feat_cols[:10]} ...")
    # Quarantine check
    from src.ingest import DOSSIER_QUARANTINE
    leaked = [c for c in fm.columns if c in DOSSIER_QUARANTINE]
    assert not leaked, f"Quarantine violated: {leaked}"
    print(f"  Quarantine check: PASS (no leakage columns)")
    # Null report
    null_pct = fm[feat_cols].isna().mean().sort_values(ascending=False)
    worst = null_pct[null_pct > 0.3]
    if not worst.empty:
        print(f"  High-missing features (>30%):")
        for c, v in worst.head(5).items():
            print(f"    {c}: {v:.0%} null")
    return fm


def step_train(
    fm: pd.DataFrame,
    output_dir: Path,
    n_splits: int = 3,
) -> Dict:
    print("\n" + "=" * 60)
    print("STEP 4: Train & Evaluate")
    print("=" * 60)

    feat_cols = get_feature_columns(fm)
    # Drop rows where target is NaN
    fm_clean = fm.dropna(subset=["log_market_value"]).copy()
    print(f"  Training samples: {len(fm_clean):,} (from {len(fm):,} total)")
    print(f"  Feature count: {len(feat_cols)}")

    if len(fm_clean) < 20:
        print("  [WARN] Very few samples for meaningful CV.")
        n_splits = 2
    if len(fm_clean) < 10:
        raise ValueError(f"Only {len(fm_clean)} clean samples -- need more data")

    results = train_all_models(fm_clean, feat_cols, n_splits=n_splits)

    # Save metrics
    metrics_path = output_dir / "baseline_metrics.json"
    save_metrics(results, metrics_path)

    # Brief summary
    print("\n  Summary (mean across folds):")
    for name, res in results.items():
        r2 = res.mean_metrics.get("r2_log", float("nan"))
        mae = res.mean_metrics.get("mae_log", float("nan"))
        da = res.mean_metrics.get("directional_accuracy", float("nan"))
        print(f"    {name:10s}  R2_log={r2:.3f}  MAE_log={mae:.3f}  DA={da:.1%}")

    return results


def run_pipeline(
    with_elo: bool = False,
    use_cache: bool = True,
    output_dir: Optional[Path] = None,
    n_splits: int = 3,
) -> Dict:
    """
    Execute the full pipeline.
    """
    t_start = time.time()
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "models").mkdir(parents=True, exist_ok=True)

    data = step_ingest(use_cache=use_cache, with_elo=with_elo)
    ph = step_elo(data) if with_elo else None
    fm = step_features(data, ph)
    results = step_train(fm, output_dir, n_splits=n_splits)

    # Save feature matrix for inspection
    fm_path = output_dir / "feature_matrix.parquet"
    try:
        fm.to_parquet(fm_path, index=False)
        print(f"\n  Feature matrix saved to {fm_path}")
    except Exception as exc:
        print(f"  Could not save feature matrix: {exc}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    return {"data": data, "player_history": ph, "feature_matrix": fm, "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time Valuation + Elo Pipeline")
    parser.add_argument("--train", action="store_true", help="Train models (default action)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate (same as --train with CV)")
    parser.add_argument("--predict", action="store_true", help="Train and predict (placeholder)")
    parser.add_argument("--with-elo", action="store_true", help="Include Elo features (requires SQL, slow)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore parquet cache")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--splits", type=int, default=3, help="Number of CV splits")
    args = parser.parse_args()

    # Default: --train if nothing specified
    if not any([args.train, args.evaluate, args.predict]):
        args.train = True

    try:
        run_pipeline(
            with_elo=args.with_elo,
            use_cache=not args.no_cache,
            output_dir=Path(args.output),
            n_splits=args.splits,
        )
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
