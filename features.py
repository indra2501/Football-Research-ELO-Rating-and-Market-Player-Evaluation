"""
src/features.py
===============
Unified feature matrix builder aligned to (player_id, valuation_date).

Each observation = one player's valuation snapshot.
Features merged AS-OF that valuation_date (never future data).

Sources
-------
- Player Elo (current, rolling trends, rank)  -- from src.elo history
- Safe financial risk metrics from team_full_dossier (quarantine-excluded)
- Transfer flags from club_transfers
- Demographics (age, position)

Quarantine is enforced strictly: none of the 15 flagged dossier columns
enter the feature matrix, and SD-DB / XLSX post-hoc targets are excluded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .ingest import DOSSIER_QUARANTINE
except ImportError:  # pragma: no cover - fallback for non-package execution
    try:
        from src.ingest import DOSSIER_QUARANTINE  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - defensive fallback
        DOSSIER_QUARANTINE = []

# ---------------------------------------------------------------------------
# Safe dossier columns (explicit allowlist)
# ---------------------------------------------------------------------------
# Everything in team_full_dossier EXCEPT the quarantine set and 100%-null cols
# and non-feature metadata. We enumerate the known safe columns.
SAFE_DOSSIER_COLS: List[str] = [
    "market_value_at_checkpoint",
    "club_id_at_checkpoint",
    "cagr_historical", "cagr_historical_ln",
    "total_return", "total_return_ln",
    "rolling_return_6m", "rolling_return_6m_ln",
    "career_value_slope_historical", "career_value_slope_1y",
    "max_drawdown_historical", "max_drawdown_since_last_transfer",
    "avg_drawdown_historical",
    "volatility_1y", "volatility_1y_ln",
    "downside_volatility_historical", "downside_volatility_historical_ln",
    "value_at_risk_var_historical", "value_at_risk_var_historical_ln",
    "value_at_risk_var_1y", "value_at_risk_var_1y_ln",
    "parametric_var_historical", "parametric_var_historical_ln",
    "expected_shortfall_cvar_historical", "expected_shortfall_cvar_historical_ln",
    "value_stability_index_historical",
    "sharpe_ratio_historical", "sharpe_ratio_historical_ln",
    "sharpe_ratio_1y", "sharpe_ratio_1y_ln",
    "sharpe_ratio_since_last_transfer", "sharpe_ratio_since_last_transfer_ln",
    "sortino_ratio_historical", "sortino_ratio_historical_ln",
    "sortino_ratio_1y", "sortino_ratio_1y_ln",
    "sortino_ratio_since_last_transfer", "sortino_ratio_since_last_transfer_ln",
    "calmar_ratio_historical", "calmar_ratio_historical_ln",
    "return_max_drawdown_ratio_historical", "return_max_drawdown_ratio_historical_ln",
    "time_on_market_pre_last_transfer",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _asof_merge(
    valuations: pd.DataFrame,
    dossier: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join the latest dossier row per player where as_of_date <= valuation_date.
    Falls back to metric_calculation_date if as_of_date missing.
    """
    if dossier.empty:
        return valuations.copy()

    # Ensure datetime
    valuations = valuations.copy()
    dossier = dossier.copy()
    # Use as_of_date as primary time key
    date_col = "as_of_date" if "as_of_date" in dossier.columns else "metric_calculation_date"
    dossier["_dossier_date"] = pd.to_datetime(dossier[date_col], errors="coerce")
    valuations["_val_date"] = pd.to_datetime(valuations["valuation_date"], errors="coerce")

    # Keep only safe cols + keys
    keep = [c for c in SAFE_DOSSIER_COLS if c in dossier.columns]
    dossier_small = dossier[["player_id", "_dossier_date"] + keep].copy()
    # Sort for merge_asof
    dossier_small = dossier_small.sort_values("_dossier_date")
    valuations = valuations.sort_values("_val_date")

    result_frames: List[pd.DataFrame] = []
    for pid, grp in valuations.groupby("player_id", sort=False):
        dsub = dossier_small[dossier_small["player_id"] == pid].sort_values("_dossier_date")
        if dsub.empty:
            # No dossier row for this player -- keep valuation rows with NaN features
            g = grp.copy()
            for c in keep:
                g[c] = np.nan
            result_frames.append(g)
            continue
        merged = pd.merge_asof(
            grp.sort_values("_val_date"),
            dsub.drop(columns=["player_id"]),
            left_on="_val_date",
            right_on="_dossier_date",
            direction="backward",
        )
        result_frames.append(merged)

    out = pd.concat(result_frames, ignore_index=True) if result_frames else valuations
    out = out.drop(columns=["_dossier_date", "_val_date"], errors="ignore")
    return out


def _add_transfer_features(
    df: pd.DataFrame,
    transfers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add per-observation transfer context:
      days_since_last_transfer, is_loan_last, market_value_at_last_transfer,
      n_transfers_to_date, was_transferred_recently (30d flag)
    All computed AS-OF valuation_date (no future transfers used).
    """
    if transfers.empty:
        df["days_since_last_transfer"] = np.nan
        df["is_loan_last"] = 0
        df["market_value_at_last_transfer"] = np.nan
        df["n_transfers_to_date"] = 0
        df["transferred_last_30d"] = 0
        df["transferred_last_90d"] = 0
        return df

    transfers = transfers.copy()
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers = transfers.sort_values("transfer_date")

    # Ensure valuation datetime
    df = df.copy()
    df["_val_date"] = pd.to_datetime(df["valuation_date"], errors="coerce")

    # Per observation, find last transfer before valuation_date.
    # We iterate per player to avoid pandas groupby include_groups quirks.
    df_sorted = df.sort_values(["player_id", "_val_date"]).copy()
    # Prepare transfer lookup per player (sorted)
    transfer_by_player: Dict[int, pd.DataFrame] = {
        pid: sub.sort_values("transfer_date")
        for pid, sub in transfers.groupby("player_id")
    }

    dslt_list, ill_list, mv_list, ntr_list, tr30_list, tr90_list = [], [], [], [], [], []
    for _, row in df_sorted.iterrows():
        pid = int(row["player_id"])
        vd = row["_val_date"]
        tsub = transfer_by_player.get(pid, pd.DataFrame())
        if tsub.empty:
            dslt_list.append(np.nan); ill_list.append(0); mv_list.append(np.nan)
            ntr_list.append(0); tr30_list.append(0); tr90_list.append(0)
            continue
        mask = tsub["transfer_date"] <= vd
        past = tsub[mask]
        ntr_list.append(int(mask.sum()))
        if past.empty:
            dslt_list.append(np.nan); ill_list.append(0); mv_list.append(np.nan)
            tr30_list.append(0); tr90_list.append(0)
        else:
            last = past.iloc[-1]
            delta = (vd - last["transfer_date"]).days
            dslt_list.append(float(delta))
            ill_list.append(int(bool(last.get("is_loan", False))))
            mv_at = last.get("market_value_at_transfer", np.nan)
            mv_list.append(float(mv_at) if pd.notna(mv_at) else np.nan)
            tr30_list.append(int(delta <= 30))
            tr90_list.append(int(delta <= 90))

    df_sorted["days_since_last_transfer"] = dslt_list
    df_sorted["is_loan_last"] = ill_list
    df_sorted["market_value_at_last_transfer"] = mv_list
    df_sorted["n_transfers_to_date"] = ntr_list
    df_sorted["transferred_last_30d"] = tr30_list
    df_sorted["transferred_last_90d"] = tr90_list
    out = df_sorted
    out = out.drop(columns=["_val_date"], errors="ignore")
    return out


def _add_demographic_features(
    df: pd.DataFrame,
    players_sql: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add age at valuation date + position group if available.
    players_sql has player_id, birth_date, role_code / role_name.
    """
    df = df.copy()
    df["_val_date"] = pd.to_datetime(df["valuation_date"], errors="coerce")

    # Age
    if players_sql is not None and not players_sql.empty and "birth_date" in players_sql.columns:
        birth_map = dict(zip(players_sql["player_id"].astype(int), pd.to_datetime(players_sql["birth_date"], errors="coerce")))
        role_map = {}
        if "role_code" in players_sql.columns:
            role_map = dict(zip(players_sql["player_id"].astype(int), players_sql["role_code"]))
        elif "role_name" in players_sql.columns:
            role_map = dict(zip(players_sql["player_id"].astype(int), players_sql["role_name"]))

        ages = []
        pos_groups = []
        for _, r in df.iterrows():
            pid = int(r["player_id"])
            b = birth_map.get(pid, pd.NaT)
            if pd.notna(b) and pd.notna(r["_val_date"]):
                ages.append((r["_val_date"] - b).days / 365.25)
            else:
                ages.append(np.nan)
            # Position category
            rc = role_map.get(pid, None)
            if rc is None or (isinstance(rc, float) and np.isnan(rc)):
                pos_groups.append("unknown")
            else:
                pos_groups.append(str(rc).strip())
        df["age_at_valuation"] = ages
        df["position_raw"] = pos_groups
        # One-hot-ish: map to coarse groups
        coarse = {
            "GK": "GK", "Goalkeeper": "GK",
            "DF": "DF", "CB": "DF", "LB": "DF", "RB": "DF", "Defender": "DF",
            "MF": "MF", "CM": "MF", "DM": "MF", "AM": "MF", "Midfielder": "MF",
            "FW": "FW", "ST": "FW", "CF": "FW", "W": "FW", "Forward": "FW", "Attacker": "FW",
        }
        df["position_group"] = df["position_raw"].map(coarse).fillna("MF")
    else:
        df["age_at_valuation"] = np.nan
        df["position_raw"] = "unknown"
        df["position_group"] = "MF"

    df = df.drop(columns=["_val_date"], errors="ignore")
    return df


def _add_elo_features(
    df: pd.DataFrame,
    player_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add Elo features AS-OF valuation_date:
      elo_current, elo_delta_5, elo_delta_10, elo_trend_5 (slope),
      elo_rank (percentile among active players), elo_n_matches
    """
    df = df.copy()

    if player_history is None or player_history.empty:
        df["elo_current"] = 1500.0
        df["elo_delta_5"] = 0.0
        df["elo_delta_10"] = 0.0
        df["elo_trend_5"] = 0.0
        df["elo_rank_pct"] = 0.5
        df["elo_n_matches"] = 0
        df["elo_surprise_last"] = 0.0
        return df

    ph = player_history.copy()
    ph["match_datetime"] = pd.to_datetime(ph["match_datetime"], errors="coerce")
    # Sort for asof logic
    ph = ph.sort_values(["player_id", "match_datetime"])

    df["_val_date"] = pd.to_datetime(df["valuation_date"], errors="coerce")

    # Precompute snapshots per valuation: for each (player_id, val_date) find last Elo
    # Do per-player merge_asof for efficiency
    elo_results: List[Dict] = []

    for pid, grp in df.groupby("player_id"):
        ph_sub = ph[ph["player_id"] == pid].sort_values("match_datetime")
        g = grp.sort_values("_val_date").copy()
        if ph_sub.empty:
            for _, r in g.iterrows():
                elo_results.append({"idx": r.name, "elo_current": 1500.0, "elo_delta_5": 0.0,
                                    "elo_delta_10": 0.0, "elo_trend_5": 0.0, "elo_rank_pct": 0.5,
                                    "elo_n_matches": 0, "elo_surprise_last": 0.0})
            continue

        # Merge asof to get elo_before/after at each valuation date
        merged = pd.merge_asof(
            g.sort_values("_val_date"),
            ph_sub[["match_datetime", "elo_after", "surprise"]].sort_values("match_datetime"),
            left_on="_val_date",
            right_on="match_datetime",
            direction="backward",
        )

        # For rolling deltas we need to look back N matches before the valuation
        for _, r in merged.iterrows():
            vd = r["_val_date"]
            past = ph_sub[ph_sub["match_datetime"] <= vd]
            n = len(past)
            if past.empty:
                elo_results.append({"idx": r.name, "elo_current": 1500.0, "elo_delta_5": 0.0,
                                    "elo_delta_10": 0.0, "elo_trend_5": 0.0, "elo_rank_pct": 0.5,
                                    "elo_n_matches": 0, "elo_surprise_last": 0.0})
                continue
            cur = float(past.iloc[-1]["elo_after"])
            surprise_last = float(past.iloc[-1].get("surprise", 0.0))
            # Delta 5
            if len(past) >= 5:
                d5 = cur - float(past.iloc[-5]["elo_after"])
            elif len(past) >= 2:
                d5 = cur - float(past.iloc[0]["elo_after"])
            else:
                d5 = 0.0
            # Delta 10
            if len(past) >= 10:
                d10 = cur - float(past.iloc[-10]["elo_after"])
            elif len(past) >= 2:
                d10 = cur - float(past.iloc[0]["elo_after"])
            else:
                d10 = 0.0
            # Trend (linear slope over last 5)
            if len(past) >= 3:
                last5 = past.tail(5)
                y = last5["elo_after"].values.astype(float)
                x = np.arange(len(y), dtype=float)
                slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0
            else:
                slope = 0.0

            elo_results.append({"idx": r.name, "elo_current": cur, "elo_delta_5": d5,
                                "elo_delta_10": d10, "elo_trend_5": slope,
                                "elo_rank_pct": 0.5,  # filled below globally per date
                                "elo_n_matches": n, "elo_surprise_last": surprise_last})

    # Map back to df
    idx_to = {r["idx"]: r for r in elo_results}
    for col in ["elo_current", "elo_delta_5", "elo_delta_10", "elo_trend_5", "elo_n_matches", "elo_surprise_last"]:
        df[col] = df.index.map(lambda i: idx_to.get(i, {}).get(col, 1500.0 if col == "elo_current" else 0.0))

    # Compute rank percentile per valuation date (among players with Elo at that date)
    # For each unique valuation date, rank players by their as-of Elo
    if not df.empty:
        for val_date, sub_idx in df.groupby(pd.to_datetime(df["valuation_date"], errors="coerce")).groups.items():
            elos = df.loc[sub_idx, "elo_current"].values
            # Percentile rank
            ranks = pd.Series(elos).rank(pct=True).values
            df.loc[sub_idx, "elo_rank_pct"] = ranks
    else:
        df["elo_rank_pct"] = 0.5

    df = df.drop(columns=["_val_date"], errors="ignore")
    # Ensure no quarantine leakage via column names
    leaked = [c for c in df.columns if c in DOSSIER_QUARANTINE]
    if leaked:
        raise ValueError(f"Leakage columns leaked into feature matrix: {leaked}")

    return df


def _add_valuation_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cheap valuation-trajectory features that use ONLY past valuations
    (never future). Features are computed per player incrementally.
    """
    df = df.copy()
    df = df.sort_values(["player_id", "valuation_date"]).reset_index(drop=True)
    df["_log_mv"] = np.log1p(df["market_value"].astype(float))

    # Lag features: value at t-1, growth since last valuation
    df["log_mv_lag1"] = df.groupby("player_id")["_log_mv"].shift(1)
    df["mv_growth_since_last"] = df["_log_mv"] - df["log_mv_lag1"]
    # Time since last valuation (days)
    df["_val_dt"] = pd.to_datetime(df["valuation_date"], errors="coerce")
    df["days_since_last_valuation"] = df.groupby("player_id")["_val_dt"].diff().dt.days.astype(float)
    # Valuation count to date
    df["n_valuations_to_date"] = df.groupby("player_id").cumcount() + 1

    df = df.drop(columns=["_log_mv", "_val_dt"], errors="ignore")
    return df


def build_feature_matrix(
    valuations: pd.DataFrame,
    dossier: pd.DataFrame,
    transfers: pd.DataFrame,
    players_sql: Optional[pd.DataFrame] = None,
    player_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the unified observation matrix.

    Parameters
    ----------
    valuations : player_valuations_full CSV (must have player_id, valuation_date, market_value)
    dossier : team_full_dossier CSV (quarantine enforced internally)
    transfers : club_transfers CSV
    players_sql : sports.players for age/position
    player_history : player Elo history from src.elo (optional; defaults to 1500)

    Returns
    -------
    DataFrame with one row per (player_id, valuation_date), enriched with
    all feature groups. Target column is 'market_value' (raw) plus 'log_market_value'.
    """
    if valuations.empty:
        raise ValueError("valuations is empty")

    valuations = valuations.copy()
    valuations["valuation_date"] = pd.to_datetime(valuations["valuation_date"], errors="coerce")
    valuations = valuations.sort_values(["player_id", "valuation_date"]).reset_index(drop=True)

    # 1) Dossier (safe columns only, asof join)
    vals = _asof_merge(valuations, dossier)

    # 2) Transfer features (asof)
    vals = _add_transfer_features(vals, transfers)

    # 3) Demographics
    vals = _add_demographic_features(vals, players_sql)

    # 4) Elo
    vals = _add_elo_features(vals, player_history)

    # 5) Valuation history (lags)
    vals = _add_valuation_history_features(vals)

    # 6) Target transforms
    vals["log_market_value"] = np.log1p(vals["market_value"].astype(float))

    # Final quarantine check
    leaked = [c for c in vals.columns if c in DOSSIER_QUARANTINE]
    assert not leaked, f"Quarantine violation: {leaked}"

    return vals


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of columns to use as X (features) for modelling.
    Excludes identifiers, dates, targets, and raw position strings.
    """
    exclude = {
        "player_id", "player_name", "valuation_date", "club", "club_id",
        "market_value", "log_market_value",
        "position_raw", "position_group",
    }
    # Also exclude anything in quarantine (defence in depth)
    exclude |= DOSSIER_QUARANTINE
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys

    # Running this file directly from C:\Football\src leaves the parent folder
    # off sys.path, which makes `import src.ingest` fail. Add the repo root as
    # a fallback so the script works from either direct execution or package use.
    if __package__ in (None, ""):
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    try:
        from src.ingest import load_all  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
        try:
            from ingest import load_all  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "Could not import ingest from either the src package or the repo root. "
                "Run this script from the repository root or install the package."
            ) from exc

    print("Loading data for feature matrix demo ...")
    data = load_all(use_cache=True)
    # Try to build without Elo first (fast path)
    fm = build_feature_matrix(
        valuations=data["valuations"],
        dossier=data["dossier"],
        transfers=data["transfers"],
        players_sql=data["sql"].get("sports.players"),
        player_history=None,
    )
    print(f"Feature matrix: {fm.shape[0]:,} rows x {fm.shape[1]} cols")
    print("Feature columns:", get_feature_columns(fm)[:15], "...")
    print(fm.head(3).to_string())
