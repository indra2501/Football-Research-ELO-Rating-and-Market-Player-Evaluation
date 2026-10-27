"""
Data Profiling Script
=====================
Reads all core CSVs and the xlsx file in C:\\Football\\Data, prints a
catalog-quality summary: schema, row counts, missing values, unique counts,
date ranges, and cross-file player overlap.

Usage:
    python -m src.data_profiling
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")




def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"  [WARN] Could not read {path.name}: {e}")
            return pd.DataFrame()
    print(f"  [WARN] Could not read {path.name}: no encoding worked")
    return pd.DataFrame()


def _read_xlsx(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}")
        return pd.DataFrame()


def _schema(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null_count": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def _player_id_set(df: pd.DataFrame, col_name: str) -> set[int]:
    if col_name not in df.columns:
        return set()
    numeric = pd.to_numeric(df[col_name], errors="coerce")
    return {int(v) for v in numeric.dropna().unique() if pd.notna(v)}


def _player_name_set(df: pd.DataFrame, col_name: str) -> set[str]:
    if col_name not in df.columns:
        return set()
    return {
        str(v).strip().lower()
        for v in df[col_name].dropna().astype(str)
        if str(v).strip()
    }


# ── per-file profiling ──────────────────────────────────────────────────────

DATE_COLS = {
    "player_valuations_full (1).csv": ["valuation_date"],
    "club_transfers (1).csv": ["transfer_date"],
    "team_full_dossier (1).csv": ["as_of_date", "metric_calculation_date", "prediction_calculation_date"],
}


def profile_csv(name: str, df: pd.DataFrame) -> None:
    _print_section(name)
    print(f"Shape : {df.shape[0]:,} rows  x  {df.shape[1]} columns\n")

    print("Schema & Missing Values:")
    print(_schema(df).to_string())

    # date range for known date columns
    for col in DATE_COLS.get(name, []):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid = parsed.dropna()
            if len(valid):
                print(f"\n  {col} range: {valid.min().date()}  ->  {valid.max().date()}  ({len(valid):,} valid / {len(df):,} total)")

    # unique player counts
    for pid_col in ["player_id", "Player"]:
        if pid_col in df.columns:
            print(f"  Unique '{pid_col}': {df[pid_col].nunique():,}")


def profile_xlsx(name: str, df: pd.DataFrame) -> None:
    _print_section(name)
    print(f"Shape : {df.shape[0]:,} rows  x  {df.shape[1]} columns\n")
    print("Schema & Missing Values:")
    print(_schema(df).to_string())
    if "Player" in df.columns:
        print(f"\n  Unique 'Player': {df['Player'].nunique():,}")


# ── overlap analysis ────────────────────────────────────────────────────────

def overlap_analysis(
    valuations: pd.DataFrame,
    transfers: pd.DataFrame,
    dossier: pd.DataFrame,
    sd_players: pd.DataFrame,
) -> None:
    _print_section("CROSS-FILE OVERLAP ANALYSIS")

    val_ids = _player_id_set(valuations, "player_id")
    val_names = _player_name_set(valuations, "player_name")

    tr_ids = _player_id_set(transfers, "player_id")
    tr_names = _player_name_set(transfers, "player_name")

    d_ids = _player_id_set(dossier, "player_id")

    sd_names = _player_name_set(sd_players, "Player")

    # ID-based overlap: transfers <-> valuations
    if val_ids and tr_ids:
        common = val_ids & tr_ids
        pct_val = len(common) / len(val_ids) * 100 if val_ids else 0
        pct_tr = len(common) / len(tr_ids) * 100 if tr_ids else 0
        print(f"\n  Transfers ↔ Valuations (player_id):")
        print(f"    Common IDs        : {len(common):,}")
        print(f"    % of val IDs      : {pct_val:.1f}%")
        print(f"    % of transfer IDs : {pct_tr:.1f}%")
        missing_in_val = tr_ids - val_ids
        if missing_in_val:
            print(f"    In transfers but NOT in valuations ({len(missing_in_val)}): {sorted(missing_in_val)[:10]}...")

    # ID-based overlap: dossier <-> valuations
    if val_ids and d_ids:
        common = val_ids & d_ids
        print(f"\n  Dossier ↔ Valuations (player_id):")
        print(f"    Common IDs        : {len(common):,}")
        print(f"    % of dossier IDs  : {len(common)/len(d_ids)*100:.1f}%")

    # Name-based overlap: SD-DB <-> valuations
    if val_names and sd_names:
        common_n = val_names & sd_names
        print(f"\n  SD-DB ↔ Valuations (player name, case-insensitive):")
        print(f"    Common names      : {len(common_n):,}")
        print(f"    % of SD-DB names  : {len(common_n)/len(sd_names)*100:.1f}%")

    # ID-based: transfers <-> dossier
    if tr_ids and d_ids:
        common = tr_ids & d_ids
        print(f"\n  Transfers ↔ Dossier (player_id):")
        print(f"    Common IDs        : {len(common):,}")

    # SD-DB doesn't have player_id — name match against dossier names
    d_names = set()
    if "player_name" in dossier.columns:
        d_names = set(dossier["player_name"].dropna().str.strip().str.lower())
    if sd_names and d_names:
        common_n = sd_names & d_names
        print(f"\n  SD-DB ↔ Dossier (player name, case-insensitive):")
        print(f"    Common names      : {len(common_n):,}")


# ── SQL schema summary ──────────────────────────────────────────────────────

def sql_table_summary() -> None:
    sql_path = (
        DATA_DIR
        / "soccersolver_export_investigacion_2023-24_a_2025-26.sql"
        / "soccersolver_export_investigacion_2023-24_a_2025-26.sql"
    )
    if not sql_path.exists():
        return

    _print_section("SQL SCHEMA SUMMARY (soccersolver)")

    tables = [
        ("sports.matches", "match_id, competition_id, season_id, match_datetime_utc, gameweek, ..."),
        ("sports.players", "player_id, short_name, first_name, last_name, birth_date, role_code, ..."),
        ("sports.teams", "team_id, name, official_name, ..."),
        ("sports.player_match_stats_total", "player_id, match_id, minutes_played, goals, assists, shots, passes, duels, xg_shot, xg_assist, ..."),
        ("sports.player_match_stats_average", "player_id, match_id, *_avg columns for all stat categories"),
        ("sports.player_match_stats_percent", "player_id, match_id, *_percent columns"),
        ("sports.player_match_stats_positions", "player_id, match_id, position_code, percent"),
        ("sports.player_match_stats", "player_id, match_id, competition_id, season_id, round_id"),
    ]
    for tname, cols in tables:
        print(f"  {tname}")
        print(f"    Columns: {cols}\n")
    print("  Seasons: 2023-2024, 2024-2025, 2025-2026")
    print("  Source: wyscout + transfermarkt + model")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Create the Data folder or point DATA_DIR to the correct location.")
        return

    files = sorted(p.name for p in DATA_DIR.iterdir())
    print(f"Data directory: {DATA_DIR}")
    print(f"Files found: {files}")

    csv_files = {
        "player_valuations_full (1).csv": None,
        "club_transfers (1).csv": None,
        "team_full_dossier (1).csv": None,
        "SD-DB(PLAYERS) (1).csv": None,
    }

    # load CSVs
    for name in csv_files:
        path = DATA_DIR / name
        if not path.exists():
            print(f"  [WARN] Missing CSV: {name}")
            csv_files[name] = pd.DataFrame()
            continue
        df = _read_csv(path)
        csv_files[name] = df
        profile_csv(name, df)

    # load xlsx
    xlsx_name = "real_madrid_la_fabrica_deep_analysis_expanded_2006_2026 (1).xlsx"
    xlsx_path = DATA_DIR / xlsx_name
    xlsx_df = pd.DataFrame()
    if xlsx_path.exists():
        xlsx_df = _read_xlsx(xlsx_path)
        profile_xlsx(xlsx_name, xlsx_df)

    # overlap
    overlap_analysis(
        valuations=csv_files["player_valuations_full (1).csv"],
        transfers=csv_files["club_transfers (1).csv"],
        dossier=csv_files["team_full_dossier (1).csv"],
        sd_players=csv_files["SD-DB(PLAYERS) (1).csv"],
    )

    # SQL summary
    sql_table_summary()

    print(f"\n{'=' * 70}")
    print("  PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
