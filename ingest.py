"""
src/ingest.py
=============
SQL + CSV ingestion with efficient chunked COPY parsing and parquet caching.

Handles the 6.2 GB PostgreSQL dump (COPY FROM stdin format) by streaming
COPY blocks directly to parquet without holding the full file in memory.

Usage:
    from src.ingest import load_all, load_valuations, load_transfers, load_dossier
    data = load_all(use_cache=True)  # returns dict of DataFrames
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PROCESSED_DIR = DATA_DIR / "processed"

# Resolve the SQL file (it's inside a directory named *.sql)
_SQL_CANDIDATES = [
    DATA_DIR / "soccersolver_export_investigacion_2023-24_a_2025-26.sql" / "soccersolver_export_investigacion_2023-24_a_2025-26.sql",
    DATA_DIR / "soccersolver_export_investigacion_2023-24_a_2025-26.sql",
]
SQL_FILE: Optional[Path] = next((p for p in _SQL_CANDIDATES if p.is_file()), None)

CSV_VALUATIONS = DATA_DIR / "player_valuations_full (1).csv"
CSV_TRANSFERS  = DATA_DIR / "club_transfers (1).csv"
CSV_DOSSIER    = DATA_DIR / "team_full_dossier (1).csv"
CSV_SDDB       = DATA_DIR / "SD-DB(PLAYERS) (1).csv"
XLSX_FABRICA   = DATA_DIR / "real_madrid_la_fabrica_deep_analysis_expanded_2006_2026 (1).xlsx"

# ---------------------------------------------------------------------------
# Quarantine -- columns that must NEVER enter feature matrices
# ---------------------------------------------------------------------------
DOSSIER_QUARANTINE: Set[str] = {
    "predicted_value_1y", "predicted_value_2y", "predicted_value_3y",
    "fair_price_1y", "fair_price_2y", "fair_price_3y",
    "accuracy_1y", "accuracy_2y", "accuracy_3y",
    "prediction_calculation_date", "prediction_season",
    "upside_potential_current",
    "return_on_salary_since_arrival", "salary_to_value_ratio_current",
    "marginal_value_growth_since_arrival",
}

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise IOError(f"Could not decode {path} with any of utf-8/latin-1/cp1252")


def load_valuations() -> pd.DataFrame:
    df = _read_csv_robust(CSV_VALUATIONS)
    df["valuation_date"] = pd.to_datetime(df["valuation_date"], errors="coerce")
    df = df.sort_values(["player_id", "valuation_date"]).reset_index(drop=True)
    return df


def load_transfers() -> pd.DataFrame:
    df = _read_csv_robust(CSV_TRANSFERS)
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    df = df.sort_values(["player_id", "transfer_date"]).reset_index(drop=True)
    return df


def load_dossier(safe_only: bool = False) -> pd.DataFrame:
    df = _read_csv_robust(CSV_DOSSIER)
    for c in ("as_of_date", "metric_calculation_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if safe_only:
        drop = [c for c in DOSSIER_QUARANTINE if c in df.columns]
        df = df.drop(columns=drop)
    return df


def load_sddb() -> pd.DataFrame:
    if not CSV_SDDB.exists():
        return pd.DataFrame()
    return _read_csv_robust(CSV_SDDB)


def load_fabrica() -> pd.DataFrame:
    if not XLSX_FABRICA.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(XLSX_FABRICA, sheet_name="Cases", engine="openpyxl")
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# SQL COPY parser -- streaming, chunk-aware
# ---------------------------------------------------------------------------

# Tables we actually need for the Elo + valuation pipeline.
# Parsing all 28 tables would be wasteful (the file is 6.2 GB).
TARGET_TABLES: Dict[str, List[str]] = {
    # table_name -> expected column list (filled dynamically from header)
}

# We parse these specifically; everything else is skipped.
WANTED_TABLES: Set[str] = {
    "sports.matches",
    "sports.player_match_stats_total",
    "sports.players",
    "sports.teams",
    "sports.competitions",
    "sports.seasons",
    "sports.player_match_stats",
    "sports.player_match_stats_positions",
    "sports.match_teams",
    "sports.player_valuations",  # SQL version of valuations (more rows than CSV)
}

# Chunk size: number of data rows buffered before flushing to a DataFrame chunk
_CHUNK_ROWS = 50_000


def _parse_copy_block(
    sql_path: Path,
    wanted: Set[str],
    chunk_rows: int = _CHUNK_ROWS,
) -> Dict[str, pd.DataFrame]:
    """
    Stream the SQL dump and extract COPY blocks for *wanted* tables.

    Returns a dict: table_name -> DataFrame (concatenated from chunks).

    Strategy:
    - Read line-by-line with large buffer (8 MB).
    - On COPY header for a wanted table, parse column names.
    - Accumulate tab-separated rows until \\. terminator.
    - Convert each chunk to DataFrame via pd.read_csv on StringIO (fast).
    """
    # Accumulators: table -> list of chunk DataFrames
    chunks: Dict[str, List[pd.DataFrame]] = {t: [] for t in wanted}
    # Current state
    current_table: Optional[str] = None
    current_cols: Optional[List[str]] = None
    buffer: List[str] = []

    header_re = re.compile(r"^COPY\s+(\S+)\s*\(([^)]+)\)\s+FROM\s+stdin;")

    def _flush():
        nonlocal buffer, current_table, current_cols
        if not buffer or current_table is None or current_cols is None:
            buffer = []
            return
        # Join buffered TSV lines and parse
        tsv = "\n".join(buffer)
        try:
            df = pd.read_csv(
                io.StringIO(tsv),
                sep="\t",
                header=None,
                names=current_cols,
                na_values=["\\N"],
                low_memory=False,
                quoting=3,  # csv.QUOTE_NONE -- PostgreSQL COPY uses \t without quoting
                dtype=str,  # read as str first, coerce after
            )
            # Coerce numeric / datetime columns best-effort
            for c in df.columns:
                # Try numeric
                numeric = pd.to_numeric(df[c], errors="coerce")
                # If at least 50% converted, keep numeric
                if numeric.notna().sum() / max(len(df), 1) > 0.5:
                    # Use nullable integer where possible
                    # Keep as float if any non-integer values
                    df[c] = numeric
                # Try datetime for columns that look like dates
                elif c in ("match_datetime_utc", "birth_date", "start_date", "end_date",
                           "valuation_date", "created_at", "updated_at"):
                    dt = pd.to_datetime(df[c], errors="coerce")
                    if dt.notna().sum() > 0:
                        df[c] = dt
        except Exception as exc:
            print(f"  [WARN] Failed to parse chunk for {current_table}: {exc}", file=sys.stderr)
        else:
            chunks[current_table].append(df)
        buffer = []

    print(f"  Streaming {sql_path.name} ({sql_path.stat().st_size / 1e9:.2f} GB) ...")

    with open(sql_path, "r", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        for line in f:
            # COPY header
            m = header_re.match(line)
            if m:
                # Flush previous block if any
                if current_table is not None and buffer:
                    _flush()
                tbl = m.group(1)
                cols_raw = m.group(2)
                if tbl in wanted:
                    current_table = tbl
                    # Column names may contain quoted identifiers like "position"
                    current_cols = [c.strip().strip('"') for c in cols_raw.split(",")]
                    buffer = []
                else:
                    current_table = None
                    current_cols = None
                continue

            # End-of-COPY marker
            if line.strip() == r"\.":
                if current_table is not None and buffer:
                    _flush()
                current_table = None
                current_cols = None
                buffer = []
                continue

            # Data row for a wanted table
            if current_table is not None:
                buffer.append(line.rstrip("\n"))
                if len(buffer) >= chunk_rows:
                    _flush()

        # Final flush
        if current_table is not None and buffer:
            _flush()

    # Concatenate chunks per table
    result: Dict[str, pd.DataFrame] = {}
    for tbl, part_list in chunks.items():
        if part_list:
            result[tbl] = pd.concat(part_list, ignore_index=True)
            print(f"    {tbl}: {len(result[tbl]):,} rows x {len(result[tbl].columns)} cols")
        else:
            result[tbl] = pd.DataFrame()
            print(f"    {tbl}: EMPTY (no COPY block matched)")

    return result


def _coerce_sql_dtypes(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Best-effort dtype coercion after parsing as strings."""
    for tbl, df in tables.items():
        if df.empty:
            continue
        for c in df.columns:
            if df[c].dtype == object:
                # Already attempted in _flush; skip
                continue
    return tables

# ---------------------------------------------------------------------------
# Parquet caching
# ---------------------------------------------------------------------------

def _cache_tables(tables: Dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for tbl, df in tables.items():
        if df.empty:
            continue
        # Parquet file name: sports.matches -> matches.parquet etc.
        short = tbl.replace("sports.", "")
        out = PROCESSED_DIR / f"{short}.parquet"
        df.to_parquet(out, index=False)
        print(f"    Cached {out.name} ({len(df):,} rows)")

def _load_cached_tables() -> Optional[Dict[str, pd.DataFrame]]:
    if not PROCESSED_DIR.exists():
        return None
    result: Dict[str, pd.DataFrame] = {}
    for short in ["matches", "player_match_stats_total", "players", "teams",
                  "competitions", "seasons", "player_match_stats",
                  "player_match_stats_positions", "match_teams", "player_valuations"]:
        p = PROCESSED_DIR / f"{short}.parquet"
        if p.exists():
            result[f"sports.{short}"] = pd.read_parquet(p)
    if not result:
        return None
    # Ensure match_datetime_utc is datetime
    for tbl in result:
        for c in result[tbl].columns:
            if c in ("match_datetime_utc", "birth_date", "valuation_date", "start_date", "end_date"):
                result[tbl][c] = pd.to_datetime(result[tbl][c], errors="coerce")
    return result

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sql_tables(
    use_cache: bool = True,
    force_reparse: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Load wanted SQL tables, using parquet cache when available.
    """
    if use_cache and not force_reparse:
        cached = _load_cached_tables()
        if cached is not None and any(not v.empty for v in cached.values()):
            # Validate that key tables are present
            if "sports.matches" in cached and "sports.player_match_stats_total" in cached:
                print("  Loaded SQL tables from parquet cache.")
                return cached
            print("  Cache incomplete, re-parsing SQL ...")

    if SQL_FILE is None or not SQL_FILE.exists():
        print(f"  [WARN] SQL file not found: {SQL_FILE}")
        return {t: pd.DataFrame() for t in WANTED_TABLES}

    tables = _parse_copy_block(SQL_FILE, WANTED_TABLES)
    tables = _coerce_sql_dtypes(tables)

    if use_cache:
        _cache_tables(tables)

    return tables


def load_all(
    use_cache: bool = True,
    force_reparse: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Load every data source. Returns a dict with keys:
      valuations, transfers, dossier, sddb, fabrica, sql (nested dict)
    """
    print("=== Loading CSV datasets ===")
    valuations = load_valuations()
    print(f"  valuations: {len(valuations):,} rows")
    transfers = load_transfers()
    print(f"  transfers:  {len(transfers):,} rows")
    dossier = load_dossier(safe_only=False)
    print(f"  dossier:    {len(dossier):,} rows")
    sddb = load_sddb()
    print(f"  sddb:       {len(sddb):,} rows")
    fabrica = load_fabrica()
    print(f"  fabrica:    {len(fabrica):,} rows")

    print("\n=== Loading SQL datasets ===")
    sql_tables = load_sql_tables(use_cache=use_cache, force_reparse=force_reparse)

    return {
        "valuations": valuations,
        "transfers": transfers,
        "dossier": dossier,
        "sddb": sddb,
        "fabrica": fabrica,
        "sql": sql_tables,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest & cache all data")
    parser.add_argument("--force", action="store_true", help="Force re-parse SQL")
    parser.add_argument("--no-cache", action="store_true", help="Disable parquet cache")
    args = parser.parse_args()
    data = load_all(use_cache=not args.no_cache, force_reparse=args.force)
    print("\nDone. Keys:", list(data.keys()))
    if "sql" in data:
        for k, v in data["sql"].items():
            print(f"  sql[{k}]: {len(v):,} rows")
