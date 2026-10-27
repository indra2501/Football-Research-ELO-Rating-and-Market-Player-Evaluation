# HANDOFF -- Real-Time Market Valuation + Elo-Style Player Rating

> **Last updated:** 2026-08-30
> **Status:** Phases 1-6 complete. Pipeline live. 20/20 tests pass. Baseline metrics in `output/baseline_metrics.json`.

---

## 1. Project Objective

Build an automated, time-aware ML pipeline that:

1. Computes **dynamic Elo-style player ratings** updated match-by-match from on-field performance.
2. Predicts **continuous transfer market valuations** using those ratings + financial/contractual features.
3. The core loop is:

```
Match Performance  ->  Expected Performance  ->  Performance Surprise  ->  Elo Rating Update
       |                                                                      |
       v                                                                      v
  Feature Matrix  ------------------------------------------------->  Market Value Prediction
```

---

## 2. Current State

### What exists

| Asset | Location | Purpose |
|-------|----------|---------|
| Raw data (5 files) | `C:\Football\Data\` | Source datasets |
| `audit_pipeline.py` | `C:\Football\Data\audit_pipeline.py` | Automated profiling + leakage audit script |
| `AUDIT_SUMMARY.md` | `C:\Football\Data\AUDIT_SUMMARY.md` | Full audit output (406 lines) |
| `data_profiling.py` | `C:\Football\src\data_profiling.py` | Earlier profiling script (superseded by audit_pipeline) |
| `__init__.py` | `C:\Football\src\__init__.py` | Empty package marker |

### What does NOT exist yet

- No `src/elo.py` (Elo rating engine)
- No `src/valuation.py` (market value model)
- No `src/features.py` (feature engineering)
- No `src/ingest.py` (match-performance ingestion from SQL/CSV)
- No `src/pipeline.py` (orchestration)
- No `requirements.txt` or `pyproject.toml`
- No tests
- No notebooks

---

## 3. Data Inventory

### 3.1 `player_valuations_full (1).csv` -- CORE LABEL TIMELINE

| Fact | Value |
|------|-------|
| Rows | 1,380 |
| Columns | 6: `player_id`, `player_name`, `valuation_date`, `market_value`, `club`, `club_id` |
| Unique players | 46 |
| Date range | 2005-08-07 to 2026-07-22 |
| Missing | `club`/`club_id` null for 10 rows (0.7%) |
| **Role in pipeline** | **Primary label source.** `market_value` at each `valuation_date` is the target variable. |

### 3.2 `club_transfers (1).csv` -- TRANSFER EVENT LOG

| Fact | Value |
|------|-------|
| Rows | 105 |
| Columns | 9: `direction`, `transfer_date`, `player_id`, `player_name`, `price`, `market_value_at_transfer`, `is_loan`, `counterpart_club_id`, `counterpart_club_name` |
| Unique players | 59 (20 not in valuations) |
| Date range | 2021-08-10 to 2026-07-08 |
| **Critical gap** | `price` is **53.3% null** (56 of 105 rows). Use `market_value_at_transfer` as fallback. |
| **Role in pipeline** | Transfer events mark regime changes (club changes trigger valuation jumps). `direction` = in/out. `is_loan` distinguishes temporary vs permanent. |

### 3.3 `team_full_dossier (1).csv` -- FINANCIAL RISK METRICS

| Fact | Value |
|------|-------|
| Rows | 260 (46 players x ~5.7 snapshots each) |
| Columns | 64 |
| Unique players | 46 (100% overlap with valuations) |
| Date range | as_of_date: 2021-12-31 to 2026-07-30 |
| **Role in pipeline** | Rich feature source: Sharpe, Sortino, Calmar, VaR, CVaR, CAGR, volatility, drawdowns. **But 15 columns leak future info -- see Section 5.** |

### 3.4 `SD-DB(PLAYERS) (1).csv` -- REAL MADRID CASE STUDY (SMALL)

| Fact | Value |
|------|-------|
| Rows | 70 |
| Columns | 33 (6 are junk `Unnamed:*`) |
| Join key | **None** -- no `player_id`. Name-only match. Only 3 of 70 names match valuations. |
| **Role in pipeline** | Case-study data for Real Madrid academy departures. Contains `PRED_1Y/2Y/3Y` (leaking) and `Highest Value Reached` (hindsight). Use only for transfer-optionality feature engineering, not as model features directly. |

### 3.5 `real_madrid_la_fabrica_deep_analysis_expanded_2006_2026 (1).xlsx` -- EXPANDED CASE STUDY

| Fact | Value |
|------|-------|
| Rows | 137 (Cases sheet) |
| Columns | 17 |
| Sheets | Cases, Coverage, Summary, Timeline, Glossary, Sources, By season |
| **Role in pipeline** | Extended version of SD-DB. Same structure. `Optionality index` and `Confidence` are post-hoc -- quarantine. |

### 3.6 `soccersolver_export_investigacion_2023-24_a_2025-26.sql` -- MATCH-PERFORMANCE SQL

| Fact | Value |
|------|-------|
| File size | ~38M lines |
| Seasons covered | **2023-24, 2024-25, 2025-26 only** |
| Key tables | `sports.player_match_stats_total` (per-match stats), `sports.matches` (match metadata), `sports.players` (demographics), `sports.teams` |
| Stats available | minutes, goals, assists, xG, xA, shots, passes, progressive passes, duels, interceptions, defensive actions, cards, and 80+ more fields |
| **Critical gap** | Valuations span 2005-2026 but SQL only covers 2023-2026. ~18 years of match data missing for Elo warm-up. |

---

## 4. Entity Resolution

### How files link

```
player_valuations  <--player_id-->  team_dossier        (100% match, 46/46)
player_valuations  <--player_id-->  club_transfers       (84.8% match, 39/46)
club_transfers     <--player_id-->  team_dossier         (84.8% match, 39/46)
sd_db_players      <--name only-->  player_valuations    (4.3% match, 3/70)
real_madrid_xlsx   <--name only-->  player_valuations    (case study, not core pipeline)
```

### 20 transfer-only players (in transfers, NOT in valuations)

These are players who were bought/sold but whose valuation timeline is absent:

```
164770  Raphael Varane       241641  Bernardo Silva     298976  Borja Mayoral
316264  Martin Odegaard      321528  Denzel Dumfries    357119  Ibrahima Konate
405398  Takefusa Kubo        495625  Victor Chust       525299  Miguel Gutierrez
537762  Sergio Arribas       623521  Juanmi Latasa      627226  Reinier
646824  Manuel Angel         662261  Rafa Marin         935231  Victor Munoz
935281  David Jimenez        948272  Cesar Palacios     948275  Alvaro Rodriguez
948294  Nico Paz             1055220 Fran Gonzalez
```

These players exist only in `club_transfers` and possibly `team_dossier`. If you want valuation labels for them, you need to source them externally.

---

## 5. Leakage Quarantine -- CRITICAL

These columns encode **future information** and MUST be excluded from any feature matrix used for real-time prediction:

### From `team_full_dossier` (15 columns)

```
predicted_value_1y              predicted_value_2y              predicted_value_3y
fair_price_1y                  fair_price_2y                   fair_price_3y
accuracy_1y                    accuracy_2y                     accuracy_3y
prediction_calculation_date    prediction_season               upside_potential_current
return_on_salary_since_arrival salary_to_value_ratio_current   marginal_value_growth_since_arrival
```

### From `sd_db_players` (9 columns)

```
PRED_1Y    1Y_FP    PRED_2Y    2Y_FP    PRED_3Y    3Y_FP
Current Value (M)    Highest Value Reached (M)    Scenario
```

### From `real_madrid_xlsx` (3 columns)

```
Direct net income (M)    Optionality index    Confidence
```

### Safe feature columns from `team_full_dossier` (48 columns)

All the risk/return metrics (CAGR, Sharpe, Sortino, Calmar, VaR, CVaR, volatility, drawdowns, career slope) plus metadata columns (`player_id`, `player_name`, `as_of_date`, `metric_calculation_date`, `market_value_at_checkpoint`, `club_id_at_checkpoint`, `time_on_market_pre_last_transfer`, `created_at`, `updated_at`).

---

## 6. Elo Rating System -- Design Spec

### 6.1 Standard Elo Formula (adapted for players)

```
R_new = R_old + K * (S - E)

where:
  R_old  = current player Elo rating
  K      = sensitivity constant (tune: 16-32 typical)
  S      = actual performance score (match-level, 0-1 scaled)
  E      = expected performance score (based on opponent/team strength)
```

### 6.2 What feeds into `S` (actual performance)

From `player_match_stats_total`, per match:

| Component | Fields | Weight hint |
|-----------|--------|-------------|
| Goal contribution | `goals`, `assists`, `xg_shot`, `xg_assist` | High |
| Creative output | `key_passes`, `progressive_passes`, `smart_passes` | Medium |
| Defensive work | `defensive_actions`, `interceptions`, `duels_won` | Medium |
| Discipline | `yellow_cards`, `red_cards` (negative) | Low-moderate |
| Playing time | `minutes_played` (normaliser) | Divisor |

### 6.3 What feeds into `E` (expected performance)

- **Opponent team Elo** -- derived from `sports.matches` results (build a separate team Elo first).
- **Home/away** -- from `matches.venue` or home/away indicator.
- **Position** -- from `player_match_stats_positions.position_code`.

### 6.4 K-factor adaptations

- Higher K for younger players (faster rating updates, more volatility).
- Lower K for veterans (rating stability).
- Consider `age = (match_date - birth_date) / 365.25` from `sports.players`.

### 6.5 Cold start

Players with no match history before their first valuation get an initial Elo of 1500 (standard). The system needs at minimum 5-10 matches to produce a meaningful rating.

---

## 7. Market Value Model -- Design Spec

### 7.1 Target variable

`market_value` from `player_valuations`, predicted at a future `valuation_date`.

### 7.2 Feature groups

| Group | Source | Examples |
|-------|--------|----------|
| Elo rating | Computed from match performance | Current Elo, Elo delta (last N matches), Elo rank |
| Financial risk | `team_full_dossier` (safe columns) | Sharpe ratio, volatility, max drawdown, CAGR |
| Contractual | `club_transfers` + `SD-DB` / `xlsx` | `is_loan`, `direction`, transfer fee, retained rights % |
| Demographic | `sports.players` | age, position, foot, height |
| Career trajectory | `player_valuations` (lagged) | valuation velocity, club history |

### 7.3 Modelling approach (implemented in `src/valuation.py:1`)

- **Regression** on `log1p(market_value)` (right-skewed, 300K to 180M EUR).
- Models: Ridge (alpha=10, median imputation + StandardScaler, <60% null features), LightGBM (n=300, lr=0.05), XGBoost (n=300, lr=0.05).
- Temporal expanding-window CV (3 splits, 50% initial train). `expanding_window_splits` in `src/valuation.py:95`.
- **Verified baseline (output/baseline_metrics.json):**

| Model   | R2_log (mean) | MAE_log | MAE_EUR | Directional Acc. |
|---------|--------------|---------|---------|-----------------|
| Ridge   | 0.834        | 0.465   | 12.7M   | 35.3%           |
| LightGBM| 0.842        | 0.231   | 3.7M    | 62.7%           |
| XGBoost | **0.854**    | **0.156** | **1.8M** | **67.4%**    |

  - 1,380 training samples, 61 feature columns, 230 test per fold.
  - XGBoost best; tree models handle sparse dossier features far better than linear.

---

## 7.4 How to run

```powershell
python -m src.pipeline --train                  # fast path (CSV only, neutral Elo=1500)
python -m src.pipeline --train --with-elo        # full Elo (parses 6.2 GB SQL, caches to Data/processed/*.parquet)
python -m src.pipeline --train --splits 5        # 5-fold expanding window
python -m src.pipeline --evaluate                # alias for --train
python -m pytest tests/test_elo.py -v            # 20/20 Elo tests
```

---

## 8. Known Gaps & Blockers

| # | Gap | Severity | Mitigation |
|---|-----|----------|------------|
| 1 | SQL match data covers only 2023-2026; valuations span 2005-2026 | **HIGH** | Source historical Wyscout/StatsBomb data for 2005-2023, or accept a shorter Elo history |
| 2 | SD-DB has no `player_id` join key | MEDIUM | Fuzzy name matching (`thefuzz` library) or manual mapping CSV |
| 3 | `club_transfers.price` is 53% null | MEDIUM | Binary `price_known` flag; use `market_value_at_transfer` as proxy |
| 4 | 20 transfer players have no valuation timeline | LOW | Source valuations from Transfermarkt for those 20 IDs |
| 5 | 4 columns in `team_dossier` are 100% null (`upside_potential_current`, etc.) | LOW | Drop from schema; no information lost |
| 6 | No injury/availability data | LOW | Optional enrichment; not blocking for MVP |
| 7 | SD-DB and xlsx have encoding issues (latin-1 / cp1252) | LOW | `_read_csv()` in `audit_pipeline.py` already handles multi-encoding |

---

## 9. Reproducing the Audit & Pipeline

```powershell
cd C:\Football\Data
python audit_pipeline.py                          # Outputs: AUDIT_SUMMARY.md

cd C:\Football
python -m pytest tests/test_elo.py -v             # 20/20 Elo tests
python -m src.pipeline --train                    # Full pipeline, 22s, outputs baseline_metrics.json
python -m src.features                            # Standalone feature matrix demo
python -m src.ingest --help                       # SQL ingestion options
```

Requires: `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `openpyxl`, `thefuzz`, `pyarrow`, `pytest` (see `requirements.txt:1`).

---

## 10. Roadmap

```
Phase 1  [DONE]     Data audit, leakage identification, gap specification
Phase 2  [DONE]     Ingest (src/ingest.py:1 -- chunked COPY parser + parquet cache)
Phase 3  [DONE]     Team Elo from match results (src/elo.py:150)
Phase 4  [DONE]     Player Elo R_new=R_old+K*(S-E) with age-decayed K (src/elo.py:240)
Phase 5  [DONE]     Feature matrix 1380x70, 61 cols, quarantine-enforced (src/features.py:1)
Phase 6  [DONE]     Ridge / LightGBM / XGBoost on log1p(market_value), expanding-window CV (src/valuation.py:1)
Phase 7  [NEXT]     Real-time update pipeline (match-by-match ingestion)
Phase 8  [NEXT]     Backtesting, deployment, dashboard
```

---

## 11. Directory Structure (Actual)

```
C:\Football\
├── Data/
│   ├── player_valuations_full (1).csv
│   ├── club_transfers (1).csv
│   ├── team_full_dossier (1).csv
│   ├── SD-DB(PLAYERS) (1).csv
│   ├── real_madrid_la_fabrica_deep_analysis_expanded_2006_2026 (1).xlsx
│   ├── soccersolver_export_investigacion_2023-24_a_2025-26.sql/
│   │   └── soccersolver_export_investigacion_2023-24_a_2025-26.sql  (6.2 GB)
│   ├── processed/             # parquet cache (generated on --with-elo)
│   ├── audit_pipeline.py
│   └── AUDIT_SUMMARY.md
├── src/
│   ├── __init__.py
│   ├── data_profiling.py
│   ├── ingest.py              # chunked SQL COPY parser + CSV loaders
│   ├── elo.py                 # Team + Player Elo (S/E/K with age decay)
│   ├── features.py            # unified (player_id, valuation_date) matrix
│   ├── valuation.py           # Ridge / LGBM / XGB + expanding-window CV
│   └── pipeline.py            # CLI orchestration (--train/--evaluate/--predict/--with-elo)
├── models/                    # saved model artifacts
├── output/
│   ├── baseline_metrics.json  # 3-model CV results (MAE/RMSE/R2/DA per fold)
│   └── feature_matrix.parquet # 1380x70 full feature matrix
├── tests/
│   ├── __init__.py
│   └── test_elo.py            # 20 smoke tests (team + player Elo)
├── notebooks/                 # EDA + experiment notebooks
└── requirements.txt
```

---

*End of handoff.*
