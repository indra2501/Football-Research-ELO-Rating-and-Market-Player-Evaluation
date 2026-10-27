"""
src/elo.py
==========
Dynamic Team & Player Elo engine.

Team Elo: standard Elo with home-advantage, updated per match result.
Player Elo: R_new = R_old + K * (S - E)
  S = weighted per-minute composite (goals, xG, key passes, etc.) scaled [0,1]
  E = sigmoid of (player Elo - opponent Team Elo - home penalty - position adj.)
  K = base_K * age_decay(player_age)

Cold start: 1500 for every team / player.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RATING: float = 1500.0
TEAM_K: float = 20.0          # K-factor for team Elo
PLAYER_BASE_K: float = 24.0   # base K for player Elo before age decay
HOME_ADVANTAGE: float = 60.0  # Elo points added to home team expectation
ELO_SCALE: float = 400.0      # logistic scale (standard chess Elo)

# Player actual-score (S) weights (per-minute normalised components).
# These are combined linearly then passed through a sigmoid-like clamp.
S_WEIGHTS: Dict[str, float] = {
    # Goal actions
    "goals":            1.00,
    "assists":          0.70,
    "xg_shot":          0.50,
    "xg_assist":        0.35,
    # Creative
    "key_passes":       0.05,
    "progressive_passes": 0.02,
    "smart_passes":     0.04,
    "successful_crosses": 0.03,
    # Defensive
    "interceptions":    0.06,
    "duels_won":        0.015,
    "defensive_actions": 0.01,
    "recoveries":       0.01,
    # Discipline (negative)
    "yellow_cards":    -0.15,
    "red_cards":       -0.60,
}

# Position groups for E adjustment (defenders have lower goal expectation etc.)
POSITION_GROUPS: Dict[str, str] = {
    # Attack
    "FW": "attack", "CF": "attack", "ST": "attack", "W": "attack",
    "LW": "attack", "RW": "attack", "AM": "attack", "SS": "attack",
    # Midfield
    "CM": "midfield", "CDM": "midfield", "CAM": "midfield", "MF": "midfield",
    "DM": "midfield", "LM": "midfield", "RM": "midfield",
    # Defence
    "CB": "defence", "LB": "defence", "RB": "defence", "DF": "defence",
    "WB": "defence", "LWB": "defence", "RWB": "defence",
    # Keeper
    "GK": "keeper",
}

POSITION_E_ADJUST: Dict[str, float] = {
    "attack":   20.0,
    "midfield": 0.0,
    "defence": -15.0,
    "keeper":  -25.0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Probability A beats B (logistic)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / ELO_SCALE))


def _age_decay_k(age_years: Optional[float], base_k: float = PLAYER_BASE_K) -> float:
    """
    Scale K inversely with age.
      age < 21 : 1.5x base
      21-25    : 1.2x
      25-30    : 1.0x
      30-33    : 0.8x
      33+      : 0.6x
    """
    if age_years is None or np.isnan(age_years):
        return base_k
    if age_years < 21:
        return base_k * 1.5
    if age_years < 25:
        return base_k * 1.2
    if age_years < 30:
        return base_k * 1.0
    if age_years < 33:
        return base_k * 0.8
    return base_k * 0.6


def _get_age_years(birth_date, ref_date) -> Optional[float]:
    try:
        b = pd.to_datetime(birth_date, errors="coerce")
        r = pd.to_datetime(ref_date, errors="coerce")
        if pd.isna(b) or pd.isna(r):
            return None
        return (r - b).days / 365.25
    except Exception:
        return None


def _position_group(position_code: Optional[str]) -> str:
    if position_code is None or (isinstance(position_code, float) and np.isnan(position_code)):
        return "midfield"
    code = str(position_code).strip().upper()
    # Handle composite codes like "CM/CDM"
    for part in code.replace("/", ",").split(","):
        part = part.strip()
        if part in POSITION_GROUPS:
            return POSITION_GROUPS[part]
    return "midfield"


def compute_actual_score(row: pd.Series) -> float:
    """
    Weighted per-minute composite scaled to [0, 1].

    Raw composite is per-90 weighted sum; we divide by minutes and squash
    with a logistic to keep in [0,1].
    """
    minutes = row.get("minutes_played", 0)
    if pd.isna(minutes) or minutes == 0:
        return 0.0
    minutes = float(minutes)
    # Per-90 normalisation
    per90 = 90.0 / max(minutes, 1)

    raw = 0.0
    for col, w in S_WEIGHTS.items():
        v = row.get(col, 0)
        if pd.isna(v):
            v = 0
        v = float(v)
        # Normalise to per-90
        raw += v * w * per90

    # Squash raw into [0,1] via logistic centred at 0.3 (typical per-90 raw ~0.2-0.8)
    # raw=0 -> ~0.43, raw=1 -> ~0.67, raw=2 -> ~0.84
    # We shift so that an "average" performance (raw~0.3) maps to 0.5
    score = 1.0 / (1.0 + math.exp(-2.5 * (raw - 0.3)))
    # Clamp
    return float(np.clip(score, 0.01, 0.99))


def compute_expected_score(
    player_elo: float,
    opponent_team_elo: float,
    is_home: bool = False,
    position_group: str = "midfield",
) -> float:
    """
    Expected performance given player vs opponent strength.
    Incorporates home advantage and position-group offset.
    """
    pos_adj = POSITION_E_ADJUST.get(position_group, 0.0)
    # Adjust opponent elo for home/away
    adj_opp = opponent_team_elo
    if is_home:
        # Playing at home makes it easier -> reduce opponent effective elo
        adj_opp -= 15.0
    else:
        adj_opp += 15.0

    effective_player = player_elo + pos_adj
    return _elo_expected(effective_player, adj_opp)

# ---------------------------------------------------------------------------
# Team Elo
# ---------------------------------------------------------------------------

@dataclass
class TeamEloState:
    ratings: Dict[int, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def get(self, team_id: int) -> float:
        return self.ratings.get(team_id, DEFAULT_RATING)

    def ensure(self, team_id: int) -> None:
        if team_id not in self.ratings:
            self.ratings[team_id] = DEFAULT_RATING


def compute_team_elo(
    matches: pd.DataFrame,
    match_teams: Optional[pd.DataFrame] = None,
    k: float = TEAM_K,
    home_advantage: float = HOME_ADVANTAGE,
) -> Tuple[TeamEloState, pd.DataFrame]:
    """
    Iterate matches chronologically and update team Elo.

    Parameters
    ----------
    matches : DataFrame with at least
        match_id, match_datetime_utc, winner_team_id
    match_teams : optional, with match_id, team_id, side
        If provided, determines home/away and handles draws correctly.
        Otherwise winner_team_id alone is used (draws = both get 0.5).

    Returns
    -------
    state : TeamEloState with final ratings + per-match history
    match_team_elo : DataFrame with match_id, team_id, elo_before, elo_after
    """
    if matches.empty:
        return TeamEloState(), pd.DataFrame(columns=["match_id", "team_id", "elo_before", "elo_after"])

    # Sort chronologically
    if "match_datetime_utc" in matches.columns:
        matches = matches.sort_values("match_datetime_utc").reset_index(drop=True)
    else:
        matches = matches.sort_values("match_id").reset_index(drop=True)

    state = TeamEloState()
    records: List[Dict] = []

    # Build lookup: match_id -> {team_id: side} if match_teams available
    team_side_lookup: Dict[int, Dict[int, str]] = {}
    if match_teams is not None and not match_teams.empty and "side" in match_teams.columns:
        for _, r in match_teams.iterrows():
            mid = int(r["match_id"])
            tid = int(r["team_id"]) if pd.notna(r["team_id"]) else None
            if tid is None:
                continue
            team_side_lookup.setdefault(mid, {})[tid] = str(r.get("side", ""))

    # Also need scores for draw detection
    has_scores = "score" in (match_teams.columns if match_teams is not None and not match_teams.empty else [])

    for _, m in matches.iterrows():
        mid = int(m["match_id"])
        winner = m.get("winner_team_id", None)
        if pd.isna(winner):
            winner = None
        else:
            winner = int(winner)

        # Get teams for this match
        teams_in_match: List[int] = []
        if mid in team_side_lookup:
            teams_in_match = list(team_side_lookup[mid].keys())
        elif match_teams is not None and not match_teams.empty:
            sub = match_teams[match_teams["match_id"] == mid]
            teams_in_match = [int(t) for t in sub["team_id"].dropna().unique()]

        if len(teams_in_match) < 2:
            # Cannot update without two teams
            continue

        # Only handle 2-team matches (standard)
        if len(teams_in_match) != 2:
            continue

        t1, t2 = teams_in_match[0], teams_in_match[1]
        state.ensure(t1)
        state.ensure(t2)

        r1, r2 = state.get(t1), state.get(t2)

        # Determine home side
        is_t1_home = False
        if mid in team_side_lookup:
            side1 = team_side_lookup[mid].get(t1, "")
            is_t1_home = side1.lower() in ("home", "h")

        # Adjust for home advantage in expected calculation
        r1_adj = r1 + (home_advantage if is_t1_home else 0)
        r2_adj = r2 + (home_advantage if not is_t1_home else 0)

        e1 = _elo_expected(r1_adj, r2_adj)
        e2 = 1 - e1

        # Actual scores
        if winner is None or (isinstance(winner, float) and np.isnan(winner)):
            # Draw -- check if we can detect via scores
            if has_scores and match_teams is not None:
                sub = match_teams[match_teams["match_id"] == mid]
                scores = sub["score"].dropna().tolist()
                if len(scores) == 2 and scores[0] == scores[1]:
                    s1, s2 = 0.5, 0.5
                elif winner is not None:
                    s1 = 1.0 if winner == t1 else 0.0
                    s2 = 1 - s1
                else:
                    s1, s2 = 0.5, 0.5
            else:
                s1, s2 = 0.5, 0.5
        else:
            s1 = 1.0 if winner == t1 else 0.0
            s2 = 1 - s1

        # Update
        new_r1 = r1 + k * (s1 - e1)
        new_r2 = r2 + k * (s2 - e2)

        records.append({"match_id": mid, "team_id": t1, "elo_before": r1, "elo_after": new_r1,
                        "opponent_id": t2, "expected": e1, "actual": s1})
        records.append({"match_id": mid, "team_id": t2, "elo_before": r2, "elo_after": new_r2,
                        "opponent_id": t1, "expected": e2, "actual": s2})

        state.ratings[t1] = new_r1
        state.ratings[t2] = new_r2
        state.history.append({"match_id": mid, "t1": t1, "t2": t2, "s1": s1, "s2": s2, "e1": e1})

    match_team_elo = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["match_id", "team_id", "elo_before", "elo_after"]
    )
    return state, match_team_elo

# ---------------------------------------------------------------------------
# Player Elo
# ---------------------------------------------------------------------------

@dataclass
class PlayerEloState:
    ratings: Dict[int, float] = field(default_factory=dict)
    match_count: Dict[int, int] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def get(self, player_id: int) -> float:
        return self.ratings.get(player_id, DEFAULT_RATING)


def compute_player_elo(
    match_stats: pd.DataFrame,
    matches: pd.DataFrame,
    players: Optional[pd.DataFrame] = None,
    match_teams: Optional[pd.DataFrame] = None,
    team_elo_state: Optional[TeamEloState] = None,
    base_k: float = PLAYER_BASE_K,
    home_advantage: float = HOME_ADVANTAGE,
    position_map: Optional[Dict[int, str]] = None,
) -> Tuple[PlayerEloState, pd.DataFrame]:
    """
    Compute player Elo ratings updated after each match appearance.

    Parameters
    ----------
    match_stats : player_match_stats_total (player_id, match_id, goals, assists, ...)
    matches : sports.matches (match_id, match_datetime_utc, competition_id, ...)
    players : sports.players (player_id, birth_date, ...) for age-adjusted K
    match_teams : sports.match_teams (match_id, team_id, side) for opponent lookup
    team_elo_state : precomputed TeamEloState (if None, team Elo is not used in E)

    Returns
    -------
    state : PlayerEloState with final ratings
    history_df : per-(player, match) Elo trajectory
    """
    if match_stats.empty:
        return PlayerEloState(), pd.DataFrame()

    # --- prepare match lookup ---
    match_lookup: Dict[int, Dict] = {}
    if not matches.empty:
        for _, r in matches.iterrows():
            mid = int(r["match_id"])
            match_lookup[mid] = {
                "datetime": r.get("match_datetime_utc", None),
                "competition_id": r.get("competition_id", None),
            }

    # --- player birth dates for age K ---
    birth_lookup: Dict[int, object] = {}
    if players is not None and not players.empty and "birth_date" in players.columns:
        for _, r in players.iterrows():
            birth_lookup[int(r["player_id"])] = r["birth_date"]

    # --- position group map ---
    if position_map is None:
        position_map = {}
    # Merge with match_stats if position columns exist
    if "position_code" in match_stats.columns:
        for _, r in match_stats.iterrows():
            pid = int(r["player_id"])
            if pid not in position_map and pd.notna(r.get("position_code")):
                position_map[pid] = str(r["position_code"])

    # --- team assignment per (player, match) ---
    # Use match_teams + match_player_appearances concept.
    # For now, we infer player's team from match_stats + match_teams:
    # If match_teams available, join via match_id -> team_id.
    # Otherwise, opponent Elo is approximated from average.

    # Build match_id -> list of team_ids
    match_to_teams: Dict[int, List[int]] = {}
    if match_teams is not None and not match_teams.empty:
        for _, r in match_teams.iterrows():
            mid = int(r["match_id"])
            tid = r.get("team_id", None)
            if pd.isna(tid):
                continue
            match_to_teams.setdefault(mid, []).append(int(tid))

    # Build (player_id, match_id) -> team_id via match_player_appearances or direct
    # If not available, we will assume opponent is average (=1500)
    # Check if match_stats has a team_id column
    player_team_lookup: Dict[Tuple[int, int], int] = {}
    if "team_id" in match_stats.columns:
        for _, r in match_stats.iterrows():
            if pd.notna(r.get("team_id")):
                player_team_lookup[(int(r["player_id"]), int(r["match_id"]))] = int(r["team_id"])

    # --- sort stats chronologically by match datetime ---
    # Add a sort key
    ms = match_stats.copy()
    ms["_match_dt"] = ms["match_id"].map(lambda mid: match_lookup.get(mid, {}).get("datetime", pd.NaT))
    ms["_match_dt"] = pd.to_datetime(ms["_match_dt"], errors="coerce")
    # Fallback: sort by match_id if no datetime
    ms = ms.sort_values(["_match_dt", "match_id", "player_id"], na_position="last").reset_index(drop=True)

    state = PlayerEloState()
    records: List[Dict] = []

    for _, row in ms.iterrows():
        pid = int(row["player_id"])
        mid = int(row["match_id"])

        old_rating = state.get(pid)

        # --- S: actual performance ---
        s = compute_actual_score(row)

        # --- E: expected performance ---
        # Find opponent team Elo
        opp_elo = DEFAULT_RATING
        is_home = False
        pos_group = _position_group(position_map.get(pid))

        if team_elo_state is not None and match_to_teams.get(mid):
            teams_in_match = match_to_teams[mid]
            my_team = player_team_lookup.get((pid, mid))
            if my_team is not None and len(teams_in_match) >= 2:
                # Opponent is the other team
                opp_candidates = [t for t in teams_in_match if t != my_team]
                if opp_candidates:
                    opp_id = opp_candidates[0]
                    # Use Elo BEFORE this match (need to look up historical)
                    # For simplicity use current team Elo; ideally would snapshot per-match
                    # We attempt to find the pre-match rating from match_teams history
                    opp_elo = team_elo_state.get(opp_id)
            elif len(teams_in_match) == 2:
                # Ambiguous team assignment: use average opponent Elo
                opp_elo = np.mean([team_elo_state.get(t) for t in teams_in_match])
            # Home check: need player's team side
            # Not critical; default to neutral

        e = compute_expected_score(old_rating, opp_elo, is_home=is_home, position_group=pos_group)

        # --- K: age-adjusted ---
        age = _get_age_years(birth_lookup.get(pid), match_lookup.get(mid, {}).get("datetime", None))
        k = _age_decay_k(age, base_k)

        new_rating = old_rating + k * (s - e)

        records.append({
            "player_id": pid,
            "match_id": mid,
            "match_datetime": match_lookup.get(mid, {}).get("datetime", None),
            "elo_before": old_rating,
            "elo_after": new_rating,
            "actual_score": s,
            "expected_score": e,
            "surprise": s - e,
            "k_factor": k,
            "age_years": age,
            "opponent_team_elo": opp_elo,
            "minutes_played": row.get("minutes_played", 0),
        })

        state.ratings[pid] = new_rating
        state.match_count[pid] = state.match_count.get(pid, 0) + 1
        state.history.append({"player_id": pid, "match_id": mid, "elo": new_rating})

    history_df = pd.DataFrame(records) if records else pd.DataFrame()
    return state, history_df


def elo_snapshot_at_date(
    history_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> Dict[int, float]:
    """
    Get each player's Elo as of a given date (last match <= as_of).
    """
    if history_df.empty:
        return {}
    mask = pd.to_datetime(history_df["match_datetime"], errors="coerce") <= pd.to_datetime(as_of)
    sub = history_df[mask]
    if sub.empty:
        return {}
    # Last per player
    idx = sub.groupby("player_id")["match_datetime"].idxmax()
    snap = sub.loc[idx]
    return dict(zip(snap["player_id"], snap["elo_after"]))


# ---------------------------------------------------------------------------
# Convenience: run both Team + Player Elo in sequence
# ---------------------------------------------------------------------------

def run_full_elo(
    matches: pd.DataFrame,
    match_teams: Optional[pd.DataFrame],
    match_stats: pd.DataFrame,
    players: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Execute Team Elo then Player Elo. Returns dict with states + history DataFrames.
    """
    print("  Computing Team Elo ...")
    team_state, team_history = compute_team_elo(matches, match_teams)
    print(f"    -> {len(team_state.ratings)} teams rated, {len(team_history)} match-team records")

    print("  Computing Player Elo ...")
    player_state, player_history = compute_player_elo(
        match_stats, matches, players, match_teams, team_elo_state=team_state,
    )
    print(f"    -> {len(player_state.ratings)} players rated, {len(player_history)} player-match records")

    return {
        "team_state": team_state,
        "team_history": team_history,
        "player_state": player_state,
        "player_history": player_history,
    }
