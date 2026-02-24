"""Player lookup service backed by Lahman data."""

from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

from config import PLAYER_INDEX_CACHE


def _normalize_name(row: pd.Series) -> str:
    first = str(row.get("namefirst", "")).strip()
    last = str(row.get("namelast", "")).strip()
    return f"{first} {last}".strip()


def _safe_load_teams(players_df: pd.DataFrame) -> pd.DataFrame:
    """Build optional player-to-team mapping from batting table."""
    try:
        import pylahman as pl

        batting = pl.Batting()
        batting.columns = batting.columns.str.lower()
        if "playerid" not in batting.columns or "teamid" not in batting.columns:
            return pd.DataFrame(columns=["playerid", "teams"])

        player_team = (
            batting[["playerid", "teamid"]]
            .dropna()
            .drop_duplicates()
            .groupby("playerid")["teamid"]
            .apply(lambda s: ", ".join(sorted(set(map(str, s)))))
            .reset_index(name="teams")
        )
        return player_team
    except Exception:
        # Team lookup is optional; fail soft and keep search usable.
        return pd.DataFrame(columns=["playerid", "teams"])


def build_player_index(force_refresh: bool = False) -> pd.DataFrame:
    """Create or load a cached player index with bbref IDs."""
    if PLAYER_INDEX_CACHE.exists() and not force_refresh:
        return pd.read_csv(PLAYER_INDEX_CACHE)

    import pylahman as pl

    people = pl.People()
    people.columns = people.columns.str.lower()

    keep_cols = ["playerid", "bbrefid", "namefirst", "namelast", "debut", "finalgame"]
    people = people[keep_cols].copy()
    people = people.dropna(subset=["bbrefid"])
    people["bbrefid"] = people["bbrefid"].astype(str).str.strip()
    people = people[people["bbrefid"] != ""]
    people["full_name"] = people.apply(_normalize_name, axis=1)

    team_map = _safe_load_teams(people)
    index_df = people.merge(team_map, on="playerid", how="left")
    index_df["teams"] = index_df["teams"].fillna("")
    index_df["search_blob"] = (
        index_df["full_name"].str.lower()
        + " "
        + index_df["playerid"].str.lower()
        + " "
        + index_df["bbrefid"].str.lower()
        + " "
        + index_df["teams"].str.lower()
    )

    PLAYER_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(PLAYER_INDEX_CACHE, index=False)
    return index_df


def _fuzzy_score(query: str, candidate: str) -> float:
    return SequenceMatcher(None, query, candidate).ratio()


def get_team_codes(force_refresh: bool = False) -> list[str]:
    """Return sorted unique Lahman-style team codes (e.g., NYA, NYN)."""
    index_df = build_player_index(force_refresh=force_refresh)
    teams = set()
    for row in index_df["teams"].fillna("").astype(str):
        for t in [p.strip() for p in row.split(",") if p.strip()]:
            teams.add(t)
    return sorted(teams)


def search_players(
    query: str,
    team_filter: Optional[str] = None,
    limit: int = 30,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Search players by id/name/team with deterministic ranking."""
    if not query and not team_filter:
        return pd.DataFrame()

    index_df = build_player_index(force_refresh=force_refresh)
    working = index_df.copy()

    if team_filter:
        team_filter_l = team_filter.lower().strip()
        working = working[working["teams"].str.lower().str.contains(team_filter_l, na=False)]

    if not query:
        cols = ["full_name", "playerid", "bbrefid", "teams", "debut", "finalgame"]
        return working[cols].head(limit)

    q = query.lower().strip()
    direct = working[
        (working["bbrefid"].str.lower() == q) | (working["playerid"].str.lower() == q)
    ].copy()
    direct["match_score"] = 1.0

    contains = working[working["search_blob"].str.contains(q, na=False)].copy()
    contains["match_score"] = contains["full_name"].str.lower().apply(lambda x: _fuzzy_score(q, x))

    merged = pd.concat([direct, contains], ignore_index=True).drop_duplicates(
        subset=["playerid", "bbrefid"], keep="first"
    )

    if merged.empty:
        # fallback fuzzy search on full name only
        fuzzy = working.copy()
        fuzzy["match_score"] = fuzzy["full_name"].str.lower().apply(lambda x: _fuzzy_score(q, x))
        merged = fuzzy[fuzzy["match_score"] > 0.45]

    merged = merged.sort_values(by=["match_score", "full_name"], ascending=[False, True])
    cols = ["full_name", "playerid", "bbrefid", "teams", "debut", "finalgame", "match_score"]
    return merged[cols].head(limit)

