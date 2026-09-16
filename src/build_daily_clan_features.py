import os

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from dotenv import load_dotenv


# ================================================================
# CONFIGURATION
# ================================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

GALACTIC_KINGS_TAG = "#Y9202P9U"
RIVER_LENGTH = 10000

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ================================================================
# LOAD GALACTIC KINGS DAILY PERIOD RESULTS
# ================================================================

query = """
SELECT
    cwpr.live_race_id,
    lr.section_index,
    lr.period_type,
    cwpr.clan_tag,
    c.clan_name,
    cwpr.period_index,
    cwpr.points_earned,
    cwpr.progress_start_of_day,
    cwpr.progress_end_of_day,
    cwpr.end_of_day_rank,
    cwpr.progress_earned,
    cwpr.progress_earned_from_defenses
FROM clan_war_period_results cwpr
JOIN live_races lr
    ON cwpr.live_race_id = lr.live_race_id
JOIN clans c
    ON cwpr.clan_tag = c.clan_tag
WHERE
    cwpr.clan_tag = %(gk_tag)s
ORDER BY
    cwpr.live_race_id,
    cwpr.period_index
"""

df = pd.read_sql(
    query,
    engine,
    params={
        "gk_tag": GALACTIC_KINGS_TAG
    },
)

print(
    "Raw GK daily rows:",
    len(df)
)


if df.empty:
    raise RuntimeError(
        "No Galactic Kings daily period results found."
    )


# ================================================================
# KEEP ONLY BATTLE DAYS
# ================================================================

df["period_position"] = (
    pd.to_numeric(
        df["period_index"],
        errors="coerce",
    )
    % 7
)

df = df[
    df["period_position"].between(
        3,
        6,
    )
].copy()


# ================================================================
# REMOVE UNFINISHED DAYS
# ================================================================

df["end_of_day_rank"] = pd.to_numeric(
    df["end_of_day_rank"],
    errors="coerce",
)

df = df[
    df["end_of_day_rank"].notna()
    & (
        df["end_of_day_rank"] >= 0
    )
].copy()

print(
    "Completed GK Battle Day rows:",
    len(df)
)


# ================================================================
# NORMALIZE NUMERIC COLUMNS
# ================================================================

numeric_columns = [
    "section_index",
    "period_index",
    "points_earned",
    "progress_start_of_day",
    "progress_end_of_day",
    "end_of_day_rank",
    "progress_earned",
    "progress_earned_from_defenses",
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


# ================================================================
# BATTLE DAY NUMBER
# ================================================================

# periodIndex % 7:
#
# 3 = Battle Day 1
# 4 = Battle Day 2
# 5 = Battle Day 3
# 6 = Battle Day 4

df["battle_day"] = (
    df["period_position"]
    - 2
).astype(int)


# ================================================================
# WAR MODE
# ================================================================

df["war_mode"] = (
    df["period_type"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

df["is_colosseum"] = (
    df["war_mode"]
    == "colosseum"
)


# ================================================================
# HUMAN-READABLE DAILY RANK
# ================================================================

# API rank is 0-based.
#
# 0 = 1st
# 1 = 2nd
# etc.

df["daily_place"] = (
    df["end_of_day_rank"]
    + 1
).astype(int)


# ================================================================
# RIVER MOVEMENT
# ================================================================

df["rank_movement"] = (
    df["progress_earned"]
    .fillna(0)
)

df["defense_movement"] = (
    df["progress_earned_from_defenses"]
    .fillna(0)
)

df["total_movement_earned"] = (
    df["rank_movement"]
    + df["defense_movement"]
)


# ================================================================
# RIVER FINISH STATUS
# ================================================================

# Colosseum does not use River Race movement / finish line.
#
# For normal River Race rows, reaching 10,000 means the clan has
# completed the River Race.

df["finished_after_day"] = (
    (~df["is_colosseum"])
    &
    (
        df["progress_end_of_day"]
        >= RIVER_LENGTH
    )
)


# ================================================================
# IDENTIFY FIRST FINISH DAY PER RACE
# ================================================================

finish_day_lookup = (
    df[
        df["finished_after_day"]
    ]
    .groupby(
        "live_race_id"
    )[
        "battle_day"
    ]
    .min()
    .to_dict()
)

df["race_finish_day"] = (
    df["live_race_id"]
    .map(
        finish_day_lookup
    )
)

df["finished_race"] = (
    df["race_finish_day"]
    .notna()
)

df["race_finished_by_day_3"] = (
    df["race_finish_day"]
    .fillna(99)
    <= 3
)


# ================================================================
# LOAD GK PLAYER-DAY PARTICIPATION
# ================================================================

# live_player_war_status stores cumulative war state for the live race,
# not a clean historical player-day fact table.
#
# Therefore, for now we derive player-day participation only when the
# currently stored live race corresponds to the period result row.
#
# Historical player-day participation should eventually be collected
# explicitly if we want a proper supervised daily player model.

player_query = """
SELECT
    lpws.live_race_id,
    lpws.player_tag,
    lpws.fame,
    lpws.decks_used,
    lpws.decks_used_today,
    lpws.boat_attacks
FROM live_player_war_status lpws
WHERE
    lpws.clan_tag = %(gk_tag)s
"""

player_status = pd.read_sql(
    player_query,
    engine,
    params={
        "gk_tag": GALACTIC_KINGS_TAG
    },
)

if not player_status.empty:

    for column in [
        "fame",
        "decks_used",
        "decks_used_today",
        "boat_attacks",
    ]:

        player_status[column] = pd.to_numeric(
            player_status[column],
            errors="coerce",
        ).fillna(0)


# ================================================================
# CURRENT LIVE-RACE PLAYER CONTEXT
# ================================================================

# We only attach these fields to rows where the live race IDs match.
# This avoids pretending we have historical player-day participation
# data when we do not.

if not player_status.empty:

    player_context = (
        player_status
        .groupby(
            "live_race_id"
        )
        .agg(
            race_players_seen=(
                "player_tag",
                "nunique",
            ),

            race_contributors_seen=(
                "player_tag",
                lambda tags:
                tags.nunique(),
            ),

            race_total_decks_seen=(
                "decks_used",
                "sum",
            ),

            race_total_fame_seen=(
                "fame",
                "sum",
            ),
        )
        .reset_index()
    )

    df = df.merge(
        player_context,
        on="live_race_id",
        how="left",
    )

else:

    df["race_players_seen"] = np.nan
    df["race_contributors_seen"] = np.nan
    df["race_total_decks_seen"] = np.nan
    df["race_total_fame_seen"] = np.nan


# ================================================================
# DAILY EFFICIENCY
# ================================================================

# At the clan/day level:
#
# medals per point of river movement is not meaningful.
# A better current daily efficiency indicator is medals earned per
# rank-movement unit only for analysis, but we keep the clearer raw
# fields instead of inventing a composite score.
#
# We do compute how much of the day's movement came from defenses.

df["defense_movement_share"] = np.where(
    df["total_movement_earned"] > 0,
    (
        df["defense_movement"]
        / df["total_movement_earned"]
    ),
    0.0,
)


# ================================================================
# GK HISTORY WITHIN EACH RACE
# ================================================================

groups = df.groupby(
    "live_race_id",
    sort=False,
)


df["previous_day_points"] = (
    groups[
        "points_earned"
    ]
    .shift(1)
)


df["previous_day_place"] = (
    groups[
        "daily_place"
    ]
    .shift(1)
)


df["previous_day_progress"] = (
    groups[
        "progress_end_of_day"
    ]
    .shift(1)
)


df["previous_day_total_movement"] = (
    groups[
        "total_movement_earned"
    ]
    .shift(1)
)


df["avg_previous_points"] = (
    groups[
        "points_earned"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


df["avg_previous_place"] = (
    groups[
        "daily_place"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


df["avg_previous_total_movement"] = (
    groups[
        "total_movement_earned"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


# ================================================================
# CROSS-RACE GK HISTORICAL FEATURES
# ================================================================

# These are shifted so every row only sees older GK Battle Days.

df = df.sort_values(
    [
        "section_index",
        "live_race_id",
        "battle_day",
    ]
).reset_index(
    drop=True
)


df["gk_last_3_day_avg_points"] = (
    df["points_earned"]
    .shift(1)
    .rolling(
        3,
        min_periods=1,
    )
    .mean()
)


df["gk_last_5_day_avg_points"] = (
    df["points_earned"]
    .shift(1)
    .rolling(
        5,
        min_periods=1,
    )
    .mean()
)


df["gk_last_3_day_avg_place"] = (
    df["daily_place"]
    .shift(1)
    .rolling(
        3,
        min_periods=1,
    )
    .mean()
)


df["gk_last_3_day_avg_movement"] = (
    df["total_movement_earned"]
    .shift(1)
    .rolling(
        3,
        min_periods=1,
    )
    .mean()
)


# ================================================================
# BATTLE-DAY-SPECIFIC HISTORICAL BASELINES
# ================================================================

# Example:
# before today's Day 2 row, what has GK historically averaged on
# previous Day 2s?

df["gk_previous_same_day_avg_points"] = (
    df.groupby(
        "battle_day"
    )[
        "points_earned"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


df["gk_previous_same_day_avg_place"] = (
    df.groupby(
        "battle_day"
    )[
        "daily_place"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


df["gk_previous_same_day_avg_movement"] = (
    df.groupby(
        "battle_day"
    )[
        "total_movement_earned"
    ]
    .transform(
        lambda x:
        x.shift(1)
        .expanding()
        .mean()
    )
)


# ================================================================
# RACE-LEVEL SUMMARY
# ================================================================

race_summary = (
    df.groupby(
        [
            "live_race_id",
            "section_index",
            "war_mode",
            "is_colosseum",
        ],
        dropna=False,
    )
    .agg(
        completed_battle_days=(
            "battle_day",
            "nunique",
        ),

        total_points_earned=(
            "points_earned",
            "sum",
        ),

        avg_daily_points=(
            "points_earned",
            "mean",
        ),

        avg_daily_place=(
            "daily_place",
            "mean",
        ),

        total_rank_movement=(
            "rank_movement",
            "sum",
        ),

        total_defense_movement=(
            "defense_movement",
            "sum",
        ),

        total_movement_earned=(
            "total_movement_earned",
            "sum",
        ),

        final_progress=(
            "progress_end_of_day",
            "max",
        ),
    )
    .reset_index()
)


race_summary["race_finish_day"] = (
    race_summary[
        "live_race_id"
    ]
    .map(
        finish_day_lookup
    )
)


race_summary["finished_race"] = (
    race_summary[
        "race_finish_day"
    ]
    .notna()
)


race_summary["finished_by_day_3"] = (
    race_summary[
        "race_finish_day"
    ]
    .fillna(99)
    <= 3
)


# ================================================================
# OVERALL GK SUMMARY
# ================================================================

normal_race_summary = race_summary[
    ~race_summary[
        "is_colosseum"
    ]
].copy()


print(
    "\nGALACTIC KINGS HISTORY SUMMARY"
)

print(
    "Completed Battle Days:",
    len(df),
)

print(
    "Recorded races:",
    race_summary[
        "live_race_id"
    ].nunique(),
)

print(
    "Normal River Races:",
    len(
        normal_race_summary
    ),
)

if not normal_race_summary.empty:

    finished_count = int(
        normal_race_summary[
            "finished_race"
        ].sum()
    )

    finished_day_3_count = int(
        normal_race_summary[
            "finished_by_day_3"
        ].sum()
    )

    print(
        "Completed River Races:",
        finished_count,
    )

    print(
        "Finished by Day 3:",
        finished_day_3_count,
    )

    if finished_count > 0:

        print(
            "Day-3 finish rate among finished races:",
            round(
                (
                    finished_day_3_count
                    / finished_count
                )
                * 100,
                2,
            ),
            "%",
        )


# ================================================================
# DISPLAY DAILY DATA
# ================================================================

display_columns = [
    "live_race_id",
    "war_mode",
    "battle_day",
    "points_earned",
    "daily_place",
    "progress_start_of_day",
    "rank_movement",
    "defense_movement",
    "total_movement_earned",
    "progress_end_of_day",
    "finished_after_day",
    "race_finish_day",
    "previous_day_points",
    "avg_previous_points",
    "gk_last_3_day_avg_points",
    "gk_previous_same_day_avg_points",
]


print(
    "\nGALACTIC KINGS DAILY DATA"
)

print(
    df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ================================================================
# SAVE DATASETS
# ================================================================

df.to_csv(
    "data/gk_daily_war_dataset.csv",
    index=False,
)

race_summary.to_csv(
    "data/gk_race_summary.csv",
    index=False,
)


print(
    "\nSaved:"
)

print(
    "data/gk_daily_war_dataset.csv"
)

print(
    "data/gk_race_summary.csv"
)
