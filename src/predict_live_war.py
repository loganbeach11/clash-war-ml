import os

import mysql.connector
import numpy as np
import pandas as pd

from dotenv import load_dotenv


# Configuration

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

GALACTIC_KINGS_TAG = "#Y9202P9U"

BATTLE_DAYS_PER_WEEK = 4
MAX_DAILY_DECKS = 4
RIVER_LENGTH = 10000

PLAYER_PREDICTIONS_FILE = "data/gk_live_player_predictions.csv"
GK_DAILY_HISTORY_FILE = "data/gk_daily_war_dataset.csv"
GK_RACE_HISTORY_FILE = "data/gk_race_summary.csv"

PLAYER_OUTPUT_FILE = "data/gk_live_war_player_projection.csv"
DAY_OUTPUT_FILE = "data/gk_live_war_day_projection.csv"
WAR_OUTPUT_FILE = "data/gk_live_war_projection.csv"


# Helpers

def safe_float(value, default=0.0):

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return float(value)


def load_optional_csv(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


def historical_mode(series):

    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    modes = clean.mode()

    if not modes.empty:
        return float(modes.iloc[0])

    return float(clean.median())


# Load GK live player predictions

predictions = pd.read_csv(
    PLAYER_PREDICTIONS_FILE
)

required_columns = [
    "player_tag",
    "clan_tag",
    "clan_name",
    "capacity_adjusted_fame",
    "daily_participation_weight",
    "decks_used_today",
]

missing_columns = [
    column
    for column in required_columns
    if column not in predictions.columns
]

if missing_columns:
    raise ValueError(
        "Missing required prediction columns: "
        + ", ".join(missing_columns)
    )


predictions = predictions[
    predictions["clan_tag"]
    == GALACTIC_KINGS_TAG
].copy()


if predictions.empty:
    raise RuntimeError(
        "No Galactic Kings players found in "
        "data/gk_live_player_predictions.csv."
    )


for column in [
    "capacity_adjusted_fame",
    "daily_participation_weight",
    "decks_used_today",
]:

    predictions[column] = pd.to_numeric(
        predictions[column],
        errors="coerce",
    ).fillna(0.0)


print(
    "Galactic Kings forecast-pool players:",
    predictions["player_tag"].nunique(),
)


# Load GK historical daily and race data

gk_daily_history = load_optional_csv(
    GK_DAILY_HISTORY_FILE
)

gk_race_history = load_optional_csv(
    GK_RACE_HISTORY_FILE
)


if not gk_daily_history.empty:

    for column in [
        "battle_day",
        "points_earned",
        "daily_place",
        "rank_movement",
        "defense_movement",
        "total_movement_earned",
        "progress_end_of_day",
    ]:

        if column in gk_daily_history.columns:

            gk_daily_history[column] = pd.to_numeric(
                gk_daily_history[column],
                errors="coerce",
            )


# Connect to MySQL

db = mysql.connector.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
)

cursor = db.cursor(
    dictionary=True
)


# Get current live race

cursor.execute(
    """
    SELECT
        live_race_id,
        section_index,
        period_index,
        period_type,
        last_seen_at
    FROM live_races
    ORDER BY last_seen_at DESC
    LIMIT 1
    """
)

live_race = cursor.fetchone()

if live_race is None:
    raise RuntimeError(
        "No live race found in live_races."
    )


live_race_id = live_race[
    "live_race_id"
]

section_index = int(
    live_race[
        "section_index"
    ]
)

period_index = int(
    live_race[
        "period_index"
    ]
)

period_type = str(
    live_race[
        "period_type"
    ]
)

period_type_normalized = (
    period_type
    .strip()
    .lower()
)

period_position = (
    period_index % 7
)

week_bucket = (
    period_index // 7
)

is_colosseum = (
    period_type_normalized
    == "colosseum"
)


# Determine current Battle Day

# periodIndex % 7:
#
# 0, 1, 2 -> Training Days
# 3       -> Battle Day 1
# 4       -> Battle Day 2
# 5       -> Battle Day 3
# 6       -> Battle Day 4

if period_position <= 2:

    current_battle_day = 0

else:

    current_battle_day = (
        period_position - 2
    )


print(
    "\nGALACTIC KINGS LIVE WAR CONTEXT"
)

print(
    "Live race ID:",
    live_race_id,
)

print(
    "Section index:",
    section_index,
)

print(
    "Period type:",
    period_type,
)

print(
    "Period index:",
    period_index,
)

print(
    "Current battle day:",
    current_battle_day,
)

print(
    "Mode:",
    (
        "COLOSSEUM"
        if is_colosseum
        else "RIVER RACE"
    ),
)


# Current GK roster count

cursor.execute(
    """
    SELECT
        COUNT(*) AS current_members
    FROM live_clan_members
    WHERE
        live_race_id = %s
        AND clan_tag = %s
        AND currently_member = TRUE
    """,
    (
        live_race_id,
        GALACTIC_KINGS_TAG,
    )
)

roster_row = cursor.fetchone()

current_members = int(
    (
        roster_row
        or {}
    ).get(
        "current_members",
        0,
    )
    or 0
)


# Load current-week GK period logs

cursor.execute(
    """
    SELECT
        clan_tag,
        period_index,
        points_earned,
        progress_start_of_day,
        progress_end_of_day,
        end_of_day_rank,
        progress_earned,
        progress_earned_from_defenses
    FROM clan_war_period_results
    WHERE
        live_race_id = %s
        AND clan_tag = %s
        AND period_index DIV 7 = %s
        AND MOD(period_index, 7) BETWEEN 3 AND 6
    ORDER BY
        period_index
    """,
    (
        live_race_id,
        GALACTIC_KINGS_TAG,
        week_bucket,
    )
)

period_rows = cursor.fetchall()

period_logs = pd.DataFrame(
    period_rows
)

expected_period_columns = [
    "clan_tag",
    "period_index",
    "points_earned",
    "progress_start_of_day",
    "progress_end_of_day",
    "end_of_day_rank",
    "progress_earned",
    "progress_earned_from_defenses",
]


if period_logs.empty:

    period_logs = pd.DataFrame(
        columns=expected_period_columns
    )

else:

    for column in expected_period_columns:

        if (
            column != "clan_tag"
            and column in period_logs.columns
        ):

            period_logs[column] = pd.to_numeric(
                period_logs[column],
                errors="coerce",
            )


# Player-based GK daily production forecast

# This is still the current player-model estimate of GK's Medal
# production. It is not treated as River movement or final place.
#
# We only use it to estimate how many Medals GK can produce on a
# Battle Day.

predictions[
    "predicted_medals_per_battle_day"
] = (
    predictions[
        "capacity_adjusted_fame"
    ]
    / BATTLE_DAYS_PER_WEEK
)


decks_today = (
    predictions[
        "decks_used_today"
    ]
    .fillna(0)
    .clip(
        0,
        MAX_DAILY_DECKS,
    )
)


if current_battle_day == 0:

    predictions[
        "current_day_remaining_fraction"
    ] = 0.0

else:

    predictions[
        "current_day_remaining_fraction"
    ] = (
        1.0
        - (
            decks_today
            / MAX_DAILY_DECKS
        )
    )


predictions[
    "predicted_current_day_remaining_medals"
] = (
    predictions[
        "predicted_medals_per_battle_day"
    ]
    * predictions[
        "current_day_remaining_fraction"
    ]
)


predicted_full_day_medals = float(
    predictions[
        "predicted_medals_per_battle_day"
    ].sum()
)

predicted_current_day_remaining_medals = float(
    predictions[
        "predicted_current_day_remaining_medals"
    ].sum()
)

expected_daily_participants = float(
    predictions[
        "daily_participation_weight"
    ].sum()
)

forecast_pool_players = int(
    predictions[
        "player_tag"
    ].nunique()
)


# Historical GK River Race baselines

normal_daily_history = pd.DataFrame()


if not gk_daily_history.empty:

    if "is_colosseum" in gk_daily_history.columns:

        colosseum_flag = (
            gk_daily_history[
                "is_colosseum"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
        )

        normal_daily_history = (
            gk_daily_history[
                ~colosseum_flag
            ]
            .copy()
        )

    else:

        normal_daily_history = (
            gk_daily_history.copy()
        )


# Exclude the current race from "history" so the forecast never uses
# the same race's future/target rows as historical evidence.

if (
    not normal_daily_history.empty
    and "live_race_id"
    in normal_daily_history.columns
):

    normal_daily_history = normal_daily_history[
        normal_daily_history[
            "live_race_id"
        ].astype(str)
        != str(
            live_race_id
        )
    ].copy()


def get_gk_historical_baseline(
    battle_day,
):

    baseline = {
        "history_rows": 0,
        "historical_avg_medals": np.nan,
        "historical_expected_place": np.nan,
        "historical_rank_movement": np.nan,
        "historical_defense_movement": np.nan,
    }

    if normal_daily_history.empty:

        return baseline


    same_day = normal_daily_history[
        normal_daily_history[
            "battle_day"
        ] == battle_day
    ].copy()


    # Prefer the same Battle Day.
    # If there are no historical rows for that day, fall back to all
    # completed GK River Race Battle Days.

    source = (
        same_day
        if not same_day.empty
        else normal_daily_history
    )


    baseline[
        "history_rows"
    ] = len(
        source
    )


    if (
        "points_earned"
        in source.columns
    ):

        baseline[
            "historical_avg_medals"
        ] = pd.to_numeric(
            source[
                "points_earned"
            ],
            errors="coerce",
        ).mean()


    if (
        "daily_place"
        in source.columns
    ):

        baseline[
            "historical_expected_place"
        ] = historical_mode(
            source[
                "daily_place"
            ]
        )


    if (
        "rank_movement"
        in source.columns
    ):

        baseline[
            "historical_rank_movement"
        ] = historical_mode(
            source[
                "rank_movement"
            ]
        )


    if (
        "defense_movement"
        in source.columns
    ):

        baseline[
            "historical_defense_movement"
        ] = pd.to_numeric(
            source[
                "defense_movement"
            ],
            errors="coerce",
        ).median()


    return baseline


# Look up current-week day row

def get_period_row(
    battle_day,
):

    day_period_index = (
        week_bucket * 7
        + 2
        + battle_day
    )

    rows = period_logs[
        period_logs[
            "period_index"
        ] == day_period_index
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


# Build GK day-by-day forecast

daily_rows = []

current_river_progress = 0.0
projected_finish_day = None
projected_total_medals = 0.0

movement_projection_available = True


for battle_day in range(
    1,
    BATTLE_DAYS_PER_WEEK + 1,
):

    period_row = get_period_row(
        battle_day
    )

    baseline = get_gk_historical_baseline(
        battle_day
    )


    actual_medals = 0.0
    actual_rank_zero_based = np.nan
    actual_progress_start = np.nan
    actual_progress_end = np.nan
    actual_rank_movement = np.nan
    actual_defense_movement = np.nan


    if period_row is not None:

        actual_medals = safe_float(
            period_row[
                "points_earned"
            ]
        )

        actual_rank_zero_based = (
            period_row[
                "end_of_day_rank"
            ]
        )

        actual_progress_start = (
            period_row[
                "progress_start_of_day"
            ]
        )

        actual_progress_end = (
            period_row[
                "progress_end_of_day"
            ]
        )

        actual_rank_movement = (
            period_row[
                "progress_earned"
            ]
        )

        actual_defense_movement = (
            period_row[
                "progress_earned_from_defenses"
            ]
        )


    is_completed = (
        period_row is not None
        and pd.notna(
            actual_rank_zero_based
        )
        and int(
            actual_rank_zero_based
        ) != -1
    )


    is_current_pending_day = (
        current_battle_day > 0
        and battle_day
        == current_battle_day
        and not is_completed
    )


    is_future_day = (
        current_battle_day == 0
        or battle_day
        > current_battle_day
    )


# Skip later days after GK finishes

    if (
        projected_finish_day is not None
        and projected_finish_day < battle_day
        and not is_colosseum
    ):

        daily_rows.append(
            {
                "clan_tag":
                    GALACTIC_KINGS_TAG,

                "clan_name":
                    predictions[
                        "clan_name"
                    ].iloc[0],

                "battle_day":
                    battle_day,

                "day_status":
                    "finished_previous_day",

                "actual_medals":
                    actual_medals,

                "projected_medals":
                    0.0,

                "historical_avg_medals":
                    baseline[
                        "historical_avg_medals"
                    ],

                "historical_expected_place":
                    baseline[
                        "historical_expected_place"
                    ],

                "projected_place":
                    np.nan,

                "projected_rank_movement":
                    0.0,

                "projected_defense_movement":
                    0.0,

                "projected_total_movement":
                    0.0,

                "projected_progress_end":
                    current_river_progress,

                "finished_after_day":
                    False,

                "history_rows_used":
                    baseline[
                        "history_rows"
                    ],
            }
        )

        continue


# Medals

    if is_completed:

        projected_medals = (
            actual_medals
        )

        day_status = (
            "completed"
        )

    elif is_current_pending_day:

        projected_medals = (
            actual_medals
            + predicted_current_day_remaining_medals
        )

        day_status = (
            "in_progress"
        )

    elif is_future_day:

        projected_medals = (
            predicted_full_day_medals
        )

        day_status = (
            "future"
        )

    else:

        projected_medals = (
            actual_medals
        )

        day_status = (
            "missing_actual"
        )


    projected_total_medals += max(
        0.0,
        projected_medals,
    )


# Colosseum

    if is_colosseum:

        projected_place = np.nan
        projected_rank_movement = np.nan
        projected_defense_movement = np.nan
        projected_total_movement = np.nan
        projected_progress_end = np.nan
        finished_after_day = False


# Normal River Race: completed day

    elif is_completed:

        projected_place = (
            int(
                actual_rank_zero_based
            )
            + 1
        )

        projected_rank_movement = safe_float(
            actual_rank_movement
        )

        projected_defense_movement = safe_float(
            actual_defense_movement
        )

        projected_total_movement = (
            projected_rank_movement
            + projected_defense_movement
        )


        if pd.notna(
            actual_progress_end
        ):

            current_river_progress = min(
                RIVER_LENGTH,
                safe_float(
                    actual_progress_end
                ),
            )

        else:

            current_river_progress = min(
                RIVER_LENGTH,
                current_river_progress
                + projected_total_movement,
            )


        projected_progress_end = (
            current_river_progress
        )

        finished_after_day = (
            current_river_progress
            >= RIVER_LENGTH
        )


        if (
            finished_after_day
            and projected_finish_day is None
        ):

            projected_finish_day = (
                battle_day
            )


# Normal River Race: future or in-progress day

    else:

        projected_place = baseline[
            "historical_expected_place"
        ]

        projected_rank_movement = baseline[
            "historical_rank_movement"
        ]

        projected_defense_movement = baseline[
            "historical_defense_movement"
        ]


        if pd.isna(
            projected_rank_movement
        ):

            movement_projection_available = False

            projected_total_movement = np.nan
            projected_progress_end = np.nan
            finished_after_day = False

        else:

            projected_rank_movement = max(
                0.0,
                safe_float(
                    projected_rank_movement
                ),
            )

            projected_defense_movement = max(
                0.0,
                safe_float(
                    projected_defense_movement,
                    0.0,
                ),
            )

            projected_total_movement = (
                projected_rank_movement
                + projected_defense_movement
            )

            current_river_progress = min(
                RIVER_LENGTH,
                current_river_progress
                + projected_total_movement,
            )

            projected_progress_end = (
                current_river_progress
            )

            finished_after_day = (
                current_river_progress
                >= RIVER_LENGTH
            )


            if (
                finished_after_day
                and projected_finish_day is None
            ):

                projected_finish_day = (
                    battle_day
                )


    daily_rows.append(
        {
            "clan_tag":
                GALACTIC_KINGS_TAG,

            "clan_name":
                predictions[
                    "clan_name"
                ].iloc[0],

            "battle_day":
                battle_day,

            "day_status":
                day_status,

            "actual_medals":
                actual_medals,

            "projected_medals":
                max(
                    0.0,
                    projected_medals,
                ),

            "historical_avg_medals":
                baseline[
                    "historical_avg_medals"
                ],

            "historical_expected_place":
                baseline[
                    "historical_expected_place"
                ],

            "projected_place":
                projected_place,

            "projected_rank_movement":
                projected_rank_movement,

            "projected_defense_movement":
                projected_defense_movement,

            "projected_total_movement":
                projected_total_movement,

            "projected_progress_end":
                projected_progress_end,

            "finished_after_day":
                finished_after_day,

            "history_rows_used":
                baseline[
                    "history_rows"
                ],
        }
    )


daily_projection = pd.DataFrame(
    daily_rows
)


# Historical GK race context

historical_finished_by_day_3_rate = np.nan
historical_avg_finish_day = np.nan
historical_races_available = 0


if not gk_race_history.empty:

    race_history = (
        gk_race_history.copy()
    )


    if "is_colosseum" in race_history.columns:

        colosseum_flag = (
            race_history[
                "is_colosseum"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
        )

        race_history = race_history[
            ~colosseum_flag
        ].copy()


    if (
        "live_race_id"
        in race_history.columns
    ):

        race_history = race_history[
            race_history[
                "live_race_id"
            ].astype(str)
            != str(
                live_race_id
            )
        ].copy()


    historical_races_available = len(
        race_history
    )


    if (
        not race_history.empty
        and "finished_by_day_3"
        in race_history.columns
    ):

        day3_flag = (
            race_history[
                "finished_by_day_3"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
        )

        historical_finished_by_day_3_rate = float(
            day3_flag.mean()
        )


    if (
        not race_history.empty
        and "race_finish_day"
        in race_history.columns
    ):

        finish_days = pd.to_numeric(
            race_history[
                "race_finish_day"
            ],
            errors="coerce",
        ).dropna()

        if not finish_days.empty:

            historical_avg_finish_day = float(
                finish_days.mean()
            )


# Final GK war summary

if is_colosseum:

    current_or_projected_progress = np.nan

    finish_status = (
        "colosseum_no_river_finish"
    )

else:

    if projected_finish_day is not None:

        finish_status = (
            f"projected_finish_day_{projected_finish_day}"
        )

    elif movement_projection_available:

        finish_status = (
            "not_projected_to_finish"
        )

    else:

        finish_status = (
            "insufficient_gk_movement_history"
        )


    if daily_projection[
        "projected_progress_end"
    ].notna().any():

        current_or_projected_progress = float(
            daily_projection[
                "projected_progress_end"
            ]
            .dropna()
            .iloc[-1]
        )

    else:

        current_or_projected_progress = np.nan


war_projection = pd.DataFrame(
    [
        {
            "clan_tag":
                GALACTIC_KINGS_TAG,

            "clan_name":
                predictions[
                    "clan_name"
                ].iloc[0],

            "war_mode":
                (
                    "colosseum"
                    if is_colosseum
                    else "river_race"
                ),

            "current_battle_day":
                current_battle_day,

            "current_members":
                current_members,

            "forecast_pool_players":
                forecast_pool_players,

            "expected_daily_participants":
                expected_daily_participants,

            "player_model_full_day_medals":
                predicted_full_day_medals,

            "projected_total_medals":
                projected_total_medals,

            "projected_finish_day":
                projected_finish_day,

            "projected_river_progress":
                current_or_projected_progress,

            "finish_status":
                finish_status,

            "historical_races_available":
                historical_races_available,

            "historical_day3_finish_rate":
                historical_finished_by_day_3_rate,

            "historical_avg_finish_day":
                historical_avg_finish_day,

            "movement_projection_available":
                movement_projection_available,
        }
    ]
)


# Round display values

for column in [
    "actual_medals",
    "projected_medals",
    "historical_avg_medals",
    "historical_expected_place",
    "projected_place",
    "projected_rank_movement",
    "projected_defense_movement",
    "projected_total_movement",
    "projected_progress_end",
]:

    if column in daily_projection.columns:

        daily_projection[column] = pd.to_numeric(
            daily_projection[column],
            errors="coerce",
        ).round(2)


for column in [
    "expected_daily_participants",
    "player_model_full_day_medals",
    "projected_total_medals",
    "projected_river_progress",
    "historical_day3_finish_rate",
    "historical_avg_finish_day",
]:

    if column in war_projection.columns:

        war_projection[column] = pd.to_numeric(
            war_projection[column],
            errors="coerce",
        ).round(4)


# Output

print(
    "\nGALACTIC KINGS DAY-BY-DAY PROJECTION"
)

daily_display_columns = [
    "battle_day",
    "day_status",
    "projected_medals",
    "historical_avg_medals",
]


if not is_colosseum:

    daily_display_columns += [
        "projected_place",
        "projected_rank_movement",
        "projected_defense_movement",
        "projected_progress_end",
        "finished_after_day",
        "history_rows_used",
    ]


print(
    daily_projection[
        daily_display_columns
    ]
    .to_string(
        index=False
    )
)


print(
    "\nGALACTIC KINGS LIVE WAR SUMMARY"
)

summary_columns = [
    "war_mode",
    "current_battle_day",
    "current_members",
    "forecast_pool_players",
    "expected_daily_participants",
    "player_model_full_day_medals",
    "projected_total_medals",
]


if not is_colosseum:

    summary_columns += [
        "projected_finish_day",
        "projected_river_progress",
        "finish_status",
        "historical_races_available",
        "historical_day3_finish_rate",
        "historical_avg_finish_day",
    ]


print(
    war_projection[
        summary_columns
    ]
    .to_string(
        index=False
    )
)


if (
    not is_colosseum
    and not movement_projection_available
):

    print(
        "\nNote: GK does not yet have enough completed daily River Race "
        "history to project future rank movement reliably. Medal production "
        "is still projected, but finish day is intentionally left unknown "
        "instead of modeling opponent clans."
    )


# Save

predictions.to_csv(
    PLAYER_OUTPUT_FILE,
    index=False,
)

daily_projection.to_csv(
    DAY_OUTPUT_FILE,
    index=False,
)

war_projection.to_csv(
    WAR_OUTPUT_FILE,
    index=False,
)


print(
    "\nSaved:"
)

print(
    PLAYER_OUTPUT_FILE
)

print(
    DAY_OUTPUT_FILE
)

print(
    WAR_OUTPUT_FILE
)


# Close database

cursor.close()
db.close()
