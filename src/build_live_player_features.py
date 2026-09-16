import os

import pandas as pd

from sqlalchemy import create_engine
from dotenv import load_dotenv


# Configuration

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

GALACTIC_KINGS_TAG = "#Y9202P9U"

ARCHETYPE_FILE = "data/gk_player_archetypes.csv"
OUTPUT_FILE = "data/gk_live_player_features.csv"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# Load historical player-war data

history_query = """
SELECT
    pwp.player_tag,
    p.player_name,
    pwp.war_id,
    w.race_date,
    SUM(pwp.fame) AS fame,
    SUM(pwp.decks_used) AS decks_used
FROM player_war_performance pwp
JOIN players p
    ON pwp.player_tag = p.player_tag
JOIN wars w
    ON pwp.war_id = w.war_id
GROUP BY
    pwp.player_tag,
    p.player_name,
    pwp.war_id,
    w.race_date
ORDER BY
    pwp.player_tag,
    w.race_date
"""

history = pd.read_sql(
    history_query,
    engine,
)

print(
    "Historical player-war rows:",
    len(history),
)


# Basic historical features

history["fame"] = pd.to_numeric(
    history["fame"],
    errors="coerce",
).fillna(0)

history["decks_used"] = pd.to_numeric(
    history["decks_used"],
    errors="coerce",
).fillna(0)

history["fame_per_deck"] = 0.0

active_mask = (
    history["decks_used"] > 0
)

history.loc[
    active_mask,
    "fame_per_deck",
] = (
    history.loc[
        active_mask,
        "fame",
    ]
    / history.loc[
        active_mask,
        "decks_used",
    ]
)

history["participated"] = (
    history["decks_used"] > 0
).astype(int)

history["full_participation"] = (
    history["decks_used"] >= 16
).astype(int)


# Build one feature row per player

feature_rows = []

for player_tag, player_history in history.groupby(
    "player_tag"
):

    player_history = (
        player_history
        .sort_values("race_date")
        .reset_index(drop=True)
    )

    previous_wars_count = len(
        player_history
    )

    if previous_wars_count == 0:
        continue

    fame_values = (
        player_history["fame"]
    )

    deck_values = (
        player_history["decks_used"]
    )

    efficiency_values = (
        player_history["fame_per_deck"]
    )

    participation_values = (
        player_history["participated"]
    )

    full_participation_values = (
        player_history["full_participation"]
    )

    previous_war_fame = (
        fame_values.iloc[-1]
    )

    previous_war_decks = (
        deck_values.iloc[-1]
    )

    fame_trend = 0.0

    if previous_wars_count >= 2:

        fame_trend = (
            fame_values.iloc[-1]
            - fame_values.iloc[-2]
        )

    feature_rows.append(
        {
            "player_tag":
                player_tag,

            "previous_wars_count":
                previous_wars_count,

            "last_3_avg_fame":
                fame_values.tail(3).mean(),

            "last_3_avg_decks":
                deck_values.tail(3).mean(),

            "last_2_avg_decks":
                deck_values.tail(2).mean(),

            "last_3_avg_efficiency":
                efficiency_values.tail(3).mean(),

            "last_5_participation_rate":
                participation_values.tail(5).mean(),

            "last_5_full_participation_rate":
                full_participation_values.tail(5).mean(),

            "previous_war_fame":
                previous_war_fame,

            "previous_war_decks":
                previous_war_decks,

            "fame_trend":
                fame_trend,

            "last_3_fame_std":
                fame_values.tail(3).std(
                    ddof=1
                ),

            "last_3_decks_std":
                deck_values.tail(3).std(
                    ddof=1
                ),
        }
    )


player_features = pd.DataFrame(
    feature_rows
)

print(
    "Players with historical features:",
    len(player_features),
)


# Load GK player archetypes

if os.path.exists(
    ARCHETYPE_FILE
):

    archetypes = pd.read_csv(
        ARCHETYPE_FILE
    )

    archetype_columns = [
        column
        for column in [
            "player_tag",
            "cluster",
            "archetype_name",
        ]
        if column in archetypes.columns
    ]

    archetypes = (
        archetypes[
            archetype_columns
        ]
        .drop_duplicates(
            subset=[
                "player_tag"
            ]
        )
    )

else:

    raise FileNotFoundError(
        f"Missing {ARCHETYPE_FILE}. "
        "Run build_player_archetypes.py first."
    )


print(
    "GK players with archetypes:",
    len(archetypes),
)


# Get latest live race

latest_race_query = """
SELECT
    live_race_id
FROM live_races
ORDER BY
    last_seen_at DESC
LIMIT 1
"""

latest_race_df = pd.read_sql(
    latest_race_query,
    engine,
)

if latest_race_df.empty:
    raise RuntimeError(
        "No live race found."
    )

latest_live_race_id = (
    latest_race_df.iloc[0][
        "live_race_id"
    ]
)


# Load GK live players only

live_query = """
SELECT
    lpws.live_race_id,
    lpws.player_tag,
    lpws.player_name,
    lpws.clan_tag,
    c.clan_name,

    lpws.fame AS current_fame,
    lpws.decks_used AS current_decks_used,
    lpws.decks_used_today,

    CASE
        WHEN lcm.currently_member = TRUE
        THEN 1
        ELSE 0
    END AS currently_member

FROM live_player_war_status lpws

JOIN clans c
    ON lpws.clan_tag = c.clan_tag

LEFT JOIN live_clan_members lcm
    ON lpws.live_race_id = lcm.live_race_id
    AND lpws.clan_tag = lcm.clan_tag
    AND lpws.player_tag = lcm.player_tag

WHERE
    lpws.live_race_id = %(live_race_id)s
    AND lpws.clan_tag = %(gk_tag)s
"""

live_players = pd.read_sql(
    live_query,
    engine,
    params={
        "live_race_id":
            latest_live_race_id,

        "gk_tag":
            GALACTIC_KINGS_TAG,
    },
)


if live_players.empty:

    raise RuntimeError(
        "No Galactic Kings players found "
        "for the latest live race."
    )


live_players = (
    live_players
    .drop_duplicates(
        subset=[
            "player_tag"
        ]
    )
    .copy()
)


print(
    "\nGALACTIC KINGS LIVE POOL"
)

print(
    "Live race ID:",
    latest_live_race_id,
)

print(
    "Rotation-pool players:",
    len(live_players),
)

print(
    "Currently on roster:",
    int(
        live_players[
            "currently_member"
        ].sum()
    ),
)

print(
    "Currently outside clan:",
    int(
        len(live_players)
        - live_players[
            "currently_member"
        ].sum()
    ),
)


# Load GK historical clan context

clan_query = """
SELECT
    cwr.clan_tag,
    w.race_date,
    cwr.clan_score
FROM clan_war_results cwr
JOIN wars w
    ON cwr.war_id = w.war_id
WHERE
    cwr.clan_tag = %(gk_tag)s
ORDER BY
    w.race_date
"""

clan_history = pd.read_sql(
    clan_query,
    engine,
    params={
        "gk_tag":
            GALACTIC_KINGS_TAG
    },
)


previous_clan_score = float(
    "nan"
)

last_3_avg_clan_score = float(
    "nan"
)


if not clan_history.empty:

    clan_history[
        "clan_score"
    ] = pd.to_numeric(
        clan_history[
            "clan_score"
        ],
        errors="coerce",
    )

    scores = (
        clan_history[
            "clan_score"
        ]
        .dropna()
    )

    if not scores.empty:

        previous_clan_score = (
            scores.iloc[-1]
        )

        last_3_avg_clan_score = (
            scores.tail(3).mean()
        )


# Merge GK live players with history

live_features = live_players.merge(
    player_features,
    on="player_tag",
    how="left",
)


# Add GK clan context

live_features[
    "previous_clan_score"
] = previous_clan_score

live_features[
    "last_3_avg_clan_score"
] = (
    last_3_avg_clan_score
)


# Merge GK archetypes

live_features = live_features.merge(
    archetypes,
    on="player_tag",
    how="left",
)

live_features[
    "archetype_name"
] = (
    live_features[
        "archetype_name"
    ]
    .fillna(
        "Not Enough History"
    )
)


# Prediction tiers

feature_columns = [
    "last_3_avg_fame",
    "last_3_avg_decks",
    "last_2_avg_decks",
    "last_3_avg_efficiency",
    "last_5_participation_rate",
    "last_5_full_participation_rate",
    "previous_war_fame",
    "previous_war_decks",
    "fame_trend",
    "last_3_fame_std",
    "last_3_decks_std",
    "previous_clan_score",
    "last_3_avg_clan_score",
]


live_features[
    "model_eligible"
] = (
    live_features[
        feature_columns
    ]
    .notna()
    .all(
        axis=1
    )
    &
    (
        live_features[
            "previous_wars_count"
        ] >= 3
    )
)


live_features[
    "prediction_tier"
] = "clan_baseline"


limited_history_mask = (
    live_features[
        "previous_wars_count"
    ]
    .between(
        1,
        2,
        inclusive="both",
    )
)


live_features.loc[
    limited_history_mask,
    "prediction_tier",
] = "limited_history"


live_features.loc[
    live_features[
        "model_eligible"
    ],
    "prediction_tier",
] = "ml_model"


# GK rotation flag

live_features[
    "rotation_eligible"
] = 1


# Summary

eligible_count = int(
    live_features[
        "model_eligible"
    ].sum()
)


print(
    "\nGALACTIC KINGS LIVE FEATURE SUMMARY"
)

print(
    "Total GK players:",
    len(live_features),
)

print(
    "Model-eligible players:",
    eligible_count,
)

print(
    "Not full-model eligible:",
    len(live_features)
    - eligible_count,
)


print(
    "\nPrediction tiers:"
)

print(
    live_features[
        "prediction_tier"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nGK archetypes:"
)

print(
    live_features[
        "archetype_name"
    ]
    .value_counts()
    .to_string()
)


# Save

live_features.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\nSaved Galactic Kings live features to:"
)

print(
    OUTPUT_FILE
)
