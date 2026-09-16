import os

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from dotenv import load_dotenv

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


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

HISTORY_FILE = "data/model_dataset.csv"

OUTPUT_FILE = "data/gk_player_archetypes.csv"
SUMMARY_FILE = "data/gk_player_archetype_summary.csv"

RANDOM_STATE = 42
MIN_WARS_FOR_CLUSTERING = 3

CLUSTER_OPTIONS = [
    3,
    4,
    5,
]

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ================================================================
# LOAD HISTORICAL MODEL DATA
# ================================================================

history = pd.read_csv(
    HISTORY_FILE
)

print(
    "Historical player-war rows loaded:",
    len(history),
)


# ================================================================
# GET LATEST LIVE RACE
# ================================================================

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
        "No live race found in live_races."
    )

latest_live_race_id = (
    latest_race_df.iloc[0][
        "live_race_id"
    ]
)

print(
    "Latest live race:",
    latest_live_race_id,
)


# ================================================================
# LOAD CURRENT GK ROTATION POOL DIRECTLY FROM MYSQL
# ================================================================

# Anyone who has appeared for GK in the current live race is part of
# the current rotation pool, whether currently in the clan or rotated
# out temporarily.

gk_pool_query = """
SELECT
    lpws.player_tag,
    lpws.player_name,

    CASE
        WHEN lcm.currently_member = TRUE
        THEN 1
        ELSE 0
    END AS currently_member

FROM live_player_war_status lpws

LEFT JOIN live_clan_members lcm
    ON lpws.live_race_id = lcm.live_race_id
    AND lpws.clan_tag = lcm.clan_tag
    AND lpws.player_tag = lcm.player_tag

WHERE
    lpws.live_race_id = %(live_race_id)s
    AND lpws.clan_tag = %(gk_tag)s
"""

gk_pool = pd.read_sql(
    gk_pool_query,
    engine,
    params={
        "live_race_id":
            latest_live_race_id,

        "gk_tag":
            GALACTIC_KINGS_TAG,
    },
)


if gk_pool.empty:
    raise RuntimeError(
        "No Galactic Kings players found in the latest live race."
    )


gk_pool = (
    gk_pool
    .drop_duplicates(
        subset=[
            "player_tag"
        ]
    )
    .copy()
)


gk_player_tags = (
    gk_pool[
        "player_tag"
    ]
    .dropna()
    .astype(str)
    .unique()
)


print(
    "GK rotation-pool players:",
    len(gk_player_tags),
)

print(
    "Currently in clan:",
    int(
        gk_pool[
            "currently_member"
        ].fillna(0).sum()
    ),
)

print(
    "Currently outside clan:",
    int(
        len(gk_pool)
        - gk_pool[
            "currently_member"
        ].fillna(0).sum()
    ),
)


# ================================================================
# FILTER HISTORY TO CURRENT GK ROTATION POOL
# ================================================================

# We study current GK players only, while preserving each player's full
# available historical performance history.

gk_history = history[
    history[
        "player_tag"
    ]
    .astype(str)
    .isin(
        gk_player_tags
    )
].copy()


print(
    "Historical rows belonging to current GK pool:",
    len(gk_history),
)


if gk_history.empty:
    raise RuntimeError(
        "No historical model rows found for current GK players."
    )


# ================================================================
# REQUIRED COLUMNS
# ================================================================

required_columns = [
    "player_tag",
    "player_name",
    "war_id",
    "fame",
    "decks_used",
    "fame_per_deck",
]


missing_columns = [
    column
    for column in required_columns
    if column not in gk_history.columns
]


if missing_columns:
    raise ValueError(
        "model_dataset.csv is missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


# ================================================================
# BUILD ONE LONG-TERM PROFILE PER GK PLAYER
# ================================================================

player_profiles = (
    gk_history
    .groupby(
        [
            "player_tag",
            "player_name",
        ]
    )
    .agg(
        wars_observed=(
            "war_id",
            "nunique",
        ),

        avg_fame=(
            "fame",
            "mean",
        ),

        avg_decks=(
            "decks_used",
            "mean",
        ),

        fame_std=(
            "fame",
            "std",
        ),

        decks_std=(
            "decks_used",
            "std",
        ),

        participation_rate=(
            "decks_used",
            lambda x:
            (
                x > 0
            ).mean(),
        ),

        full_participation_rate=(
            "decks_used",
            lambda x:
            (
                x >= 16
            ).mean(),
        ),

        avg_efficiency=(
            "fame_per_deck",
            "mean",
        ),
    )
    .reset_index()
)


# ================================================================
# CLEAN PLAYER PROFILES
# ================================================================

player_profiles[
    "fame_std"
] = (
    player_profiles[
        "fame_std"
    ]
    .fillna(0)
)


player_profiles[
    "decks_std"
] = (
    player_profiles[
        "decks_std"
    ]
    .fillna(0)
)


player_profiles[
    "avg_efficiency"
] = (
    player_profiles[
        "avg_efficiency"
    ]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
)


# ================================================================
# ELIGIBLE PLAYERS
# ================================================================

cluster_data = player_profiles[
    player_profiles[
        "wars_observed"
    ] >= MIN_WARS_FOR_CLUSTERING
].copy()


if cluster_data.empty:
    raise RuntimeError(
        "No GK players have enough historical wars for clustering."
    )


cluster_data[
    "avg_efficiency"
] = (
    cluster_data[
        "avg_efficiency"
    ]
    .fillna(
        cluster_data[
            "avg_efficiency"
        ].median()
    )
)


print(
    "GK players eligible for clustering:",
    len(cluster_data),
)


# ================================================================
# CLUSTER FEATURES
# ================================================================

CLUSTER_FEATURES = [
    "avg_fame",
    "avg_decks",
    "participation_rate",
    "full_participation_rate",
    "avg_efficiency",
    "fame_std",
    "decks_std",
]


X = cluster_data[
    CLUSTER_FEATURES
].copy()


# ================================================================
# SCALE FEATURES
# ================================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ================================================================
# VALID CLUSTER COUNTS
# ================================================================

valid_cluster_options = [
    k
    for k in CLUSTER_OPTIONS
    if (
        k >= 2
        and k < len(
            cluster_data
        )
    )
]


if not valid_cluster_options:
    raise RuntimeError(
        "Not enough GK players to test configured cluster counts."
    )


# ================================================================
# TEST DIFFERENT NUMBERS OF CLUSTERS
# ================================================================

scores = {}


print(
    "\nGK SILHOUETTE SCORES"
)


for k in valid_cluster_options:

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    labels = model.fit_predict(
        X_scaled
    )

    if len(
        np.unique(
            labels
        )
    ) < 2:

        continue


    score = silhouette_score(
        X_scaled,
        labels,
    )


    scores[
        k
    ] = score


    print(
        f"k={k}: {score:.4f}"
    )


if not scores:
    raise RuntimeError(
        "Unable to calculate a valid silhouette score."
    )


# ================================================================
# SELECT BEST K
# ================================================================

best_k = max(
    scores,
    key=scores.get,
)


print(
    "\nSelected number of GK archetype clusters:",
    best_k,
)


# ================================================================
# FINAL K-MEANS MODEL
# ================================================================

final_model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=50,
)


cluster_data[
    "cluster"
] = final_model.fit_predict(
    X_scaled
)


# ================================================================
# RAW CLUSTER SUMMARY
# ================================================================

cluster_summary = (
    cluster_data
    .groupby(
        "cluster"
    )
    .agg(
        players=(
            "player_tag",
            "nunique",
        ),

        avg_wars_observed=(
            "wars_observed",
            "mean",
        ),

        avg_fame=(
            "avg_fame",
            "mean",
        ),

        avg_decks=(
            "avg_decks",
            "mean",
        ),

        participation_rate=(
            "participation_rate",
            "mean",
        ),

        full_participation_rate=(
            "full_participation_rate",
            "mean",
        ),

        avg_efficiency=(
            "avg_efficiency",
            "mean",
        ),

        fame_std=(
            "fame_std",
            "mean",
        ),

        decks_std=(
            "decks_std",
            "mean",
        ),
    )
    .reset_index()
)


# ================================================================
# BEHAVIOR SCORE
# ================================================================

summary_for_score = (
    cluster_summary.copy()
)


score_columns_positive = [
    "avg_fame",
    "avg_decks",
    "participation_rate",
    "full_participation_rate",
    "avg_efficiency",
]


score_columns_negative = [
    "fame_std",
    "decks_std",
]


for column in (
    score_columns_positive
    + score_columns_negative
):

    std = (
        summary_for_score[
            column
        ].std(
            ddof=0
        )
    )


    if (
        pd.isna(
            std
        )
        or std == 0
    ):

        summary_for_score[
            f"{column}_z"
        ] = 0.0

    else:

        summary_for_score[
            f"{column}_z"
        ] = (
            summary_for_score[
                column
            ]
            - summary_for_score[
                column
            ].mean()
        ) / std


summary_for_score[
    "behavior_score"
] = 0.0


for column in score_columns_positive:

    summary_for_score[
        "behavior_score"
    ] += (
        summary_for_score[
            f"{column}_z"
        ]
    )


for column in score_columns_negative:

    summary_for_score[
        "behavior_score"
    ] -= (
        0.35
        * summary_for_score[
            f"{column}_z"
        ]
    )


cluster_summary = cluster_summary.merge(
    summary_for_score[
        [
            "cluster",
            "behavior_score",
        ]
    ],
    on="cluster",
    how="left",
)


# ================================================================
# DYNAMIC ARCHETYPE NAMES
# ================================================================

ordered_clusters = (
    cluster_summary
    .sort_values(
        "behavior_score",
        ascending=False,
    )[
        "cluster"
    ]
    .tolist()
)


if best_k == 3:

    ordered_names = [
        "Reliable High Performer",
        "Inconsistent Contributor",
        "Low-Participation Player",
    ]


elif best_k == 4:

    ordered_names = [
        "Reliable High Performer",
        "Strong Contributor",
        "Inconsistent Contributor",
        "Low-Participation Player",
    ]


else:

    ordered_names = [
        "Elite Reliable Performer",
        "Reliable High Performer",
        "Steady Contributor",
        "Inconsistent Contributor",
        "Low-Participation Player",
    ]


archetype_names = {
    cluster_id: archetype_name
    for cluster_id, archetype_name
    in zip(
        ordered_clusters,
        ordered_names,
    )
}


cluster_data[
    "archetype_name"
] = (
    cluster_data[
        "cluster"
    ]
    .map(
        archetype_names
    )
)


cluster_summary[
    "archetype_name"
] = (
    cluster_summary[
        "cluster"
    ]
    .map(
        archetype_names
    )
)


# ================================================================
# ADD CURRENT GK MEMBERSHIP
# ================================================================

membership = (
    gk_pool[
        [
            "player_tag",
            "currently_member",
        ]
    ]
    .drop_duplicates(
        subset=[
            "player_tag"
        ]
    )
)


cluster_data = cluster_data.merge(
    membership,
    on="player_tag",
    how="left",
)


cluster_data[
    "rotation_eligible"
] = 1


# ================================================================
# ORDER SUMMARY
# ================================================================

cluster_summary = cluster_summary[
    [
        "cluster",
        "archetype_name",
        "players",
        "avg_wars_observed",
        "avg_fame",
        "avg_decks",
        "participation_rate",
        "full_participation_rate",
        "avg_efficiency",
        "fame_std",
        "decks_std",
        "behavior_score",
    ]
]


# ================================================================
# OUTPUT
# ================================================================

print(
    "\nGALACTIC KINGS ARCHETYPE SUMMARY"
)


print(
    cluster_summary
    .sort_values(
        "behavior_score",
        ascending=False,
    )
    .round(
        {
            "avg_wars_observed": 2,
            "avg_fame": 2,
            "avg_decks": 2,
            "participation_rate": 3,
            "full_participation_rate": 3,
            "avg_efficiency": 2,
            "fame_std": 2,
            "decks_std": 2,
            "behavior_score": 3,
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\nGK PLAYERS BY ARCHETYPE"
)


for archetype_name in ordered_names:

    group = cluster_data[
        cluster_data[
            "archetype_name"
        ] == archetype_name
    ].copy()


    if group.empty:
        continue


    print(
        f"\n{archetype_name} ({len(group)} players)"
    )


    print(
        group[
            [
                "player_name",
                "currently_member",
                "wars_observed",
                "avg_fame",
                "avg_decks",
                "participation_rate",
                "full_participation_rate",
                "avg_efficiency",
            ]
        ]
        .sort_values(
            "avg_fame",
            ascending=False,
        )
        .head(
            15
        )
        .round(
            {
                "avg_fame": 2,
                "avg_decks": 2,
                "participation_rate": 3,
                "full_participation_rate": 3,
                "avg_efficiency": 2,
            }
        )
        .to_string(
            index=False
        )
    )


# ================================================================
# PLAYERS WITHOUT ENOUGH HISTORY
# ================================================================

eligible_tags = set(
    cluster_data[
        "player_tag"
    ]
    .astype(str)
)


not_enough_history = (
    gk_pool[
        ~gk_pool[
            "player_tag"
        ]
        .astype(str)
        .isin(
            eligible_tags
        )
    ]
    .copy()
)


print(
    "\nGK players without enough history for clustering:",
    len(
        not_enough_history
    ),
)


# ================================================================
# SAVE
# ================================================================

cluster_data.to_csv(
    OUTPUT_FILE,
    index=False,
)


cluster_summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)
