import pandas as pd
import numpy as np


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = "data/gk_live_player_predictions.csv"

EXPECTED_LINEUP_OUTPUT = (
    "data/gk_rotation_expected_value.csv"
)

ACTIVE_LINEUP_OUTPUT = (
    "data/gk_rotation_active_lineup.csv"
)

BACKUP_OUTPUT = (
    "data/gk_rotation_backups.csv"
)

SUMMARY_OUTPUT = (
    "data/gk_rotation_summary.csv"
)

DAILY_LINEUP_SIZE = 50
BATTLE_DAYS_PER_WEEK = 4


# ================================================================
# LOAD GK LIVE PLAYER PREDICTIONS
# ================================================================

gk = pd.read_csv(
    INPUT_FILE
)


required_columns = [
    "player_tag",
    "player_name",
    "currently_member",
    "prediction_tier",
    "participation_probability",
    "final_predicted_fame",
    "hard_predicted_fame",
]


missing_columns = [
    column
    for column in required_columns
    if column not in gk.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns in "
        "gk_live_player_predictions.csv: "
        + ", ".join(
            missing_columns
        )
    )


if gk.empty:

    raise RuntimeError(
        "No Galactic Kings players found."
    )


gk = (
    gk
    .drop_duplicates(
        subset=[
            "player_tag"
        ]
    )
    .copy()
)


print(
    "Galactic Kings rotation-pool players:",
    len(gk),
)


# ================================================================
# CLEAN INPUT VALUES
# ================================================================

gk[
    "currently_member"
] = pd.to_numeric(
    gk[
        "currently_member"
    ],
    errors="coerce",
).fillna(
    0
).astype(
    int
)


gk[
    "participation_probability"
] = pd.to_numeric(
    gk[
        "participation_probability"
    ],
    errors="coerce",
).fillna(
    0
).clip(
    0,
    1,
)


gk[
    "final_predicted_fame"
] = pd.to_numeric(
    gk[
        "final_predicted_fame"
    ],
    errors="coerce",
).fillna(
    0
).clip(
    lower=0,
)


gk[
    "hard_predicted_fame"
] = pd.to_numeric(
    gk[
        "hard_predicted_fame"
    ],
    errors="coerce",
)


# ================================================================
# EXPECTED-VALUE DAILY PERFORMANCE
# ================================================================

# final_predicted_fame reflects the current whole-war player forecast
# with attendance uncertainty included.
#
# This ranking is useful before we know exactly who will be available.

gk[
    "expected_daily_fame"
] = (
    gk[
        "final_predicted_fame"
    ]
    / BATTLE_DAYS_PER_WEEK
)


# ================================================================
# CONFIRMED-ACTIVE DAILY PERFORMANCE
# ================================================================

# For full-ML players, hard_predicted_fame represents production when
# the participation model says they are active.
#
# Once a player is confirmed available, this is more appropriate than
# dividing their expected fame by participation probability.
#
# Limited-history / baseline players do not have hard predictions, so
# they fall back to final_predicted_fame.

gk[
    "confirmed_active_war_fame"
] = (
    gk[
        "hard_predicted_fame"
    ]
    .where(
        gk[
            "hard_predicted_fame"
        ].notna(),
        gk[
            "final_predicted_fame"
        ],
    )
    .clip(
        lower=0,
    )
)


gk[
    "confirmed_active_daily_fame"
] = (
    gk[
        "confirmed_active_war_fame"
    ]
    / BATTLE_DAYS_PER_WEEK
)


# ================================================================
# RELIABILITY
# ================================================================

gk[
    "reliability_score"
] = (
    gk[
        "participation_probability"
    ]
)


# ================================================================
# SCORE 1: EXPECTED VALUE
# ================================================================

# Expected production is primary.
# Reliability is only a tiny deterministic tie-breaker.

gk[
    "expected_value_score"
] = (
    gk[
        "expected_daily_fame"
    ]
    + (
        gk[
            "reliability_score"
        ]
        * 0.01
    )
)


# ================================================================
# SCORE 2: CONFIRMED ACTIVE
# ================================================================

# Once availability is known, active production is primary.
# Reliability remains only a tiny tie-breaker.

gk[
    "active_lineup_score"
] = (
    gk[
        "confirmed_active_daily_fame"
    ]
    + (
        gk[
            "reliability_score"
        ]
        * 0.01
    )
)


# ================================================================
# EXPECTED-VALUE RANKING
# ================================================================

expected_ranking = (
    gk
    .sort_values(
        [
            "expected_value_score",
            "reliability_score",
            "player_name",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


expected_ranking[
    "expected_value_rank"
] = (
    np.arange(
        len(
            expected_ranking
        )
    )
    + 1
)


expected_ranking[
    "expected_value_status"
] = np.where(
    expected_ranking[
        "expected_value_rank"
    ] <= DAILY_LINEUP_SIZE,
    "START",
    "BACKUP",
)


# ================================================================
# CONFIRMED-ACTIVE RANKING
# ================================================================

active_ranking = (
    gk
    .sort_values(
        [
            "active_lineup_score",
            "reliability_score",
            "player_name",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


active_ranking[
    "active_lineup_rank"
] = (
    np.arange(
        len(
            active_ranking
        )
    )
    + 1
)


active_ranking[
    "active_lineup_status"
] = np.where(
    active_ranking[
        "active_lineup_rank"
    ] <= DAILY_LINEUP_SIZE,
    "START",
    "BACKUP",
)


# ================================================================
# MERGE BOTH RANKINGS
# ================================================================

comparison_columns = [
    "player_tag",
    "player_name",
    "currently_member",
    "prediction_tier",
    "participation_probability",
    "expected_daily_fame",
    "confirmed_active_daily_fame",
]


if (
    "archetype_name"
    in gk.columns
):

    comparison_columns.append(
        "archetype_name"
    )


rank_comparison = (
    gk[
        comparison_columns
    ]
    .merge(
        expected_ranking[
            [
                "player_tag",
                "expected_value_rank",
                "expected_value_status",
            ]
        ],
        on="player_tag",
        how="left",
    )
    .merge(
        active_ranking[
            [
                "player_tag",
                "active_lineup_rank",
                "active_lineup_status",
            ]
        ],
        on="player_tag",
        how="left",
    )
)


rank_comparison[
    "rank_change_if_confirmed"
] = (
    rank_comparison[
        "expected_value_rank"
    ]
    - rank_comparison[
        "active_lineup_rank"
    ]
)


# ================================================================
# LINEUP / BACKUP SETS
# ================================================================

expected_starters = (
    rank_comparison[
        rank_comparison[
            "expected_value_status"
        ] == "START"
    ]
    .sort_values(
        "expected_value_rank"
    )
    .copy()
)


active_starters = (
    rank_comparison[
        rank_comparison[
            "active_lineup_status"
        ] == "START"
    ]
    .sort_values(
        "active_lineup_rank"
    )
    .copy()
)


active_backups = (
    rank_comparison[
        rank_comparison[
            "active_lineup_status"
        ] == "BACKUP"
    ]
    .sort_values(
        "active_lineup_rank"
    )
    .copy()
)


# ================================================================
# VALIDATION
# ================================================================

if len(
    gk
) >= DAILY_LINEUP_SIZE:

    if len(
        expected_starters
    ) != DAILY_LINEUP_SIZE:

        raise RuntimeError(
            "Expected-value lineup does not contain exactly "
            f"{DAILY_LINEUP_SIZE} players."
        )

    if len(
        active_starters
    ) != DAILY_LINEUP_SIZE:

        raise RuntimeError(
            "Confirmed-active lineup does not contain exactly "
            f"{DAILY_LINEUP_SIZE} players."
        )


# ================================================================
# SUMMARY METRICS
# ================================================================

expected_daily_total = float(
    expected_starters[
        "expected_daily_fame"
    ].sum()
)


confirmed_active_daily_total = float(
    active_starters[
        "confirmed_active_daily_fame"
    ].sum()
)


expected_current_members = int(
    expected_starters[
        "currently_member"
    ].sum()
)


active_current_members = int(
    active_starters[
        "currently_member"
    ].sum()
)


summary = pd.DataFrame(
    [
        {
            "rotation_pool_players":
                len(
                    gk
                ),

            "lineup_size":
                min(
                    DAILY_LINEUP_SIZE,
                    len(
                        gk
                    ),
                ),

            "expected_value_daily_fame":
                expected_daily_total,

            "expected_value_avg_participation_probability":
                expected_starters[
                    "participation_probability"
                ].mean(),

            "expected_value_current_members":
                expected_current_members,

            "expected_value_outside_clan":
                len(
                    expected_starters
                )
                - expected_current_members,

            "confirmed_active_daily_fame":
                confirmed_active_daily_total,

            "confirmed_active_avg_participation_probability":
                active_starters[
                    "participation_probability"
                ].mean(),

            "confirmed_active_current_members":
                active_current_members,

            "confirmed_active_outside_clan":
                len(
                    active_starters
                )
                - active_current_members,

            "backup_players":
                len(
                    active_backups
                ),
        }
    ]
)


# ================================================================
# TERMINAL OUTPUT
# ================================================================

print(
    "\nGK ROTATION OPTIMIZATION"
)


print(
    "Rotation pool:",
    len(
        gk
    ),
)


print(
    "Lineup size:",
    min(
        DAILY_LINEUP_SIZE,
        len(
            gk
        ),
    ),
)


print(
    "\nEXPECTED-VALUE LINEUP"
)


print(
    "Expected daily fame:",
    round(
        expected_daily_total,
        2,
    ),
)


print(
    "Average participation probability:",
    round(
        expected_starters[
            "participation_probability"
        ].mean(),
        3,
    ),
)


print(
    "Currently in clan:",
    expected_current_members,
)


print(
    "Currently outside clan:",
    len(
        expected_starters
    )
    - expected_current_members,
)


print(
    "\nCONFIRMED-ACTIVE LINEUP"
)


print(
    "Projected confirmed-active daily fame:",
    round(
        confirmed_active_daily_total,
        2,
    ),
)


print(
    "Average historical participation probability:",
    round(
        active_starters[
            "participation_probability"
        ].mean(),
        3,
    ),
)


print(
    "Currently in clan:",
    active_current_members,
)


print(
    "Currently outside clan:",
    len(
        active_starters
    )
    - active_current_members,
)


# ================================================================
# ARCHETYPE BREAKDOWN
# ================================================================

if (
    "archetype_name"
    in rank_comparison.columns
):

    print(
        "\nCONFIRMED-ACTIVE ARCHETYPE BREAKDOWN"
    )

    print(
        active_starters[
            "archetype_name"
        ]
        .fillna(
            "Not Enough History"
        )
        .value_counts()
        .to_string()
    )


# ================================================================
# BIGGEST RISERS
# ================================================================

print(
    "\nBIGGEST RISERS IF CONFIRMED AVAILABLE"
)


riser_columns = [
    "player_name",
]


if (
    "archetype_name"
    in rank_comparison.columns
):

    riser_columns.append(
        "archetype_name"
    )


riser_columns += [
    "expected_value_rank",
    "active_lineup_rank",
    "rank_change_if_confirmed",
    "participation_probability",
    "expected_daily_fame",
    "confirmed_active_daily_fame",
]


print(
    rank_comparison[
        riser_columns
    ]
    .sort_values(
        [
            "rank_change_if_confirmed",
            "confirmed_active_daily_fame",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(
        10
    )
    .round(
        {
            "participation_probability": 2,
            "expected_daily_fame": 2,
            "confirmed_active_daily_fame": 2,
        }
    )
    .to_string(
        index=False
    )
)


# ================================================================
# CONFIRMED-ACTIVE RECOMMENDED 50
# ================================================================

print(
    "\nCONFIRMED-ACTIVE RECOMMENDED 50"
)


display_columns = [
    "active_lineup_rank",
    "player_name",
    "currently_member",
]


if (
    "archetype_name"
    in rank_comparison.columns
):

    display_columns.append(
        "archetype_name"
    )


display_columns += [
    "prediction_tier",
    "participation_probability",
    "expected_daily_fame",
    "confirmed_active_daily_fame",
]


active_display = (
    active_starters[
        display_columns
    ]
    .sort_values(
        "active_lineup_rank"
    )
    .copy()
)


for column in [
    "participation_probability",
    "expected_daily_fame",
    "confirmed_active_daily_fame",
]:

    active_display[
        column
    ] = (
        active_display[
            column
        ]
        .astype(
            float
        )
        .round(
            2
        )
    )


print(
    active_display.to_string(
        index=False
    )
)


# ================================================================
# TOP BACKUPS
# ================================================================

print(
    "\nACTIVE-LINEUP TOP 10 BACKUPS"
)


backup_display = (
    active_backups[
        display_columns
    ]
    .sort_values(
        "active_lineup_rank"
    )
    .head(
        10
    )
    .copy()
)


for column in [
    "participation_probability",
    "expected_daily_fame",
    "confirmed_active_daily_fame",
]:

    backup_display[
        column
    ] = (
        backup_display[
            column
        ]
        .astype(
            float
        )
        .round(
            2
        )
    )


print(
    backup_display.to_string(
        index=False
    )
)


# ================================================================
# SAVE
# ================================================================

expected_starters.to_csv(
    EXPECTED_LINEUP_OUTPUT,
    index=False,
)


active_starters.to_csv(
    ACTIVE_LINEUP_OUTPUT,
    index=False,
)


active_backups.to_csv(
    BACKUP_OUTPUT,
    index=False,
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


print(
    "\nSaved:"
)


print(
    EXPECTED_LINEUP_OUTPUT
)


print(
    ACTIVE_LINEUP_OUTPUT
)


print(
    BACKUP_OUTPUT
)


print(
    SUMMARY_OUTPUT
)
