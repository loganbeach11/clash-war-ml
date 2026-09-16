import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


# Configuration

GALACTIC_KINGS_TAG = "#Y9202P9U"
GALACTIC_KINGS_DAILY_SLOTS = 50

PARTICIPATION_THRESHOLD = 0.40

HISTORY_FILE = "data/model_dataset.csv"
LIVE_FEATURE_FILE = "data/gk_live_player_features.csv"

PLAYER_OUTPUT_FILE = "data/gk_live_player_predictions.csv"
SUMMARY_OUTPUT_FILE = "data/gk_live_player_summary.csv"


# Feature list

FEATURE_COLUMNS = [
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


# Normalize player weights to 50 daily slots

def normalize_to_slots(
    probabilities,
    target_slots,
):
    """
    Convert raw participation probabilities into expected daily
    participation weights that:

    - remain between 0 and 1
    - preserve relative likelihood as much as possible
    - sum to target_slots when enough players exist

    This is a capped proportional allocation.
    """

    probs = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(probs) == 0:
        return probs

    target_slots = min(
        float(target_slots),
        float(len(probs)),
    )

    probs = np.clip(
        probs,
        0,
        1,
    )

    if probs.sum() == 0:

        return np.full(
            len(probs),
            target_slots / len(probs),
        )

    weights = np.zeros(
        len(probs),
        dtype=float,
    )

    remaining = np.ones(
        len(probs),
        dtype=bool,
    )

    slots_left = target_slots

    while (
        slots_left > 1e-9
        and remaining.any()
    ):

        remaining_probs = probs[
            remaining
        ]

        if remaining_probs.sum() == 0:

            allocation = (
                slots_left
                / remaining.sum()
            )

            weights[
                remaining
            ] += allocation

            break

        scale = (
            slots_left
            / remaining_probs.sum()
        )

        proposed = (
            remaining_probs
            * scale
        )

        remaining_indices = np.where(
            remaining
        )[0]

        capped = (
            proposed >= 1
        )

        if not capped.any():

            weights[
                remaining_indices
            ] = proposed

            slots_left = 0

            break

        capped_indices = (
            remaining_indices[
                capped
            ]
        )

        weights[
            capped_indices
        ] = 1

        slots_left -= len(
            capped_indices
        )

        remaining[
            capped_indices
        ] = False

    return np.clip(
        weights,
        0,
        1,
    )


# Load historical training data

# Keep the larger historical dataset for model training for now.
#
# The project OUTPUT is GK-only, but using more historical players
# gives the supervised models more examples and is likely better than
# retraining on a much smaller GK-only sample before we validate that
# choice separately.

history = pd.read_csv(
    HISTORY_FILE
)

print(
    "Historical training rows:",
    len(history),
)


# Load GK live features

live = pd.read_csv(
    LIVE_FEATURE_FILE
)

print(
    "All live feature rows:",
    len(live),
)


gk_live = live[
    live[
        "clan_tag"
    ] == GALACTIC_KINGS_TAG
].copy()


if gk_live.empty:
    raise RuntimeError(
        "No Galactic Kings players found in "
        "data/live_player_features.csv."
    )


# GK rotation pool

# Any GK player observed in this live race is part of the rotation
# pool, whether currently in the clan or temporarily outside it.
#
# This matches GK's operational behavior: players rotate in/out, while
# the clan still fills up to 50 participants each Battle Day.

gk_live[
    "rotation_eligible"
] = 1


forecast_live = (
    gk_live.copy()
)


print(
    "Galactic Kings rotation-pool players:",
    len(
        forecast_live
    ),
)

print(
    "Currently in clan:",
    int(
        forecast_live[
            "currently_member"
        ]
        .fillna(0)
        .sum()
    ),
)

print(
    "Currently outside clan:",
    int(
        len(
            forecast_live
        )
        - forecast_live[
            "currently_member"
        ]
        .fillna(0)
        .sum()
    ),
)


# Validate features

missing_history_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in history.columns
]

missing_live_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in forecast_live.columns
]


if missing_history_features:

    raise ValueError(
        "Historical dataset is missing model features: "
        + ", ".join(
            missing_history_features
        )
    )


if missing_live_features:

    raise ValueError(
        "GK live features are missing model features: "
        + ", ".join(
            missing_live_features
        )
    )


# Train linear regression model

X_history = history[
    FEATURE_COLUMNS
]

y_fame = history[
    "fame"
]


linear_model = LinearRegression()

linear_model.fit(
    X_history,
    y_fame,
)


# Train participation model

history[
    "participated"
] = (
    history[
        "decks_used"
    ] > 0
).astype(int)


y_participation = history[
    "participated"
]


participation_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=3,
    class_weight="balanced",
)


participation_model.fit(
    X_history,
    y_participation,
)


# Train active-deck model

active_history = history[
    history[
        "decks_used"
    ] > 0
].copy()


X_active = active_history[
    FEATURE_COLUMNS
]

y_active_decks = active_history[
    "decks_used"
]


deck_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=3,
)


deck_model.fit(
    X_active,
    y_active_decks,
)


# Create prediction columns

prediction_columns = [
    "linear_predicted_fame",
    "participation_probability",
    "predicted_participation",
    "predicted_active_decks",
    "predicted_efficiency",
    "hard_predicted_fame",
    "expected_predicted_fame",
    "final_predicted_fame",
    "daily_participation_weight",
    "capacity_adjusted_fame",
]


for column in prediction_columns:

    forecast_live[
        column
    ] = np.nan


# Tier 1: full ML model

ml_mask = (
    forecast_live[
        "prediction_tier"
    ] == "ml_model"
)


ml_live = forecast_live[
    ml_mask
].copy()


if not ml_live.empty:

    X_live = ml_live[
        FEATURE_COLUMNS
    ]


# Linear regression

    linear_predictions = (
        linear_model.predict(
            X_live
        )
    )

    linear_predictions = np.clip(
        linear_predictions,
        0,
        None,
    )


# Participation

    participation_probability = (
        participation_model.predict_proba(
            X_live
        )[:, 1]
    )


    predicted_participation = (
        participation_probability
        >= PARTICIPATION_THRESHOLD
    ).astype(int)


# Active decks

    predicted_active_decks = (
        deck_model.predict(
            X_live
        )
    )


    predicted_active_decks = np.clip(
        predicted_active_decks,
        0,
        16,
    )


# Efficiency

    predicted_efficiency = (
        ml_live[
            "last_3_avg_efficiency"
        ]
        .fillna(0)
        .to_numpy()
    )


# Pipeline fame

    hard_predicted_fame = (
        predicted_participation
        * predicted_active_decks
        * predicted_efficiency
    )


    expected_predicted_fame = (
        participation_probability
        * predicted_active_decks
        * predicted_efficiency
    )


# Save full-ML results

    forecast_live.loc[
        ml_mask,
        "linear_predicted_fame",
    ] = linear_predictions

    forecast_live.loc[
        ml_mask,
        "participation_probability",
    ] = participation_probability

    forecast_live.loc[
        ml_mask,
        "predicted_participation",
    ] = predicted_participation

    forecast_live.loc[
        ml_mask,
        "predicted_active_decks",
    ] = predicted_active_decks

    forecast_live.loc[
        ml_mask,
        "predicted_efficiency",
    ] = predicted_efficiency

    forecast_live.loc[
        ml_mask,
        "hard_predicted_fame",
    ] = hard_predicted_fame

    forecast_live.loc[
        ml_mask,
        "expected_predicted_fame",
    ] = expected_predicted_fame


# Full-ML player performance prediction

# Preserve the current midpoint ensemble so this remains comparable
# with the previously evaluated live pipeline.
#
# Important: this midpoint is still a heuristic ensemble and should
# be evaluated separately before we call it the strongest model.

forecast_live.loc[
    ml_mask,
    "final_predicted_fame",
] = (
    forecast_live.loc[
        ml_mask,
        "linear_predicted_fame",
    ]
    +
    forecast_live.loc[
        ml_mask,
        "hard_predicted_fame",
    ]
) / 2


# Tier 2: limited history

limited_mask = (
    forecast_live[
        "prediction_tier"
    ] == "limited_history"
)


forecast_live.loc[
    limited_mask,
    "final_predicted_fame",
] = (
    forecast_live.loc[
        limited_mask,
        "last_3_avg_fame",
    ]
)


forecast_live.loc[
    limited_mask,
    "participation_probability",
] = (
    forecast_live.loc[
        limited_mask,
        "last_5_participation_rate",
    ]
)


# GK baselines from full-ML players

gk_ml_fame_baseline = (
    forecast_live.loc[
        ml_mask,
        "final_predicted_fame",
    ]
    .mean()
)


gk_ml_participation_baseline = (
    forecast_live.loc[
        ml_mask,
        "participation_probability",
    ]
    .mean()
)


# Safety fallback if there happen to be no full-ML GK players.
if pd.isna(
    gk_ml_fame_baseline
):

    gk_ml_fame_baseline = (
        history[
            "fame"
        ]
        .mean()
    )


if pd.isna(
    gk_ml_participation_baseline
):

    gk_ml_participation_baseline = (
        history[
            "participated"
        ]
        .mean()
    )


# Tier 3: no history / GK baseline

baseline_mask = (
    forecast_live[
        "prediction_tier"
    ] == "clan_baseline"
)


forecast_live.loc[
    baseline_mask,
    "final_predicted_fame",
] = (
    gk_ml_fame_baseline
)


forecast_live.loc[
    baseline_mask,
    "participation_probability",
] = (
    gk_ml_participation_baseline
)


# Fallback safety

forecast_live[
    "final_predicted_fame"
] = pd.to_numeric(
    forecast_live[
        "final_predicted_fame"
    ],
    errors="coerce",
).fillna(
    gk_ml_fame_baseline
).clip(
    lower=0,
)


forecast_live[
    "participation_probability"
] = pd.to_numeric(
    forecast_live[
        "participation_probability"
    ],
    errors="coerce",
).fillna(
    gk_ml_participation_baseline
).clip(
    0,
    1,
)


# Normalize GK daily participation to 50 slots

# GK fills all 50 participant slots on each Battle Day.
#
# Therefore the live probability model is used to determine relative
# participation likelihood within the rotation pool, and the weights
# are normalized so the expected daily participation total equals 50
# whenever the rotation pool contains at least 50 players.

gk_probabilities = (
    forecast_live[
        "participation_probability"
    ]
    .to_numpy()
)


gk_weights = normalize_to_slots(
    gk_probabilities,
    GALACTIC_KINGS_DAILY_SLOTS,
)


forecast_live[
    "daily_participation_weight"
] = (
    gk_weights
)


print(
    "\nGalactic Kings expected daily participants:",
    round(
        forecast_live[
            "daily_participation_weight"
        ].sum(),
        4,
    ),
)


# Capacity-adjusted performance

# Convert the existing whole-war performance estimate into a
# rotation-capacity-adjusted value.
#
# We divide out the original participation probability, then apply the
# normalized GK daily participation weight.
#
# This remains a heuristic adjustment, not a direct daily Medal model.

safe_probability = (
    forecast_live[
        "participation_probability"
    ]
    .clip(
        lower=0.05
    )
)


forecast_live[
    "capacity_adjusted_fame"
] = (
    forecast_live[
        "final_predicted_fame"
    ]
    / safe_probability
    * forecast_live[
        "daily_participation_weight"
    ]
)


# Prevent unstable inflation for very low original probabilities.
forecast_live[
    "capacity_adjusted_fame"
] = np.minimum(
    forecast_live[
        "capacity_adjusted_fame"
    ],
    forecast_live[
        "final_predicted_fame"
    ] * 3,
)


forecast_live[
    "capacity_adjusted_fame"
] = np.clip(
    forecast_live[
        "capacity_adjusted_fame"
    ],
    0,
    None,
)


# Safety checks

missing_predictions = (
    forecast_live[
        "final_predicted_fame"
    ]
    .isna()
    .sum()
)


missing_adjusted_predictions = (
    forecast_live[
        "capacity_adjusted_fame"
    ]
    .isna()
    .sum()
)


print(
    "Players without base prediction:",
    missing_predictions,
)

print(
    "Players without capacity-adjusted prediction:",
    missing_adjusted_predictions,
)


# GK player summary fields

forecast_live[
    "projected_daily_fame"
] = (
    forecast_live[
        "capacity_adjusted_fame"
    ]
    / 4
)


forecast_live[
    "availability_bucket"
] = pd.cut(
    forecast_live[
        "participation_probability"
    ],
    bins=[
        -0.001,
        0.40,
        0.70,
        1.001,
    ],
    labels=[
        "Low",
        "Medium",
        "High",
    ],
)


# GK summary

tier_counts = (
    forecast_live[
        "prediction_tier"
    ]
    .value_counts()
    .to_dict()
)


archetype_counts = {}

if (
    "archetype_name"
    in forecast_live.columns
):

    archetype_counts = (
        forecast_live[
            "archetype_name"
        ]
        .fillna(
            "Not Enough History"
        )
        .value_counts()
        .to_dict()
    )


summary = pd.DataFrame(
    [
        {
            "clan_tag":
                GALACTIC_KINGS_TAG,

            "clan_name":
                forecast_live[
                    "clan_name"
                ].iloc[0],

            "rotation_pool_players":
                forecast_live[
                    "player_tag"
                ].nunique(),

            "current_members":
                int(
                    forecast_live[
                        "currently_member"
                    ]
                    .fillna(0)
                    .sum()
                ),

            "players_outside_clan":
                int(
                    len(
                        forecast_live
                    )
                    - forecast_live[
                        "currently_member"
                    ]
                    .fillna(0)
                    .sum()
                ),

            "expected_daily_participants":
                forecast_live[
                    "daily_participation_weight"
                ].sum(),

            "ml_players":
                int(
                    (
                        forecast_live[
                            "prediction_tier"
                        ] == "ml_model"
                    ).sum()
                ),

            "limited_history_players":
                int(
                    (
                        forecast_live[
                            "prediction_tier"
                        ] == "limited_history"
                    ).sum()
                ),

            "baseline_players":
                int(
                    (
                        forecast_live[
                            "prediction_tier"
                        ] == "clan_baseline"
                    ).sum()
                ),

            "ml_coverage":
                (
                    (
                        forecast_live[
                            "prediction_tier"
                        ] == "ml_model"
                    ).sum()
                    / len(
                        forecast_live
                    )
                ),

            "base_predicted_fame":
                forecast_live[
                    "final_predicted_fame"
                ].sum(),

            "capacity_adjusted_fame":
                forecast_live[
                    "capacity_adjusted_fame"
                ].sum(),

            "projected_daily_fame":
                forecast_live[
                    "projected_daily_fame"
                ].sum(),

            "average_participation_probability":
                forecast_live[
                    "participation_probability"
                ].mean(),
        }
    ]
)


# Output

display_columns = [
    "player_name",
    "currently_member",
    "archetype_name",
    "prediction_tier",
    "previous_wars_count",
    "participation_probability",
    "daily_participation_weight",
    "final_predicted_fame",
    "capacity_adjusted_fame",
    "projected_daily_fame",
]


display_columns = [
    column
    for column in display_columns
    if column in forecast_live.columns
]


print(
    "\nGALACTIC KINGS LIVE PLAYER PREDICTIONS"
)


print(
    forecast_live[
        display_columns
    ]
    .sort_values(
        "capacity_adjusted_fame",
        ascending=False,
    )
    .head(
        25
    )
    .round(
        {
            "participation_probability": 3,
            "daily_participation_weight": 3,
            "final_predicted_fame": 2,
            "capacity_adjusted_fame": 2,
            "projected_daily_fame": 2,
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\nGALACTIC KINGS PLAYER MODEL SUMMARY"
)


print(
    summary.round(
        {
            "expected_daily_participants": 2,
            "ml_coverage": 3,
            "base_predicted_fame": 2,
            "capacity_adjusted_fame": 2,
            "projected_daily_fame": 2,
            "average_participation_probability": 3,
        }
    )
    .to_string(
        index=False
    )
)


if archetype_counts:

    print(
        "\nGALACTIC KINGS ARCHETYPE COUNTS"
    )

    for archetype, count in archetype_counts.items():

        print(
            f"{archetype}: {count}"
        )


# Save

forecast_live.to_csv(
    PLAYER_OUTPUT_FILE,
    index=False,
)


summary.to_csv(
    SUMMARY_OUTPUT_FILE,
    index=False,
)


print(
    "\nSaved:"
)

print(
    PLAYER_OUTPUT_FILE
)

print(
    SUMMARY_OUTPUT_FILE
)
