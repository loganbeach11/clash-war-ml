import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Load data

df = pd.read_csv(
    "data/model_dataset.csv"
)

df["race_date"] = pd.to_datetime(
    df["race_date"]
)

df["race_week"] = (
    df["race_date"]
    .dt.to_period("W")
    .apply(
        lambda period:
        period.start_time
    )
)

df = df.sort_values(
    "race_date"
)


# Features

feature_columns = [
    # Player history
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

    # Clan context
    "previous_clan_score",
    "last_3_avg_clan_score"
]


# Split by whole weeks

weeks = sorted(
    df["race_week"].unique()
)

split_index = int(
    len(weeks) * 0.80
)

train_weeks = weeks[
    :split_index
]

test_weeks = weeks[
    split_index:
]

train_df = df[
    df["race_week"].isin(
        train_weeks
    )
].copy()

test_df = df[
    df["race_week"].isin(
        test_weeks
    )
].copy()


print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)

print(
    "Training weeks:",
    train_df["race_week"].min(),
    "to",
    train_df["race_week"].max()
)

print(
    "Testing weeks:",
    test_df["race_week"].min(),
    "to",
    test_df["race_week"].max()
)


# Stage 1: participation classifier

train_df["participated_target"] = (
    train_df["decks_used"] > 0
).astype(int)

test_df["participated_target"] = (
    test_df["decks_used"] > 0
).astype(int)


participation_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=3,
    class_weight="balanced"
)

participation_model.fit(
    train_df[
        feature_columns
    ],
    train_df[
        "participated_target"
    ]
)


participation_probability = (
    participation_model.predict_proba(
        test_df[
            feature_columns
        ]
    )[:, 1]
)


# Best threshold from participation testing
participation_threshold = 0.40

predicted_participation = (
    participation_probability
    >= participation_threshold
).astype(int)


# Stage 2: active-player deck model

active_train_df = train_df[
    train_df["decks_used"] > 0
].copy()


decks_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=3
)

decks_model.fit(
    active_train_df[
        feature_columns
    ],
    active_train_df[
        "decks_used"
    ]
)


predicted_active_decks = (
    decks_model.predict(
        test_df[
            feature_columns
        ]
    )
)

predicted_active_decks = (
    predicted_active_decks.clip(
        0,
        16
    )
)


# Stage 3: recent efficiency

predicted_efficiency = (
    test_df[
        "last_3_avg_efficiency"
    ].copy()
)


# Hard-threshold fame prediction

hard_predicted_fame = (
    predicted_participation
    * predicted_active_decks
    * predicted_efficiency
)


# Probability-weighted expected fame

expected_predicted_fame = (
    participation_probability
    * predicted_active_decks
    * predicted_efficiency
)


# Evaluate hard-threshold version

actual_fame = test_df[
    "fame"
]


hard_mae = mean_absolute_error(
    actual_fame,
    hard_predicted_fame
)

hard_rmse = (
    mean_squared_error(
        actual_fame,
        hard_predicted_fame
    ) ** 0.5
)

hard_r2 = r2_score(
    actual_fame,
    hard_predicted_fame
)


print(
    "\nHARD-THRESHOLD PIPELINE"
)

print(
    "MAE:",
    hard_mae
)

print(
    "RMSE:",
    hard_rmse
)

print(
    "R^2:",
    hard_r2
)


# Evaluate expected-value version

expected_mae = mean_absolute_error(
    actual_fame,
    expected_predicted_fame
)

expected_rmse = (
    mean_squared_error(
        actual_fame,
        expected_predicted_fame
    ) ** 0.5
)

expected_r2 = r2_score(
    actual_fame,
    expected_predicted_fame
)


print(
    "\nPROBABILITY-WEIGHTED PIPELINE"
)

print(
    "MAE:",
    expected_mae
)

print(
    "RMSE:",
    expected_rmse
)

print(
    "R^2:",
    expected_r2
)


# Sample results

results = test_df[
    [
        "player_name",
        "race_date",
        "race_week",
        "decks_used",
        "fame"
    ]
].copy()

results[
    "participation_probability"
] = participation_probability

results[
    "predicted_participation"
] = predicted_participation

results[
    "predicted_active_decks"
] = predicted_active_decks

results[
    "predicted_efficiency"
] = predicted_efficiency

results[
    "hard_predicted_fame"
] = hard_predicted_fame

results[
    "expected_predicted_fame"
] = expected_predicted_fame


# Save player predictions

prediction_output = test_df[
    [
        "player_tag",
        "player_name",
        "clan_tag",
        "war_id",
        "race_date",
        "race_week",
        "fame",
        "decks_used"
    ]
].copy()

prediction_output[
    "participation_probability"
] = participation_probability

prediction_output[
    "predicted_active_decks"
] = predicted_active_decks

prediction_output[
    "predicted_efficiency"
] = predicted_efficiency

prediction_output[
    "hard_predicted_fame"
] = hard_predicted_fame

prediction_output[
    "expected_predicted_fame"
] = expected_predicted_fame

prediction_output.to_csv(
    "data/player_test_predictions.csv",
    index=False
)

print(
    "\nSaved player predictions to "
    "data/player_test_predictions.csv"
)


print(
    "\nSample predictions:"
)

print(
    results[
        [
            "player_name",
            "fame",
            "participation_probability",
            "predicted_active_decks",
            "predicted_efficiency",
            "hard_predicted_fame",
            "expected_predicted_fame"
        ]
    ]
    .head(25)
    .to_string(
        index=False
    )
)
