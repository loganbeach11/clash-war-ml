import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
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


# Keep only active players

active_df = df[
    df["decks_used"] > 0
].copy()

print(
    "Active-player rows:",
    len(active_df)
)


# Split by whole weeks

weeks = sorted(
    active_df["race_week"].unique()
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

train_df = active_df[
    active_df["race_week"].isin(
        train_weeks
    )
].copy()

test_df = active_df[
    active_df["race_week"].isin(
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


# Train model

X_train = train_df[
    feature_columns
]

y_train = train_df[
    "decks_used"
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    "decks_used"
]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=3
)

model.fit(
    X_train,
    y_train
)


# Raw predictions

raw_predictions = model.predict(
    X_test
)

raw_predictions = raw_predictions.clip(
    0,
    16
)


# Round to realistic deck values

allowed_decks = np.array(
    [
        4,
        8,
        12,
        16
    ]
)

rounded_predictions = np.array(
    [
        allowed_decks[
            np.abs(
                allowed_decks
                - prediction
            ).argmin()
        ]
        for prediction
        in raw_predictions
    ]
)


# Evaluate raw model

raw_mae = mean_absolute_error(
    y_test,
    raw_predictions
)

raw_rmse = (
    mean_squared_error(
        y_test,
        raw_predictions
    ) ** 0.5
)

raw_r2 = r2_score(
    y_test,
    raw_predictions
)

print(
    "\nRAW DECK PREDICTIONS"
)

print(
    "MAE:",
    raw_mae
)

print(
    "RMSE:",
    raw_rmse
)

print(
    "R^2:",
    raw_r2
)


# Evaluate rounded model

rounded_mae = mean_absolute_error(
    y_test,
    rounded_predictions
)

rounded_rmse = (
    mean_squared_error(
        y_test,
        rounded_predictions
    ) ** 0.5
)

rounded_r2 = r2_score(
    y_test,
    rounded_predictions
)

print(
    "\nROUNDED DECK PREDICTIONS"
)

print(
    "MAE:",
    rounded_mae
)

print(
    "RMSE:",
    rounded_rmse
)

print(
    "R^2:",
    rounded_r2
)


# Sample results

results = test_df[
    [
        "player_name",
        "race_date",
        "race_week",
        "decks_used"
    ]
].copy()

results[
    "raw_predicted_decks"
] = raw_predictions

results[
    "rounded_predicted_decks"
] = rounded_predictions

results["error"] = (
    results["decks_used"]
    - results[
        "rounded_predicted_decks"
    ]
).abs()


print(
    "\nSample predictions:"
)

print(
    results[
        [
            "player_name",
            "decks_used",
            "raw_predicted_decks",
            "rounded_predicted_decks",
            "error"
        ]
    ]
    .head(25)
    .to_string(
        index=False
    )
)
