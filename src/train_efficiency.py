import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Load data

df = pd.read_csv("data/model_dataset.csv")

df["race_date"] = pd.to_datetime(df["race_date"])

df = df.sort_values("race_date")


# Features

feature_columns = [
    "last_3_avg_fame",
    "last_3_avg_decks",
    "last_3_avg_efficiency",
    "last_5_participation_rate",
    "last_5_full_participation_rate",
    "previous_war_fame",
    "previous_war_decks",
    "fame_trend",
    "last_3_fame_std",
    "last_3_decks_std"
]


# Keep only active players

active_df = df[df["decks_used"] > 0].copy()

active_df["fame_per_deck_target"] = (
    active_df["fame"]
    / active_df["decks_used"]
)

print("Active-player rows:", len(active_df))


# Split by whole wars

war_dates = sorted(active_df["race_date"].unique())

split_index = int(len(war_dates) * 0.80)

train_dates = war_dates[:split_index]
test_dates = war_dates[split_index:]

train_df = active_df[
    active_df["race_date"].isin(train_dates)
].copy()

test_df = active_df[
    active_df["race_date"].isin(test_dates)
].copy()


print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "Training dates:",
    train_df["race_date"].min(),
    "to",
    train_df["race_date"].max()
)

print(
    "Testing dates:",
    test_df["race_date"].min(),
    "to",
    test_df["race_date"].max()
)


# Train model

X_train = train_df[feature_columns]
y_train = train_df["fame_per_deck_target"]

X_test = test_df[feature_columns]
y_test = test_df["fame_per_deck_target"]


model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=3
)

model.fit(
    X_train,
    y_train
)


# Predict

predictions = model.predict(X_test)

predictions = predictions.clip(min=0)


# Evaluate

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = (
    mean_squared_error(
        y_test,
        predictions
    ) ** 0.5
)

r2 = r2_score(
    y_test,
    predictions
)


print("\nFAME PER DECK MODEL")

print("MAE:", mae)
print("RMSE:", rmse)
print("R^2:", r2)


# Sample results

results = test_df[
    [
        "player_name",
        "race_date",
        "decks_used",
        "fame",
        "fame_per_deck_target"
    ]
].copy()

results["predicted_fame_per_deck"] = predictions

results["error"] = (
    results["fame_per_deck_target"]
    - results["predicted_fame_per_deck"]
).abs()


print("\nSample predictions:")

print(
    results[
        [
            "player_name",
            "decks_used",
            "fame",
            "fame_per_deck_target",
            "predicted_fame_per_deck",
            "error"
        ]
    ]
    .head(25)
    .to_string(index=False)
)
