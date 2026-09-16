import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("data/model_dataset.csv")

df["race_date"] = pd.to_datetime(df["race_date"])

df = df.sort_values("race_date")


# --------------------------------
# FEATURES
# --------------------------------

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


# --------------------------------
# TIME-BASED SPLIT BY WHOLE WARS
# --------------------------------

war_dates = sorted(df["race_date"].unique())

split_index = int(len(war_dates) * 0.80)

train_dates = war_dates[:split_index]
test_dates = war_dates[split_index:]

train_df = df[df["race_date"].isin(train_dates)].copy()
test_df = df[df["race_date"].isin(test_dates)].copy()

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


# ================================================================
# MODEL 1: PREDICT DECKS USED
# ================================================================

X_train_decks = train_df[feature_columns]
y_train_decks = train_df["decks_used"]

X_test_decks = test_df[feature_columns]
y_test_decks = test_df["decks_used"]


decks_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=3
)

decks_model.fit(
    X_train_decks,
    y_train_decks
)

predicted_decks = decks_model.predict(X_test_decks)


# Deck usage must stay between 0 and 16
predicted_decks = predicted_decks.clip(0, 16)
# --------------------------------
# EVALUATE DECK MODEL
# --------------------------------

decks_mae = mean_absolute_error(
    y_test_decks,
    predicted_decks
)

decks_rmse = (
    mean_squared_error(
        y_test_decks,
        predicted_decks
    ) ** 0.5
)

decks_r2 = r2_score(
    y_test_decks,
    predicted_decks
)

print("\nDECKS USED MODEL")

print("MAE:", decks_mae)
print("RMSE:", decks_rmse)
print("R^2:", decks_r2)

# ================================================================
# MODEL 2: PREDICT FAME PER DECK
# ================================================================

# Only train this model on wars where the player actually played
active_train_df = train_df[
    train_df["decks_used"] > 0
].copy()

active_test_df = test_df[
    test_df["decks_used"] > 0
].copy()


X_train_efficiency = active_train_df[feature_columns]

y_train_efficiency = (
    active_train_df["fame"]
    / active_train_df["decks_used"]
)


efficiency_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=3
)

efficiency_model.fit(
    X_train_efficiency,
    y_train_efficiency
)

# --------------------------------
# PREDICT EFFICIENCY FOR ALL TEST PLAYERS
# --------------------------------

predicted_efficiency = efficiency_model.predict(
    test_df[feature_columns]
)


# Fame per deck should never be negative
predicted_efficiency = predicted_efficiency.clip(min=0)


# ================================================================
# COMBINE BOTH MODELS
# ================================================================

predicted_fame = (
    predicted_decks
    * predicted_efficiency
)


# --------------------------------
# FINAL FAME EVALUATION
# --------------------------------

actual_fame = test_df["fame"]

fame_mae = mean_absolute_error(
    actual_fame,
    predicted_fame
)

fame_rmse = (
    mean_squared_error(
        actual_fame,
        predicted_fame
    ) ** 0.5
)

fame_r2 = r2_score(
    actual_fame,
    predicted_fame
)

print("\nCOMBINED FAME MODEL")

print("MAE:", fame_mae)
print("RMSE:", fame_rmse)
print("R^2:", fame_r2)


# --------------------------------
# SAMPLE PREDICTIONS
# --------------------------------

results = test_df[
    [
        "player_name",
        "race_date",
        "decks_used",
        "fame"
    ]
].copy()

results["predicted_decks"] = predicted_decks

results["predicted_efficiency"] = predicted_efficiency

results["predicted_fame"] = predicted_fame

results["fame_error"] = (
    results["fame"]
    - results["predicted_fame"]
).abs()


print("\nSample predictions:")

print(
    results[
        [
            "player_name",
            "race_date",
            "decks_used",
            "predicted_decks",
            "fame",
            "predicted_fame",
            "fame_error"
        ]
    ]
    .head(20)
    .to_string(index=False)
)
