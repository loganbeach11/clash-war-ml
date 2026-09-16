import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(
    "data/model_dataset.csv",
    parse_dates=["race_date"]
)

print("Dataset shape:", df.shape)


# -----------------------------
# FEATURES
# -----------------------------

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

target_column = "fame"


# -----------------------------
# TIME-BASED SPLIT
# -----------------------------

war_dates = sorted(
    df["race_date"].unique()
)

split_index = int(
    len(war_dates) * 0.80
)

train_dates = war_dates[:split_index]
test_dates = war_dates[split_index:]

train_df = df[
    df["race_date"].isin(train_dates)
].copy()

test_df = df[
    df["race_date"].isin(test_dates)
].copy()


print()
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


# -----------------------------
# X AND Y
# -----------------------------

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]


# -----------------------------
# GRADIENT BOOSTING MODEL
# -----------------------------

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# -----------------------------
# PREDICTIONS
# -----------------------------

predictions = model.predict(
    X_test
)


# Fame cannot realistically be negative
predictions = predictions.clip(min=0)


# -----------------------------
# METRICS
# -----------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print()
print("GRADIENT BOOSTING RESULTS")
print("MAE:", mae)
print("RMSE:", rmse)
print("R^2:", r2)


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print()
print("FEATURE IMPORTANCE")
print(
    importance_df.to_string(
        index=False
    )
)


# -----------------------------
# SAMPLE PREDICTIONS
# -----------------------------

results = test_df[
    [
        "player_name",
        "race_date",
        "fame"
    ]
].copy()

results["predicted_fame"] = predictions

results["error"] = (
    results["fame"]
    - results["predicted_fame"]
).abs()


print()
print("Sample predictions:")

print(
    results.head(25).to_string(
        index=False
    )
)