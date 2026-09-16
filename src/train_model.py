import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# LOAD ML DATASET
# --------------------------------

df = pd.read_csv("data/model_dataset.csv")

print("Dataset shape:", df.shape)


# --------------------------------
# SELECT FEATURES AND TARGET
# --------------------------------

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


# --------------------------------
# TIME-BASED TRAIN / TEST SPLIT
# --------------------------------

# Convert race_date back to datetime
df["race_date"] = pd.to_datetime(df["race_date"])
df["race_week"] = (
    df["race_date"]
    .dt.to_period("W")
    .apply(
        lambda period:
        period.start_time
    )
)

# Sort by time so older wars come before newer wars
df = df.sort_values("race_date")

# --------------------------------
# TIME-BASED SPLIT BY WHOLE WARS
# --------------------------------

# Get unique war dates in chronological order
weeks = sorted(
    df["race_week"].unique()
)

# Use the first 80% of wars for training
# and the newest 20% of wars for testing
split_index = int(len(weeks) * 0.80)

train_weeks = weeks[:split_index]
test_weeks = weeks[split_index:]

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

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

print("\nTraining rows:", len(train_df))
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


# --------------------------------
# TRAIN BASELINE MODEL
# --------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# --------------------------------
# MAKE PREDICTIONS
# --------------------------------

predictions = model.predict(X_test)


# --------------------------------
# EVALUATE MODEL
# --------------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)

print("\nLinear Regression Results")

print("MAE:", mae)
print("RMSE:", rmse)
print("R^2:", r2)


# --------------------------------
# VIEW SOME PREDICTIONS
# --------------------------------

results = test_df[
    [
        "player_tag",
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

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

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
