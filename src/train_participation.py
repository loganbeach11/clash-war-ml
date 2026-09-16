import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("data/model_dataset.csv")

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


# --------------------------------
# FEATURES
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


# --------------------------------
# CREATE CLASSIFICATION TARGET
# --------------------------------

# 1 = player participated
# 0 = player did not participate

df["participated_target"] = (
    df["decks_used"] > 0
).astype(int)


# --------------------------------
# SPLIT BY WHOLE WEEKS
# --------------------------------

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


X_train = train_df[
    feature_columns
]

y_train = train_df[
    "participated_target"
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    "participated_target"
]


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


# --------------------------------
# TRAIN RANDOM FOREST CLASSIFIER
# --------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=3,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# --------------------------------
# PREDICT PROBABILITIES
# --------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]


# --------------------------------
# TEST MULTIPLE THRESHOLDS
# --------------------------------

thresholds = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

print(
    "\nTHRESHOLD COMPARISON"
)

for threshold in thresholds:

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Accuracy:  {accuracy:.3f}"
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall:    {recall:.3f}"
    )

    print(
        f"F1:        {f1:.3f}"
    )

    print(
        f"TN: {tn} | FP: {fp} | "
        f"FN: {fn} | TP: {tp}"
    )