import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# --------------------------------
# LOAD PLAYER PREDICTIONS
# --------------------------------

predictions = pd.read_csv(
    "data/player_test_predictions.csv",
    parse_dates=["race_date"]
)

print(
    "Player prediction rows:",
    len(predictions)
)

# --------------------------------
# AGGREGATE TO CLAN LEVEL
# --------------------------------

clan_predictions = (
    predictions
    .groupby(
        [
            "war_id",
            "race_date",
            "clan_tag"
        ]
    )
    .agg(
        actual_player_fame=(
            "fame",
            "sum"
        ),

        predicted_hard_fame=(
            "hard_predicted_fame",
            "sum"
        ),

        predicted_expected_fame=(
            "expected_predicted_fame",
            "sum"
        ),

        predicted_active_players=(
            "participation_probability",
            "sum"
        ),

        predicted_total_decks=(
            "predicted_active_decks",
            "sum"
        ),

        players_predicted=(
            "player_tag",
            "nunique"
        )
    )
    .reset_index()
)

# --------------------------------
# LOAD ACTUAL CLAN RESULTS
# --------------------------------

query = """
SELECT
    cwr.war_id,
    cwr.clan_tag,
    c.clan_name,
    w.race_date,
    w.section_index,
    cwr.fame AS actual_clan_fame,
    cwr.rank_position
FROM clan_war_results cwr
JOIN clans c
    ON cwr.clan_tag = c.clan_tag
JOIN wars w
    ON cwr.war_id = w.war_id
"""

actual = pd.read_sql(
    query,
    engine
)

# --------------------------------
# MERGE
# --------------------------------

results = clan_predictions.merge(
    actual,
    on=[
        "war_id",
        "clan_tag"
    ],
    how="left",
    suffixes=(
        "_player",
        "_clan"
    )
)

# Only evaluate normal River Race weeks.
results = results[
    results["section_index"] != 4
].copy()

# --------------------------------
# ERROR
# --------------------------------

results["hard_error"] = (
    results["actual_clan_fame"]
    - results["predicted_hard_fame"]
).abs()

results["expected_error"] = (
    results["actual_clan_fame"]
    - results["predicted_expected_fame"]
).abs()

# --------------------------------
# OUTPUT
# --------------------------------

columns = [
    "clan_name",
    "race_date_clan",
    "actual_clan_fame",
    "predicted_hard_fame",
    "predicted_expected_fame",
    "hard_error",
    "expected_error",
    "predicted_active_players",
    "players_predicted",
    "rank_position"
]

print(
    "\nNORMAL-WEEK CLAN PROJECTIONS"
)

print(
    results[
        columns
    ].sort_values(
        [
            "race_date_clan",
            "rank_position"
        ]
    ).to_string(
        index=False
    )
)

print(
    "\nAverage hard projection error:",
    results["hard_error"].mean()
)

print(
    "Average expected projection error:",
    results["expected_error"].mean()
)