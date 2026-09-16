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


# LOAD CLAN WAR RESULTS

clan_query = """
SELECT
    cwr.clan_tag,
    c.clan_name,
    cwr.war_id,
    w.race_date,
    w.season_id,
    w.section_index,
    cwr.rank_position,
    cwr.fame,
    cwr.clan_score,
    cwr.trophy_change
FROM clan_war_results cwr
JOIN clans c
    ON cwr.clan_tag = c.clan_tag
JOIN wars w
    ON cwr.war_id = w.war_id
ORDER BY
    cwr.clan_tag,
    w.race_date
"""

clan_df = pd.read_sql(
    clan_query,
    engine
)


# WAR TYPE

# Section 4 is the Colosseum week.
# Normal River Race weeks are sections 0-3.
clan_df["is_colosseum"] = (
    clan_df["section_index"] == 4
).astype(int)

print("Clan-war rows:")
print(clan_df.head(20))

print(
    "\nClan-war shape:",
    clan_df.shape
)


# LOAD PLAYER PERFORMANCE

player_query = """
SELECT
    pwp.player_tag,
    pwp.clan_tag,
    pwp.war_id,
    pwp.fame,
    pwp.decks_used
FROM player_war_performance pwp
"""

player_df = pd.read_sql(
    player_query,
    engine
)


# AGGREGATE PLAYER PERFORMANCE
# TO CLAN-WAR LEVEL

player_df["participated"] = (
    player_df["decks_used"] > 0
).astype(int)

player_df["full_participation"] = (
    player_df["decks_used"] == 16
).astype(int)

clan_player_stats = (
    player_df
    .groupby(
        [
            "clan_tag",
            "war_id"
        ]
    )
    .agg(
        roster_size=(
            "player_tag",
            "nunique"
        ),
        active_players=(
            "participated",
            "sum"
        ),
        full_participation_players=(
            "full_participation",
            "sum"
        ),
        total_player_fame=(
            "fame",
            "sum"
        ),
        total_decks_used=(
            "decks_used",
            "sum"
        ),
        avg_player_fame=(
            "fame",
            "mean"
        ),
        avg_decks_used=(
            "decks_used",
            "mean"
        )
    )
    .reset_index()
)


# MERGE CLAN + PLAYER STATS

df = clan_df.merge(
    clan_player_stats,
    on=[
        "clan_tag",
        "war_id"
    ],
    how="left"
)

# Participation rate for current completed war
df["participation_rate"] = (
    df["active_players"]
    / df["roster_size"]
)

df["full_participation_rate"] = (
    df["full_participation_players"]
    / df["roster_size"]
)

print(
    "\nMerged clan dataset:"
)

print(
    df.head(20).to_string(
        index=False
    )
)


# CLAN HISTORY FEATURES

clan_groups = df.groupby(
    "clan_tag"
)

# Previous war values
df["previous_clan_fame"] = (
    clan_groups["fame"]
    .shift(1)
)

df["previous_clan_rank"] = (
    clan_groups["rank_position"]
    .shift(1)
)

df["previous_clan_score"] = (
    clan_groups["clan_score"]
    .shift(1)
)

df["previous_active_players"] = (
    clan_groups["active_players"]
    .shift(1)
)

df["previous_participation_rate"] = (
    clan_groups["participation_rate"]
    .shift(1)
)

df["previous_total_decks"] = (
    clan_groups["total_decks_used"]
    .shift(1)
)


# ROLLING CLAN FEATURES

df["last_3_avg_clan_fame"] = (
    clan_groups["fame"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

df["last_3_avg_clan_rank"] = (
    clan_groups["rank_position"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

df["last_3_avg_clan_score"] = (
    clan_groups["clan_score"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

df["last_3_avg_active_players"] = (
    clan_groups["active_players"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

df["last_3_avg_participation_rate"] = (
    clan_groups["participation_rate"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

df["last_3_avg_total_decks"] = (
    clan_groups["total_decks_used"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)


# TREND FEATURES

df["clan_fame_trend"] = (
    clan_groups["fame"].shift(1)
    - clan_groups["fame"].shift(2)
)

df["clan_score_change"] = (
    clan_groups["clan_score"].shift(1)
    - clan_groups["clan_score"].shift(2)
)

df["active_players_change"] = (
    clan_groups["active_players"].shift(1)
    - clan_groups["active_players"].shift(2)
)


# HISTORY COUNT

df["previous_wars_count"] = (
    df.groupby("clan_tag")
    .cumcount()
)

# Require at least 3 prior wars
model_df = df[
    df["previous_wars_count"] >= 3
].copy()

# Normal River Race dataset
normal_model_df = model_df[
    model_df["is_colosseum"] == 0
].copy()

# Colosseum dataset
colosseum_model_df = model_df[
    model_df["is_colosseum"] == 1
].copy()

# FEATURES

feature_columns = [
    "previous_clan_fame",
    "previous_clan_rank",
    "previous_clan_score",
    "previous_active_players",
    "previous_participation_rate",
    "previous_total_decks",
    "last_3_avg_clan_fame",
    "last_3_avg_clan_rank",
    "last_3_avg_clan_score",
    "last_3_avg_active_players",
    "last_3_avg_participation_rate",
    "last_3_avg_total_decks",
    "clan_fame_trend",
    "clan_score_change",
    "active_players_change"
]

target_columns = [
    "fame",
    "rank_position"
]

# Remove rows with missing historical features
normal_model_df = normal_model_df.dropna(
    subset=feature_columns + [
        "fame",
        "rank_position"
    ]
)

colosseum_model_df = colosseum_model_df.dropna(
    subset=feature_columns + [
        "fame",
        "rank_position"
    ]
)


# OUTPUT

print("\nNormal-week ML-ready dataset shape:")
print(normal_model_df.shape)

print("\nColosseum ML-ready dataset shape:")
print(colosseum_model_df.shape)

print("\nNormal-week targets:")

print(
    normal_model_df[
        [
            "clan_name",
            "race_date",
            "section_index",
            "fame",
            "rank_position"
        ]
    ].to_string(index=False)
)

print("\nColosseum targets:")

print(
    colosseum_model_df[
        [
            "clan_name",
            "race_date",
            "section_index",
            "fame",
            "rank_position"
        ]
    ].to_string(index=False)
)

normal_model_df.to_csv(
    "data/clan_normal_model_dataset.csv",
    index=False
)

colosseum_model_df.to_csv(
    "data/clan_colosseum_model_dataset.csv",
    index=False
)

print(
    "\nSaved normal-week dataset to "
    "data/clan_normal_model_dataset.csv"
)

print(
    "Saved Colosseum dataset to "
    "data/clan_colosseum_model_dataset.csv"
)