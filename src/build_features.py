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

# Load player-war data

# Combine cases where a player appeared in more than one clan
# during the same River Race into one player-war observation.
query = """
WITH ranked_clans AS (
    SELECT
        pwp.player_tag,
        pwp.war_id,
        pwp.clan_tag,
        pwp.decks_used,
        pwp.fame,
        ROW_NUMBER() OVER (
            PARTITION BY pwp.player_tag, pwp.war_id
            ORDER BY pwp.decks_used DESC, pwp.fame DESC
        ) AS clan_rank
    FROM player_war_performance pwp
),

primary_clan AS (
    SELECT
        player_tag,
        war_id,
        clan_tag
    FROM ranked_clans
    WHERE clan_rank = 1
)

SELECT
    pwp.player_tag,
    p.player_name,
    w.war_id,
    w.race_date,
    pc.clan_tag,
    SUM(pwp.fame) AS fame,
    SUM(pwp.decks_used) AS decks_used
FROM player_war_performance pwp

JOIN players p
    ON pwp.player_tag = p.player_tag

JOIN wars w
    ON pwp.war_id = w.war_id

JOIN primary_clan pc
    ON pwp.player_tag = pc.player_tag
    AND pwp.war_id = pc.war_id

GROUP BY
    pwp.player_tag,
    p.player_name,
    w.war_id,
    w.race_date,
    pc.clan_tag

ORDER BY
    pwp.player_tag,
    w.race_date
"""

df = pd.read_sql(query, engine)

print(df.head(20))
print("\nShape:", df.shape)

# Load clan-war history

clan_query = """
SELECT
    cwr.clan_tag,
    w.war_id,
    w.race_date,
    cwr.rank_position,
    cwr.clan_score
FROM clan_war_results cwr
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

# Build clan history features

clan_groups = clan_df.groupby(
    "clan_tag"
)

# Previous war's clan rank
clan_df["previous_clan_rank"] = (
    clan_groups["rank_position"]
    .shift(1)
)

# Previous war's persistent clan score
clan_df["previous_clan_score"] = (
    clan_groups["clan_score"]
    .shift(1)
)

# Average clan rank over previous 3 wars
clan_df["last_3_avg_clan_rank"] = (
    clan_groups["rank_position"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

# Average clan score over previous 3 wars
clan_df["last_3_avg_clan_score"] = (
    clan_groups["clan_score"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

# Change in clan score from two wars ago
# to the previous war
clan_df["clan_score_change"] = (
    clan_groups["clan_score"].shift(1)
    - clan_groups["clan_score"].shift(2)
)

# Keep only the columns needed for merging
clan_feature_df = clan_df[
    [
        "clan_tag",
        "war_id",
        "previous_clan_rank",
        "previous_clan_score",
        "last_3_avg_clan_rank",
        "last_3_avg_clan_score",
        "clan_score_change"
    ]
].copy()

# Merge clan features into players

df = df.merge(
    clan_feature_df,
    on=[
        "clan_tag",
        "war_id"
    ],
    how="left"
)

# Basic feature engineering

# Fame earned per deck.
# Players who did not participate get 0.
df["fame_per_deck"] = 0.0

df.loc[df["decks_used"] > 0, "fame_per_deck"] = (
    df.loc[df["decks_used"] > 0, "fame"]
    / df.loc[df["decks_used"] > 0, "decks_used"]
)

# Whether the player used all 16 available decks.
df["full_participation"] = (
    df["decks_used"] == 16
).astype(int)

# Whether the player participated at all.
df["participated"] = (
    df["decks_used"] > 0
).astype(int)

# Player groups

player_groups = df.groupby("player_tag")

# Rolling historical features

# Average fame over previous 3 wars.
df["last_3_avg_fame"] = (
    player_groups["fame"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

# Average decks used over previous 3 wars.
df["last_3_avg_decks"] = (
    player_groups["decks_used"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

# Average decks used over previous 2 wars.
# This captures very recent participation behavior.
df["last_2_avg_decks"] = (
    player_groups["decks_used"]
    .transform(
        lambda x: x.shift(1).rolling(
            2,
            min_periods=1
        ).mean()
    )
)

# Average fame-per-deck over previous 3 wars.
df["last_3_avg_efficiency"] = (
    player_groups["fame_per_deck"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )
)

# Participation features

# Percentage of previous 5 wars in which
# the player used at least one deck.
df["last_5_participation_rate"] = (
    player_groups["participated"]
    .transform(
        lambda x: x.shift(1).rolling(
            5,
            min_periods=1
        ).mean()
    )
)

# Percentage of previous 2 wars in which
# the player participated.
df["last_2_participation_rate"] = (
    player_groups["participated"]
    .transform(
        lambda x: x.shift(1).rolling(
            2,
            min_periods=1
        ).mean()
    )
)

# Percentage of previous 5 wars in which
# the player used all 16 decks.
df["last_5_full_participation_rate"] = (
    player_groups["full_participation"]
    .transform(
        lambda x: x.shift(1).rolling(
            5,
            min_periods=1
        ).mean()
    )
)

# Number of the previous 3 wars in which
# the player participated.
#
# Despite the name, this is not technically a true
# "consecutive streak." It is an active-war count
# over the previous 3 wars.
df["recent_active_wars"] = (
    player_groups["participated"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=1
        ).sum()
    )
)

# Previous war features

df["previous_war_fame"] = (
    player_groups["fame"]
    .shift(1)
)

df["previous_war_decks"] = (
    player_groups["decks_used"]
    .shift(1)
)

# Trend features

# Change in fame from two wars ago
# to the previous war.
#
# Positive = fame increased.
# Negative = fame decreased.
df["fame_trend"] = (
    player_groups["fame"].shift(1)
    - player_groups["fame"].shift(2)
)

# Change in deck usage from two wars ago
# to the previous war.
#
# Positive = player used more decks recently.
# Negative = player used fewer decks recently.
df["decks_change"] = (
    player_groups["decks_used"].shift(1)
    - player_groups["decks_used"].shift(2)
)

# Consistency features

# Standard deviation of fame over previous 3 wars.
# Lower = more consistent.
df["last_3_fame_std"] = (
    player_groups["fame"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=2
        ).std()
    )
)

# Standard deviation of deck usage over previous 3 wars.
# Lower = more consistent.
df["last_3_decks_std"] = (
    player_groups["decks_used"]
    .transform(
        lambda x: x.shift(1).rolling(
            3,
            min_periods=2
        ).std()
    )
)

# Sanity check: Logan

logan = df[
    df["player_tag"] == "#8RQCGY8CQ"
]

columns_to_show = [
    "race_date",
    "fame",
    "decks_used",
    "fame_per_deck",
    "last_3_avg_fame",
    "last_3_avg_decks",
    "last_2_avg_decks",
    "last_3_avg_efficiency",
    "last_5_participation_rate",
    "last_2_participation_rate",
    "last_5_full_participation_rate",
    "recent_active_wars",
    "previous_war_fame",
    "previous_war_decks",
    "fame_trend",
    "decks_change",
    "last_3_fame_std",
    "last_3_decks_std"
]

print("\nLogan feature history:")

print(
    logan[
        columns_to_show
    ].to_string(index=False)
)

# Build ML-ready dataset

# Number of historical wars that occurred before
# each player-war observation.
df["previous_wars_count"] = (
    df.groupby("player_tag")
    .cumcount()
)

# Require at least 3 previous wars.
model_df = df[
    df["previous_wars_count"] >= 3
].copy()

# Model features

feature_columns = [
    "last_3_avg_fame",
    "last_3_avg_decks",
    "last_2_avg_decks",
    "last_3_avg_efficiency",
    "last_5_participation_rate",
    "last_2_participation_rate",
    "last_5_full_participation_rate",
    "recent_active_wars",
    "previous_war_fame",
    "previous_war_decks",
    "fame_trend",
    "decks_change",
    "last_3_fame_std",
    "last_3_decks_std",

    # Clan context
    "previous_clan_rank",
    "previous_clan_score",
    "last_3_avg_clan_rank",
    "last_3_avg_clan_score",
    "clan_score_change"
]

target_column = "fame"

# Remove observations that still have missing
# values in required features.
model_df = model_df.dropna(
    subset=feature_columns + [target_column]
)

# Output

print("\nML-ready dataset shape:")
print(model_df.shape)

print("\nML features:")
print(
    model_df[
        feature_columns
    ].head()
)

print("\nTarget:")
print(
    model_df[
        target_column
    ].head()
)

model_df.to_csv(
    "data/model_dataset.csv",
    index=False
)

print(
    "\nSaved ML dataset to "
    "data/model_dataset.csv"
)
