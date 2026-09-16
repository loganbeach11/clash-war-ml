import os
import requests
import pandas as pd
import mysql.connector

from dotenv import load_dotenv


# ================================================================
# CONFIGURATION
# ================================================================

load_dotenv()

API_TOKEN = os.getenv("CLASH_ROYALE_API_TOKEN")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

BASE_URL = "https://api.clashroyale.com/v1"


headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Galactic Kings is the source clan used to retrieve
# River Race information.
# Galactic Kings remains the source for the current live race.
SOURCE_CLAN_TAG = "#Y9202P9U"

encoded_tag = SOURCE_CLAN_TAG.replace(
    "#",
    "%23"
)

# Use every clan in the current race to expand
# our historical training data.
HISTORY_SOURCE_CLAN_TAGS = [
    "#Y9202P9U",  # Galactic Kings
    "#YV8JLPC2",  # ILLUMINATI 2.0
    "#LUG9LVC0",  # 37XU
    "#9VLJRY2J",  # Gentleman Jack
    "#J29J90P"    # isao
]


# ================================================================
# GET HISTORICAL RIVER RACE DATA
# ================================================================

all_races = {}

for history_clan_tag in HISTORY_SOURCE_CLAN_TAGS:

    history_encoded_tag = (
        history_clan_tag.replace(
            "#",
            "%23"
        )
    )

    river_log_url = (
        f"{BASE_URL}/clans/"
        f"{history_encoded_tag}/riverracelog"
    )

    river_log_response = requests.get(
        river_log_url,
        headers=headers
    )

    print(
        "River Race Log Status",
        history_clan_tag,
        ":",
        river_log_response.status_code
    )

    if river_log_response.status_code != 200:

        print(
            "River Race Log Response:",
            river_log_response.json()
        )

        river_log_response.raise_for_status()

    river_log_data = (
        river_log_response.json()
    )

    clan_races = river_log_data.get(
        "items",
        []
    )

    print(
        "Historical wars returned for",
        history_clan_tag,
        ":",
        len(clan_races)
    )

    # ------------------------------------------------------------
    # DEDUPLICATE OVERLAPPING RACES
    # ------------------------------------------------------------

    for race in clan_races:

        race_key = (
            race["seasonId"],
            race["sectionIndex"],
            race["createdDate"]
        )

        # Several source clans may have participated
        # in the same race. Store that race only once.
        if race_key not in all_races:
            all_races[race_key] = race


races = list(
    all_races.values()
)

print(
    "\nUnique historical wars returned:",
    len(races)
)


# ================================================================
# GET CURRENT RIVER RACE DATA
# ================================================================

current_race_url = (
    f"{BASE_URL}/clans/"
    f"{encoded_tag}/currentriverrace"
)

current_race_response = requests.get(
    current_race_url,
    headers=headers
)

print(
    "Current River Race Status:",
    current_race_response.status_code
)

current_race_data = None

if current_race_response.status_code == 200:

    current_race_data = (
        current_race_response.json()
    )

    print(
        "Current period type:",
        current_race_data.get(
            "periodType"
        )
    )

    print(
        "Current period index:",
        current_race_data.get(
            "periodIndex"
        )
    )

    print(
        "Current section index:",
        current_race_data.get(
            "sectionIndex"
        )
    )

    print(
        "Period logs returned:",
        len(
            current_race_data.get(
                "periodLogs",
                []
            )
        )
    )

elif current_race_response.status_code == 404:

    # The API can temporarily return 404 when there is
    # no active River Race.
    print(
        "No active current River Race."
    )

else:

    print(
        "Current River Race Response:",
        current_race_response.json()
    )

    current_race_response.raise_for_status()


# ================================================================
# SORT HISTORICAL RACES
# ================================================================

# Oldest -> newest.
#
# This also means the most recently observed clan
# for a player becomes current_clan_tag.
races = sorted(
    races,
    key=lambda race: race["createdDate"]
)


# ================================================================
# BUILD RECORD LISTS
# ================================================================

war_records = []
clan_records = []
clan_war_records = []
player_records = []
performance_records = []
period_records = []
live_player_records = []
live_member_records = []


# ================================================================
# PROCESS HISTORICAL RIVER RACES
# ================================================================

for race in races:

    race_date_raw = race["createdDate"]
    season_id = race["seasonId"]
    section_index = race["sectionIndex"]

    war_id = (
        f"{season_id}-"
        f"{section_index}-"
        f"{race_date_raw}"
    )

    race_date = pd.to_datetime(
        race_date_raw
    ).to_pydatetime()

    # ------------------------------------------------------------
    # WAR RECORD
    # ------------------------------------------------------------

    war_records.append(
        {
            "war_id": war_id,
            "season_id": season_id,
            "section_index": section_index,
            "race_date": race_date
        }
    )

    # ------------------------------------------------------------
    # EVERY CLAN IN THIS WAR
    # ------------------------------------------------------------

    for standing in race["standings"]:

        clan = standing["clan"]

        clan_tag = clan["tag"]
        clan_name = clan["name"]

        # Store clan.
        clan_records.append(
            {
                "clan_tag": clan_tag,
                "clan_name": clan_name
            }
        )

        # Store clan-level war result.
        clan_war_records.append(
            {
                "war_id": war_id,
                "clan_tag": clan_tag,
                "rank_position": standing[
                    "rank"
                ],
                "trophy_change": standing.get(
                    "trophyChange"
                ),
                "fame": clan.get(
                    "fame",
                    0
                ),
                "repair_points": clan.get(
                    "repairPoints",
                    0
                ),
                "clan_score": clan.get(
                    "clanScore"
                )
            }
        )

        # --------------------------------------------------------
        # EVERY PLAYER IN THIS CLAN
        # --------------------------------------------------------

        participants = clan.get(
            "participants",
            []
        )

        for player in participants:

            player_records.append(
                {
                    "player_tag": player[
                        "tag"
                    ],
                    "player_name": player[
                        "name"
                    ],
                    "clan_tag": clan_tag
                }
            )

            performance_records.append(
                {
                    "player_tag": player[
                        "tag"
                    ],
                    "war_id": war_id,
                    "clan_tag": clan_tag,
                    "fame": player.get(
                        "fame",
                        0
                    ),
                    "decks_used": player.get(
                        "decksUsed",
                        0
                    ),
                    "repair_points": player.get(
                        "repairPoints",
                        0
                    ),
                    "boat_attacks": player.get(
                        "boatAttacks",
                        0
                    )
                }
            )


# ================================================================
# ADD CLANS FROM THE LIVE RIVER RACE
# ================================================================

if current_race_data is not None:

    live_clans = current_race_data.get(
        "clans",
        []
    )

    source_live_clan = current_race_data.get(
        "clan"
    )

    if source_live_clan:

        source_tag = source_live_clan.get(
            "tag"
        )

        already_in_clans = any(
            clan.get("tag") == source_tag
            for clan in live_clans
        )

        if not already_in_clans:
            live_clans.append(
                source_live_clan
            )

    for clan in live_clans:

        clan_tag = clan.get("tag")
        clan_name = clan.get("name")

        if (
            clan_tag
            and clan_name
        ):
            clan_records.append(
                {
                    "clan_tag": clan_tag,
                    "clan_name": clan_name
                }
            )


# ================================================================
# PROCESS CURRENT RIVER RACE PERIOD LOGS
# ================================================================

live_race_record = None

if current_race_data is not None:

    current_section_index = (
        current_race_data.get(
            "sectionIndex"
        )
    )

    current_period_index = (
        current_race_data.get(
            "periodIndex"
        )
    )

    current_period_type = (
        current_race_data.get(
            "periodType"
        )
    )

    period_logs = (
        current_race_data.get(
            "periodLogs",
            []
        )
    )

    # ------------------------------------------------------------
    # CREATE A STABLE LIVE-RACE ID
    # ------------------------------------------------------------

    # periodIndex increases continuously through the season.
    # Dividing by 7 gives the week bucket.
    #
    # Include the source clan tag so this remains unique
    # if we later collect live races from multiple clans.

    clean_source_tag = (
        SOURCE_CLAN_TAG.replace(
            "#",
            ""
        )
    )

    now = pd.Timestamp.now()

    week_start = (
        now.normalize()
        - pd.Timedelta(
            days=now.weekday()
        )
    )

    week_start_str = (
        week_start.strftime(
            "%Y%m%d"
        )
    )

    live_race_id = (
        f"{clean_source_tag}-"
        f"{week_start_str}-"
        f"section-{current_section_index}"
    )

    current_time = (
        pd.Timestamp.now()
        .to_pydatetime()
    )

    live_race_record = {
        "live_race_id": live_race_id,
        "section_index": current_section_index,
        "period_index": current_period_index,
        "period_type": current_period_type,
        "first_seen_at": current_time,
        "last_seen_at": current_time
    }

    print(
        "\nLive race ID:",
        live_race_id
    )
    # ================================================================
    # GET CURRENT CLAN ROSTERS
    # ================================================================

    for clan in live_clans:

        clan_tag = clan.get("tag")

        if not clan_tag:
            continue

        encoded_clan_tag = clan_tag.replace(
            "#",
            "%23"
        )

        clan_url = (
            f"{BASE_URL}/clans/"
            f"{encoded_clan_tag}"
        )

        clan_response = requests.get(
            clan_url,
            headers=headers
        )

        print(
            "Clan roster status",
            clan_tag,
            ":",
            clan_response.status_code
        )

        if clan_response.status_code != 200:
            continue

        clan_data = clan_response.json()

        members = clan_data.get(
            "memberList",
            []
        )

        print(
            "Current roster members",
            clan_tag,
            ":",
            len(members)
        )

        for member in members:

            live_member_records.append(
                {
                    "live_race_id": live_race_id,
                    "clan_tag": clan_tag,
                    "player_tag": member.get(
                        "tag"
                    ),
                    "player_name": member.get(
                        "name"
                    ),
                    "first_seen_at": current_time,
                    "last_seen_at": current_time,
                    "currently_member": True
                }
            )

    
    # ------------------------------------------------------------
    # BUILD LIVE PLAYER STATUS RECORDS
    # ------------------------------------------------------------

    for clan in live_clans:

        clan_tag = clan.get(
            "tag"
        )

        participants = clan.get(
            "participants",
            []
        )

        for player in participants:

            player_tag = player.get(
                "tag"
            )

            if not player_tag:
                continue

            live_player_records.append(
                {
                    "live_race_id": live_race_id,
                    "player_tag": player_tag,
                    "clan_tag": clan_tag,
                    "player_name": player.get(
                        "name"
                    ),
                    "fame": player.get(
                        "fame",
                        0
                    ),
                    "decks_used": player.get(
                        "decksUsed",
                        0
                    ),
                    "decks_used_today": player.get(
                        "decksUsedToday",
                        0
                    ),
                    "boat_attacks": player.get(
                        "boatAttacks",
                        0
                    ),
                    "last_seen_at": current_time
                }
            )

    print(
        "Live player records:",
        len(live_player_records)
    )
    # ------------------------------------------------------------
    # BUILD PERIOD RECORDS
    # ------------------------------------------------------------

    for period_log in period_logs:

        period_index = (
            period_log.get(
                "periodIndex"
            )
        )

        items = (
            period_log.get(
                "items",
                []
            )
        )

        for item in items:

            clan_info = (
                item.get(
                    "clan",
                    {}
                )
            )

            clan_tag = (
                clan_info.get(
                    "tag"
                )
            )

            clan_name = (
                clan_info.get(
                    "name"
                )
            )

            if not clan_tag:
                continue

            # Make sure every live-period clan
            # exists in the clans table.
            if clan_name:
                clan_records.append(
                    {
                        "clan_tag": clan_tag,
                        "clan_name": clan_name
                    }
                )

            period_records.append(
                {
                    "live_race_id": (
                        live_race_id
                    ),
                    "clan_tag": (
                        clan_tag
                    ),
                    "period_index": (
                        period_index
                    ),
                    "points_earned": (
                        item.get(
                            "pointsEarned"
                        )
                    ),
                    "progress_start_of_day": (
                        item.get(
                            "progressStartOfDay"
                        )
                    ),
                    "progress_end_of_day": (
                        item.get(
                            "progressEndOfDay"
                        )
                    ),
                    "end_of_day_rank": (
                        item.get(
                            "endOfDayRank"
                        )
                    ),
                    "progress_earned": (
                        item.get(
                            "progressEarned"
                        )
                    ),
                    "defenses_remaining": (
                        item.get(
                            "numOfDefensesRemaining"
                        )
                    ),
                    "progress_earned_from_defenses": (
                        item.get(
                            "progressEarnedFromDefenses"
                        )
                    )
                }
            )


# ================================================================
# BASIC DATA SUMMARY
# ================================================================

print(
    "\nWar records:",
    len(war_records)
)

print(
    "Clan-war records:",
    len(clan_war_records)
)

print(
    "Player-war performance records:",
    len(performance_records)
)

print(
    "Period-log records:",
    len(period_records)
)


# ================================================================
# DUPLICATE PLAYER-WAR CHECK
# ================================================================

performance_df = pd.DataFrame(
    performance_records
)

duplicates = performance_df[
    performance_df.duplicated(
        subset=[
            "player_tag",
            "war_id"
        ],
        keep=False
    )
].sort_values(
    [
        "war_id",
        "player_tag"
    ]
)

print(
    "\nDuplicate player-war records:"
)

if len(duplicates) > 0:

    print(
        duplicates[
            [
                "player_tag",
                "war_id",
                "clan_tag",
                "fame",
                "decks_used"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "None"
    )

print(
    "\nNumber of duplicate rows:",
    len(duplicates)
)

print(
    "\nUnique players:",
    performance_df[
        "player_tag"
    ].nunique()
)

print(
    "Unique clans:",
    performance_df[
        "clan_tag"
    ].nunique()
)

print(
    "\nPerformance DataFrame shape:",
    performance_df.shape
)


# ================================================================
# CONNECT TO MYSQL
# ================================================================

db = mysql.connector.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)

cursor = db.cursor()

print(
    "\nConnected to MySQL successfully."
)


# ================================================================
# 1. INSERT WARS
# ================================================================

war_sql = """
INSERT INTO wars (
    war_id,
    season_id,
    section_index,
    race_date
)
VALUES (%s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    season_id = VALUES(season_id),
    section_index = VALUES(section_index),
    race_date = VALUES(race_date)
"""

for record in war_records:

    cursor.execute(
        war_sql,
        (
            record["war_id"],
            record["season_id"],
            record["section_index"],
            record["race_date"]
        )
    )


# ================================================================
# 2. INSERT / UPDATE CLANS
# ================================================================

clan_sql = """
INSERT INTO clans (
    clan_tag,
    clan_name
)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE
    clan_name = VALUES(clan_name)
"""

for record in clan_records:

    cursor.execute(
        clan_sql,
        (
            record["clan_tag"],
            record["clan_name"]
        )
    )

# ================================================================
# 3. INSERT / UPDATE LIVE RACE
# ================================================================

if live_race_record is not None:

    live_race_sql = """
    INSERT INTO live_races (
        live_race_id,
        section_index,
        period_index,
        period_type,
        first_seen_at,
        last_seen_at
    )
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        section_index =
            VALUES(section_index),
        period_index =
            VALUES(period_index),
        period_type =
            VALUES(period_type),
        last_seen_at =
            VALUES(last_seen_at)
    """

    cursor.execute(
        live_race_sql,
        (
            live_race_record[
                "live_race_id"
            ],
            live_race_record[
                "section_index"
            ],
            live_race_record[
                "period_index"
            ],
            live_race_record[
                "period_type"
            ],
            live_race_record[
                "first_seen_at"
            ],
            live_race_record[
                "last_seen_at"
            ]
        )
    )

# ================================================================
# 4. INSERT / UPDATE CURRENT CLAN MEMBERS
# ================================================================

if live_race_record is not None:

    # Mark everyone previously seen in this live race as not current.
    # Current members fetched above will be flipped back to TRUE below.
    cursor.execute(
        """
        UPDATE live_clan_members
        SET currently_member = FALSE
        WHERE live_race_id = %s
        """,
        (
            live_race_id,
        )
    )

    live_member_sql = """
    INSERT INTO live_clan_members (
        live_race_id,
        clan_tag,
        player_tag,
        player_name,
        first_seen_at,
        last_seen_at,
        currently_member
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s
    )
    ON DUPLICATE KEY UPDATE
        player_name = VALUES(player_name),
        last_seen_at = VALUES(last_seen_at),
        currently_member = TRUE
    """

    for record in live_member_records:

        cursor.execute(
            live_member_sql,
            (
                record["live_race_id"],
                record["clan_tag"],
                record["player_tag"],
                record["player_name"],
                record["first_seen_at"],
                record["last_seen_at"],
                record["currently_member"]
            )
        )


# ================================================================
# 5. INSERT / UPDATE LIVE PLAYER WAR STATUS
# ================================================================


live_player_sql = """
INSERT INTO live_player_war_status (
    live_race_id,
    player_tag,
    clan_tag,
    player_name,
    fame,
    decks_used,
    decks_used_today,
    boat_attacks,
    last_seen_at
)
VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    player_name =
        VALUES(player_name),
    fame =
        VALUES(fame),
    decks_used =
        VALUES(decks_used),
    decks_used_today =
        VALUES(decks_used_today),
    boat_attacks =
        VALUES(boat_attacks),
    last_seen_at =
        VALUES(last_seen_at)
"""

for record in live_player_records:

    cursor.execute(
        live_player_sql,
        (
            record["live_race_id"],
            record["player_tag"],
            record["clan_tag"],
            record["player_name"],
            record["fame"],
            record["decks_used"],
            record["decks_used_today"],
            record["boat_attacks"],
            record["last_seen_at"]
        )
    )

# ================================================================
# 6. INSERT / UPDATE PLAYERS
# ================================================================

player_sql = """
INSERT INTO players (
    player_tag,
    player_name,
    current_clan_tag
)
VALUES (%s, %s, %s)
ON DUPLICATE KEY UPDATE
    player_name = VALUES(player_name),
    current_clan_tag = VALUES(current_clan_tag)
"""

for record in player_records:

    cursor.execute(
        player_sql,
        (
            record["player_tag"],
            record["player_name"],
            record["clan_tag"]
        )
    )


# ================================================================
# 7. INSERT CLAN WAR RESULTS
# ================================================================

clan_war_sql = """
INSERT INTO clan_war_results (
    war_id,
    clan_tag,
    rank_position,
    trophy_change,
    fame,
    repair_points,
    clan_score
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    rank_position = VALUES(rank_position),
    trophy_change = VALUES(trophy_change),
    fame = VALUES(fame),
    repair_points = VALUES(repair_points),
    clan_score = VALUES(clan_score)
"""

for record in clan_war_records:

    cursor.execute(
        clan_war_sql,
        (
            record["war_id"],
            record["clan_tag"],
            record["rank_position"],
            record["trophy_change"],
            record["fame"],
            record["repair_points"],
            record["clan_score"]
        )
    )


# ================================================================
# 8. INSERT PLAYER WAR PERFORMANCE
# ================================================================

performance_sql = """
INSERT INTO player_war_performance (
    player_tag,
    war_id,
    clan_tag,
    fame,
    decks_used,
    repair_points,
    boat_attacks
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    fame = VALUES(fame),
    decks_used = VALUES(decks_used),
    repair_points = VALUES(repair_points),
    boat_attacks = VALUES(boat_attacks)
"""

for record in performance_records:

    cursor.execute(
        performance_sql,
        (
            record["player_tag"],
            record["war_id"],
            record["clan_tag"],
            record["fame"],
            record["decks_used"],
            record["repair_points"],
            record["boat_attacks"]
        )
    )


# ================================================================
# 9. INSERT CURRENT RIVER RACE PERIOD RESULTS
# ================================================================

period_sql = """
INSERT INTO clan_war_period_results (
    live_race_id,
    clan_tag,
    period_index,
    points_earned,
    progress_start_of_day,
    progress_end_of_day,
    end_of_day_rank,
    progress_earned,
    defenses_remaining,
    progress_earned_from_defenses
)
VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    points_earned =
        VALUES(points_earned),
    progress_start_of_day =
        VALUES(progress_start_of_day),
    progress_end_of_day =
        VALUES(progress_end_of_day),
    end_of_day_rank =
        VALUES(end_of_day_rank),
    progress_earned =
        VALUES(progress_earned),
    defenses_remaining =
        VALUES(defenses_remaining),
    progress_earned_from_defenses =
        VALUES(progress_earned_from_defenses)
"""

for record in period_records:

    cursor.execute(
        period_sql,
        (
            record["live_race_id"],
            record["clan_tag"],
            record["period_index"],
            record["points_earned"],
            record[
                "progress_start_of_day"
            ],
            record[
                "progress_end_of_day"
            ],
            record[
                "end_of_day_rank"
            ],
            record[
                "progress_earned"
            ],
            record[
                "defenses_remaining"
            ],
            record[
                "progress_earned_from_defenses"
            ]
        )
    )


# ================================================================
# SAVE DATABASE CHANGES
# ================================================================

db.commit()

print(
    "\nDatabase load complete."
)


# ================================================================
# VERIFY DATABASE COUNTS
# ================================================================

cursor.execute(
    "SELECT COUNT(*) FROM players"
)

player_count = cursor.fetchone()[0]


cursor.execute(
    "SELECT COUNT(*) FROM clans"
)

clan_count = cursor.fetchone()[0]


cursor.execute(
    "SELECT COUNT(*) FROM wars"
)

war_count = cursor.fetchone()[0]


cursor.execute(
    """
    SELECT COUNT(*)
    FROM clan_war_results
    """
)

clan_war_count = cursor.fetchone()[0]


cursor.execute(
    """
    SELECT COUNT(*)
    FROM player_war_performance
    """
)

performance_count = cursor.fetchone()[0]
cursor.execute(
    """
    SELECT COUNT(*)
    FROM live_player_war_status
    """
)

live_player_count = cursor.fetchone()[0]

cursor.execute(
    """
    SELECT COUNT(*)
    FROM live_clan_members
    """
)

live_member_count = cursor.fetchone()[0]
cursor.execute(
    """
    SELECT COUNT(*)
    FROM live_races
    """
)

live_race_count = cursor.fetchone()[0]

cursor.execute(
    """
    SELECT COUNT(*)
    FROM clan_war_period_results
    """
)

period_count = cursor.fetchone()[0]


print(
    "\nDATABASE COUNTS"
)

print(
    "Players stored:",
    player_count
)

print(
    "Clans stored:",
    clan_count
)

print(
    "Wars stored:",
    war_count
)

print(
    "Clan-war results stored:",
    clan_war_count
)

print(
    "Player performance records stored:",
    performance_count
)

print(
    "Period-log records stored:",
    period_count
)
print(
    "Live races stored:",
    live_race_count
)
print(
    "Live player status records stored:",
    live_player_count
)

print(
    "Live clan-member records stored:",
    live_member_count
)

# ================================================================
# CLOSE CONNECTION
# ================================================================

cursor.close()
db.close()

print(
    "\nMySQL connection closed."
)