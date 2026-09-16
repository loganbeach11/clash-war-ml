CREATE DATABASE IF NOT EXISTS clash_war_ml;

USE clash_war_ml;

CREATE TABLE IF NOT EXISTS players (
    player_tag VARCHAR(20) PRIMARY KEY,
    player_name VARCHAR(100) NOT NULL,
    current_clan_tag VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS wars (
    war_id VARCHAR(100) PRIMARY KEY,
    season_id INT NOT NULL,
    section_index INT NOT NULL,
    race_date DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS clans (
    clan_tag VARCHAR(20) PRIMARY KEY,
    clan_name VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS player_war_performance (
    player_tag VARCHAR(20) NOT NULL,
    war_id VARCHAR(100) NOT NULL,
    clan_tag VARCHAR(20) NOT NULL,
    fame INT NOT NULL,
    decks_used INT NOT NULL,
    repair_points INT NOT NULL,
    boat_attacks INT NOT NULL,

    PRIMARY KEY (
        player_tag,
        war_id,
        clan_tag
    ),

    FOREIGN KEY (player_tag)
        REFERENCES players(player_tag),

    FOREIGN KEY (war_id)
        REFERENCES wars(war_id),

    FOREIGN KEY (clan_tag)
        REFERENCES clans(clan_tag)
);

CREATE TABLE IF NOT EXISTS clan_war_results (
    war_id VARCHAR(100) NOT NULL,
    clan_tag VARCHAR(20) NOT NULL,
    rank_position INT NOT NULL,
    trophy_change INT,
    fame INT,
    repair_points INT,
    clan_score INT,

    PRIMARY KEY (war_id, clan_tag),

    FOREIGN KEY (war_id)
        REFERENCES wars(war_id),

    FOREIGN KEY (clan_tag)
        REFERENCES clans(clan_tag)
);
CREATE TABLE IF NOT EXISTS live_clan_members (
    live_race_id VARCHAR(100) NOT NULL,
    clan_tag VARCHAR(20) NOT NULL,
    player_tag VARCHAR(20) NOT NULL,
    player_name VARCHAR(100),

    first_seen_at DATETIME NOT NULL,
    last_seen_at DATETIME NOT NULL,

    currently_member BOOLEAN NOT NULL DEFAULT TRUE,

    PRIMARY KEY (
        live_race_id,
        clan_tag,
        player_tag
    ),

    FOREIGN KEY (live_race_id)
        REFERENCES live_races(live_race_id),

    FOREIGN KEY (clan_tag)
        REFERENCES clans(clan_tag)
);