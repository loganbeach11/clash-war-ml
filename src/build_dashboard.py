import os
from datetime import datetime

import pandas as pd


# ================================================================
# CONFIGURATION
# ================================================================

OUTPUT_FILE = "reports/gk_war_dashboard.html"

FILES = {
    "war_projection":
        "data/gk_live_war_projection.csv",

    "day_projection":
        "data/gk_live_war_day_projection.csv",

    "player_predictions":
        "data/gk_live_player_predictions.csv",

    "player_summary":
        "data/gk_live_player_summary.csv",

    "archetype_summary":
        "data/gk_player_archetype_summary.csv",

    "rotation_summary":
        "data/gk_rotation_summary.csv",

    "active_lineup":
        "data/gk_rotation_active_lineup.csv",

    "backups":
        "data/gk_rotation_backups.csv",
}


# ================================================================
# HELPERS
# ================================================================

def load_csv(path):

    if not os.path.exists(path):

        print(
            f"Warning: missing {path}"
        )

        return pd.DataFrame()

    return pd.read_csv(
        path
    )


def fmt_number(
    value,
    decimals=0,
):

    if pd.isna(
        value
    ):

        return "-"

    return f"{float(value):,.{decimals}f}"


def fmt_percent(
    value,
    decimals=0,
):

    if pd.isna(
        value
    ):

        return "-"

    return (
        f"{float(value) * 100:.{decimals}f}%"
    )


def fmt_finish_day(
    value,
):

    if pd.isna(
        value
    ):

        return "Unknown"

    return (
        f"Battle Day {int(value)}"
    )


def yes_no(
    value,
):

    if pd.isna(
        value
    ):

        return "-"

    if isinstance(
        value,
        str,
    ):

        normalized = (
            value
            .strip()
            .lower()
        )

        return (
            "Yes"
            if normalized in [
                "true",
                "1",
                "yes",
            ]
            else "No"
        )

    try:

        return (
            "Yes"
            if int(value) == 1
            else "No"
        )

    except Exception:

        return "No"


def dataframe_to_html(
    df,
    columns,
    rename=None,
):

    if df.empty:

        return (
            '<div class="empty">'
            "No data available."
            "</div>"
        )

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    if not available_columns:

        return (
            '<div class="empty">'
            "No matching display columns available."
            "</div>"
        )

    display = df[
        available_columns
    ].copy()

    if rename:

        display = display.rename(
            columns=rename
        )

    return display.to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=False,
    )


# ================================================================
# LOAD DATA
# ================================================================

war_projection = load_csv(
    FILES[
        "war_projection"
    ]
)

day_projection = load_csv(
    FILES[
        "day_projection"
    ]
)

player_predictions = load_csv(
    FILES[
        "player_predictions"
    ]
)

player_summary = load_csv(
    FILES[
        "player_summary"
    ]
)

archetype_summary = load_csv(
    FILES[
        "archetype_summary"
    ]
)

rotation_summary = load_csv(
    FILES[
        "rotation_summary"
    ]
)

active_lineup = load_csv(
    FILES[
        "active_lineup"
    ]
)

backups = load_csv(
    FILES[
        "backups"
    ]
)


# ================================================================
# WAR MODE
# ================================================================

war_mode = "unknown"

if (
    not war_projection.empty
    and "war_mode"
    in war_projection.columns
):

    war_mode = str(
        war_projection.iloc[
            0
        ][
            "war_mode"
        ]
    ).strip().lower()


is_colosseum = (
    war_mode
    == "colosseum"
)


mode_label = (
    "Colosseum"
    if is_colosseum
    else "River Race"
)


# ================================================================
# WAR SUMMARY VALUES
# ================================================================

war_row = (
    war_projection.iloc[
        0
    ]
    if not war_projection.empty
    else pd.Series(
        dtype=object
    )
)


current_battle_day = (
    int(
        war_row[
            "current_battle_day"
        ]
    )
    if (
        "current_battle_day"
        in war_row.index
        and pd.notna(
            war_row[
                "current_battle_day"
            ]
        )
    )
    else 0
)


projected_finish_day = (
    fmt_finish_day(
        war_row[
            "projected_finish_day"
        ]
    )
    if (
        "projected_finish_day"
        in war_row.index
        and not is_colosseum
    )
    else "-"
)


projected_river_progress = (
    (
        fmt_number(
            war_row[
                "projected_river_progress"
            ],
            0,
        )
        + " / 10,000"
    )
    if (
        "projected_river_progress"
        in war_row.index
        and not is_colosseum
    )
    else "-"
)


projected_total_medals = (
    fmt_number(
        war_row[
            "projected_total_medals"
        ],
        0,
    )
    if (
        "projected_total_medals"
        in war_row.index
    )
    else "-"
)


finish_status = (
    str(
        war_row[
            "finish_status"
        ]
    )
    if (
        "finish_status"
        in war_row.index
    )
    else "-"
)


historical_day3_finish_rate = (
    fmt_percent(
        war_row[
            "historical_day3_finish_rate"
        ],
        0,
    )
    if (
        "historical_day3_finish_rate"
        in war_row.index
    )
    else "-"
)


historical_avg_finish_day = (
    fmt_number(
        war_row[
            "historical_avg_finish_day"
        ],
        2,
    )
    if (
        "historical_avg_finish_day"
        in war_row.index
    )
    else "-"
)


historical_races_available = (
    fmt_number(
        war_row[
            "historical_races_available"
        ],
        0,
    )
    if (
        "historical_races_available"
        in war_row.index
    )
    else "-"
)


# ================================================================
# PLAYER SUMMARY VALUES
# ================================================================

player_row = (
    player_summary.iloc[
        0
    ]
    if not player_summary.empty
    else pd.Series(
        dtype=object
    )
)


rotation_pool_players = (
    int(
        player_row[
            "rotation_pool_players"
        ]
    )
    if (
        "rotation_pool_players"
        in player_row.index
        and pd.notna(
            player_row[
                "rotation_pool_players"
            ]
        )
    )
    else len(
        player_predictions
    )
)


current_members = (
    int(
        player_row[
            "current_members"
        ]
    )
    if (
        "current_members"
        in player_row.index
        and pd.notna(
            player_row[
                "current_members"
            ]
        )
    )
    else 0
)


players_outside_clan = (
    int(
        player_row[
            "players_outside_clan"
        ]
    )
    if (
        "players_outside_clan"
        in player_row.index
        and pd.notna(
            player_row[
                "players_outside_clan"
            ]
        )
    )
    else (
        rotation_pool_players
        - current_members
    )
)


expected_daily_participants = (
    fmt_number(
        player_row[
            "expected_daily_participants"
        ],
        0,
    )
    if (
        "expected_daily_participants"
        in player_row.index
    )
    else "-"
)


ml_coverage = (
    fmt_percent(
        player_row[
            "ml_coverage"
        ],
        0,
    )
    if (
        "ml_coverage"
        in player_row.index
    )
    else "-"
)


projected_daily_fame = (
    fmt_number(
        player_row[
            "projected_daily_fame"
        ],
        0,
    )
    if (
        "projected_daily_fame"
        in player_row.index
    )
    else "-"
)


# ================================================================
# ROTATION SUMMARY VALUES
# ================================================================

rotation_row = (
    rotation_summary.iloc[
        0
    ]
    if not rotation_summary.empty
    else pd.Series(
        dtype=object
    )
)


confirmed_active_daily_fame = (
    fmt_number(
        rotation_row[
            "confirmed_active_daily_fame"
        ],
        0,
    )
    if (
        "confirmed_active_daily_fame"
        in rotation_row.index
    )
    else "-"
)


confirmed_active_outside_clan = (
    int(
        rotation_row[
            "confirmed_active_outside_clan"
        ]
    )
    if (
        "confirmed_active_outside_clan"
        in rotation_row.index
        and pd.notna(
            rotation_row[
                "confirmed_active_outside_clan"
            ]
        )
    )
    else 0
)


# ================================================================
# FORMAT DAY PROJECTION
# ================================================================

day_table = (
    day_projection.copy()
)


if not day_table.empty:

    for column in [
        "projected_medals",
        "historical_avg_medals",
        "projected_rank_movement",
        "projected_defense_movement",
        "projected_total_movement",
        "projected_progress_end",
    ]:

        if column in day_table.columns:

            day_table[
                column
            ] = day_table[
                column
            ].map(
                lambda x:
                fmt_number(
                    x,
                    0,
                )
            )


    if (
        "projected_place"
        in day_table.columns
    ):

        day_table[
            "projected_place"
        ] = day_table[
            "projected_place"
        ].map(
            lambda x:
            (
                "-"
                if pd.isna(
                    x
                )
                else str(
                    int(
                        float(
                            x
                        )
                    )
                )
            )
        )


    if (
        "finished_after_day"
        in day_table.columns
    ):

        day_table[
            "finished_after_day"
        ] = day_table[
            "finished_after_day"
        ].map(
            yes_no
        )


# ================================================================
# FORMAT ARCHETYPE SUMMARY
# ================================================================

archetype_table = (
    archetype_summary.copy()
)


if not archetype_table.empty:

    for column in [
        "avg_fame",
        "avg_decks",
        "avg_efficiency",
        "avg_wars_observed",
    ]:

        if column in archetype_table.columns:

            archetype_table[
                column
            ] = archetype_table[
                column
            ].map(
                lambda x:
                fmt_number(
                    x,
                    1,
                )
            )


    for column in [
        "participation_rate",
        "full_participation_rate",
    ]:

        if column in archetype_table.columns:

            archetype_table[
                column
            ] = archetype_table[
                column
            ].map(
                lambda x:
                fmt_percent(
                    x,
                    0,
                )
            )


# ================================================================
# TOP PLAYER PREDICTIONS
# ================================================================

top_players = (
    player_predictions.copy()
)


if not top_players.empty:

    if (
        "capacity_adjusted_fame"
        in top_players.columns
    ):

        top_players = (
            top_players
            .sort_values(
                "capacity_adjusted_fame",
                ascending=False,
            )
            .head(
                20
            )
            .copy()
        )


    for column in [
        "participation_probability",
        "daily_participation_weight",
    ]:

        if column in top_players.columns:

            top_players[
                column
            ] = top_players[
                column
            ].map(
                lambda x:
                fmt_percent(
                    x,
                    0,
                )
            )


    for column in [
        "final_predicted_fame",
        "capacity_adjusted_fame",
        "projected_daily_fame",
    ]:

        if column in top_players.columns:

            top_players[
                column
            ] = top_players[
                column
            ].map(
                lambda x:
                fmt_number(
                    x,
                    0,
                )
            )


# ================================================================
# FORMAT ACTIVE LINEUP
# ================================================================

active_table = (
    active_lineup.copy()
)


if not active_table.empty:

    if (
        "currently_member"
        in active_table.columns
    ):

        active_table[
            "currently_member"
        ] = active_table[
            "currently_member"
        ].map(
            yes_no
        )


    if (
        "participation_probability"
        in active_table.columns
    ):

        active_table[
            "participation_probability"
        ] = active_table[
            "participation_probability"
        ].map(
            lambda x:
            fmt_percent(
                x,
                0,
            )
        )


    for column in [
        "expected_daily_fame",
        "confirmed_active_daily_fame",
    ]:

        if column in active_table.columns:

            active_table[
                column
            ] = active_table[
                column
            ].map(
                lambda x:
                fmt_number(
                    x,
                    0,
                )
            )


# ================================================================
# FORMAT BACKUPS
# ================================================================

backup_table = (
    backups.copy()
)


if not backup_table.empty:

    if (
        "active_lineup_rank"
        in backup_table.columns
    ):

        backup_table = (
            backup_table
            .sort_values(
                "active_lineup_rank"
            )
            .head(
                10
            )
            .copy()
        )


    if (
        "currently_member"
        in backup_table.columns
    ):

        backup_table[
            "currently_member"
        ] = backup_table[
            "currently_member"
        ].map(
            yes_no
        )


    if (
        "participation_probability"
        in backup_table.columns
    ):

        backup_table[
            "participation_probability"
        ] = backup_table[
            "participation_probability"
        ].map(
            lambda x:
            fmt_percent(
                x,
                0,
            )
        )


    if (
        "confirmed_active_daily_fame"
        in backup_table.columns
    ):

        backup_table[
            "confirmed_active_daily_fame"
        ] = backup_table[
            "confirmed_active_daily_fame"
        ].map(
            lambda x:
            fmt_number(
                x,
                0,
            )
        )


# ================================================================
# MODE-SPECIFIC DASHBOARD TEXT
# ================================================================

if is_colosseum:

    hero_subtitle = (
        "Galactic Kings Colosseum analytics, player performance, "
        "and rotation optimization"
    )

    war_description = (
        "Current Galactic Kings Colosseum projection. "
        "Colosseum does not use the River Race 10,000-point finish line."
    )

else:

    hero_subtitle = (
        "Galactic Kings River Race analytics, player performance, "
        "and rotation optimization"
    )

    war_description = (
        "Current Galactic Kings River Race forecast using GK player "
        "performance and GK historical Battle Day behavior."
    )


# ================================================================
# MODE-SPECIFIC HTML BLOCKS
# ================================================================

if is_colosseum:

    river_cards_html = ""
    river_note_html = ""

else:

    river_cards_html = f"""
    <div class="card">

        <div class="card-label">
            Projected Finish Day
        </div>

        <div class="card-value compact">
            {projected_finish_day}
        </div>

    </div>

    <div class="card">

        <div class="card-label">
            Projected River Progress
        </div>

        <div class="card-value compact">
            {projected_river_progress}
        </div>

    </div>

    <div class="card">

        <div class="card-label">
            Historical Day-3 Finish Rate
        </div>

        <div class="card-value">
            {historical_day3_finish_rate}
        </div>

    </div>
    """

    river_note_html = f"""
    <div class="note">
        Finish status: {finish_status}
        · Historical races available: {historical_races_available}
        · Historical average finish day: {historical_avg_finish_day}
    </div>
    """


# ================================================================
# HTML
# ================================================================

generated_time = (
    datetime.now()
    .strftime(
        "%Y-%m-%d %H:%M"
    )
)


html = f"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
    Galactic Kings War Analytics
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;

    background:
        #0f172a;

    color:
        #e2e8f0;
}}

.container {{
    width:
        min(
            1450px,
            94%
        );

    margin:
        0 auto;

    padding:
        36px
        0
        70px;
}}

.hero {{
    padding:
        38px;

    border-radius:
        22px;

    background:
        linear-gradient(
            135deg,
            #172554,
            #1e293b
        );

    box-shadow:
        0
        18px
        50px
        rgba(
            0,
            0,
            0,
            0.28
        );
}}

.hero-top {{
    display:
        flex;

    justify-content:
        space-between;

    align-items:
        flex-start;

    gap:
        20px;
}}

.hero h1 {{
    margin:
        0
        0
        8px;

    font-size:
        38px;
}}

.hero p {{
    margin:
        0;

    color:
        #94a3b8;
}}

.badge {{
    display:
        inline-block;

    padding:
        8px
        13px;

    border-radius:
        999px;

    background:
        #2563eb;

    font-size:
        13px;

    font-weight:
        700;

    white-space:
        nowrap;
}}

.cards {{
    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                185px,
                1fr
            )
        );

    gap:
        16px;

    margin:
        24px
        0;
}}

.card {{
    background:
        #1e293b;

    border:
        1px
        solid
        #334155;

    border-radius:
        16px;

    padding:
        20px;
}}

.card-label {{
    color:
        #94a3b8;

    font-size:
        12px;

    text-transform:
        uppercase;

    letter-spacing:
        0.08em;
}}

.card-value {{
    margin-top:
        8px;

    font-size:
        29px;

    font-weight:
        700;
}}

.compact {{
    font-size:
        21px;
}}

.section {{
    background:
        #111827;

    border:
        1px
        solid
        #273449;

    border-radius:
        18px;

    padding:
        24px;

    margin-top:
        22px;

    box-shadow:
        0
        12px
        30px
        rgba(
            0,
            0,
            0,
            0.18
        );
}}

.section h2 {{
    margin:
        0
        0
        6px;

    font-size:
        23px;
}}

.section p {{
    margin-top:
        0;

    color:
        #94a3b8;
}}

.two-column {{
    display:
        grid;

    grid-template-columns:
        repeat(
            2,
            minmax(
                0,
                1fr
            )
        );

    gap:
        22px;
}}

.table-wrap {{
    overflow-x:
        auto;

    margin-top:
        16px;
}}

.data-table {{
    width:
        100%;

    border-collapse:
        collapse;

    font-size:
        14px;
}}

.data-table th {{
    background:
        #1e293b;

    color:
        #cbd5e1;

    text-align:
        left;

    padding:
        12px;

    border-bottom:
        1px
        solid
        #334155;

    white-space:
        nowrap;
}}

.data-table td {{
    padding:
        11px
        12px;

    border-bottom:
        1px
        solid
        #243244;

    white-space:
        nowrap;
}}

.data-table tr:hover {{
    background:
        #182235;
}}

.note {{
    margin-top:
        14px;

    padding:
        14px
        16px;

    border-radius:
        12px;

    background:
        #172033;

    color:
        #94a3b8;

    font-size:
        13px;
}}

.empty {{
    color:
        #94a3b8;

    padding:
        16px
        0;
}}

.footer {{
    margin-top:
        28px;

    text-align:
        center;

    color:
        #64748b;

    font-size:
        13px;
}}

@media (
    max-width:
        900px
) {{

    .two-column {{
        grid-template-columns:
            1fr;
    }}

    .hero-top {{
        display:
            block;
    }}

    .badge {{
        margin-top:
            16px;
    }}

    .hero h1 {{
        font-size:
            28px;
    }}
}}

</style>

</head>

<body>

<div class="container">

    <div class="hero">

        <div class="hero-top">

            <div>

                <h1>
                    Galactic Kings War Analytics
                </h1>

                <p>
                    {hero_subtitle}
                </p>

            </div>

            <div class="badge">
                {mode_label}
            </div>

        </div>

    </div>


    <div class="cards">

        <div class="card">

            <div class="card-label">
                Rotation Pool
            </div>

            <div class="card-value">
                {rotation_pool_players}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                Current Members
            </div>

            <div class="card-value">
                {current_members}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                Rotation Players Outside Clan
            </div>

            <div class="card-value">
                {players_outside_clan}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                Expected Daily Players
            </div>

            <div class="card-value">
                {expected_daily_participants}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                ML Coverage
            </div>

            <div class="card-value">
                {ml_coverage}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                Player Model Daily Output
            </div>

            <div class="card-value">
                {projected_daily_fame}
            </div>

        </div>


        <div class="card">

            <div class="card-label">
                Confirmed-Active Lineup Output
            </div>

            <div class="card-value">
                {confirmed_active_daily_fame}
            </div>

        </div>

    </div>


    <div class="section">

        <h2>
            Current {mode_label} Outlook
        </h2>

        <p>
            {war_description}
        </p>

        <div class="cards">

            <div class="card">

                <div class="card-label">
                    Current Battle Day
                </div>

                <div class="card-value">
                    {current_battle_day}
                </div>

            </div>


            <div class="card">

                <div class="card-label">
                    Projected Total Medals
                </div>

                <div class="card-value">
                    {projected_total_medals}
                </div>

            </div>

            {river_cards_html}

        </div>


        {river_note_html}

    </div>


    <div class="section">

        <h2>
            GK Battle Day Projection
        </h2>

        <p>
            Day-by-day Galactic Kings forecast compared with GK's historical
            Battle Day behavior.
        </p>

        <div class="table-wrap">

            {
                dataframe_to_html(
                    day_table,
                    (
                        [
                            "battle_day",
                            "day_status",
                            "projected_medals",
                            "historical_avg_medals",
                        ]
                        if is_colosseum
                        else [
                            "battle_day",
                            "day_status",
                            "projected_medals",
                            "historical_avg_medals",
                            "projected_place",
                            "projected_rank_movement",
                            "projected_defense_movement",
                            "projected_progress_end",
                            "finished_after_day",
                            "history_rows_used",
                        ]
                    ),
                    (
                        {
                            "battle_day":
                                "Battle Day",
                            "day_status":
                                "Status",
                            "projected_medals":
                                "Projected Medals",
                            "historical_avg_medals":
                                "Historical Avg Medals",
                        }
                        if is_colosseum
                        else {
                            "battle_day":
                                "Battle Day",
                            "day_status":
                                "Status",
                            "projected_medals":
                                "Projected Medals",
                            "historical_avg_medals":
                                "Historical Avg Medals",
                            "projected_place":
                                "Expected Place",
                            "projected_rank_movement":
                                "Rank Movement",
                            "projected_defense_movement":
                                "Defense Movement",
                            "projected_progress_end":
                                "River Position",
                            "finished_after_day":
                                "Finished",
                            "history_rows_used":
                                "History Rows",
                        }
                    ),
                )
            }

        </div>

    </div>


    <div class="two-column">

        <div class="section">

            <h2>
                GK Player Archetypes
            </h2>

            <p>
                Unsupervised behavioral clusters built only from the
                current Galactic Kings rotation pool.
            </p>

            <div class="table-wrap">

                {
                    dataframe_to_html(
                        archetype_table,
                        [
                            "archetype_name",
                            "players",
                            "avg_wars_observed",
                            "avg_fame",
                            "avg_decks",
                            "participation_rate",
                            "full_participation_rate",
                            "avg_efficiency",
                        ],
                        {
                            "archetype_name":
                                "Archetype",
                            "players":
                                "Players",
                            "avg_wars_observed":
                                "Avg Wars",
                            "avg_fame":
                                "Avg Fame",
                            "avg_decks":
                                "Avg Decks",
                            "participation_rate":
                                "Participation",
                            "full_participation_rate":
                                "Full Participation",
                            "avg_efficiency":
                                "Fame / Deck",
                        },
                    )
                }

            </div>

        </div>


        <div class="section">

            <h2>
                Top GK Player Forecasts
            </h2>

            <p>
                Highest current Galactic Kings player performance forecasts.
            </p>

            <div class="table-wrap">

                {
                    dataframe_to_html(
                        top_players,
                        [
                            "player_name",
                            "currently_member",
                            "archetype_name",
                            "prediction_tier",
                            "participation_probability",
                            "daily_participation_weight",
                            "final_predicted_fame",
                            "capacity_adjusted_fame",
                            "projected_daily_fame",
                        ],
                        {
                            "player_name":
                                "Player",
                            "currently_member":
                                "In Clan",
                            "archetype_name":
                                "Archetype",
                            "prediction_tier":
                                "Prediction Tier",
                            "participation_probability":
                                "Historical Participation",
                            "daily_participation_weight":
                                "Daily Slot Weight",
                            "final_predicted_fame":
                                "Base Forecast",
                            "capacity_adjusted_fame":
                                "Rotation-Adjusted",
                            "projected_daily_fame":
                                "Projected Daily",
                        },
                    )
                }

            </div>

            <div class="note">
                Player forecasts are model-based performance estimates.
                They are inputs to GK war planning, not direct River movement.
            </div>

        </div>

    </div>


    <div class="section">

        <h2>
            Recommended 50 — Confirmed Active
        </h2>

        <p>
            Strongest projected 50-player lineup when those players are
            confirmed available for the Battle Day.
        </p>

        <div class="note">
            {confirmed_active_outside_clan} recommended players are currently
            outside the clan and would need to rotate in.
        </div>

        <div class="table-wrap">

            {
                dataframe_to_html(
                    active_table,
                    [
                        "active_lineup_rank",
                        "player_name",
                        "currently_member",
                        "archetype_name",
                        "prediction_tier",
                        "participation_probability",
                        "expected_daily_fame",
                        "confirmed_active_daily_fame",
                    ],
                    {
                        "active_lineup_rank":
                            "Rank",
                        "player_name":
                            "Player",
                        "currently_member":
                            "Currently In Clan",
                        "archetype_name":
                            "Archetype",
                        "prediction_tier":
                            "Prediction Tier",
                        "participation_probability":
                            "Historical Participation",
                        "expected_daily_fame":
                            "Expected Daily",
                        "confirmed_active_daily_fame":
                            "Confirmed-Active Daily",
                    },
                )
            }

        </div>

    </div>


    <div class="section">

        <h2>
            Top Backups
        </h2>

        <p>
            Next Galactic Kings players in line if a recommended player
            is unavailable.
        </p>

        <div class="table-wrap">

            {
                dataframe_to_html(
                    backup_table,
                    [
                        "active_lineup_rank",
                        "player_name",
                        "currently_member",
                        "archetype_name",
                        "prediction_tier",
                        "participation_probability",
                        "confirmed_active_daily_fame",
                    ],
                    {
                        "active_lineup_rank":
                            "Overall Rank",
                        "player_name":
                            "Player",
                        "currently_member":
                            "Currently In Clan",
                        "archetype_name":
                            "Archetype",
                        "prediction_tier":
                            "Prediction Tier",
                        "participation_probability":
                            "Historical Participation",
                        "confirmed_active_daily_fame":
                            "Confirmed-Active Daily",
                    },
                )
            }

        </div>

    </div>


    <div class="footer">

        Generated {generated_time}
        · Galactic Kings Clash Royale War Analytics

    </div>

</div>

</body>

</html>
"""


# ================================================================
# SAVE REPORT
# ================================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True,
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        html
    )


print(
    "Dashboard generated:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "War mode:",
    mode_label
)

print()

print(
    "Open it with:"
)

print(
    f"open {OUTPUT_FILE}"
)
