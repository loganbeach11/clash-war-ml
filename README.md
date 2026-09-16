Galactic Kings Clan War Analytics

A machine learning and data analytics system for Galactic Kings, a competitive Clash Royale clan.

The project collects Clan War data from the Clash Royale API, stores historical and live data in MySQL, builds player-level behavioral features, predicts player participation and performance, clusters players into behavioral archetypes, recommends a 50-player Battle Day rotation, and generates a live HTML dashboard for clan leadership.

Project Goals

The system is designed to answer questions such as:

Which Galactic Kings players are the most reliable war contributors?

Which players are likely to participate in the next war?

How much production can we expect from the current rotation pool?

Which 50 players should be prioritized for a Battle Day lineup?

Which players should serve as backups?

What behavioral archetypes exist within Galactic Kings?

How does the current war compare with Galactic Kings' historical performance?

When is Galactic Kings projected to finish the River Race?

The final analytics and dashboard are intentionally focused only on Galactic Kings.

Current Pipeline

Clash Royale API
        |
        v
collect_data.py
        |
        v
MySQL
        |
        +---------------------------+
        |                           |
        v                           v
build_features.py          build_player_archetypes.py
        |                           |
        v                           v
Historical ML Dataset      GK Player Archetypes
        |                           |
        +-------------+-------------+
                      |
                      v
          build_live_player_features.py
                      |
                      v
             predict_live_players.py
                      |
                      v
        gk_live_player_predictions.csv
                      |
          +-----------+------------+
          |                        |
          v                        v
build_daily_clan_features.py   optimize_gk_rotation.py
          |                        |
          v                        v
    GK War History          Recommended 50 + Backups
          |                        |
          +-----------+------------+
                      |
                      v
              predict_live_war.py
                      |
                      v
              build_dashboard.py
                      |
                      v
         reports/gk_war_dashboard.html

Main Features

Live Galactic Kings Player Modeling

The live player pipeline builds one feature row for every player observed in the current Galactic Kings war rotation pool.

Players can remain part of the rotation pool even if they are temporarily outside the clan, which is important because Galactic Kings rotates war-only players in and out to fill the available Battle Day participant slots.

Current player prediction tiers:

ml_model — enough historical data for the full model

limited_history — one or two historical wars

clan_baseline — insufficient individual history

Participation Model

A Random Forest classifier estimates each player's probability of participating.

That probability is also used to normalize expected participation across the current rotation pool so the system reflects the approximately 50 available Battle Day participant slots.

Player Performance Forecast

The main player performance model uses historical features such as:

recent average Fame

recent average decks used

recent efficiency

historical participation rate

full-participation rate

previous-war Fame

previous-war decks

recent Fame trend

recent Fame variance

recent deck-use variance

Galactic Kings historical clan score context

The current live forecast combines the primary regression estimate with a confirmed-active estimate when available.

Player Archetypes

K-Means clustering is used to identify behavioral groups among Galactic Kings players.

The number of clusters is selected by comparing silhouette scores rather than fixing a cluster count in advance.

Current examples include:

Reliable High Performer

Inconsistent Contributor

Low-Participation Player

Cluster IDs are not hard-coded to names. Archetype names are assigned from the actual behavior of each cluster after clustering.

Rotation Optimization

The system produces two GK lineup rankings:

Expected-Value Lineup

Useful when player availability is uncertain.

Confirmed-Active Lineup

Useful when leadership knows which players are available for that Battle Day.

The optimizer outputs:

recommended 50-player lineup

ranked backups

expected production

confirmed-active production

current-member vs. rotated-out counts

Galactic Kings War Forecast

The war forecast uses only Galactic Kings data.

It does not attempt to model or rank opponent clans.

For River Race weeks, the forecast can use Galactic Kings' historical Battle Day behavior to estimate:

projected daily Medals

expected placement context

rank movement

defense movement

River progress

projected finish day

Colosseum weeks are handled separately because they do not use the normal 10,000-point River Race finish line.

Dashboard

The project generates a GK-only dashboard at:

reports/gk_war_dashboard.html

The dashboard includes:

rotation-pool size

current clan members

players currently outside the clan

expected daily participants

ML coverage

player-model projected production

current River Race / Colosseum outlook

projected finish day

historical Day-3 finish rate

day-by-day GK forecast

GK player archetypes

top player forecasts

recommended 50-player lineup

top backups

Generate and open it with:

python src/build_dashboard.py
open reports/gk_war_dashboard.html

Project Structure

clash-war-ml/
|
├── data/
|   ├── model_dataset.csv
|   ├── gk_player_archetypes.csv
|   ├── gk_player_archetype_summary.csv
|   ├── gk_live_player_features.csv
|   ├── gk_live_player_predictions.csv
|   ├── gk_live_player_summary.csv
|   ├── gk_daily_war_dataset.csv
|   ├── gk_race_summary.csv
|   ├── gk_live_war_projection.csv
|   ├── gk_live_war_day_projection.csv
|   ├── gk_rotation_active_lineup.csv
|   ├── gk_rotation_backups.csv
|   └── gk_rotation_summary.csv
|
├── reports/
|   └── gk_war_dashboard.html
|
├── sql/
|   └── schema.sql
|
├── src/
|   ├── collect_data.py
|   ├── build_features.py
|   ├── build_player_archetypes.py
|   ├── build_live_player_features.py
|   ├── predict_live_players.py
|   ├── build_daily_clan_features.py
|   ├── predict_live_war.py
|   ├── optimize_gk_rotation.py
|   └── build_dashboard.py
|
├── .env
├── .gitignore
└── README.md

Additional training and evaluation scripts may also be present in src/.

Technologies

Python

pandas

NumPy

scikit-learn

SQLAlchemy

PyMySQL

MySQL

Docker

Clash Royale API

HTML/CSS

Git / GitHub

Setup

1. Clone the repository

git clone <your-repository-url>
cd clash-war-ml

2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

At minimum, the project uses:

pip install pandas numpy scikit-learn sqlalchemy pymysql python-dotenv requests

If a requirements.txt file is included, use:

pip install -r requirements.txt

4. Configure environment variables

Create a local .env file:

CLASH_ROYALE_API_TOKEN=your_api_token_here

DB_HOST=127.0.0.1
DB_PORT=33306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=clash_war_ml

Do not commit .env.

5. Start MySQL

The project was developed with MySQL running in Docker.

Example:

docker start mysql-server-x370

Create the database schema using:

sql/schema.sql

Running the Analytics Pipeline

A typical refresh is:

python src/collect_data.py
python src/build_features.py
python src/build_player_archetypes.py
python src/build_live_player_features.py
python src/predict_live_players.py
python src/build_daily_clan_features.py
python src/predict_live_war.py
python src/optimize_gk_rotation.py
python src/build_dashboard.py

Then:

open reports/gk_war_dashboard.html

Some historical model-building scripts do not need to be rerun every time live data is refreshed.

Current Model Notes

The historical supervised dataset contains thousands of player-war observations.

The project currently keeps the larger historical training dataset for supervised model training because it provides substantially more examples than Galactic Kings alone.

However:
 - live predictions are Galactic Kings-only
 - archetype outputs are Galactic Kings-only
 - rotation optimization is Galactic Kings-only
 - war forecasting is Galactic Kings-only
 - the final dashboard is Galactic Kings-only

This allows the models to learn from a larger historical sample while keeping the final product focused on Galactic Kings.

Important Modeling Limitations

This is an analytics and portfolio project, not an official Clash Royale system.

Several values should be interpreted as model estimates rather than guaranteed outcomes.

In particular:
 - player performance predictions are historical estimates
 - capacity_adjusted_fame is a rotation-planning heuristic
 - player-level Fame estimates are not identical to Clan      War River movement
 - future River movement uses Galactic Kings historical behavior when direct future outcomes are unavailable
 - player-day historical data is still limited compared with  player-war historical data
 - new players with little or no history require fallback estimates

The project intentionally avoids presenting these heuristics as guaranteed or proven optimal outcomes.

Future Improvements

 Potential next steps include:
  - store true historical player-day performance
  - evaluate predictions against each completed Battle Day
  - track prediction error over time
  - retrain and compare additional regression models
  - measure model performance specifically on Galactic Kings holdout data
  - add uncertainty intervals

create a web-hosted dashboard

connect predictions to the Galactic Kings Discord bot

automatically refresh data and forecasts

recommend daily rotations based on confirmed player availability

add historical player trend visualizations

Author

Built by Logan Beach as a machine learning / data science portfolio project using real Clash Royale Clan War data.