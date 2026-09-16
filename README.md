# Galactic Kings Clan War Analytics

An end-to-end machine learning and data analytics system built for **Galactic Kings**, a competitive Clash Royale clan.

The project collects Clan War data from the Clash Royale API, stores historical and live data in MySQL, engineers player-level features, predicts player participation and performance, clusters players into behavioral archetypes, recommends a 50-player Battle Day rotation, forecasts Galactic Kings' River Race performance, and generates an interactive-style HTML analytics dashboard.

## Highlights

* 6,000+ historical player-war observations used for supervised modeling
* Live predictions across the current Galactic Kings rotation pool
* Random Forest participation modeling
* Regression-based player performance forecasting
* K-Means player behavior clustering
* Automatic selection of archetype cluster count using silhouette score
* 50-player Battle Day rotation optimization
* Ranked backup recommendations
* Galactic Kings-specific River Race forecasting
* MySQL + Docker data pipeline
* Automated HTML analytics dashboard
* Separate handling for River Race and Colosseum weeks

---

## Project Goal

Galactic Kings regularly rotates players in and out of the clan to maximize participation during Clan Wars.

Because the clan can use up to 50 unique participants during a Battle Day, the current in-game roster does not necessarily represent the full group of players available for war.

This project was built to help answer questions such as:

* Which GK players are the most reliable war contributors?
* Which players are most likely to participate?
* How much production can we expect from the current rotation pool?
* Which 50 players should be prioritized for a Battle Day?
* Which players should be backups?
* Which players consistently perform at a high level?
* Which players are inconsistent or participate infrequently?
* How does the current war compare with GK's historical performance?
* When is Galactic Kings projected to finish the River Race?

The final analytics system is intentionally focused only on **Galactic Kings**.

---

# System Architecture

```text
Clash Royale API
        |
        v
collect_data.py
        |
        v
      MySQL
        |
        +------------------------------+
        |                              |
        v                              v
build_features.py          build_player_archetypes.py
        |                              |
        v                              v
Historical ML Dataset          GK Player Archetypes
        |                              |
        +---------------+--------------+
                        |
                        v
            build_live_player_features.py
                        |
                        v
               predict_live_players.py
                        |
                        v
          GK Live Player Predictions
                        |
             +----------+----------+
             |                     |
             v                     v
build_daily_clan_features.py  optimize_gk_rotation.py
             |                     |
             v                     v
        GK War History      Recommended 50 + Backups
             |                     |
             +----------+----------+
                        |
                        v
                predict_live_war.py
                        |
                        v
                build_dashboard.py
                        |
                        v
           reports/gk_war_dashboard.html
```

---

# Machine Learning Pipeline

## 1. Historical Feature Engineering

The historical dataset contains one row per player-war observation.

Player features include:

* recent average Fame
* recent average decks used
* recent Fame per deck
* participation rate
* full-participation rate
* previous-war Fame
* previous-war decks
* recent Fame trend
* recent Fame variance
* recent deck-use variance
* Galactic Kings clan-score context

These features are constructed so that predictions use information from previous wars rather than future target information.

---

## 2. Participation Prediction

A **Random Forest classifier** estimates the probability that a player participates in a war.

Participation probability is useful because player performance alone does not tell the full story.

A highly skilled player who rarely participates may be less useful for an expected-value lineup than a slightly lower-performing player who consistently completes war attacks.

The system therefore models:

```text
P(player participates)
```

separately from expected player production.

---

## 3. Player Performance Prediction

The system predicts player war production using historical player behavior.

The live pipeline supports three prediction tiers:

### Full ML Model

Used when a player has sufficient historical data and all required model features.

```text
prediction_tier = ml_model
```

### Limited History

Used when a player has only one or two historical wars.

```text
prediction_tier = limited_history
```

### Clan Baseline

Used for players with insufficient individual history.

```text
prediction_tier = clan_baseline
```

This allows every player in the current GK rotation pool to receive a usable estimate, including new players.

---

# Galactic Kings Player Archetypes

The project uses **K-Means clustering** to identify different types of war players.

Only players in the current Galactic Kings rotation pool are included in the final archetype system.

Each player's full historical performance can still be used to describe their behavior.

Features used for clustering include:

* average Fame
* average decks
* participation rate
* full-participation rate
* Fame per deck
* Fame consistency
* deck-use consistency

Before clustering, features are standardized with `StandardScaler`.

The system compares several possible cluster counts using the **silhouette score** rather than choosing the number of clusters manually.

Example archetypes include:

* **Reliable High Performer**
* **Inconsistent Contributor**
* **Low-Participation Player**

K-Means cluster IDs are arbitrary, so the project does not permanently map cluster numbers to labels.

Instead, archetype names are assigned based on the actual behavior of each cluster after training.

---

# Live Galactic Kings Rotation Pool

One important design decision is that the system does not assume the current clan roster represents the entire available war roster.

Galactic Kings frequently rotates war players in and out.

The live rotation pool therefore includes:

```text
Current GK members
+
GK players already observed in the current live race
```

This means a player can remain part of the war-planning pool even when temporarily outside the clan.

The pipeline tracks:

* total rotation-pool players
* currently active clan members
* players currently outside the clan
* historical performance
* prediction tier
* participation probability
* archetype
* expected production

---

# 50-Player Rotation Optimization

Galactic Kings attempts to fill all 50 available Battle Day participant slots.

The optimizer creates two separate rankings.

## Expected-Value Lineup

Used when player availability is not yet confirmed.

The ranking considers predicted production while participation remains uncertain.

This answers:

> Who gives GK the strongest expected lineup before we know exactly who will play?

---

## Confirmed-Active Lineup

Used when a player is known to be available.

Once availability is confirmed, the system emphasizes projected active production rather than historical attendance probability.

This answers:

> If these players are definitely available, which 50 should GK use?

The optimizer produces:

* recommended 50-player lineup
* expected-value ranking
* confirmed-active ranking
* ranked backups
* current-member count
* outside-clan rotation count
* projected lineup production
* archetype breakdown

---

# Galactic Kings War Forecast

The final war forecasting system models **Galactic Kings only**.

Opponent clans are not individually modeled or displayed in the final dashboard.

For River Race weeks, the system combines:

* current GK player predictions
* projected participation
* historical GK Battle Day performance
* historical movement
* historical finish behavior
* current River progress

to estimate:

* projected Battle Day production
* projected movement
* projected River progress
* projected finish day

If historical evidence is unavailable for a particular value, the system avoids inventing a result.

---

# Colosseum Handling

Colosseum weeks are treated separately from normal River Race weeks.

Normal River Race analysis includes the 10,000-point River finish condition.

Colosseum does not use the same finish structure, so the dashboard adjusts its displayed metrics accordingly.

---

# Dashboard

The project generates a Galactic Kings-only HTML dashboard:

```text
reports/gk_war_dashboard.html
```

The dashboard includes:

### Current War Overview

* current Battle Day
* projected total production
* projected River progress
* projected finish day
* historical GK finish behavior

### Rotation Pool

* total available GK rotation players
* current members
* players currently outside the clan
* expected daily participants
* ML coverage

### Player Analytics

* top live player forecasts
* participation probability
* prediction tier
* behavioral archetype
* projected daily production

### Player Archetypes

* cluster sizes
* average Fame
* average deck usage
* participation rate
* full-participation rate
* efficiency

### Rotation Optimization

* recommended 50-player lineup
* current-member status
* expected-value production
* confirmed-active production
* ranked backups

---

# Repository Structure

```text
clash-war-ml/
│
├── data/
│   ├── model_dataset.csv
│   ├── player_test_predictions.csv
│   │
│   ├── gk_player_archetypes.csv
│   ├── gk_player_archetype_summary.csv
│   │
│   ├── gk_live_player_features.csv
│   ├── gk_live_player_predictions.csv
│   ├── gk_live_player_summary.csv
│   │
│   ├── gk_daily_war_dataset.csv
│   ├── gk_race_summary.csv
│   │
│   ├── gk_live_war_player_projection.csv
│   ├── gk_live_war_day_projection.csv
│   ├── gk_live_war_projection.csv
│   │
│   ├── gk_rotation_expected_value.csv
│   ├── gk_rotation_active_lineup.csv
│   ├── gk_rotation_backups.csv
│   └── gk_rotation_summary.csv
│
├── reports/
│   └── gk_war_dashboard.html
│
├── sql/
│   └── schema.sql
│
├── src/
│   ├── collect_data.py
│   ├── build_features.py
│   │
│   ├── train_model.py
│   ├── train_participation.py
│   ├── train_active_decks.py
│   ├── train_efficiency.py
│   ├── train_two_stage.py
│   ├── train_final_pipeline.py
│   ├── train_gradient_boosting.py
│   │
│   ├── build_player_archetypes.py
│   ├── build_live_player_features.py
│   ├── predict_live_players.py
│   ├── build_daily_clan_features.py
│   ├── predict_live_war.py
│   ├── optimize_gk_rotation.py
│   └── build_dashboard.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Technologies

### Programming

* Python

### Data Analysis

* pandas
* NumPy

### Machine Learning

* scikit-learn
* Linear Regression
* Random Forest
* Gradient Boosting experimentation
* K-Means clustering
* StandardScaler
* silhouette analysis

### Database

* MySQL
* SQLAlchemy
* PyMySQL
* Docker

### Data Source

* Clash Royale API

### Visualization / Reporting

* HTML
* CSS
* pandas HTML table generation

### Development

* Git
* GitHub
* Python virtual environments

---

# Setup

## 1. Clone the Repository

```bash
git clone https://github.com/loganbeach11/clash-war-ml.git
cd clash-war-ml
```

---

## 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Copy the example environment configuration:

```bash
cp .env.example .env
```

Then update `.env` with your own credentials.

Example:

```env
CLASH_ROYALE_API_TOKEN=your_api_token_here

DB_HOST=127.0.0.1
DB_PORT=33306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=clash_war_ml
```

The real `.env` file is excluded from Git.

---

## 5. Start MySQL

The project was developed using MySQL in Docker.

Example:

```bash
docker start mysql-server-x370
```

The database structure is defined in:

```text
sql/schema.sql
```

---

# Running the Pipeline

A full analytics refresh can be run in this order:

```bash
python src/collect_data.py

python src/build_features.py

python src/build_player_archetypes.py

python src/build_live_player_features.py

python src/predict_live_players.py

python src/build_daily_clan_features.py

python src/predict_live_war.py

python src/optimize_gk_rotation.py

python src/build_dashboard.py
```

Then open the dashboard:

```bash
open reports/gk_war_dashboard.html
```

---

# Historical Training vs. Live GK Analysis

The supervised player models currently retain a larger historical training dataset containing thousands of player-war observations.

This provides substantially more training examples than using Galactic Kings history alone.

However, the final live system is GK-specific:

```text
Historical model training → larger dataset

Live features              → Galactic Kings only
Player predictions         → Galactic Kings only
Player archetypes          → Galactic Kings only
Rotation optimization      → Galactic Kings only
War forecasting            → Galactic Kings only
Dashboard                  → Galactic Kings only
```

This approach allows the models to benefit from a larger supervised training sample while keeping the final application entirely focused on Galactic Kings.

---

# Modeling Limitations

This project is an independent analytics and machine learning project and is not affiliated with or endorsed by Supercell.

Predictions should be interpreted as estimates rather than guaranteed outcomes.

Important limitations include:

* historical player behavior may not perfectly predict future participation
* new players have limited historical information
* player-war data is more complete than player-day historical data
* `capacity_adjusted_fame` is a planning heuristic
* predicted player Fame is not equivalent to River Race movement
* future movement estimates rely partly on historical GK behavior
* Battle Day rotation availability may change after predictions are generated
* the current combined live forecast is a practical heuristic and should continue to be evaluated against future completed wars

The project intentionally distinguishes between trained model outputs and planning heuristics rather than presenting every estimate as a proven optimal result.

---

# Future Improvements

Planned improvements include:

* collect true historical player-day performance
* save prediction snapshots before each Battle Day
* compare predictions against actual outcomes
* calculate prediction error over time
* evaluate models specifically on Galactic Kings holdout data
* compare additional regression algorithms
* add uncertainty intervals
* improve daily participation forecasting
* automatically detect confirmed player availability
* build historical player trend visualizations
* deploy the dashboard as a hosted web application
* integrate predictions into the Galactic Kings Discord bot
* automate scheduled data collection
* automatically generate Battle Day rotation recommendations
* track model performance over multiple seasons

---

# Why I Built This

I wanted to build a machine learning project around a real system with continuously changing data rather than a static classroom dataset.

Galactic Kings provided an opportunity to combine:

* API data collection
* relational database design
* feature engineering
* supervised machine learning
* unsupervised learning
* optimization
* live prediction
* data visualization

into a single end-to-end application with a real use case.

---

# Author

**Logan Beach**

Computer Science
Applied Data Science

University of Georgia

GitHub: [loganbeach11](https://github.com/loganbeach11)

---

## Disclaimer

This project is an independent educational and portfolio project.

Clash Royale and all related game assets and terminology are trademarks of Supercell. This project is not affiliated with, endorsed by, sponsored by, or specifically approved by Supercell.
