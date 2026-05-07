# JRA Predictive Model: Recommended Next Steps

Based on the audit of the successful tennis prediction AI and the current architecture of the UmaEdge JRA machine learning pipeline, we should prioritize the following concrete next steps:

## 1. Implement an Expected Value (EV) / ROI Voting Engine
Raw win accuracy in large-field horse racing is an insufficient metric. The most critical next step is to build an engine that compares our model's predicted probabilities against the public implied betting odds.
- **Action:** Integrate real-time (or historical closing) odds data into the pipeline.
- **Action:** Create an Expected Value (EV) function: `(Model Probability * Decimal Odds) - 1`. If EV > 0, the bet has positive expected value.
- **Action:** Shift the primary evaluation metric from "Accuracy" to "Simulated ROI".

## 2. Formalize Strict Time-Series Cross-Validation
To guarantee the model generalizes to unseen data (like the tennis model predicting the Australian Open), we must completely eliminate lookup bias and future data leakage.
- **Action:** Implement a strict chronological split for training and testing. For example, train the models on all data from 2018–2022, and test exclusively on 2023 data.
- **Action:** Build a backtesting simulation that walks forward in time, updating the model predictions exactly as they would have been produced on the morning of a race day.

## 3. Expand High-Impact Domain Feature Engineering
Since we are using efficient tabular models (like XGBoost/LightGBM) built to run locally, our competitive advantage relies entirely on the quality of our features, not compute power.
- **Action:** Build upon the existing speed figures and running styles by adding pace-scenario modeling (e.g., predicting the shape of the race based on the density of early speed horses).
- **Action:** Implement jockey and trainer "course specialty" metrics (e.g., how well does Jockey X perform on Tokyo Dirt 1600m specifically, rather than their overall win rate).
- **Action:** Calculate distance-traveled metrics for horses shipping from different training centers (e.g., Miho vs. Ritto).

## 4. Continuous Data Acquisition & Health
Machine learning thrives on volume and cleanliness.
- **Action:** Set up automated, recurring jobs to backfill any new or missing data points (similar to the 17k+ broodmare sire recovery effort).
- **Action:** Implement database health monitoring to immediately flag any newly scraped races that have missing sectional times or anomalous weight formats.
