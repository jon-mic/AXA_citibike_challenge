# AXA_citibike_challenge
_Author: jomic@tutanota.com_


This project encompasses two predictive modeling challenges:

1. **Accident Likelihood Prediction**: Predicting the likelihood of accidents based on weather data and historical accident records.
2. **Trip Data Modeling**: Predicting the most likely end station for a new bicycle trip using NYC CitiBike trip data.

## Features

### Accident Likelihood Prediction
- Utilizes timestamped weather data and historical accidents (one-class classification problem).
- Custom feature engineering (e.g., cyclic features for time-based patterns).
- XGBoost model with out-of-core support for large datasets.

### Trip Data Modeling
- Predicts end station based on trip start data (e.g., start time, location).
- Feature engineering includes spatial and temporal insights.
- Scalable data handling with Polars and optimized workflows.

## Installation

1. Clone this repository: `git clone https://github.com/jon-mic/AXA_citibike_challenge.git`
2. Install dependencies using `uv`(https://docs.astral.sh/uv/): `uv sync`

## Datasets

- **Accident Prediction**: Weather and historical accident records from NYPD.
- **Trip Data Modeling**: NYC CitiBike trip data, including station and timestamp information.

## Key Techniques

- **Accident Model**:
  - Timestamp feature extraction using sine/cosine transformations.
  - XGBoost implementation for large datasets.
- **Trip Data Model**:
  - Spatial and temporal feature engineering.
  - Efficient data processing using Polars.

## Results

- Accident model outputs likelihood scores and performance metrics.
- Trip data model predicts end stations with evaluation metrics.
