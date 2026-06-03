# Traffic Demand Prediction — Technical Approach Documentation

## 1. Problem Statement

Regression task to predict traffic `demand` at specific locations and times.  
**Metric:** `max(0, 100 × R²(actual, predicted))`  
**Training data:** 77,299 rows × 11 columns  
**Test data:** 41,778 rows × 10 columns  

## 2. Data Analysis & Preprocessing

- **Imputation**: Missing values in `Temperature` are filled with the training median. Categorical missing values (`RoadType`, `Weather`, `LargeVehicles`, `Landmarks`) are filled with "Unknown".
- **Categorical Encoding**: Label encoding is applied to all categorical features (`geohash`, `geo_prefix3`, `geo_prefix4`, `geo_prefix5`, `RoadType`, `Weather`, `LargeVehicles`, `Landmarks`), fitted on the combined train and test sets to ensure all unique values are covered.

## 3. Feature Engineering (25 features total)

### 3.1 Temporal Features (7 features)
- `hour`, `minute`, `time_minutes` — extracted from timestamp
- `hour_sin`, `hour_cos` — cyclic encoding of hour (period=24)
- `minute_sin`, `minute_cos` — cyclic encoding of minute (period=60)

### 3.2 Spatial / Geohash Features (4 features)
- `geohash` — full geohash label encoded
- `geo_prefix3`, `geo_prefix4`, `geo_prefix5` — 3, 4, and 5-character geohash prefixes

### 3.3 Geohash Target Encoding (7 features)
- `geo_mean`, `geo_median`, `geo_std`, `geo_count` — aggregate statistics of demand grouped by the full geohash
- `geo_prefix3_mean`, `geo_prefix4_mean`, `geo_prefix5_mean` — mean demand aggregated at the 3, 4, and 5-character geohash prefix levels

### 3.4 Raw Numeric / Categorical Features (7 features)
- `day`, `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`

## 4. Model Architecture

The final approach uses a 3-model stack trained across 3 different random seeds (42, 7, 123) to reduce variance.

### 4.1 LightGBM
- `n_estimators`: 2000
- `learning_rate`: 0.03
- `num_leaves`: 127
- `subsample`: 0.8
- `colsample_bytree`: 0.8

### 4.2 XGBoost
- `n_estimators`: 2000
- `learning_rate`: 0.03
- `max_depth`: 7
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `tree_method`: "hist"

### 4.3 CatBoost
- `iterations`: 1000
- `learning_rate`: 0.05
- `depth`: 8

## 5. Ensembling Strategy

The predictions from the 9 models (3 models × 3 seeds) are averaged per model type. The final blend uses weights derived from a 5-fold Out-Of-Fold (OOF) Ridge regression meta-learner (alpha=1.0).

**Optimal Ridge Weights:**
- **LightGBM:** 0.486
- **XGBoost:** 0.330
- **CatBoost:** 0.184

The final predictions are clipped to `[0, 1]` to ensure they remain within the valid demand bounds.

## 6. Deliverables

| File | Description |
|------|-------------|
| `submission.csv` | 41,778 predictions (Index, demand) |
| `traffic_demand_pred.ipynb` | Complete reproducible notebook |
| `approach_documentation.md` | This document |
