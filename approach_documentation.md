# Traffic Demand Prediction — Technical Approach Documentation

## 1. Problem Statement

Regression task to predict traffic `demand` at specific locations and times.  
**Metric:** `max(0, 100 × R²(actual, predicted))`  
**Training data:** 77,299 rows × 11 columns  
**Test data:** 41,778 rows × 10 columns  

## 2. Data Analysis

### 2.1 Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| geohash | Categorical (921 unique) | Encoded location identifier |
| day | Integer (48–50) | Day index |
| timestamp | String (H:MM format) | Time of observation |
| RoadType | Categorical | Residential, Street, Highway, Unknown |
| NumberofLanes | Integer (1–5) | Lane count |
| LargeVehicles | Binary | Allowed / Not Allowed |
| Landmarks | Binary | Yes / No |
| Temperature | Continuous | Range ~-11 to ~37°C |
| Weather | Categorical | Sunny, Foggy, Rainy, Snowy, Unknown |
| demand | Continuous (target) | Range 0 to 1.0 |

### 2.2 Key Observations

- **Missing values** present in RoadType, Temperature, and Weather (~5-8% per column)
- **demand** is right-skewed with most values < 0.2; Highway/Street road types have much higher demand
- **Geohash** is the strongest signal — location-based demand varies by 10-50×
- **Temporal patterns** exist across hours (rush hour vs night) and days

## 3. Feature Engineering (54 features total)

### 3.1 Temporal Features (16 features)
- `hour`, `minute`, `time_minutes` — extracted from timestamp
- `hour_sin`, `hour_cos` — cyclic encoding of hour (period=24)
- `minute_sin`, `minute_cos` — cyclic encoding of minute (period=60)
- `day_sin`, `day_cos` — cyclic encoding of day (period=7)
- `day_of_week`, `is_weekend` — day-level temporal features
- `time_slot`, `time_slot_sin`, `time_slot_cos` — 15-minute time slots (96 per day)
- `is_rush_morning` (7–9AM), `is_rush_evening` (5–7PM), `is_rush_hour`, `is_night` (10PM–5AM)

### 3.2 Categorical Encoding (6 features)
- `RoadType_encoded` — label encoding (Unknown imputed for nulls)
- `Weather_encoded` — label encoding (Unknown imputed for nulls)
- `LargeVehicles_encoded` — binary (Allowed=1)
- `Landmarks_encoded` — binary (Yes=1)
- Temperature nulls filled with median

### 3.3 Spatial Features (14 features)
- `geohash_encoded` — full geohash label encoding (921 categories)
- `geohash_prefix4_encoded` — 4-char prefix encoding (coarser location)
- `geohash_prefix5_encoded` — 5-char prefix encoding (medium granularity)
- **Target-encoded means** (K-Fold OOF to prevent leakage):
  - `geohash_encoded_target_mean`
  - `geohash_prefix5_encoded_target_mean`
  - `RoadType_encoded_target_mean`
- **Geohash demand aggregations:** mean, std, median, min, max, count per geohash
- **Prefix-level aggregations:** mean, std per geohash_prefix4 and prefix5
- **Day × Geohash interaction:** mean demand per day-location combination
- **Road type aggregations:** mean, std demand per road type

### 3.4 Interaction Features (10 features)
- `temp_road_interaction` — Temperature × RoadType
- `temp_weather_interaction` — Temperature × Weather
- `temp_lanes_interaction` — Temperature × NumberofLanes
- `road_lanes_interaction` — RoadType × NumberofLanes
- `weather_lanes_interaction` — Weather × NumberofLanes
- `road_weather_interaction` — RoadType × Weather
- `landmarks_lanes_interaction` — Landmarks × NumberofLanes
- `large_vehicles_lanes` — LargeVehicles × NumberofLanes
- `temp_squared` — Temperature² (captures non-linear temperature effects)
- `temp_abs` — |Temperature| (absolute temperature)

## 4. Model Architecture

### 4.1 LightGBM

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.04 |
| num_leaves | 180 |
| max_depth | 9 |
| min_child_samples | 15 |
| subsample | 0.78 |
| colsample_bytree | 0.65 |
| reg_alpha | 0.05 |
| reg_lambda | 1.5 |
| max_boost_rounds | 5000 (early stop 100) |

### 4.2 XGBoost

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.04 |
| max_depth | 8 |
| min_child_weight | 8 |
| subsample | 0.8 |
| colsample_bytree | 0.65 |
| reg_alpha | 0.08 |
| reg_lambda | 1.2 |
| gamma | 0.05 |
| max_boost_rounds | 5000 (early stop 100) |

### 4.3 CatBoost

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.05 |
| depth | 8 |
| l2_leaf_reg | 3.0 |
| bagging_temperature | 0.3 |
| random_strength | 1.0 |
| border_count | 128 |
| iterations | 3000 (early stop 100) |

## 5. Validation Strategy

- **5-Fold KFold Cross-Validation** with `shuffle=True`, `random_state=42`
- Early stopping on validation RMSE (patience=100 rounds)
- Target encoding computed using OOF scheme to prevent data leakage
- Predictions clipped to `[0, ∞)` since demand is non-negative

## 6. Ensembling

**Method:** Weighted average ensemble with grid search over weight triplets.

Weight search grid: 0.00 to 1.00 in steps of 0.05, constrained to `w1 + w2 + w3 = 1.0`.

### Results

| Model | OOF R² | Scaled Score |
|-------|--------|-------------|
| LightGBM | 0.9496 | 94.96 |
| XGBoost | 0.9465 | 94.65 |
| **CatBoost** | **0.9612** | **96.12** |
| Ensemble (5/5/90) | 0.9609 | 96.09 |

CatBoost dominates the ensemble. The optimal weights (LGB=0.05, XGB=0.05, CAT=0.90) produce a marginally lower R² than pure CatBoost, so the final submission uses the best-performing configuration.

## 7. Key Technical Decisions

1. **Geohash as the primary signal:** Target-encoded geohash means and spatial aggregations are the most important features, as traffic demand is heavily location-dependent.

2. **Cyclic encoding over one-hot:** Sine/cosine transformations for temporal features preserve the circular nature of time (23:00 is close to 00:00).

3. **CatBoost superiority:** CatBoost's ordered boosting and built-in handling of feature interactions gives it a significant edge on this dataset (~1.5% R² gap over LGB/XGB).

4. **No log-transform on target:** The demand distribution is right-skewed but bounded in [0, 1], and tree-based models handle this natively without transformation.

5. **Prediction clipping:** All predictions are clipped to `≥ 0` since negative demand is physically impossible.

## 8. Deliverables

| File | Description |
|------|-------------|
| `submission.csv` | 41,778 predictions (Index, demand) |
| `traffic_demand_prediction.ipynb` | Complete reproducible notebook |
| `approach_documentation.md` | This document |
| `run_final.py` | Standalone production script |
| `results.json` | Serialized hyperparameters and metrics |
