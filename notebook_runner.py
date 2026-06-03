import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = "./dataset"
np.random.seed(SEED)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target stats:\n{train['demand'].describe()}")
train.head()

def parse_timestamp(df):
    parts = df["timestamp"].astype(str).str.split(":", expand=True)
    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)
    df["time_minutes"] = df["hour"] * 60 + df["minute"]
    return df


def engineer_features(train, test):
    target = train["demand"].copy()
    train_idx = train["Index"].copy()
    test_idx = test["Index"].copy()

    combined = pd.concat([train.drop(columns=["demand"]), test], axis=0, ignore_index=True)
    combined = parse_timestamp(combined)

    combined["hour_sin"] = np.sin(2 * np.pi * combined["hour"] / 24)
    combined["hour_cos"] = np.cos(2 * np.pi * combined["hour"] / 24)
    combined["minute_sin"] = np.sin(2 * np.pi * combined["minute"] / 60)
    combined["minute_cos"] = np.cos(2 * np.pi * combined["minute"] / 60)
    combined["day_sin"] = np.sin(2 * np.pi * combined["day"] / 7)
    combined["day_cos"] = np.cos(2 * np.pi * combined["day"] / 7)
    combined["day_of_week"] = combined["day"] % 7
    combined["is_weekend"] = (combined["day_of_week"] >= 5).astype(int)

    combined["RoadType"] = combined["RoadType"].fillna("Unknown")
    combined["RoadType_encoded"] = LabelEncoder().fit_transform(combined["RoadType"])
    combined["Weather"] = combined["Weather"].fillna("Unknown")
    combined["Weather_encoded"] = LabelEncoder().fit_transform(combined["Weather"])
    combined["LargeVehicles_encoded"] = (combined["LargeVehicles"] == "Allowed").astype(int)
    combined["Landmarks_encoded"] = (combined["Landmarks"] == "Yes").astype(int)
    combined["Temperature"] = combined["Temperature"].fillna(combined["Temperature"].median())

    combined["temp_road_interaction"] = combined["Temperature"] * combined["RoadType_encoded"]
    combined["temp_weather_interaction"] = combined["Temperature"] * combined["Weather_encoded"]
    combined["temp_lanes_interaction"] = combined["Temperature"] * combined["NumberofLanes"]
    combined["road_lanes_interaction"] = combined["RoadType_encoded"] * combined["NumberofLanes"]
    combined["weather_lanes_interaction"] = combined["Weather_encoded"] * combined["NumberofLanes"]
    combined["road_weather_interaction"] = combined["RoadType_encoded"] * combined["Weather_encoded"]
    combined["landmarks_lanes_interaction"] = combined["Landmarks_encoded"] * combined["NumberofLanes"]
    combined["large_vehicles_lanes"] = combined["LargeVehicles_encoded"] * combined["NumberofLanes"]
    combined["temp_squared"] = combined["Temperature"] ** 2
    combined["temp_abs"] = combined["Temperature"].abs()

    combined["geohash_prefix4"] = combined["geohash"].str[:4]
    combined["geohash_prefix5"] = combined["geohash"].str[:5]
    combined["geohash_prefix4_encoded"] = LabelEncoder().fit_transform(combined["geohash_prefix4"])
    combined["geohash_prefix5_encoded"] = LabelEncoder().fit_transform(combined["geohash_prefix5"])
    combined["geohash_encoded"] = LabelEncoder().fit_transform(combined["geohash"])

    combined["time_slot"] = combined["time_minutes"] // 15
    combined["time_slot_sin"] = np.sin(2 * np.pi * combined["time_slot"] / 96)
    combined["time_slot_cos"] = np.cos(2 * np.pi * combined["time_slot"] / 96)
    combined["is_rush_morning"] = ((combined["hour"] >= 7) & (combined["hour"] <= 9)).astype(int)
    combined["is_rush_evening"] = ((combined["hour"] >= 17) & (combined["hour"] <= 19)).astype(int)
    combined["is_rush_hour"] = (combined["is_rush_morning"] | combined["is_rush_evening"]).astype(int)
    combined["is_night"] = ((combined["hour"] >= 22) | (combined["hour"] <= 5)).astype(int)

    drop_cols = ["Index", "geohash", "timestamp", "RoadType", "LargeVehicles",
                 "Landmarks", "Weather", "geohash_prefix4", "geohash_prefix5"]
    combined = combined.drop(columns=drop_cols)

    train_fe = combined.iloc[:len(train)].reset_index(drop=True)
    test_fe = combined.iloc[len(train):].reset_index(drop=True)
    return train_fe, test_fe, target, train_idx, test_idx


train_fe, test_fe, target, train_idx, test_idx = engineer_features(train, test)
print(f"Engineered features: {train_fe.shape[1]}")

def add_target_encoding(train_fe, test_fe, target, cols):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for col in cols:
        train_fe[f"{col}_target_mean"] = np.nan
        global_mean = target.mean()
        for _, (tr_idx, val_idx) in enumerate(kf.split(train_fe)):
            means = target.iloc[tr_idx].groupby(train_fe[col].iloc[tr_idx]).mean()
            train_fe.loc[val_idx, f"{col}_target_mean"] = train_fe[col].iloc[val_idx].map(means)
        train_fe[f"{col}_target_mean"] = train_fe[f"{col}_target_mean"].fillna(global_mean)
        overall_means = target.groupby(train_fe[col]).mean()
        test_fe[f"{col}_target_mean"] = test_fe[col].map(overall_means).fillna(global_mean)
    return train_fe, test_fe


def add_geohash_aggregations(train_fe, test_fe, target):
    train_fe["_target"] = target.values
    for group_col, prefix in [("geohash_encoded", "geo"), ("geohash_prefix4_encoded", "geo4"),
                               ("geohash_prefix5_encoded", "geo5")]:
        agg_funcs = ["mean", "std"] if prefix != "geo" else ["mean", "std", "median", "min", "max", "count"]
        stats = train_fe.groupby(group_col)["_target"].agg(agg_funcs).reset_index()
        stats.columns = [group_col] + [f"{prefix}_demand_{f}" for f in agg_funcs]
        if f"{prefix}_demand_std" in stats.columns:
            stats[f"{prefix}_demand_std"] = stats[f"{prefix}_demand_std"].fillna(0)
        train_fe = train_fe.merge(stats, on=group_col, how="left")
        test_fe = test_fe.merge(stats, on=group_col, how="left")
        for c in stats.columns[1:]:
            test_fe[c] = test_fe[c].fillna(train_fe[c].median())

    day_geo = train_fe.groupby(["day", "geohash_encoded"])["_target"].mean().reset_index()
    day_geo.columns = ["day", "geohash_encoded", "day_geo_demand_mean"]
    train_fe = train_fe.merge(day_geo, on=["day", "geohash_encoded"], how="left")
    test_fe = test_fe.merge(day_geo, on=["day", "geohash_encoded"], how="left")
    test_fe["day_geo_demand_mean"] = test_fe["day_geo_demand_mean"].fillna(test_fe["geo_demand_mean"])

    road_stats = train_fe.groupby("RoadType_encoded")["_target"].agg(["mean", "std"]).reset_index()
    road_stats.columns = ["RoadType_encoded", "road_demand_mean", "road_demand_std"]
    train_fe = train_fe.merge(road_stats, on="RoadType_encoded", how="left")
    test_fe = test_fe.merge(road_stats, on="RoadType_encoded", how="left")
    for c in ["road_demand_mean", "road_demand_std"]:
        test_fe[c] = test_fe[c].fillna(train_fe[c].median())
    train_fe = train_fe.drop(columns=["_target"])
    return train_fe, test_fe


te_cols = ["geohash_encoded", "geohash_prefix5_encoded", "RoadType_encoded"]
train_fe, test_fe = add_target_encoding(train_fe, test_fe, target, te_cols)
train_fe, test_fe = add_geohash_aggregations(train_fe, test_fe, target)

features = list(train_fe.columns)
print(f"Final feature count: {len(features)}")
print(f"Features: {features}")

lgb_params = {
    "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
    "verbosity": -1, "seed": SEED, "n_jobs": -1,
    "learning_rate": 0.04, "num_leaves": 180, "max_depth": 9,
    "min_child_samples": 15, "subsample": 0.78, "colsample_bytree": 0.65,
    "reg_alpha": 0.05, "reg_lambda": 1.5, "min_split_gain": 0.01,
}

xgb_params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "tree_method": "hist", "seed": SEED, "nthread": -1,
    "learning_rate": 0.04, "max_depth": 8, "min_child_weight": 8,
    "subsample": 0.8, "colsample_bytree": 0.65,
    "reg_alpha": 0.08, "reg_lambda": 1.2, "gamma": 0.05,
}

cat_params = {
    "loss_function": "RMSE", "random_seed": SEED, "iterations": 3000,
    "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0,
    "bagging_temperature": 0.3, "random_strength": 1.0,
    "border_count": 128, "verbose": 0,
}

def train_lgb_kfold(X, y, X_test, params):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof, test_preds = np.zeros(len(X)), np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        dtrain = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=5000, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof[val_idx] = model.predict(X.iloc[val_idx])
        test_preds += model.predict(X_test) / N_FOLDS
        print(f"  Fold {fold+1} R2={r2_score(y.iloc[val_idx], np.clip(oof[val_idx], 0, None)):.5f}")
    return np.clip(oof, 0, None), np.clip(test_preds, 0, None)


def train_xgb_kfold(X, y, X_test, params):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof, test_preds = np.zeros(len(X)), np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
        dtest = xgb.DMatrix(X_test)
        model = xgb.train(params, dtrain, num_boost_round=5000, evals=[(dval, "val")],
                          early_stopping_rounds=100, verbose_eval=False)
        oof[val_idx] = model.predict(dval)
        test_preds += model.predict(dtest) / N_FOLDS
        print(f"  Fold {fold+1} R2={r2_score(y.iloc[val_idx], np.clip(oof[val_idx], 0, None)):.5f}")
    return np.clip(oof, 0, None), np.clip(test_preds, 0, None)


def train_cat_kfold(X, y, X_test, params):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof, test_preds = np.zeros(len(X)), np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        model = cb.CatBoostRegressor(**params)
        model.fit(cb.Pool(X.iloc[tr_idx], label=y.iloc[tr_idx]),
                  eval_set=cb.Pool(X.iloc[val_idx], label=y.iloc[val_idx]),
                  early_stopping_rounds=100, verbose=0)
        oof[val_idx] = model.predict(X.iloc[val_idx])
        test_preds += model.predict(X_test) / N_FOLDS
        print(f"  Fold {fold+1} R2={r2_score(y.iloc[val_idx], np.clip(oof[val_idx], 0, None)):.5f}")
    return np.clip(oof, 0, None), np.clip(test_preds, 0, None)

print("Training LightGBM...")
oof_lgb, test_lgb = train_lgb_kfold(train_fe[features], target, test_fe[features], lgb_params)
lgb_r2 = r2_score(target, oof_lgb)
print(f"LGB Overall R2: {lgb_r2:.6f} | Scaled: {max(0, 100*lgb_r2):.2f}\n")

print("Training XGBoost...")
oof_xgb, test_xgb = train_xgb_kfold(train_fe[features], target, test_fe[features], xgb_params)
xgb_r2 = r2_score(target, oof_xgb)
print(f"XGB Overall R2: {xgb_r2:.6f} | Scaled: {max(0, 100*xgb_r2):.2f}\n")

print("Training CatBoost...")
oof_cat, test_cat = train_cat_kfold(train_fe[features], target, test_fe[features], cat_params)
cat_r2 = r2_score(target, oof_cat)
print(f"CAT Overall R2: {cat_r2:.6f} | Scaled: {max(0, 100*cat_r2):.2f}")

def find_best_weights(oof_lgb, oof_xgb, oof_cat, y):
    best_r2, best_w = -np.inf, (1/3, 1/3, 1/3)
    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            r2 = r2_score(y, np.clip(w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat, 0, None))
            if r2 > best_r2:
                best_r2, best_w = r2, (round(w1, 2), round(w2, 2), round(w3, 2))
    return best_w, best_r2


best_w, ensemble_r2 = find_best_weights(oof_lgb, oof_xgb, oof_cat, target)

print(f"Optimal Weights -> LGB: {best_w[0]}, XGB: {best_w[1]}, CAT: {best_w[2]}")
print(f"Ensemble OOF R2: {ensemble_r2:.6f} | Scaled: {max(0, 100*ensemble_r2):.2f}")
print(f"\nIndividual model comparison:")
print(f"  LGB R2: {lgb_r2:.6f} | Scaled: {max(0, 100*lgb_r2):.2f}")
print(f"  XGB R2: {xgb_r2:.6f} | Scaled: {max(0, 100*xgb_r2):.2f}")
print(f"  CAT R2: {cat_r2:.6f} | Scaled: {max(0, 100*cat_r2):.2f}")
print(f"  ENS R2: {ensemble_r2:.6f} | Scaled: {max(0, 100*ensemble_r2):.2f}")

use_ensemble = ensemble_r2 >= cat_r2

if use_ensemble:
    final_preds = np.clip(best_w[0]*test_lgb + best_w[1]*test_xgb + best_w[2]*test_cat, 0, None)
    print(f"Using ensemble predictions (R2={ensemble_r2:.6f})")
else:
    final_preds = np.clip(test_cat, 0, None)
    print(f"Using pure CatBoost predictions (R2={cat_r2:.6f})")

submission = pd.DataFrame({"Index": test_idx.values, "demand": final_preds})
submission.to_csv("submission.csv", index=False)

print(f"\nSubmission shape: {submission.shape}")
print(f"Demand stats:\n{submission['demand'].describe()}")
submission.head(10)