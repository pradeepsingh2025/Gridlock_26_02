import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import json
import os

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_FOLDS = 5
DATA_DIR = "./dataset"
np.random.seed(SEED)


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, test


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

    le_road = LabelEncoder()
    combined["RoadType"] = combined["RoadType"].fillna("Unknown")
    combined["RoadType_encoded"] = le_road.fit_transform(combined["RoadType"])

    le_weather = LabelEncoder()
    combined["Weather"] = combined["Weather"].fillna("Unknown")
    combined["Weather_encoded"] = le_weather.fit_transform(combined["Weather"])

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

    le_geo4 = LabelEncoder()
    combined["geohash_prefix4_encoded"] = le_geo4.fit_transform(combined["geohash_prefix4"])
    le_geo5 = LabelEncoder()
    combined["geohash_prefix5_encoded"] = le_geo5.fit_transform(combined["geohash_prefix5"])
    le_geo = LabelEncoder()
    combined["geohash_encoded"] = le_geo.fit_transform(combined["geohash"])

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


def add_target_encoding(train_fe, test_fe, target, cols, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for col in cols:
        train_fe[f"{col}_target_mean"] = np.nan
        global_mean = target.mean()

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_fe)):
            means = target.iloc[tr_idx].groupby(train_fe[col].iloc[tr_idx]).mean()
            train_fe.loc[val_idx, f"{col}_target_mean"] = train_fe[col].iloc[val_idx].map(means)

        train_fe[f"{col}_target_mean"] = train_fe[f"{col}_target_mean"].fillna(global_mean)

        overall_means = target.groupby(train_fe[col]).mean()
        test_fe[f"{col}_target_mean"] = test_fe[col].map(overall_means).fillna(global_mean)

    return train_fe, test_fe


def add_geohash_aggregations(train_fe, test_fe, target):
    train_fe["_target"] = target.values

    geo_stats = train_fe.groupby("geohash_encoded")["_target"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    ).reset_index()
    geo_stats.columns = ["geohash_encoded", "geo_demand_mean", "geo_demand_std",
                         "geo_demand_median", "geo_demand_min", "geo_demand_max", "geo_count"]
    geo_stats["geo_demand_std"] = geo_stats["geo_demand_std"].fillna(0)

    train_fe = train_fe.merge(geo_stats, on="geohash_encoded", how="left")
    test_fe = test_fe.merge(geo_stats, on="geohash_encoded", how="left")

    for c in ["geo_demand_mean", "geo_demand_std", "geo_demand_median",
              "geo_demand_min", "geo_demand_max", "geo_count"]:
        test_fe[c] = test_fe[c].fillna(train_fe[c].median())

    geo4_stats = train_fe.groupby("geohash_prefix4_encoded")["_target"].agg(
        ["mean", "std"]
    ).reset_index()
    geo4_stats.columns = ["geohash_prefix4_encoded", "geo4_demand_mean", "geo4_demand_std"]
    geo4_stats["geo4_demand_std"] = geo4_stats["geo4_demand_std"].fillna(0)

    train_fe = train_fe.merge(geo4_stats, on="geohash_prefix4_encoded", how="left")
    test_fe = test_fe.merge(geo4_stats, on="geohash_prefix4_encoded", how="left")
    for c in ["geo4_demand_mean", "geo4_demand_std"]:
        test_fe[c] = test_fe[c].fillna(train_fe[c].median())

    geo5_stats = train_fe.groupby("geohash_prefix5_encoded")["_target"].agg(
        ["mean", "std"]
    ).reset_index()
    geo5_stats.columns = ["geohash_prefix5_encoded", "geo5_demand_mean", "geo5_demand_std"]
    geo5_stats["geo5_demand_std"] = geo5_stats["geo5_demand_std"].fillna(0)

    train_fe = train_fe.merge(geo5_stats, on="geohash_prefix5_encoded", how="left")
    test_fe = test_fe.merge(geo5_stats, on="geohash_prefix5_encoded", how="left")
    for c in ["geo5_demand_mean", "geo5_demand_std"]:
        test_fe[c] = test_fe[c].fillna(train_fe[c].median())

    day_geo_stats = train_fe.groupby(["day", "geohash_encoded"])["_target"].agg(
        ["mean"]
    ).reset_index()
    day_geo_stats.columns = ["day", "geohash_encoded", "day_geo_demand_mean"]

    train_fe = train_fe.merge(day_geo_stats, on=["day", "geohash_encoded"], how="left")
    test_fe = test_fe.merge(day_geo_stats, on=["day", "geohash_encoded"], how="left")
    test_fe["day_geo_demand_mean"] = test_fe["day_geo_demand_mean"].fillna(
        test_fe["geo_demand_mean"]
    )

    road_stats = train_fe.groupby("RoadType_encoded")["_target"].agg(["mean", "std"]).reset_index()
    road_stats.columns = ["RoadType_encoded", "road_demand_mean", "road_demand_std"]

    train_fe = train_fe.merge(road_stats, on="RoadType_encoded", how="left")
    test_fe = test_fe.merge(road_stats, on="RoadType_encoded", how="left")
    for c in ["road_demand_mean", "road_demand_std"]:
        test_fe[c] = test_fe[c].fillna(train_fe[c].median())

    train_fe = train_fe.drop(columns=["_target"])

    return train_fe, test_fe


def get_feature_columns(train_fe):
    return [c for c in train_fe.columns]


def train_full_lgb(X, y, X_test, params, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds

    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)
    return oof_preds, test_preds


def train_full_xgb(X, y, X_test, params, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        oof_preds[val_idx] = model.predict(dval)
        test_preds += model.predict(dtest) / n_folds

    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)
    return oof_preds, test_preds


def train_full_cat(X, y, X_test, params, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        train_pool = cb.Pool(X_tr, label=y_tr)
        val_pool = cb.Pool(X_val, label=y_val)

        model = cb.CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            verbose=0
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_folds

    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)
    return oof_preds, test_preds


def optimize_cat(X, y, n_trials=10):
    def objective(trial):
        params = {
            "loss_function": "RMSE",
            "random_seed": SEED,
            "iterations": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "depth": trial.suggest_int("depth", 5, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 64, 255),
            "verbose": 0,
        }

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof_preds = np.zeros(len(X))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            train_pool = cb.Pool(X_tr, label=y_tr)
            val_pool = cb.Pool(X_val, label=y_val)

            model = cb.CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=0)

            preds = model.predict(X_val)
            oof_preds[val_idx] = np.clip(preds, 0, None)

        return r2_score(y, oof_preds)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_params.update({
        "loss_function": "RMSE",
        "random_seed": SEED,
        "iterations": 3000,
        "verbose": 0,
    })
    print(f"  CAT Best R2: {study.best_value:.6f}")
    return best_params, study.best_value


def find_best_weights(oof_lgb, oof_xgb, oof_cat, y):
    best_r2 = -np.inf
    best_weights = (1/3, 1/3, 1/3)

    for w1 in np.arange(0.1, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.8 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            ensemble = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
            ensemble = np.clip(ensemble, 0, None)
            r2 = r2_score(y, ensemble)
            if r2 > best_r2:
                best_r2 = r2
                best_weights = (w1, w2, w3)

    return best_weights, best_r2


def main():
    print("=" * 60)
    print("TRAFFIC DEMAND PREDICTION PIPELINE v2")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    train, test = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")

    print("\n[2/7] Engineering features...")
    train_fe, test_fe, target, train_idx, test_idx = engineer_features(train, test)

    te_cols = ["geohash_encoded", "geohash_prefix5_encoded", "RoadType_encoded"]
    train_fe, test_fe = add_target_encoding(train_fe, test_fe, target, te_cols)

    train_fe, test_fe = add_geohash_aggregations(train_fe, test_fe, target)

    features = get_feature_columns(train_fe)
    print(f"  Features: {len(features)}")

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.01,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": -1,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    }

    print("\n[3/7] Optimizing LightGBM with Optuna (20 trials)...")
    def lgb_objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": SEED,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros(len(train_fe))
        for fold, (tr_idx, val_idx) in enumerate(kf.split(train_fe)):
            dtrain = lgb.Dataset(train_fe[features].iloc[tr_idx], label=target.iloc[tr_idx])
            dval = lgb.Dataset(train_fe[features].iloc[val_idx], label=target.iloc[val_idx], reference=dtrain)
            model = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            oof[val_idx] = np.clip(model.predict(train_fe[features].iloc[val_idx]), 0, None)
        return r2_score(target, oof)

    lgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    lgb_study.optimize(lgb_objective, n_trials=20, show_progress_bar=True)
    lgb_params = lgb_study.best_params
    lgb_params.update({"objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
                       "verbosity": -1, "seed": SEED, "n_jobs": -1})
    print(f"  LGB Best R2: {lgb_study.best_value:.6f}")

    print("\n[4/7] Optimizing XGBoost with Optuna (20 trials)...")
    def xgb_objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "seed": SEED,
            "nthread": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros(len(train_fe))
        for fold, (tr_idx, val_idx) in enumerate(kf.split(train_fe)):
            dtrain = xgb.DMatrix(train_fe[features].iloc[tr_idx], label=target.iloc[tr_idx])
            dval = xgb.DMatrix(train_fe[features].iloc[val_idx], label=target.iloc[val_idx])
            model = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                              early_stopping_rounds=50, verbose_eval=False)
            oof[val_idx] = np.clip(model.predict(dval), 0, None)
        return r2_score(target, oof)

    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    xgb_study.optimize(xgb_objective, n_trials=20, show_progress_bar=True)
    xgb_params = xgb_study.best_params
    xgb_params.update({"objective": "reg:squarederror", "eval_metric": "rmse",
                       "tree_method": "hist", "seed": SEED, "nthread": -1})
    print(f"  XGB Best R2: {xgb_study.best_value:.6f}")

    print("\n[5/7] Optimizing CatBoost with Optuna (10 trials)...")
    cat_params, cat_best_r2 = optimize_cat(train_fe[features], target, n_trials=10)

    print("\n[6/7] Training final models with 5-Fold CV...")
    oof_lgb, test_lgb = train_full_lgb(train_fe[features], target, test_fe[features], lgb_params)
    lgb_r2 = r2_score(target, oof_lgb)
    print(f"  LGB OOF R2: {lgb_r2:.6f} (Scaled: {max(0, 100 * lgb_r2):.2f})")

    oof_xgb, test_xgb = train_full_xgb(train_fe[features], target, test_fe[features], xgb_params)
    xgb_r2 = r2_score(target, oof_xgb)
    print(f"  XGB OOF R2: {xgb_r2:.6f} (Scaled: {max(0, 100 * xgb_r2):.2f})")

    oof_cat, test_cat = train_full_cat(train_fe[features], target, test_fe[features], cat_params)
    cat_r2 = r2_score(target, oof_cat)
    print(f"  CAT OOF R2: {cat_r2:.6f} (Scaled: {max(0, 100 * cat_r2):.2f})")

    print("\n[7/7] Finding optimal ensemble weights...")
    best_weights, ensemble_r2 = find_best_weights(oof_lgb, oof_xgb, oof_cat, target)
    print(f"  Weights -> LGB: {best_weights[0]:.2f}, XGB: {best_weights[1]:.2f}, CAT: {best_weights[2]:.2f}")
    print(f"  Ensemble OOF R2: {ensemble_r2:.6f} (Scaled: {max(0, 100 * ensemble_r2):.2f})")

    final_preds = (best_weights[0] * test_lgb +
                   best_weights[1] * test_xgb +
                   best_weights[2] * test_cat)
    final_preds = np.clip(final_preds, 0, None)

    submission = pd.DataFrame({
        "Index": test_idx.values,
        "demand": final_preds
    })
    submission.to_csv("submission.csv", index=False)
    print(f"\n  Submission saved: submission.csv ({len(submission)} rows)")

    results = {
        "lgb_params": {k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in lgb_params.items()},
        "xgb_params": {k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in xgb_params.items()},
        "cat_params": {k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in cat_params.items()},
        "lgb_r2": lgb_r2,
        "xgb_r2": xgb_r2,
        "cat_r2": cat_r2,
        "ensemble_r2": ensemble_r2,
        "best_weights": list(best_weights),
        "features": features,
        "n_features": len(features),
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved: results.json")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Final Ensemble R2 Score (Scaled): {max(0, 100 * ensemble_r2):.2f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
