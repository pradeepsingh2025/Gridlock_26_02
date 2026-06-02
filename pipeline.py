import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import warnings
import json
import pickle
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


def add_geohash_aggregations(train_fe, test_fe, target, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    train_fe["_target"] = target.values
    global_median = target.median()

    def calc_oof_stats(group_cols, stat_funcs, prefix):
        stat_names = [f"{prefix}_{stat}" if stat != "count" else f"{prefix}_count" for stat in stat_funcs]
        for col in stat_names:
            train_fe[col] = np.nan
            
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_fe)):
            tr_data = train_fe.iloc[tr_idx]
            val_data = train_fe.iloc[val_idx]
            stats = tr_data.groupby(group_cols)["_target"].agg(stat_funcs).reset_index()
            stats.columns = group_cols + stat_names
            val_merged = val_data[group_cols].merge(stats, on=group_cols, how="left")
            for col in stat_names:
                train_fe.loc[val_idx, col] = val_merged[col].values
                
        full_stats = train_fe.groupby(group_cols)["_target"].agg(stat_funcs).reset_index()
        full_stats.columns = group_cols + stat_names
        return full_stats, stat_names

    # 1. geohash_encoded
    gh_stats, gh_cols = calc_oof_stats(["geohash_encoded"], ["mean", "std", "median", "min", "max", "count"], "geo_demand")
    test_fe = test_fe.merge(gh_stats, on=["geohash_encoded"], how="left")
    
    # 2. geohash_prefix4_encoded
    gh4_stats, gh4_cols = calc_oof_stats(["geohash_prefix4_encoded"], ["mean", "std"], "geo4_demand")
    test_fe = test_fe.merge(gh4_stats, on=["geohash_prefix4_encoded"], how="left")
    
    # 3. geohash_prefix5_encoded
    gh5_stats, gh5_cols = calc_oof_stats(["geohash_prefix5_encoded"], ["mean", "std"], "geo5_demand")
    test_fe = test_fe.merge(gh5_stats, on=["geohash_prefix5_encoded"], how="left")
    
    # 4. RoadType_encoded
    road_stats, road_cols = calc_oof_stats(["RoadType_encoded"], ["mean", "std"], "road_demand")
    test_fe = test_fe.merge(road_stats, on=["RoadType_encoded"], how="left")
    
    # 5. Day & Geohash Interaction
    day_geo_stats, day_geo_cols = calc_oof_stats(["day", "geohash_encoded"], ["mean"], "day_geo_demand")
    test_fe = test_fe.merge(day_geo_stats, on=["day", "geohash_encoded"], how="left")

    # 6. Advanced Temporal/Spatial Interactions
    adv_interactions = [
        (["geohash_encoded", "hour"], ["mean", "count"], "hour_geo"),
        (["geohash_encoded", "is_rush_hour"], ["mean"], "rush_geo"),
        (["geohash_encoded", "Weather_encoded"], ["mean"], "weather_geo"),
        (["geohash_encoded", "day_of_week"], ["mean"], "dow_geo")
    ]
    adv_all_cols = []
    for group_cols, stat_funcs, prefix in adv_interactions:
        adv_stats, adv_cols = calc_oof_stats(group_cols, stat_funcs, prefix)
        test_fe = test_fe.merge(adv_stats, on=group_cols, how="left")
        adv_all_cols.extend(adv_cols)
        
    all_stat_cols = gh_cols + gh4_cols + gh5_cols + road_cols + day_geo_cols + adv_all_cols
    for c in all_stat_cols:
        if "std" in c:
            train_fe[c] = train_fe[c].fillna(0)
            test_fe[c] = test_fe[c].fillna(0)
        else:
            train_fe[c] = train_fe[c].fillna(global_median)
            test_fe[c] = test_fe[c].fillna(train_fe[c].median())
            
    test_fe["day_geo_demand_mean"] = test_fe["day_geo_demand_mean"].fillna(test_fe["geo_demand_mean"])
    train_fe = train_fe.drop(columns=["_target"])

    return train_fe, test_fe


def get_feature_columns(train_fe):
    return [c for c in train_fe.columns]


def cross_validate_lgb(X, y, params, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    scores = []

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

        preds = model.predict(X_val)
        preds = np.clip(preds, 0, 1.0)
        oof_preds[val_idx] = preds
        score = r2_score(y_val, preds)
        scores.append(score)

    overall_r2 = r2_score(y, oof_preds)
    return overall_r2, oof_preds, scores


def cross_validate_xgb(X, y, params, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False
        )

        preds = model.predict(dval)
        preds = np.clip(preds, 0, 1.0)
        oof_preds[val_idx] = preds
        score = r2_score(y_val, preds)
        scores.append(score)

    overall_r2 = r2_score(y, oof_preds)
    return overall_r2, oof_preds, scores


def cross_validate_cat(X, y, params, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    scores = []

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

        preds = model.predict(X_val)
        preds = np.clip(preds, 0, 1.0)
        oof_preds[val_idx] = preds
        score = r2_score(y_val, preds)
        scores.append(score)

    overall_r2 = r2_score(y, oof_preds)
    return overall_r2, oof_preds, scores


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

    oof_preds = np.clip(oof_preds, 0, 1.0)
    test_preds = np.clip(test_preds, 0, 1.0)
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

    oof_preds = np.clip(oof_preds, 0, 1.0)
    test_preds = np.clip(test_preds, 0, 1.0)
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

    oof_preds = np.clip(oof_preds, 0, 1.0)
    test_preds = np.clip(test_preds, 0, 1.0)
    return oof_preds, test_preds


def optimize_lgb(X, y, n_trials=40):
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": SEED,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        r2, _, _ = cross_validate_lgb(X, y, params)
        return r2

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_params.update({
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
    })
    print(f"  LGB Best R2: {study.best_value:.6f}")
    return best_params, study.best_value


def optimize_xgb(X, y, n_trials=40):
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "seed": SEED,
            "nthread": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        r2, _, _ = cross_validate_xgb(X, y, params)
        return r2

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_params.update({
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": -1,
    })
    print(f"  XGB Best R2: {study.best_value:.6f}")
    return best_params, study.best_value


def optimize_cat(X, y, n_trials=30):
    def objective(trial):
        params = {
            "loss_function": "RMSE",
            "random_seed": SEED,
            "iterations": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "verbose": 0,
        }
        r2, _, _ = cross_validate_cat(X, y, params)
        return r2

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_params.update({
        "loss_function": "RMSE",
        "random_seed": SEED,
        "iterations": 5000,
        "verbose": 0,
    })
    print(f"  CAT Best R2: {study.best_value:.6f}")
    return best_params, study.best_value


def train_meta_learner(oof_lgb, oof_xgb, oof_cat, test_lgb, test_xgb, test_cat, train_fe, test_fe, target):
    meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cat])
    meta_test = np.column_stack([test_lgb, test_xgb, test_cat])

    important_feats = ["geohash_encoded_target_mean", "geo_demand_mean", "time_minutes"]
    for feat in important_feats:
        if feat in train_fe.columns:
            meta_train = np.column_stack([meta_train, train_fe[feat].values])
            meta_test = np.column_stack([meta_test, test_fe[feat].values])

    meta_model = Ridge(alpha=1.0, random_state=SEED)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(meta_train))
    final_test_preds = np.zeros(len(meta_test))

    for tr_idx, val_idx in kf.split(meta_train):
        X_tr, X_val = meta_train[tr_idx], meta_train[val_idx]
        y_tr = target.iloc[tr_idx]

        meta_model.fit(X_tr, y_tr)
        oof_preds[val_idx] = meta_model.predict(X_val)
        final_test_preds += meta_model.predict(meta_test) / N_FOLDS

    oof_preds = np.clip(oof_preds, 0, 1.0)
    final_test_preds = np.clip(final_test_preds, 0, 1.0)

    ensemble_r2 = r2_score(target, oof_preds)
    return oof_preds, final_test_preds, ensemble_r2


def main():
    print("=" * 60)
    print("TRAFFIC DEMAND PREDICTION PIPELINE")
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
    print(f"  Feature list: {features[:10]}... (+ {len(features)-10} more)")

    print("\n[3/7] Optimizing LightGBM hyperparameters...")
    lgb_params, lgb_best_r2 = optimize_lgb(train_fe[features], target, n_trials=40)

    print("\n[4/7] Optimizing XGBoost hyperparameters...")
    xgb_params, xgb_best_r2 = optimize_xgb(train_fe[features], target, n_trials=40)

    print("\n[5/7] Optimizing CatBoost hyperparameters...")
    cat_params, cat_best_r2 = optimize_cat(train_fe[features], target, n_trials=30)

    print("\n[6/7] Training final models with K-Fold...")
    oof_lgb, test_lgb = train_full_lgb(train_fe[features], target, test_fe[features], lgb_params)
    lgb_r2 = r2_score(target, oof_lgb)
    print(f"  LGB OOF R2: {lgb_r2:.6f} (Scaled: {max(0, 100 * lgb_r2):.2f})")

    oof_xgb, test_xgb = train_full_xgb(train_fe[features], target, test_fe[features], xgb_params)
    xgb_r2 = r2_score(target, oof_xgb)
    print(f"  XGB OOF R2: {xgb_r2:.6f} (Scaled: {max(0, 100 * xgb_r2):.2f})")

    oof_cat, test_cat = train_full_cat(train_fe[features], target, test_fe[features], cat_params)
    cat_r2 = r2_score(target, oof_cat)
    print(f"  CAT OOF R2: {cat_r2:.6f} (Scaled: {max(0, 100 * cat_r2):.2f})")

    print("\n[7/7] Training Stacking Meta-Learner...")
    oof_ensemble, final_preds, ensemble_r2 = train_meta_learner(
        oof_lgb, oof_xgb, oof_cat, test_lgb, test_xgb, test_cat, train_fe, test_fe, target
    )
    print(f"  Ensemble OOF R2: {ensemble_r2:.6f} (Scaled: {max(0, 100 * ensemble_r2):.2f})")

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
