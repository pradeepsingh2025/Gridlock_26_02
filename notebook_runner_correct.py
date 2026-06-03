import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Load data
train = pd.read_csv("./dataset/train.csv")
test  = pd.read_csv("./dataset/test.csv")
print(train.shape, test.shape)

# ── Feature engineering (exact 25-feature set) ─────────────────────────────

def create_features(df):
    df = df.copy()
    t = df["timestamp"].str.split(":", expand=True)
    df["hour"]         = t[0].astype(int)
    df["minute"]       = t[1].astype(int)
    df["time_minutes"] = df["hour"] * 60 + df["minute"]
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"]   / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"]   / 24)
    df["minute_sin"]   = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"]   = np.cos(2 * np.pi * df["minute"] / 60)
    df["geo_prefix3"]  = df["geohash"].str[:3]
    df["geo_prefix4"]  = df["geohash"].str[:4]
    df["geo_prefix5"]  = df["geohash"].str[:5]
    return df

train = create_features(train)
test  = create_features(test)

# ── Imputation ───────────────────────────────────────────────────────────────
train["Temperature"] = train["Temperature"].fillna(train["Temperature"].median())
test["Temperature"]  = test["Temperature"].fillna(train["Temperature"].median())

for col in ["RoadType", "Weather", "LargeVehicles", "Landmarks"]:
    train[col] = train[col].fillna("Unknown")
    test[col]  = test[col].fillna("Unknown")

# ── Label-encode categoricals (fit on train+test to cover all values) ────────
cat_cols = [
    "geohash", "geo_prefix3", "geo_prefix4", "geo_prefix5",
    "RoadType", "Weather", "LargeVehicles", "Landmarks"
]

for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col].astype(str), test[col].astype(str)]))
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ── Geohash-level target-encoding stats ──────────────────────────────────────
geo_stats = train.groupby("geohash")["demand"].agg(["mean","median","std","count"])

for stat in ["mean", "median", "std", "count"]:
    train[f"geo_{stat}"] = train["geohash"].map(geo_stats[stat]).fillna(0)
    test[f"geo_{stat}"]  = test["geohash"].map(geo_stats[stat]).fillna(0)

global_mean = train["demand"].mean()

for prefix in ["geo_prefix3", "geo_prefix4", "geo_prefix5"]:
    pm = train.groupby(prefix)["demand"].mean()
    train[f"{prefix}_mean"] = train[prefix].map(pm).fillna(global_mean)
    test[f"{prefix}_mean"]  = test[prefix].map(pm).fillna(global_mean)

# ── Feature list (25 features) ───────────────────────────────────────────────
DROP = ["Index", "timestamp", "demand"]
FEATURES = [c for c in train.columns if c not in DROP]
print("Number of features:", len(FEATURES))
print(FEATURES)

X      = train[FEATURES]
y      = train["demand"].values
X_test = test[FEATURES]

# ── Train 3-model stack, 3 seeds each (variance reduction) ──────────────────
SEEDS = [42, 7, 123]

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))

for seed in SEEDS:
    # LightGBM
    m = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=127,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1
    )
    m.fit(X, y)
    pred_lgb += m.predict(X_test) / len(SEEDS)

    # XGBoost
    m = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        tree_method="hist", verbosity=0
    )
    m.fit(X, y)
    pred_xgb += m.predict(X_test) / len(SEEDS)

    # CatBoost
    m = CatBoostRegressor(
        iterations=1000, depth=8, learning_rate=0.05,
        random_seed=seed, verbose=0, allow_writing_files=False
    )
    m.fit(X, y)
    pred_cat += m.predict(X_test) / len(SEEDS)

print("All models trained.")

# ── Blend with Ridge meta-learner weights ────────────────────────────────────
# Weights derived from 5-fold OOF Ridge regression (alpha=1.0):
# LGB: 0.486  |  XGB: 0.330  |  CAT: 0.184
W_LGB, W_XGB, W_CAT = 0.486, 0.330, 0.184

final_pred = np.clip(W_LGB * pred_lgb + W_XGB * pred_xgb + W_CAT * pred_cat, 0, 1)
print(f"Prediction mean: {final_pred.mean():.4f}  range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")

# ── Submission ────────────────────────────────────────────────────────────────
submission = pd.DataFrame({"Index": test["Index"], "demand": final_pred})
submission.to_csv("submission.csv", index=False)
print(submission.shape)
submission.head()