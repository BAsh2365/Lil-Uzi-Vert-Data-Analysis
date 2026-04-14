import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load & clean
uzi = pd.read_csv("uzi_songs.csv")
print(uzi.head())
print(f"\nDtypes:\n{uzi.dtypes}\n")

# Force numeric in case they load as strings
uzi["Streams"] = pd.to_numeric(uzi["Streams"].astype(str).str.replace(",", ""), errors="coerce")
uzi["Daily"]   = pd.to_numeric(uzi["Daily"].astype(str).str.replace(",", ""), errors="coerce")

uzi["is_feature"] = uzi["song_title"].str.startswith("*").astype(int)
uzi["song_title"] = uzi["song_title"].str.lstrip("*")
uzi = uzi[(uzi["Streams"] > 0) & (uzi["Daily"] > 0)].reset_index(drop=True)
print(f"Loaded {len(uzi)} tracks\n")

# Base transforms
uzi["log_streams"]    = np.log1p(uzi["Streams"].astype(float))
uzi["log_daily"]      = np.log1p(uzi["Daily"].astype(float))
uzi["velocity_ratio"] = uzi["Daily"].astype(float) / uzi["Streams"].astype(float)

# Polynomial features
uzi["log_streams_sq"]    = uzi["log_streams"] ** 2
uzi["log_daily_sq"]      = uzi["log_daily"] ** 2
uzi["velocity_ratio_sq"] = uzi["velocity_ratio"] ** 2

# Interaction features
uzi["log_streams_x_log_daily"] = uzi["log_streams"] * uzi["log_daily"]
uzi["streams_x_feature"]       = uzi["log_streams"] * uzi["is_feature"]
uzi["daily_x_feature"]         = uzi["log_daily"] * uzi["is_feature"]
uzi["velocity_x_feature"]      = uzi["velocity_ratio"] * uzi["is_feature"]

# Ratio features
uzi["daily_per_million"] = uzi["Daily"].astype(float) / (uzi["Streams"].astype(float) / 1e6)
uzi["streams_daily_gap"] = uzi["log_streams"] - uzi["log_daily"]

# Percentile rank — manual calculation for compatibility
uzi["stream_percentile"] = uzi["Streams"].rank(method="average").astype(float) / len(uzi)
uzi["daily_percentile"]  = uzi["Daily"].rank(method="average").astype(float) / len(uzi)

# Target: blend of catalog size + current momentum
uzi["rerelease_score"] = (
    0.5 * uzi["stream_percentile"] +
    0.5 * uzi["daily_percentile"]
)

# Sanity check — this should NOT be all zeros
print("rerelease_score sanity check:")
print(f"  min:  {uzi['rerelease_score'].min():.4f}")
print(f"  max:  {uzi['rerelease_score'].max():.4f}")
print(f"  mean: {uzi['rerelease_score'].mean():.4f}")
assert uzi["rerelease_score"].max() > 0, "ERROR: rerelease_score is all zeros — check dtypes!"
print("  ✓ Looks good\n")

# Train / test split
FEATURE_COLS = [
    "log_streams",
    "log_daily",
    "is_feature",
    "velocity_ratio",
    "log_streams_sq",
    "log_daily_sq",
    "velocity_ratio_sq",
    "log_streams_x_log_daily",
    "streams_x_feature",
    "daily_x_feature",
    "velocity_x_feature",
    "daily_per_million",
    "streams_daily_gap",
    "stream_percentile",
    "daily_percentile",
]

X = uzi[FEATURE_COLS].astype(float)
y = uzi["rerelease_score"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# XGBoost model
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# Evaluation metrics
print("=" * 50)
print("  MODEL PERFORMANCE")
print("=" * 50)
print(f"\n  {'Metric':<20} {'Train':>10} {'Test':>10}")
print(f"  {'-'*40}")
print(f"  {'R² Score':<20} {r2_score(y_train, y_pred_train):>10.4f} {r2_score(y_test, y_pred_test):>10.4f}")
print(f"  {'RMSE':<20} {np.sqrt(mean_squared_error(y_train, y_pred_train)):>10.4f} {np.sqrt(mean_squared_error(y_test, y_pred_test)):>10.4f}")
print(f"  {'MAE':<20} {mean_absolute_error(y_train, y_pred_train):>10.4f} {mean_absolute_error(y_test, y_pred_test):>10.4f}")

# Cross-validation
cv_scores = cross_val_score(
    model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2"
)
print(f"\n  5-Fold CV R²:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Fold scores:   {[round(float(s), 4) for s in cv_scores]}")
print()

# Feature importance plot
importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importance["feature"], importance["importance"], color="#7B2FF7", height=0.6)
ax.set_xlabel("Importance (Gain)", fontsize=12)
ax.set_title("XGBoost Feature Importance — Re-Release Predictor", fontsize=14, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Saved: feature_importance.png")

# Actual vs predicted plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred_test, alpha=0.6, color="#FF4365", edgecolors="white", s=60)
ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual Re-Release Score", fontsize=12)
ax.set_ylabel("Predicted Re-Release Score", fontsize=12)
ax.set_title("Actual vs Predicted — Test Set", fontsize=14, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()
print("Saved: actual_vs_predicted.png")

# Top re-release candidates
uzi["predicted_score"] = model.predict(X)
uzi["prediction_rank"] = uzi["predicted_score"].rank(ascending=False).astype(int)
uzi["upside"] = uzi["predicted_score"] - uzi["rerelease_score"]

print("\n" + "=" * 60)
print("  TOP 20 RE-RELEASE CANDIDATES (by predicted score)")
print("=" * 60)
top20 = uzi.nlargest(20, "predicted_score")[
    ["prediction_rank", "song_title", "is_feature", "Streams", "Daily",
     "rerelease_score", "predicted_score"]
].copy()
top20["Streams"] = top20["Streams"].apply(lambda x: f"{x/1e6:.1f}M")
top20["Daily"]   = top20["Daily"].apply(lambda x: f"{x/1e3:.1f}K")
top20["rerelease_score"] = top20["rerelease_score"].round(3)
top20["predicted_score"] = top20["predicted_score"].round(3)
print(top20.to_string(index=False))

print("\n" + "=" * 60)
print("  TOP 10 HIDDEN GEMS (highest upside — undervalued songs)")
print("=" * 60)
gems = uzi.nlargest(10, "upside")[
    ["song_title", "is_feature", "Streams", "Daily",
     "rerelease_score", "predicted_score", "upside"]
].copy()
gems["Streams"] = gems["Streams"].apply(lambda x: f"{x/1e6:.1f}M")
gems["Daily"]   = gems["Daily"].apply(lambda x: f"{x/1e3:.1f}K")
gems["rerelease_score"] = gems["rerelease_score"].round(3)
gems["predicted_score"] = gems["predicted_score"].round(3)
gems["upside"]          = gems["upside"].round(3)
print(gems.to_string(index=False))

# Export predictions
export_cols = [
    "prediction_rank", "song_title", "is_feature",
    "Streams", "Daily", "velocity_ratio",
    "rerelease_score", "predicted_score", "upside"
]
uzi[export_cols].sort_values("prediction_rank").to_csv(
    "uzi_predictions.csv", index=False
)
print("\nSaved: uzi_predictions.csv")
print("\n✓ XGBoost pipeline complete.")