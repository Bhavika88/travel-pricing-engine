"""
train_pricing_model.py
Trains an XGBoost regression model to predict price_paid.
Outputs: models/pricing_model.pkl + evaluation metrics.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
import xgboost as xgb

FEATURES_PATH = "data/processed/features.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/pricing_model.pkl"

# Columns NOT to use as features
DROP_COLS = [
    "price_paid",           # target
    "villa_id",             # ID, not predictive
    "villa_avg_price_30d",  # ⚠️ This is a LEAKY feature — it IS price-related
]

# Note on villa_avg_price_30d:
# We built it in Step 3, but using it to predict price_paid is circular.
# A real pricing model must predict price WITHOUT seeing past prices of the same villa.
# We'll keep it in the dataset for other analyses but exclude it from modeling.
# Document this in your README — it's a senior-level consideration.


# ============================================================
# 1. LOAD AND PREPARE
# ============================================================
def load_features():
    df = pd.read_csv(FEATURES_PATH)
    # Drop leaky + ID columns
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df["price_paid"]
    return X, y, df


# ============================================================
# 2. CHRONOLOGICAL SPLIT
# ============================================================
def chronological_split(X, y, test_size=0.2):
    """
    Split based on time, not random.
    Critical for time-series data — random split causes leakage.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 3. BASELINE MODEL
# ============================================================
def train_baseline(X_train, y_train, X_test, y_test):
    """
    Baseline = predict the training mean.
    Any real model MUST beat this.
    """
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    preds = baseline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n📊 BASELINE (predict mean):")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")
    return rmse, r2


# ============================================================
# 4. TRAIN XGBOOST (default params)
# ============================================================
def train_xgboost_default(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    print(f"\n📊 XGBOOST (default params):")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    return model, rmse, r2


# ============================================================
# 5. HYPERPARAMETER TUNING
# ============================================================
def tune_xgboost(X_train, y_train):
    """
    RandomizedSearchCV with TimeSeriesSplit — the right CV for time data.
    """
    param_grid = {
        "n_estimators": [200, 300, 500, 800],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"\n🎯 BEST PARAMS: {search.best_params_}")
    print(f"   Best CV RMSE: ${-search.best_score_:,.2f}")
    return search.best_estimator_


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
def analyze_importance(model, X_train):
    """
    XGBoost gives us gain-based feature importances.
    Compare against EDA hypotheses from Step 3.
    """
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n🏆 TOP 15 FEATURE IMPORTANCES (gain):")
    print(importance.head(15).to_string(index=False))
    return importance


# ============================================================
# 7. PRICING SIMULATION
# ============================================================
def simulate_pricing(model, X_test, y_test):
    """
    Given the model predicts "market-clearing price",
    simulate a pricing strategy that adjusts based on demand.

    Strategy:
    - Base price = model prediction
    - If occupancy is HIGH (peak season, event) → add 15% premium
    - If occupancy is LOW (off-season, no event) → apply 10% discount
    """
    preds = model.predict(X_test)
    sim_prices = []
    for i, (pred, x) in enumerate(zip(preds, X_test.to_dict("records"))):
        price = pred
        # High-demand signals
        high_demand = (
            x.get("is_peak_season", 0) == 1
            or x.get("has_event", 0) == 1
            or x.get("is_holiday", 0) == 1
        )
        low_demand = (
            x.get("season_encoded", 2) == 0  # off season
            and x.get("has_event", 0) == 0
        )
        if high_demand:
            price *= 1.15
        elif low_demand:
            price *= 0.90
        sim_prices.append(price)

    sim_prices = np.array(sim_prices)

    # Compare simulated revenue vs. static pricing (always use mean)
    static_price = np.full_like(y_test, y_test.mean(), dtype=float)
    rev_simulated = sim_prices.sum()
    rev_static = static_price.sum()
    uplift = (rev_simulated / rev_static - 1) * 100

    print(f"\n💰 PRICING SIMULATION:")
    print(f"  Static pricing revenue:   ${rev_static:,.2f}")
    print(f"  Simulated pricing revenue: ${rev_simulated:,.2f}")
    print(f"  Revenue uplift:           {uplift:.2f}%")
    return sim_prices, uplift


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, df = load_features()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = chronological_split(X, y)

    baseline_rmse, baseline_r2 = train_baseline(X_train, y_train, X_test, y_test)
    default_model, default_rmse, default_r2 = train_xgboost_default(
        X_train, y_train, X_test, y_test
    )

    print("\n🔍 Starting hyperparameter tuning...")
    best_model = tune_xgboost(X_train, y_train)

    # Evaluate tuned model
    preds = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    print(f"\n📊 TUNED XGBOOST (test set):")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    print(f"  Improvement over baseline RMSE: {(1 - rmse/baseline_rmse)*100:.1f}%")

    importance = analyze_importance(best_model, X_train)
    importance.to_csv("models/feature_importance.csv", index=False)

    sim_prices, uplift = simulate_pricing(best_model, X_test, y_test)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": best_model,
            "feature_names": list(X_train.columns),
        }, f)
    print(f"\n✅ Model saved to {MODEL_PATH} with {len(X_train.columns)} feature names")