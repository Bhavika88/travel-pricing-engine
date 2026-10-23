"""
train_demand_model.py
Trains an LSTM to forecast daily booking volume for the next 7 days.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from prepare_demand_data import load_and_aggregate, create_sequences, FEATURE_COLS

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/demand_lstm.keras"
SCALER_PATH = f"{MODEL_DIR}/demand_scaler.pkl"

LOOKBACK = 30
HORIZON = 7


# ============================================================
# 1. PREPARE DATA
# ============================================================
def prepare_data(test_days=60):
    df = load_and_aggregate()
    X, y, dates = create_sequences(df, LOOKBACK, HORIZON)

    # Chronological split: last `test_days` sequences for testing
    split = len(X) - test_days
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    # Scale target & n_bookings feature
    # (LSTM works best with normalized inputs)
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten().reshape(y_train.shape)
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten().reshape(y_test.shape)

    # Scale n_bookings inside X (first feature)
    n_bookings_train = X_train[:, :, 0].reshape(-1, 1)
    scaler_X = StandardScaler().fit(n_bookings_train)
    X_train[:, :, 0] = scaler_X.transform(X_train[:, :, 0].reshape(-1, 1)).reshape(X_train.shape[0], LOOKBACK)
    X_test[:, :, 0] = scaler_X.transform(X_test[:, :, 0].reshape(-1, 1)).reshape(X_test.shape[0], LOOKBACK)

    return X_train, X_test, y_train_scaled, y_test_scaled, y_test, scaler, scaler_X, dates_test


# ============================================================
# 2. BUILD LSTM
# ============================================================
def build_lstm(n_features):
    model = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(HORIZON),  # 7-day forecast
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="mse",
                  metrics=["mae"])
    return model


# ============================================================
# 3. TRAIN
# ============================================================
def train():
    X_train, X_test, y_train, y_test, y_test_unscaled, scaler, scaler_X, dates_test = prepare_data()

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    model = build_lstm(n_features=X_train.shape[2])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
    y_true = y_test_unscaled

    mae = mean_absolute_error(y_true.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), preds.flatten()))
    print(f"\n📊 LSTM Demand Model (test set):")
    print(f"  MAE:  {mae:.2f} bookings/day")
    print(f"  RMSE: {rmse:.2f} bookings/day")

    # Save model & scalers
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"target_scaler": scaler, "feature_scaler": scaler_X}, f)
    print(f"✅ Saved to {MODEL_PATH}")

    return model, history, preds, y_true, dates_test


# ============================================================
# 4. VISUALIZE
# ============================================================
def plot_forecast(preds, y_true, dates):
    """
    Plot actual vs predicted for a single test window
    (plus aggregate metrics).
    """
    os.makedirs("docs/images", exist_ok=True)

    # Plot training loss curve
    # (pass history)

    # Example: pick one window and show 7-day forecast vs actual
    sample_idx = 20
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, HORIZON + 1), y_true[sample_idx], marker="o", label="Actual", color="black")
    plt.plot(range(1, HORIZON + 1), preds[sample_idx], marker="s", label="Forecast", color="crimson", linestyle="--")
    plt.title(f"7-Day Demand Forecast (Window starting {pd.Timestamp(dates[sample_idx]).date()})")
    plt.xlabel("Days ahead")
    plt.ylabel("Bookings")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/images/lstm_forecast_sample.png", dpi=150)
    plt.close()
    print("✅ Saved docs/images/lstm_forecast_sample.png")

    # Plot aggregate: mean predicted vs mean actual across all windows
    mean_pred = preds.mean(axis=0)
    mean_true = y_true.mean(axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, HORIZON + 1), mean_true, marker="o", label="Avg Actual", color="black")
    plt.plot(range(1, HORIZON + 1), mean_pred, marker="s", label="Avg Forecast", color="crimson", linestyle="--")
    plt.title("Average 7-Day Demand: Forecast vs Actual (across all test windows)")
    plt.xlabel("Days ahead")
    plt.ylabel("Avg bookings/day")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/images/lstm_forecast_aggregate.png", dpi=150)
    plt.close()
    print("✅ Saved docs/images/lstm_forecast_aggregate.png")


if __name__ == "__main__":
    model, history, preds, y_true, dates = train()
    plot_forecast(preds, y_true, dates)