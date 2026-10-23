"""
prepare_demand_data.py
Builds a daily time series of booking volume + features,
then creates sliding windows for LSTM training.
"""

import pandas as pd
import numpy as np
import sqlite3
import os

DB_PATH = "data/processed/travel_pricing.db"
OUTPUT_PATH = "data/processed/demand_timeseries.csv"


def load_and_aggregate():
    """Aggregate bookings into a daily time series."""
    conn = sqlite3.connect(DB_PATH)
    bookings = pd.read_sql("SELECT * FROM bookings", conn)
    calendar = pd.read_sql("SELECT * FROM calendar", conn)
    events = pd.read_sql("SELECT * FROM external_events", conn)
    conn.close()

    bookings["booking_date"] = pd.to_datetime(bookings["booking_date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    # Daily booking volume — this is our target
    daily_volume = (
        bookings.groupby("booking_date")
        .size()
        .reset_index(name="n_bookings")
        .rename(columns={"booking_date": "date"})
    )

    # Merge with calendar (for temporal features)
    df = calendar.merge(daily_volume, on="date", how="left")
    df["n_bookings"] = df["n_bookings"].fillna(0).astype(int)

    # Add event flag: does any event start on this date?
    event_dates = set()
    for _, ev in events.iterrows():
        for d in pd.date_range(ev["start_date"], ev["end_date"]):
            event_dates.add(d)
    df["has_event"] = df["date"].isin(event_dates).astype(int)

    # Add daily average price (another useful signal)
    daily_price = (
        bookings.groupby("booking_date")["price_paid"]
        .mean()
        .reset_index()
        .rename(columns={"booking_date": "date", "price_paid": "avg_price"})
    )
    df = df.merge(daily_price, on="date", how="left")
    df["avg_price"] = df["avg_price"].interpolate()  # smooth gaps

    # Cyclical encoding for day_of_week (preserves weekday proximity)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.sort_values("date").reset_index(drop=True)
    return df


FEATURE_COLS = [
    "n_bookings",       # target for lagged input
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday", "has_event",
]


def create_sequences(df, lookback=30, horizon=7):
    """
    Create (X, y) sliding windows.

    X: (n_samples, lookback, n_features)
    y: (n_samples, horizon)  — next `horizon` days of booking volume
    """
    features = df[FEATURE_COLS].values.astype(np.float32)
    target = df["n_bookings"].values.astype(np.float32)

    X, y, dates = [], [], []
    for i in range(len(df) - lookback - horizon + 1):
        X.append(features[i : i + lookback])
        y.append(target[i + lookback : i + lookback + horizon])
        dates.append(df["date"].iloc[i + lookback])

    return np.array(X), np.array(y), np.array(dates)


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = load_and_aggregate()
    df.to_csv(OUTPUT_PATH, index=False)

    X, y, dates = create_sequences(df)
    print(f"✅ Daily time series: {len(df)} days")
    print(f"✅ Sequences: X={X.shape}, y={y.shape}")
    print(f"✅ Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nTarget stats (n_bookings):")
    print(df["n_bookings"].describe())