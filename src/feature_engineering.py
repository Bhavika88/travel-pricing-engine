"""
feature_engineering.py
Builds model-ready features from raw data in SQLite.
Output: data/processed/features.csv (one row per booking)
"""

import pandas as pd
import numpy as np
import sqlite3
import os

DB_PATH = "data/processed/travel_pricing.db"
OUTPUT_PATH = "data/processed/features.csv"


# ============================================================
# 1. LOAD RAW DATA
# ============================================================
def load_data():
    conn = sqlite3.connect(DB_PATH)
    villas = pd.read_sql("SELECT * FROM villas", conn)
    bookings = pd.read_sql("SELECT * FROM bookings", conn)
    calendar = pd.read_sql("SELECT * FROM calendar", conn)
    events = pd.read_sql("SELECT * FROM external_events", conn)
    conn.close()

    # Convert date columns
    bookings["check_in_date"] = pd.to_datetime(bookings["check_in_date"])
    bookings["check_out_date"] = pd.to_datetime(bookings["check_out_date"])
    bookings["booking_date"] = pd.to_datetime(bookings["booking_date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])
    return villas, bookings, calendar, events


# ============================================================
# 2. VILLA FEATURES
# ============================================================
def build_villa_features(villas):
    v = villas.copy()
    v["amenity_score"] = (
        v["has_pool"].astype(int)
        + v["has_butler"].astype(int)
        + v["has_private_beach"].astype(int)
    )
    v["price_per_bedroom"] = v["base_price"] / v["bedrooms"]
    v["guest_bedroom_ratio"] = v["max_guests"] / v["bedrooms"]
    return v[[
        "villa_id", "bedrooms", "max_guests", "has_pool", "has_butler",
        "has_private_beach", "rating", "amenity_score",
        "price_per_bedroom", "guest_bedroom_ratio", "base_price", "location"
    ]]


# ============================================================
# 3. TEMPORAL FEATURES (merge with calendar)
# ============================================================
def build_temporal_features(bookings, calendar):
    b = bookings.merge(
        calendar.rename(columns={"date": "check_in_date"}),
        on="check_in_date", how="left"
    )
    b["quarter"] = b["check_in_date"].dt.quarter
    b["week_of_year"] = b["check_in_date"].dt.isocalendar().week.astype(int)
    b["is_peak_season"] = (b["season"] == "Peak").astype(int)
    b["is_shoulder_season"] = (b["season"] == "Shoulder").astype(int)
    return b


# ============================================================
# 4. BOOKING BEHAVIOR FEATURES
# ============================================================
def build_booking_features(b):
    b = b.copy()
    b["is_last_minute"] = (b["lead_time_days"] < 14).astype(int)
    b["is_early_bird"] = (b["lead_time_days"] > 120).astype(int)
    b["lead_time_bucket"] = pd.cut(
        b["lead_time_days"],
        bins=[0, 14, 60, 120, 365],
        labels=["last_minute", "standard", "advance", "early_bird"]
    ).astype(str)
    b["nights_bucket"] = pd.cut(
        b["nights"],
        bins=[0, 3, 7, 14, 30],
        labels=["short", "medium", "long", "extended"]
    ).astype(str)
    return b


# ============================================================
# 5. EVENT FEATURES (the magic sauce)
# ============================================================
def build_event_features(bookings, events):
    """
    For each booking, check if the stay overlaps with any event
    in the same location. Compute proximity and size features.
    """
    b = bookings.copy()
    b["has_event"] = 0
    b["n_events_during_stay"] = 0
    b["event_size"] = 0
    b["days_until_event"] = 999  # sentinel

    for idx, row in b.iterrows():
        # Find events in same location that overlap with stay
        loc_events = events[events["location"] == row["location"]] \
            if "location" in b.columns else events.iloc[0:0]

        overlapping = loc_events[
            (loc_events["start_date"] <= row["check_out_date"]) &
            (loc_events["end_date"] >= row["check_in_date"])
        ]

        if len(overlapping) > 0:
            b.at[idx, "has_event"] = 1
            b.at[idx, "n_events_during_stay"] = len(overlapping)
            b.at[idx, "event_size"] = int(overlapping["expected_attendees"].max())
        else:
            # Nearest future event in the same location
            future = loc_events[loc_events["start_date"] > row["check_in_date"]]
            if len(future) > 0:
                nearest = future.sort_values("start_date").iloc[0]
                delta = (nearest["start_date"] - row["check_in_date"]).days
                b.at[idx, "days_until_event"] = min(delta, 999)

    return b


# ============================================================
# 6. HISTORICAL AGGREGATES (no leakage!)
# ============================================================
def build_lag_features(bookings):
    """
    For each booking, compute rolling aggregates using ONLY
    bookings that happened BEFORE it. Critical to avoid leakage.
    """
    b = bookings.sort_values("booking_date").copy()

    # Per-villa rolling 30-day avg price (based on booking_date)
    b["villa_avg_price_30d"] = (
        b.groupby("villa_id")["price_paid"]
         .transform(lambda s: s.rolling(window=30, min_periods=1).mean().shift(1))
    )

    # Per-villa lifetime booking count (before this booking)
    b["villa_total_bookings"] = b.groupby("villa_id").cumcount()

    # Per-villa avg lead time (before this booking)
    b["villa_avg_lead_time"] = (
        b.groupby("villa_id")["lead_time_days"]
         .transform(lambda s: s.rolling(window=30, min_periods=1).mean().shift(1))
    )

    # Fill NaNs for first bookings (no history yet)
    for col in ["villa_avg_price_30d", "villa_avg_lead_time"]:
        b[col] = b[col].fillna(b[col].median())

    return b


# ============================================================
# 7. ENCODE CATEGORICALS
# ============================================================
def encode_categoricals(b):
    b = b.copy()
    # Location target encoding would require care; here we one-hot
    loc_dummies = pd.get_dummies(b["location"], prefix="loc", drop_first=True)
    b = pd.concat([b, loc_dummies], axis=1)

    # Ordinal encode seasons
    season_map = {"Off": 0, "Shoulder": 1, "Peak": 2}
    b["season_encoded"] = b["season"].map(season_map).fillna(0).astype(int)

    # Bucket encodes
    lead_map = {"last_minute": 0, "standard": 1, "advance": 2, "early_bird": 3}
    b["lead_time_bucket_enc"] = b["lead_time_bucket"].map(lead_map).fillna(1).astype(int)

    nights_map = {"short": 0, "medium": 1, "long": 2, "extended": 3}
    b["nights_bucket_enc"] = b["nights_bucket"].map(nights_map).fillna(1).astype(int)

    return b


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    villas, bookings, calendar, events = load_data()

    villa_f = build_villa_features(villas)
    merged = bookings.merge(villa_f, on="villa_id", how="left", suffixes=("", "_villa"))

    temporal = build_temporal_features(merged, calendar)
    booking_f = build_booking_features(temporal)
    event_f = build_event_features(booking_f, events)
    lag_f = build_lag_features(event_f)
    final = encode_categoricals(lag_f)

    # Drop columns not needed for modeling
    drop_cols = [
        "booking_id", "name", "booking_date", "check_in_date",
        "check_out_date", "lead_time_bucket", "nights_bucket",
        "season", "location"
    ]
    final = final.drop(columns=[c for c in drop_cols if c in final.columns])

    final.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Features saved to {OUTPUT_PATH}")
    print(f"Shape: {final.shape}")
    print(f"Columns: {list(final.columns)}")
    print(f"\nTarget distribution (price_paid):")
    print(final["price_paid"].describe())