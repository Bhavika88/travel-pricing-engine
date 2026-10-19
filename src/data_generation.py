"""
data_generation.py
Generates realistic synthetic data for the Travel Pricing Engine.
4 tables: villas, bookings, calendar, external_events.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os
import sqlite3

# Reproducibility — critical for interviews
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

# ---- CONFIG ----
N_VILLAS = 200
N_BOOKINGS = 15000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

LOCATIONS = {
    # location: (base_price_multiplier, season_peak_months)
    "Santorini":    (1.3, [6, 7, 8]),
    "Maldives":     (1.5, [12, 1, 2, 3]),
    "Cannes":       (1.4, [5, 6, 7]),
    "Aspen":        (1.6, [12, 1, 2, 3]),
    "Bali":         (1.0, [6, 7, 8]),
    "Tuscany":      (1.2, [5, 6, 9, 10]),
    "Dubai":        (1.3, [11, 12, 1, 2]),
    "Amalfi Coast": (1.35, [6, 7, 8, 9]),
}

EVENT_TEMPLATES = [
    ("Film Festival", 40000),
    ("F1 Grand Prix", 100000),
    ("Art Biennale", 25000),
    ("Wine Expo", 15000),
    ("Music Festival", 50000),
    ("Sailing Regatta", 10000),
]

# ============================================================
# TABLE 1: VILLAS
# ============================================================
def generate_villas():
    print("Generating villas...")
    villas = []
    for i in range(1, N_VILLAS + 1):
        loc = random.choice(list(LOCATIONS.keys()))
        loc_mult = LOCATIONS[loc][0]

        bedrooms = np.random.choice([2, 3, 4, 5, 6], p=[0.15, 0.3, 0.3, 0.15, 0.1])
        max_guests = bedrooms * 2
        has_pool = np.random.rand() < 0.85  # 85% have pools
        has_butler = np.random.rand() < 0.4  # 40% have butlers
        has_private_beach = np.random.rand() < 0.25  # 25% have beaches
        rating = round(np.clip(np.random.normal(4.6, 0.3), 3.5, 5.0), 2)

        # Base price logic: location * bedrooms * amenities + noise
        base = 300 * loc_mult
        base += bedrooms * 150
        base += 200 if has_pool else 0
        base += 350 if has_butler else 0
        base += 500 if has_private_beach else 0
        base *= (rating / 4.5)  # ratings boost price
        base = round(base * np.random.uniform(0.9, 1.1), 2)

        villas.append({
            "villa_id": i,
            "name": f"Villa {fake.last_name()}",
            "location": loc,
            "bedrooms": int(bedrooms),
            "max_guests": int(max_guests),
            "has_pool": bool(has_pool),
            "has_butler": bool(has_butler),
            "has_private_beach": bool(has_private_beach),
            "rating": rating,
            "base_price": base,
        })
    return pd.DataFrame(villas)


# ============================================================
# TABLE 2: BOOKINGS (with embedded signals)
# ============================================================
def generate_bookings(villas_df):
    print("Generating bookings (with seasonality, lead time, events)...")
    bookings = []
    total_days = (END_DATE - START_DATE).days

    for i in range(1, N_BOOKINGS + 1):
        villa = villas_df.sample(1).iloc[0]
        loc = villa["location"]
        peak_months = LOCATIONS[loc][1]

        # --- SIGNAL 1: Seasonality ---
        # Bias booking dates toward peak season
        for _ in range(100):  # rejection sampling
            check_in = START_DATE + timedelta(days=random.randint(0, total_days - 10))
            if check_in.month in peak_months:
                if np.random.rand() < 0.7:  # 70% chance to keep peak booking
                    break
            else:
                if np.random.rand() < 0.4:  # 40% chance to keep off-peak
                    break

        nights = np.random.choice([3, 4, 5, 7, 10, 14], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        check_out = check_in + timedelta(days=int(nights))

        # --- SIGNAL 2: Lead Time ---
        # Luxury: most bookings made 30-180 days in advance
        lead_time = int(np.clip(np.random.lognormal(mean=4.2, sigma=0.7), 1, 300))
        booking_date = check_in - timedelta(days=lead_time)

        guests = random.randint(2, villa["max_guests"])

        # --- SIGNAL 3: Pricing Logic (Luxury Premium + Seasonality + Lead Time) ---
        price = villa["base_price"]
        if check_in.month in peak_months:
            price *= 1.5  # peak multiplier
        if check_in.weekday() >= 5:  # weekend
            price *= 1.15
        if lead_time < 14:  # last-minute premium
            price *= 1.25
        elif lead_time > 120:  # early bird discount
            price *= 0.92
        price *= np.random.uniform(0.95, 1.05)  # noise
        price = round(price * nights, 2)

        bookings.append({
            "booking_id": i,
            "villa_id": int(villa["villa_id"]),
            "booking_date": booking_date.date(),
            "check_in_date": check_in.date(),
            "check_out_date": check_out.date(),
            "nights": int(nights),
            "guests": guests,
            "price_paid": price,
            "lead_time_days": lead_time,
        })
    return pd.DataFrame(bookings)


# ============================================================
# TABLE 3: CALENDAR
# ============================================================
def generate_calendar():
    print("Generating calendar...")
    holidays = [
        # (month, day) — simplified global holidays
        (1, 1), (12, 25), (12, 31), (7, 4), (5, 1), (8, 15), (11, 1),
    ]
    rows = []
    current = START_DATE
    while current <= END_DATE:
        is_weekend = current.weekday() >= 5
        is_holiday = any(current.month == m and current.day == d for m, d in holidays)

        # Season assignment based on month
        if current.month in [6, 7, 8, 12]:
            season = "Peak"
        elif current.month in [4, 5, 9, 10]:
            season = "Shoulder"
        else:
            season = "Off"

        rows.append({
            "date": current.date(),
            "day_of_week": current.weekday(),
            "month": current.month,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "season": season,
        })
        current += timedelta(days=1)
    return pd.DataFrame(rows)


# ============================================================
# TABLE 4: EXTERNAL EVENTS
# ============================================================
def generate_events():
    print("Generating external events...")
    events = []
    event_id = 1
    for loc in LOCATIONS.keys():
        # 3-5 events per location over 2 years
        n_events = random.randint(3, 5)
        for _ in range(n_events):
            name, attendees = random.choice(EVENT_TEMPLATES)
            start = START_DATE + timedelta(days=random.randint(0, 700))
            duration = random.randint(3, 14)
            end = start + timedelta(days=duration)
            events.append({
                "event_id": event_id,
                "location": loc,
                "event_name": f"{loc} {name}",
                "start_date": start.date(),
                "end_date": end.date(),
                "expected_attendees": attendees,
            })
            event_id += 1
    return pd.DataFrame(events)


# ============================================================
# SAVE TO SQLITE + CSV
# ============================================================
def save_all(villas, bookings, calendar, events):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Save CSVs (for easy inspection)
    villas.to_csv("data/raw/villas.csv", index=False)
    bookings.to_csv("data/raw/bookings.csv", index=False)
    calendar.to_csv("data/raw/calendar.csv", index=False)
    events.to_csv("data/raw/external_events.csv", index=False)

    # Save SQLite DB
    conn = sqlite3.connect("data/processed/travel_pricing.db")
    villas.to_sql("villas", conn, if_exists="replace", index=False)
    bookings.to_sql("bookings", conn, if_exists="replace", index=False)
    calendar.to_sql("calendar", conn, if_exists="replace", index=False)
    events.to_sql("external_events", conn, if_exists="replace", index=False)
    conn.close()
    print("✅ Saved to data/raw/*.csv and data/processed/travel_pricing.db")


if __name__ == "__main__":
    villas = generate_villas()
    bookings = generate_bookings(villas)
    calendar = generate_calendar()
    events = generate_events()
    save_all(villas, bookings, calendar, events)

    # Quick sanity check
    print("\n=== DATA SUMMARY ===")
    print(f"Villas: {len(villas)}")
    print(f"Bookings: {len(bookings)}")
    print(f"Calendar days: {len(calendar)}")
    print(f"Events: {len(events)}")
    print("\nSample villa:")
    print(villas.head(2).to_string())
    print("\nSample booking:")
    print(bookings.head(2).to_string())