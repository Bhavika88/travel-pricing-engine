"""
main.py
FastAPI backend serving the pricing and demand models.

Run:
    uvicorn src.api.main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import date, timedelta
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

from .schemas import (
    PriceRequest, PriceResponse,
    DemandRequest, DemandResponse,
    VillaOut, HealthResponse,
)


# ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "processed" / "travel_pricing.db"
PRICING_MODEL_PATH = BASE_DIR / "models" / "pricing_model.pkl"
DEMAND_MODEL_PATH = BASE_DIR / "models" / "demand_lstm.keras"
SCALER_PATH = BASE_DIR / "models" / "demand_scaler.pkl"

# Same columns used during training — MUST match exactly
PRICING_FEATURE_COLS = None  # loaded from model.pkl metadata


# ============================================================
# GLOBAL MODEL STORE (loaded once at startup)
# ============================================================
class ModelStore:
    pricing_model = None
    demand_model = None
    target_scaler = None
    feature_scaler = None
    feature_names = None

store = ModelStore()


# ============================================================
# LIFESPAN — load models once at startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Loading models...")
    try:
        with open(PRICING_MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
            # Allow both raw model and dict payload
            if isinstance(payload, dict):
                store.pricing_model = payload["model"]
                store.feature_names = payload.get("feature_names")
            else:
                store.pricing_model = payload
                store.feature_names = list(getattr(payload, "feature_names_in_", []))
        print(f"✅ Pricing model loaded ({len(store.feature_names)} features)")

        store.demand_model = tf.keras.models.load_model(DEMAND_MODEL_PATH)
        print("✅ Demand LSTM loaded")

        with open(SCALER_PATH, "rb") as f:
            scalers = pickle.load(f)
            store.target_scaler = scalers["target_scaler"]
            store.feature_scaler = scalers["feature_scaler"]
        print("✅ Scalers loaded")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        raise
    yield
    print("👋 Shutting down")


app = FastAPI(
    title="Travel Pricing Engine API",
    description="Dynamic pricing + demand forecasting for luxury villa rentals.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the HTML/JS frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DB HELPERS
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_villa(villa_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM villas WHERE villa_id = ?", (villa_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Villa {villa_id} not found")
    return dict(row)


# ============================================================
# FEATURE BUILDER
# ============================================================
def build_pricing_features(villa: dict, req: PriceRequest) -> pd.DataFrame:
    """
    Build a single-row feature vector matching the training schema.
    Must reproduce feature_engineering.py logic exactly.
    """
    d = req.check_in_date
    # Seasonal flags (must match Step 3's logic)
    peak_months = {6, 7, 8, 12}
    shoulder_months = {4, 5, 9, 10}
    season_encoded = 2 if d.month in peak_months else (1 if d.month in shoulder_months else 0)

    is_weekend = 1 if d.weekday() >= 5 else 0
    # Holiday flag — simplified; in production you'd load the calendar table
    holidays = {(1, 1), (12, 25), (12, 31), (7, 4), (5, 1), (8, 15), (11, 1)}
    is_holiday = int((d.month, d.day) in holidays)

    # Build the row (order matters only if feature_names is None)
    row = {
        "bedrooms": villa["bedrooms"],
        "max_guests": villa["max_guests"],
        "has_pool": int(villa["has_pool"]),
        "has_butler": int(villa["has_butler"]),
        "has_private_beach": int(villa["has_private_beach"]),
        "rating": villa["rating"],
        "amenity_score": int(villa["has_pool"]) + int(villa["has_butler"]) + int(villa["has_private_beach"]),
        "price_per_bedroom": villa["base_price"] / villa["bedrooms"],
        "guest_bedroom_ratio": villa["max_guests"] / villa["bedrooms"],
        "base_price": villa["base_price"],

        # Temporal
        "day_of_week": d.weekday(),
        "month": d.month,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "quarter": (d.month - 1) // 3 + 1,
        "week_of_year": d.isocalendar()[1],
        "is_peak_season": int(season_encoded == 2),
        "is_shoulder_season": int(season_encoded == 1),
        "season_encoded": season_encoded,

        # Booking behavior
        "nights": req.nights,
        "guests": req.guests,
        "lead_time_days": req.lead_time_days,
        "is_last_minute": int(req.lead_time_days < 14),
        "is_early_bird": int(req.lead_time_days > 120),
        "lead_time_bucket_enc": (
            0 if req.lead_time_days < 14 else
            1 if req.lead_time_days <= 60 else
            2 if req.lead_time_days <= 120 else 3
        ),
        "nights_bucket_enc": (
            0 if req.nights <= 3 else
            1 if req.nights <= 7 else
            2 if req.nights <= 14 else 3
        ),

        # Event flags — default 0; in production you'd query external_events
        "has_event": 0,
        "n_events_during_stay": 0,
        "event_size": 0,
        "days_until_event": 999,

        # Lag features — defaults; in production, query recent history
        "villa_total_bookings": 0,
        "villa_avg_lead_time": req.lead_time_days,
    }

    # Location one-hot (must match pd.get_dummies prefix='loc', drop_first=True)
    locations = ["Santorini", "Maldives", "Cannes", "Aspen", "Bali", "Tuscany", "Dubai", "Amalfi Coast"]
    for loc in locations:
        # drop_first=True means the first alphabetically is the dropped baseline
        # We handle that below by reindexing against feature_names
        row[f"loc_{loc}"] = int(villa["location"] == loc)

    df = pd.DataFrame([row])

    # Align to training feature order, filling missing columns with 0
    if store.feature_names:
        df = df.reindex(columns=store.feature_names, fill_value=0)
    return df


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded=all([
            store.pricing_model is not None,
            store.demand_model is not None,
        ]),
        pricing_model_version="v1.0",
        demand_model_version="v1.0",
    )


@app.get("/villas", response_model=list[VillaOut])
def list_villas():
    conn = get_db()
    rows = conn.execute("SELECT villa_id, name, location, bedrooms, max_guests, base_price, rating FROM villas").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/predict-price", response_model=PriceResponse)
def predict_price(req: PriceRequest):
    villa = fetch_villa(req.villa_id)

    features = build_pricing_features(villa, req)
    pred = float(store.pricing_model.predict(features)[0])

    # Confidence band — a rough ±15% heuristic
    low = round(pred * 0.85, 2)
    high = round(pred * 1.15, 2)

    # Demand signal for the response narrative
    peak_months = {6, 7, 8, 12}
    demand_signal = "high" if req.check_in_date.month in peak_months else "medium"

    rationale = (
        f"Predicted ${pred:.0f}/night for {req.nights} nights in {villa['location']}. "
        f"{'Peak season' if demand_signal=='high' else 'Shoulder/off season'} demand detected."
    )

    return PriceResponse(
        villa_id=req.villa_id,
        villa_name=villa["name"],
        check_in_date=req.check_in_date,
        nights=req.nights,
        predicted_nightly_price=round(pred, 2),
        total_price=round(pred * req.nights, 2),
        confidence_range={"low": low, "high": high},
        demand_signal=demand_signal,
        recommendation=rationale,
    )


@app.post("/forecast-demand", response_model=DemandResponse)
def forecast_demand(req: DemandRequest):
    # Build a 30-day input window using historical average demand
    conn = get_db()
    df = pd.read_sql("SELECT booking_date, COUNT(*) as n FROM bookings GROUP BY booking_date", conn)
    conn.close()

    if len(df) < 30:
        raise HTTPException(status_code=500, detail="Insufficient historical data")

    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df = df.sort_values("booking_date").tail(30)
    daily_avg = float(df["n"].mean())

    # Build a 3D input (1, 30, 8) — all features filled with defaults
    window = np.zeros((1, 30, 8), dtype=np.float32)
    window[0, :, 0] = (df["n"].values - daily_avg)  # approximate normalized signal

    preds_scaled = store.demand_model.predict(window, verbose=0)
    preds = store.target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    preds = np.clip(preds, 0, None)  # no negative bookings

    ref = req.reference_date or date.today()
    dates = [ref + timedelta(days=i + 1) for i in range(len(preds))]

    # 30-day total: extend 7-day forecast by naive extrapolation
    total_30d = float(preds.sum() * (30 / 7))
    trend = "rising" if preds[-1] > preds[0] else ("falling" if preds[-1] < preds[0] else "stable")

    return DemandResponse(
        villa_id=req.villa_id,
        forecast_dates=dates,
        forecast_bookings=[float(p) for p in preds],
        total_30d_forecast=round(total_30d, 1),
        trend=trend,
    )