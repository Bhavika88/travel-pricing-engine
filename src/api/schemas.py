"""
schemas.py
Pydantic models define the API contract.
FastAPI validates every request/response against these.
"""

from pydantic import BaseModel, Field
from datetime import date
from typing import Optional


# ============ REQUESTS ============
class PriceRequest(BaseModel):
    villa_id: int = Field(..., gt=0, description="Villa ID from /villas")
    check_in_date: date = Field(..., description="YYYY-MM-DD format")
    nights: int = Field(..., ge=1, le=30, description="1-30 nights")
    guests: int = Field(..., ge=1, description="Number of guests")
    lead_time_days: int = Field(..., ge=0, le=365, description="Days until check-in")

    model_config = {
        "json_schema_extra": {
            "example": {
                "villa_id": 42,
                "check_in_date": "2025-07-15",
                "nights": 7,
                "guests": 6,
                "lead_time_days": 60,
            }
        }
    }


class DemandRequest(BaseModel):
    villa_id: int = Field(..., gt=0)
    reference_date: Optional[date] = Field(None, description="Date to forecast from; defaults to today")


# ============ RESPONSES ============
class PriceResponse(BaseModel):
    villa_id: int
    villa_name: str
    check_in_date: date
    nights: int
    predicted_nightly_price: float
    total_price: float
    confidence_range: dict   # { "low": float, "high": float }
    demand_signal: str       # "high" | "medium" | "low"
    recommendation: str      # human-readable rationale


class DemandResponse(BaseModel):
    villa_id: int
    forecast_dates: list[date]
    forecast_bookings: list[float]
    total_30d_forecast: float
    trend: str               # "rising" | "stable" | "falling"


class VillaOut(BaseModel):
    villa_id: int
    name: str
    location: str
    bedrooms: int
    max_guests: int
    base_price: float
    rating: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    pricing_model_version: str
    demand_model_version: str