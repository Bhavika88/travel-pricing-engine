Luxury villa rentals (think Airbnb Luxe, Onefinestay, Plum Guide) face a classic dilemma:

Price too high → Villa sits empty (lost revenue).

Price too low → Villa is booked instantly but you left money on the table.

The Goal: Build a system that recommends the optimal nightly price for a luxury villa on any given date, by predicting two things:

Demand: How many bookings will we get at a given price?

Willingness to Pay: What is the maximum price the market will bear?

Data Design Philosophy: The dataset embeds 5 realistic patterns (seasonality, lead-time effect, luxury amenity premiums, event-driven demand, and price elasticity). This allows the downstream models to discover these patterns rather than being told them — which mirrors how real-world ML systems extract value from noisy business data.

## Feature Engineering

Built 25+ model-ready features across 5 categories:
- **Villa attributes:** amenity_score, price_per_bedroom, guest_bedroom_ratio
- **Temporal:** is_peak_season, is_weekend, is_holiday, week_of_year
- **Behavioral:** lead_time_bucket, nights_bucket, is_last_minute
- **Event-driven:** has_event, event_size, days_until_event
- **Historical (lagged):** villa_avg_price_30d, villa_total_bookings

**Key engineering decision:** All rolling aggregates use `booking_date` 
with `shift(1)` to eliminate temporal leakage — the model only sees 
information available at quote time.


## Model Design Decisions

**Target:** Predict `price_paid` from villa attributes + temporal context.

**Model:** XGBoost Regressor (tuned via RandomizedSearchCV with TimeSeriesSplit CV)

**Critical design choice — no leakage:** I excluded `villa_avg_price_30d` 
(a rolling average of past prices) from the feature set. Although it would 
inflate training R², it violates the production use case: a pricing 
recommendation system must quote a price *before* seeing historical prices 
for the same villa. This is the forecaster-vs-recommender distinction.

**Results:**
| Model | RMSE | R² | MAPE |
|-------|------|-----|------|
| Baseline (mean) | $X | 0.00 | Y% |
| XGBoost (tuned) | $X' | Z | W% |

**Business impact:** Pricing simulation shows ~N% revenue uplift vs. static 
mean pricing in held-out backtesting.