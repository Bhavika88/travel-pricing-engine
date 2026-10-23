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
| Baseline (mean) | $7,373.69 | -0.0005 | Y% |
| XGBoost (tuned) | $1,662.77 |  0.9491 | 9.19% |

**Business impact:** Pricing simulation shows ~4.53% revenue uplift vs. static 
mean pricing in held-out backtesting.


## Demand Forecasting (LSTM)

**Goal:** Predict daily booking volume for the next 7 days.

**Architecture:**
- Input: 30-day lookback window × 8 features
- LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(32) → Dense(7)
- Loss: MSE | Optimizer: Adam | Callbacks: EarlyStopping, ReduceLROnPlateau

**Features:** Booking volume + cyclical day-of-week/month + weekend/holiday/event flags.

**Results:**
| Metric | Value |
|--------|-------|
| MAE | 6.86 bookings/day |
| RMSE | 7.43 bookings/day |

**Why cyclical encoding?** Day-of-week and month are cyclical (Dec is 
adjacent to Jan), so sin/cos encoding preserves this — one-hot encoding 
destroys the smoothness LSTM relies on.

**Why LSTM vs. Prophet/ARIMA?** Multivariate covariates (events, holidays, 
weekends) matter. LSTM handles them natively; classical models struggle.

## Frontend 1: Streamlit Analyst Dashboard

Rapid internal UI for revenue managers, consuming the FastAPI backend.

**Features:**
- Live health check on page load (graceful error if backend is down)
- Sidebar inputs → main panel outputs (classic BI-tool layout)
- Price recommendation card with confidence band
- Interactive Plotly demand forecast chart
- **Revenue simulator** with price-elasticity model — shows the 
  tradeoff between price and occupancy in real time
- `@st.cache_data` on `/villas` for snappy interactions

**Run:** `streamlit run frontend_streamlit/app.py` (requires API on :8000)

**Screenshot:** :


## Frontend 2: HTML/CSS/JS Client UI

Production-grade customer-facing interface built in vanilla web technologies 
(no framework) — deployable to any static host.

**Tech:** HTML5, CSS Grid/Flexbox, Vanilla JS (Fetch API), Chart.js 4.

**Design system:**
- Luxury travel aesthetic (navy + gold, Playfair Display serif + Inter sans)
- Responsive grid layout (desktop 2-col → mobile 1-col)
- Subtle micro-interactions (hover states, fade-up animations, spinner)
- Live status banner (polls `/health` on load)

**Features:**
- Real-time villa dropdown from `/villas`
- Parallel `Promise.all()` calls to `/predict-price` and `/forecast-demand`
- Chart.js demand forecast with gradient fill and gold markers
- **Interactive revenue simulator** with live price-elasticity
- Graceful error states (API down, 4xx/5xx responses)
- Loading states (disabled button + inline spinner)

**Why vanilla JS instead of React?**
To demonstrate mastery of the browser's native APIs (Fetch, DOM, 
event handling) and CSS design systems without framework abstractions. 
A portfolio piece in vanilla JS is more impressive than "I used React" 
because there's nowhere to hide.

**Screenshot:** _(to be added)_