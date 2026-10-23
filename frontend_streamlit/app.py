"""
app.py
Streamlit dashboard — internal analyst cockpit.
Consumes the FastAPI backend at http://localhost:8000.

Run:
    streamlit run frontend_streamlit/app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

API_URL = "http://localhost:8000"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Travel Pricing Engine",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — subtle polish
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3c5c;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1a3c5c 0%, #2e5e8a 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
    }
    .price-card h2 {
        font-size: 3rem;
        margin: 0.3rem 0;
    }
    .price-card p {
        margin: 0;
        opacity: 0.85;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }
    .stButton > button {
        width: 100%;
        background: #1a3c5c;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# API HELPERS
# ============================================================
@st.cache_data(ttl=300)
def fetch_villas():
    """Fetch villa list — cached for 5 min."""
    try:
        r = requests.get(f"{API_URL}/villas", timeout=5)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.error(f"❌ Could not reach API. Is the backend running? ({e})")
        return None


def call_predict_price(payload):
    try:
        r = requests.post(f"{API_URL}/predict-price", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Pricing API error: {e}")
        return None


def call_forecast_demand(payload):
    try:
        r = requests.post(f"{API_URL}/forecast-demand", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Demand API error: {e}")
        return None


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🏝️ Travel Pricing Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">XGBoost pricing · LSTM demand forecasting · Real-time recommendations</div>', unsafe_allow_html=True)

# Health check
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("models_loaded"):
        st.success("✅ API online · models loaded", icon="🟢")
    else:
        st.warning("⚠️ API online but models not loaded", icon="🟡")
except Exception:
    st.error("❌ Backend is not running. Start it with: `uvicorn src.api.main:app --port 8000`")
    st.stop()


# ============================================================
# LOAD VILLAS
# ============================================================
villas = fetch_villas()
if villas is None or villas.empty:
    st.stop()


# ============================================================
# SIDEBAR — INPUTS
# ============================================================
st.sidebar.header("🔎 Configure Booking")

villa_options = villas.apply(
    lambda r: f"#{r['villa_id']} · {r['name']} · {r['location']} · {r['bedrooms']}BR",
    axis=1
).tolist()

selected_label = st.sidebar.selectbox("Select Villa", villa_options)
selected_villa_id = int(selected_label.split("·")[0].strip().replace("#", ""))

selected_villa = villas[villas["villa_id"] == selected_villa_id].iloc[0]

check_in = st.sidebar.date_input("Check-in Date", value=date.today() + timedelta(days=30))
nights = st.sidebar.slider("Nights", 1, 30, 7)
guests = st.sidebar.slider("Guests", 1, int(selected_villa["max_guests"]), min(4, int(selected_villa["max_guests"])))
lead_time = st.sidebar.slider("Lead time (days until check-in)", 0, 365, 60)

get_btn = st.sidebar.button("💰 Get Recommendation", use_container_width=True)


# ============================================================
# MAIN PANEL
# ============================================================
if get_btn:
    payload = {
        "villa_id": selected_villa_id,
        "check_in_date": str(check_in),
        "nights": nights,
        "guests": guests,
        "lead_time_days": lead_time,
    }

    with st.spinner("Analyzing pricing and demand..."):
        price_data = call_predict_price(payload)
        demand_data = call_forecast_demand({"villa_id": selected_villa_id, "reference_date": str(check_in)})

    if price_data is None or demand_data is None:
        st.stop()

    # ---------- 1. PRICE CARD ----------
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="price-card">
            <p class="metric-label">Recommended Nightly Price</p>
            <h2>${price_data['predicted_nightly_price']:,.0f}</h2>
            <p>{price_data['demand_signal'].upper()} DEMAND · {nights} nights · {guests} guests</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.metric(
            "Total Stay",
            f"${price_data['total_price']:,.0f}",
            help="Predicted nightly price × nights",
        )
    with col3:
        lo = price_data["confidence_range"]["low"]
        hi = price_data["confidence_range"]["high"]
        st.metric("Confidence Band", f"${lo:,.0f} – ${hi:,.0f}")

    st.markdown(f"*{price_data['recommendation']}*")
    st.divider()

    # ---------- 2. DEMAND FORECAST ----------
    st.subheader("📈 7-Day Demand Forecast")

    forecast_df = pd.DataFrame({
        "date": pd.to_datetime(demand_data["forecast_dates"]),
        "bookings": demand_data["forecast_bookings"],
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["bookings"],
        mode="lines+markers",
        line=dict(color="#1a3c5c", width=3),
        marker=dict(size=10, color="#2e5e8a"),
        fill="tozeroy",
        fillcolor="rgba(26, 60, 92, 0.1)",
        name="Predicted Bookings",
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted bookings",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("7-Day Forecast Total", f"{sum(demand_data['forecast_bookings']):.1f} bookings")
    c2.metric("30-Day Projection", f"{demand_data['total_30d_forecast']:.0f} bookings",
              delta=demand_data["trend"].capitalize())

    st.divider()

    # ---------- 3. REVENUE SIMULATOR ----------
    st.subheader("💡 Revenue Simulator")
    st.caption("Adjust the nightly price and see projected revenue (assumes occupancy is inversely proportional to price)")

    base_price = price_data["predicted_nightly_price"]
    proposed_price = st.slider(
        "Proposed Nightly Price ($)",
        min_value=int(base_price * 0.5),
        max_value=int(base_price * 2.0),
        value=int(base_price),
        step=50,
    )

    # Simple elasticity model: occupancy falls as price rises
    # (Real system would call the demand model; this is a simulation)
    price_ratio = proposed_price / base_price
    occupancy = max(0.1, min(1.0, 1.0 / price_ratio ** 0.6))  # elasticity ~0.6
    booked_nights = nights * occupancy
    projected_revenue = proposed_price * booked_nights

    baseline_revenue = base_price * nights * 1.0  # baseline assumes full occupancy

    m1, m2, m3 = st.columns(3)
    m1.metric("Simulated Occupancy", f"{occupancy*100:.0f}%")
    m2.metric("Projected Revenue", f"${projected_revenue:,.0f}")
    delta = projected_revenue - baseline_revenue
    m3.metric(
        "vs. Baseline",
        f"${projected_revenue:,.0f}",
        delta=f"${delta:,.0f}" if delta != 0 else None,
    )

    st.divider()

    # ---------- 4. VILLA INFO ----------
    st.subheader("🏡 Villa Details")
    info_cols = st.columns(4)
    info_cols[0].metric("Location", selected_villa["location"])
    info_cols[1].metric("Bedrooms", f"{int(selected_villa['bedrooms'])} BR")
    info_cols[2].metric("Max Guests", f"{int(selected_villa['max_guests'])}")
    info_cols[3].metric("Base Price", f"${selected_villa['base_price']:,.0f}")

else:
    # Landing state — friendly hero
    st.info("👈 Configure a booking in the sidebar and click **Get Recommendation** to begin.")

    st.subheader("🏆 Top-Rated Villas")
    top = villas.sort_values("rating", ascending=False).head(5)
    for _, row in top.iterrows():
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            c1.markdown(f"**#{row['villa_id']} · {row['name']}**  \n{row['location']}")
            c2.metric("Rating", f"⭐ {row['rating']:.1f}")
            c3.metric("Bedrooms", f"{int(row['bedrooms'])}")
            c4.metric("Base Price", f"${row['base_price']:,.0f}")