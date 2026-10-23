/**
 * script.js
 * Vanilla JS client for the Travel Pricing Engine API.
 * Handles: villa loading, form submission, DOM rendering, Chart.js.
 */

const API_URL = "http://localhost:8000";

// ============================================================
// ELEMENT REFERENCES
// ============================================================
const els = {
    form:         document.getElementById("booking-form"),
    villaSelect:  document.getElementById("villa-select"),
    checkIn:      document.getElementById("check-in"),
    nights:       document.getElementById("nights"),
    nightsVal:    document.getElementById("nights-val"),
    guests:       document.getElementById("guests"),
    guestsVal:    document.getElementById("guests-val"),
    lead:         document.getElementById("lead-time"),
    leadVal:      document.getElementById("lead-val"),
    submitBtn:    document.getElementById("submit-btn"),
    btnText:      document.querySelector("#submit-btn .btn-text"),
    btnSpinner:   document.querySelector("#submit-btn .btn-spinner"),
    errorMsg:     document.getElementById("error-msg"),
    statusBanner: document.getElementById("status-banner"),
    results:      document.getElementById("results"),
    placeholder:  document.getElementById("placeholder"),
};

// State
let villas = [];
let chart = null;
let currentBasePrice = 0;
let currentNights = 7;

// ============================================================
// UTILITIES
// ============================================================
const fmtMoney = (n) => "$" + Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });
const todayPlus = (days) => {
    const d = new Date();
    d.setDate(d.getDate() + days);
    return d.toISOString().split("T")[0];
};

function setStatus(state, text) {
    els.statusBanner.querySelector(".status-dot").className = `status-dot ${state}`;
    els.statusBanner.querySelector(".status-text").textContent = text;
}

function showError(msg) {
    els.errorMsg.textContent = msg;
    els.errorMsg.hidden = false;
}

function clearError() {
    els.errorMsg.hidden = true;
}

// ============================================================
// INITIAL LOAD — health check + villa list
// ============================================================
async function init() {
    // Set default date
    els.checkIn.value = todayPlus(30);
    els.checkIn.min = todayPlus(0);

    try {
        // Health check
        const healthRes = await fetch(`${API_URL}/health`);
        if (!healthRes.ok) throw new Error("Health check failed");
        const health = await healthRes.json();
        if (!health.models_loaded) {
            setStatus("error", "Models not loaded on server");
            return;
        }
        setStatus("online", `Engine online · ${health.pricing_model_version}`);

        // Load villas
        const villasRes = await fetch(`${API_URL}/villas`);
        villas = await villasRes.json();

        els.villaSelect.innerHTML = '<option value="">— Choose a villa —</option>';
        villas.forEach(v => {
            const opt = document.createElement("option");
            opt.value = v.villa_id;
            opt.textContent = `#${v.villa_id} · ${v.name} · ${v.location} · ${v.bedrooms} BR · ${fmtMoney(v.base_price)}`;
            els.villaSelect.appendChild(opt);
        });

    } catch (err) {
        console.error(err);
        setStatus("error", "Cannot reach API — is the backend running?");
        els.villaSelect.innerHTML = '<option value="">API offline</option>';
    }
}

// ============================================================
// LIVE SLIDER LABELS
// ============================================================
els.nights.addEventListener("input", () => els.nightsVal.textContent = els.nights.value);
els.guests.addEventListener("input", () => els.guestsVal.textContent = els.guests.value);
els.lead.addEventListener("input", () => els.leadVal.textContent = els.lead.value);

// ============================================================
// FORM SUBMIT
// ============================================================
els.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    const villaId = parseInt(els.villaSelect.value, 10);
    if (!villaId) return showError("Please select a villa.");

    const payload = {
        villa_id: villaId,
        check_in_date: els.checkIn.value,
        nights: parseInt(els.nights.value, 10),
        guests: parseInt(els.guests.value, 10),
        lead_time_days: parseInt(els.lead.value, 10),
    };

    // Loading state
    els.submitBtn.disabled = true;
    els.btnText.textContent = "Analyzing…";
    els.btnSpinner.hidden = false;

    try {
        const [priceRes, demandRes] = await Promise.all([
            fetch(`${API_URL}/predict-price`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }),
            fetch(`${API_URL}/forecast-demand`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ villa_id: villaId, reference_date: els.checkIn.value }),
            }),
        ]);

        if (!priceRes.ok) throw new Error(`Pricing failed (${priceRes.status})`);
        if (!demandRes.ok) throw new Error(`Demand failed (${demandRes.status})`);

        const price = await priceRes.json();
        const demand = await demandRes.json();

        renderResults(price, demand);

    } catch (err) {
        console.error(err);
        showError(err.message || "Something went wrong. Try again.");
    } finally {
        els.submitBtn.disabled = false;
        els.btnText.textContent = "Get Recommendation";
        els.btnSpinner.hidden = true;
    }
});

// ============================================================
// RENDER RESULTS
// ============================================================
function renderResults(price, demand) {
    currentBasePrice = price.predicted_nightly_price;
    currentNights = price.nights;

    // Swap panels
    els.placeholder.hidden = true;
    els.results.hidden = false;
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });

    // Price card
    document.getElementById("price-value").textContent = fmtMoney(price.predicted_nightly_price);
    document.getElementById("price-meta").textContent =
        `${price.villa_name} · ${price.nights} nights · ${price.demand_signal.toUpperCase()} demand`;

    // Stats
    document.getElementById("total-price").textContent = fmtMoney(price.total_price);
    document.getElementById("confidence").textContent =
        `${fmtMoney(price.confidence_range.low)} – ${fmtMoney(price.confidence_range.high)}`;
    document.getElementById("demand-signal").textContent = price.demand_signal.toUpperCase();

    // Recommendation text
    document.getElementById("recommendation").textContent = price.recommendation;

    // Chart
    renderChart(demand.forecast_dates, demand.forecast_bookings);

    // Simulator
    setupSimulator(currentBasePrice, currentNights);
}

// ============================================================
// CHART.JS
// ============================================================
function renderChart(dates, bookings) {
    const ctx = document.getElementById("demand-chart").getContext("2d");

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, 220);
    gradient.addColorStop(0, "rgba(26,60,92,0.25)");
    gradient.addColorStop(1, "rgba(26,60,92,0.02)");

    if (chart) chart.destroy();
    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: dates.map(d => new Date(d).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" })),
            datasets: [{
                label: "Predicted Bookings",
                data: bookings,
                borderColor: "#1a3c5c",
                backgroundColor: gradient,
                borderWidth: 2.5,
                tension: 0.35,
                fill: true,
                pointBackgroundColor: "#c9a961",
                pointBorderColor: "#fff",
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#0f2438",
                    padding: 12,
                    titleFont: { family: "Playfair Display", size: 14 },
                    bodyFont: { family: "Inter", size: 13 },
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `${ctx.parsed.y.toFixed(1)} bookings`,
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: "rgba(0,0,0,0.05)" },
                    ticks: { font: { family: "Inter", size: 11 }, color: "#6b7280" },
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { family: "Inter", size: 11 }, color: "#6b7280" },
                },
            },
        },
    });
}

// ============================================================
// REVENUE SIMULATOR
// ============================================================
function setupSimulator(basePrice, nights) {
    const slider = document.getElementById("sim-price");
    const priceLabel = document.getElementById("sim-price-val");

    slider.min = Math.round(basePrice * 0.5);
    slider.max = Math.round(basePrice * 2);
    slider.value = Math.round(basePrice);

    const update = () => {
        const proposed = Number(slider.value);
        priceLabel.textContent = fmtMoney(proposed);

        // Price-elasticity model (same as Streamlit)
        const ratio = proposed / basePrice;
        const occupancy = Math.max(0.1, Math.min(1.0, 1 / Math.pow(ratio, 0.6)));
        const bookedNights = nights * occupancy;
        const revenue = proposed * bookedNights;

        document.getElementById("sim-occupancy").textContent = `${Math.round(occupancy * 100)}%`;
        document.getElementById("sim-revenue").textContent = fmtMoney(revenue);
    };

    slider.oninput = update;
    update();
}

// ============================================================
// BOOT
// ============================================================
init();