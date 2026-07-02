"""
Streamlit app: Monte Carlo Simulation of AAPL Stock Price (GBM)

Run with:
    streamlit run aapl_monte_carlo_streamlit.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="AAPL Monte Carlo Simulation", layout="wide")

# ----------------------------- Sidebar controls -----------------------------
st.sidebar.header("Simulation Parameters")

ticker       = st.sidebar.text_input("Ticker", "AAPL").upper()
lookback     = st.sidebar.selectbox("Historical lookback", ["6mo", "1y", "2y", "5y"], index=2)
n_sims       = st.sidebar.select_slider(
    "Number of paths (terminal distribution)",
    options=[100, 1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000,
             5_000_000, 10_000_000, 25_000_000, 50_000_000],
    value=10_000,
    format_func=lambda x: f"{x:,}",
)
if n_sims >= 5_000_000:
    st.sidebar.warning(
        "⚠️ At this scale the app simulates *terminal* prices only "
        "(closed-form, chunked) — full day-by-day paths aren't stored "
        "in memory. The fan chart below uses a smaller subsample. "
        "50M draws may take 10-30s depending on the server."
    )
path_display_cap = st.sidebar.slider(
    "Paths to draw on fan chart", 50, 5_000, 500, step=50,
    help="Independent of the count above — controls how many full "
         "day-by-day paths are simulated just for the chart."
)
horizon_days = st.sidebar.slider("Horizon (trading days)", 5, 504, 252, step=1)
use_manual   = st.sidebar.checkbox("Override mu / sigma manually")

manual_mu = manual_sigma = None
if use_manual:
    manual_mu    = st.sidebar.number_input("Annualised drift (mu)", value=0.10, step=0.01, format="%.4f")
    manual_sigma = st.sidebar.number_input("Annualised vol (sigma)", value=0.28, step=0.01, format="%.4f")

seed_input = st.sidebar.number_input("Random seed (0 = random)", value=42, step=1)
run_button = st.sidebar.button("Run Simulation", type="primary")


# ----------------------------- Core functions -----------------------------
@st.cache_data(ttl=3600)
def fetch_market_params(ticker: str, lookback: str):
    data = yf.Ticker(ticker).history(period=lookback)["Close"].dropna()
    if len(data) < 30:
        raise ValueError("Not enough price history returned.")
    s0 = float(data.iloc[-1])
    log_ret = np.log(data / data.shift(1)).dropna()
    mu = float(log_ret.mean()) * 252
    sigma = float(log_ret.std(ddof=1)) * np.sqrt(252)
    return s0, mu, sigma, data


def simulate_terminal_prices(s0, mu, sigma, n_sims, n_days, dt=1/252,
                              seed=None, chunk_size=2_000_000, progress_cb=None):
    """
    Memory-safe terminal-price sampler using the closed-form GBM solution:
        S_T = S0 * exp[(mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z],  Z ~ N(0,1)
    Draws n_sims samples in chunks so memory stays bounded regardless of
    how large n_sims is (works fine up to tens of millions of paths).
    """
    rng = np.random.default_rng(seed if seed != 0 else None)
    T = n_days * dt
    drift = (mu - 0.5 * sigma**2) * T
    vol_term = sigma * np.sqrt(T)

    terminal = np.empty(n_sims, dtype=np.float64)
    done = 0
    while done < n_sims:
        n = min(chunk_size, n_sims - done)
        z = rng.standard_normal(n)
        terminal[done:done + n] = s0 * np.exp(drift + vol_term * z)
        done += n
        if progress_cb is not None:
            progress_cb(done / n_sims)
    return terminal


def simulate_gbm(s0, mu, sigma, n_paths, n_days, dt=1/252, seed=None):
    """Full day-by-day path simulation — only used for the fan chart,
    so n_paths should stay small (hundreds to a few thousand)."""
    rng = np.random.default_rng(seed if seed != 0 else None)
    z = rng.standard_normal((n_days, n_paths))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros(n_paths), np.cumsum(drift + diffusion, axis=0)])
    return s0 * np.exp(log_paths)


# ----------------------------- Main -----------------------------
st.title(f"📈 Monte Carlo Simulation — {ticker}")
st.caption("Geometric Brownian Motion price paths estimated from historical daily log returns.")

if run_button or "paths" not in st.session_state:
    try:
        s0, mu, sigma, hist = fetch_market_params(ticker, lookback)
        data_source = "live"
    except Exception as e:
        st.warning(f"Could not fetch live data for {ticker} ({e}). Using fallback parameters.")
        s0, mu, sigma = 230.0, 0.10, 0.28
        hist = None
        data_source = "fallback"

    if use_manual:
        mu, sigma = manual_mu, manual_sigma

    seed = int(seed_input)

    # Full day-by-day paths, capped small — for the fan chart only
    n_chart_paths = min(path_display_cap, n_sims)
    paths = simulate_gbm(s0, mu, sigma, n_chart_paths, horizon_days, seed=seed)

    # Terminal-only distribution at full requested scale, chunked
    if n_sims > 1_000_000:
        progress = st.progress(0.0, text=f"Simulating {n_sims:,} terminal prices…")
        terminal = simulate_terminal_prices(
            s0, mu, sigma, n_sims, horizon_days, seed=seed,
            progress_cb=lambda frac: progress.progress(frac, text=f"Simulating {n_sims:,} terminal prices… {frac:.0%}"),
        )
        progress.empty()
    else:
        terminal = simulate_terminal_prices(s0, mu, sigma, n_sims, horizon_days, seed=seed)

    st.session_state.update(
        paths=paths, terminal=terminal, s0=s0, mu=mu, sigma=sigma, hist=hist,
        data_source=data_source, ticker=ticker, n_sims=n_sims,
    )

paths       = st.session_state["paths"]
terminal    = st.session_state["terminal"]
s0          = st.session_state["s0"]
mu          = st.session_state["mu"]
sigma       = st.session_state["sigma"]
hist        = st.session_state["hist"]
data_source = st.session_state["data_source"]
n_sims      = st.session_state["n_sims"]

# ----------------------------- Summary metrics -----------------------------
st.subheader("Parameters used")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot price", f"${s0:,.2f}")
c2.metric("Annualised drift (μ)", f"{mu:.2%}")
c3.metric("Annualised vol (σ)", f"{sigma:.2%}")
c4.metric("Data source", data_source)

st.subheader("Terminal price distribution")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean", f"${terminal.mean():,.2f}")
c2.metric("Median", f"${np.median(terminal):,.2f}")
c3.metric("5th pct", f"${np.percentile(terminal, 5):,.2f}")
c4.metric("95th pct", f"${np.percentile(terminal, 95):,.2f}")
var95 = np.percentile(terminal / s0 - 1, 5)
c5.metric("95% VaR (return)", f"{var95:.2%}")

st.metric("P(terminal price > spot)", f"{(terminal > s0).mean():.2%}")

# ----------------------------- Path fan chart -----------------------------
st.subheader("Simulated price paths")
st.caption(
    f"Terminal statistics below are computed from all {n_sims:,} simulated "
    f"draws. This chart shows a subsample of {paths.shape[1]:,} full "
    f"day-by-day paths (controlled separately in the sidebar)."
)

days = np.arange(paths.shape[0])
n_show = min(150, paths.shape[1])

fig1 = go.Figure()
for i in range(n_show):
    fig1.add_trace(go.Scatter(
        x=days, y=paths[:, i], mode="lines",
        line=dict(width=0.5, color="rgba(70,130,180,0.15)"),
        showlegend=False, hoverinfo="skip",
    ))

for q, color, dash in [(5, "crimson", "dash"), (50, "crimson", "solid"), (95, "crimson", "dash")]:
    fig1.add_trace(go.Scatter(
        x=days, y=np.percentile(paths, q, axis=1), mode="lines",
        line=dict(width=2, color=color, dash=dash),
        name=f"{q}th percentile",
    ))

fig1.add_hline(y=s0, line=dict(color="black", dash="dot", width=1),
               annotation_text="Spot", annotation_position="bottom right")
fig1.update_layout(
    xaxis_title="Trading days ahead", yaxis_title="Price ($)",
    height=500, template="plotly_white",
    title=f"{ticker} — {paths.shape[1]:,} simulated paths (showing {n_show})",
)
st.plotly_chart(fig1, use_container_width=True)

# ----------------------------- Terminal histogram -----------------------------
st.subheader("Terminal price histogram")

fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=terminal, nbinsx=80, marker_color="steelblue", name="Terminal price"))
fig2.add_vline(x=s0, line=dict(color="black", dash="dot", width=1.5), annotation_text="Spot")
fig2.add_vline(x=terminal.mean(), line=dict(color="crimson", width=1.5), annotation_text="Mean")
fig2.update_layout(xaxis_title="Price ($)", yaxis_title="Frequency",
                    height=400, template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------- Historical price (optional) -----------------------------
if hist is not None:
    st.subheader(f"Historical {ticker} price ({lookback})")
    st.line_chart(hist)

st.caption(
    "Model: dS = μS dt + σS dW (Geometric Brownian Motion). "
    "μ and σ are estimated from historical daily log returns and annualised. "
    "For risk-neutral pricing, replace μ with the risk-free rate."
)
