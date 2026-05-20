import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="BSE/NSE Market Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2d3149;
        border-radius: 10px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8b8fa8 !important; font-size: 0.78rem; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] { color: #e8eaf6 !important; }
    .signal-box {
        border-radius: 10px;
        padding: 20px 28px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-top: 6px;
    }
    .signal-buy  { background: #0d2b1f; border: 1px solid #1db954; color: #1db954; }
    .signal-sell { background: #2b0d0d; border: 1px solid #e05252; color: #e05252; }
    .signal-neutral { background: #1a1d27; border: 1px solid #8b8fa8; color: #8b8fa8; }
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        color: #8b8fa8;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1.4rem 0 0.6rem;
        border-bottom: 1px solid #2d3149;
        padding-bottom: 6px;
    }
    .stTextInput input, .stSelectbox select {
        background: #1a1d27 !important;
        border: 1px solid #2d3149 !important;
        color: #e8eaf6 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: #4f6ef7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #3a57e8; }
    .stSidebar { background: #13151f !important; }
    h1 { color: #e8eaf6 !important; font-weight: 700; font-size: 1.6rem !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .stAlert { border-radius: 8px; }
    .footer-note {
        font-size: 0.72rem;
        color: #555878;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #2d3149;
    }
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font_color="#e8eaf6",
    xaxis=dict(gridcolor="#1e2030", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2030", showgrid=True, zeroline=False),
    legend=dict(bgcolor="#1a1d27", bordercolor="#2d3149", borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std

def composite_signal(data, rsi, macd_line, signal_line):
    """Score-based signal: sum votes from RSI, MACD, SMA cross."""
    score = 0
    reasons = []
    latest_rsi = rsi.iloc[-1]
    if latest_rsi < 40:
        score += 1; reasons.append(f"RSI oversold ({latest_rsi:.1f})")
    elif latest_rsi > 65:
        score -= 1; reasons.append(f"RSI overbought ({latest_rsi:.1f})")
    else:
        reasons.append(f"RSI neutral ({latest_rsi:.1f})")

    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        score += 2; reasons.append("MACD bullish crossover")
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        score -= 2; reasons.append("MACD bearish crossover")
    elif macd_line.iloc[-1] > signal_line.iloc[-1]:
        score += 1; reasons.append("MACD above signal")
    else:
        score -= 1; reasons.append("MACD below signal")

    close = data['Close'].squeeze()
    sma50  = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    price  = close.iloc[-1]
    if price > sma50 > sma200:
        score += 1; reasons.append("Price above SMA50 & SMA200")
    elif price < sma50 < sma200:
        score -= 1; reasons.append("Price below SMA50 & SMA200")

    avg_ret = close.pct_change().dropna().mean()
    if avg_ret > 0:
        score += 1
    else:
        score -= 1

    if score >= 2:
        signal, css = "BUY", "signal-buy"
    elif score <= -2:
        signal, css = "SELL", "signal-sell"
    else:
        signal, css = "NEUTRAL", "signal-neutral"

    return signal, css, score, reasons

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    ticker = st.text_input("Ticker Symbol", "TCS.NS", help="Append .NS for NSE or .BO for BSE")

    preset = st.selectbox("Date Range", ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"])
    today = datetime.today()
    preset_map = {
        "1 Month":  today - timedelta(days=30),
        "3 Months": today - timedelta(days=90),
        "6 Months": today - timedelta(days=180),
        "1 Year":   today - timedelta(days=365),
        "3 Years":  today - timedelta(days=1095),
        "5 Years":  today - timedelta(days=1825),
    }
    if preset != "Custom":
        start_date = preset_map[preset].date()
        end_date   = today.date()
        st.caption(f"{start_date} → {end_date}")
    else:
        start_date = st.date_input("Start Date", datetime(2022, 1, 1))
        end_date   = st.date_input("End Date", today)

    st.markdown("### 📊 Indicators")
    show_bb    = st.toggle("Bollinger Bands", True)
    show_sma   = st.toggle("SMA 50 / 200", True)
    show_vol   = st.toggle("Volume", True)
    show_macd  = st.toggle("MACD", True)
    show_rsi   = st.toggle("RSI", True)

    st.markdown("---")
    analyze = st.button("Run Analysis", use_container_width=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("## 📈 BSE / NSE Market Intelligence")
st.caption("Technical analysis · Signal scoring · Multi-indicator charting")

if not analyze:
    st.info("Configure your ticker and date range in the sidebar, then click **Run Analysis**.")
    st.stop()

# ─── DATA FETCH ───────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}**…"):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    info = {}
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        pass

if data.empty:
    st.error("No data returned. Check the ticker symbol and date range.")
    st.stop()

close  = data['Close'].squeeze()
volume = data['Volume'].squeeze() if 'Volume' in data.columns else None

# ─── COMPUTED INDICATORS ──────────────────────────────────────────────────────
rsi                       = compute_rsi(close)
macd_line, sig_line, hist = compute_macd(close)
bb_up, bb_mid, bb_low     = compute_bollinger(close)
sma50                     = close.rolling(50).mean()
sma200                    = close.rolling(200).mean()

signal, signal_css, score, reasons = composite_signal(data, rsi, macd_line, sig_line)

returns      = close.pct_change().dropna()
avg_return   = returns.mean() * 100
volatility   = returns.std() * 100
sharpe       = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
max_drawdown = ((close / close.cummax()) - 1).min() * 100
ytd_return   = ((close.iloc[-1] / close.iloc[0]) - 1) * 100

# ─── COMPANY HEADER ───────────────────────────────────────────────────────────
company_name = info.get("longName", ticker.upper())
currency     = info.get("currency", "INR")
sector       = info.get("sector", "—")
industry     = info.get("industry", "—")
market_cap   = info.get("marketCap")
cap_str      = f"₹{market_cap/1e7:,.0f} Cr" if market_cap else "—"

st.markdown(f"### {company_name}")
st.caption(f"{sector}  ·  {industry}  ·  Market Cap: {cap_str}  ·  Currency: {currency}")

# ─── KPI ROW ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Key Metrics</p>', unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
latest_price = close.iloc[-1]
prev_price   = close.iloc[-2]
price_delta  = latest_price - prev_price
delta_pct    = (price_delta / prev_price) * 100

k1.metric("Current Price",       f"₹{latest_price:,.2f}", f"{delta_pct:+.2f}%")
k2.metric("Period Return",       f"{ytd_return:+.1f}%")
k3.metric("Avg Daily Return",    f"{avg_return:.3f}%")
k4.metric("Volatility (daily)",  f"{volatility:.2f}%")
k5.metric("Sharpe Ratio",        f"{sharpe:.2f}")
k6.metric("Max Drawdown",        f"{max_drawdown:.1f}%")

# ─── SIGNAL BOX ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Composite Signal</p>', unsafe_allow_html=True)
sig_col, reason_col = st.columns([1, 3])
with sig_col:
    icon = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪")
    st.markdown(f'<div class="signal-box {signal_css}">{icon} {signal}<br><span style="font-size:0.85rem;font-weight:400;opacity:0.8">Score: {score:+d} / 5</span></div>', unsafe_allow_html=True)
with reason_col:
    st.markdown("**Signal drivers:**")
    for r in reasons:
        bullet = "✅" if any(w in r.lower() for w in ["bullish","above","oversold"]) else ("⚠️" if "neutral" in r.lower() else "🔻")
        st.markdown(f"{bullet} {r}")

# ─── PRICE CHART ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Price Chart</p>', unsafe_allow_html=True)

row_heights = []
subplot_titles = ["Price"]
specs_list = [[{"secondary_y": False}]]

if show_vol:
    subplot_titles.append("Volume"); row_heights.append(0.15); specs_list.append([{"secondary_y": False}])
if show_macd:
    subplot_titles.append("MACD"); row_heights.append(0.18); specs_list.append([{"secondary_y": False}])
if show_rsi:
    subplot_titles.append("RSI"); row_heights.append(0.15); specs_list.append([{"secondary_y": False}])

n_rows = len(subplot_titles)
price_h = 1 - sum(row_heights)
row_heights = [price_h] + row_heights

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    subplot_titles=subplot_titles,
    vertical_spacing=0.03,
    specs=specs_list,
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'].squeeze(), high=data['High'].squeeze(),
    low=data['Low'].squeeze(),   close=close,
    name="Price",
    increasing_line_color="#1db954", decreasing_line_color="#e05252",
    increasing_fillcolor="#1db954", decreasing_fillcolor="#e05252",
), row=1, col=1)

if show_bb:
    for band, lbl, dash in [(bb_up, "BB Upper", "dot"), (bb_mid, "BB Mid", "dash"), (bb_low, "BB Lower", "dot")]:
        fig.add_trace(go.Scatter(x=data.index, y=band, name=lbl, line=dict(color="#7b68ee", width=1, dash=dash), opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=bb_up, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=bb_low, fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(123,104,238,0.07)', showlegend=False), row=1, col=1)

if show_sma:
    fig.add_trace(go.Scatter(x=data.index, y=sma50,  name="SMA 50",  line=dict(color="#f0a500", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=sma200, name="SMA 200", line=dict(color="#4fc3f7", width=1.2)), row=1, col=1)

current_row = 2
if show_vol and volume is not None:
    colors = ["#1db954" if close.iloc[i] >= close.iloc[i-1] else "#e05252" for i in range(len(close))]
    fig.add_trace(go.Bar(x=data.index, y=volume, name="Volume", marker_color=colors, opacity=0.7), row=current_row, col=1)
    current_row += 1

if show_macd:
    fig.add_trace(go.Scatter(x=data.index, y=macd_line, name="MACD",   line=dict(color="#4fc3f7", width=1.2)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=sig_line,  name="Signal", line=dict(color="#f0a500", width=1.2)), row=current_row, col=1)
    hist_colors = ["#1db954" if v >= 0 else "#e05252" for v in hist]
    fig.add_trace(go.Bar(x=data.index, y=hist, name="Histogram", marker_color=hist_colors, opacity=0.6), row=current_row, col=1)
    current_row += 1

if show_rsi:
    fig.add_trace(go.Scatter(x=data.index, y=rsi, name="RSI", line=dict(color="#ce93d8", width=1.2)), row=current_row, col=1)
    for level, color in [(70, "rgba(224,82,82,0.3)"), (30, "rgba(29,185,84,0.3)")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, row=current_row, col=1)
    current_row += 1

fig.update_layout(
    height=820,
    xaxis_rangeslider_visible=False,
    showlegend=True,
    title=dict(text=f"{company_name} — Technical Analysis", font=dict(size=14, color="#e8eaf6")),
    **PLOT_THEME,
)
for i in range(1, n_rows + 1):
    fig.update_xaxes(gridcolor="#1e2030", row=i, col=1)
    fig.update_yaxes(gridcolor="#1e2030", row=i, col=1)

st.plotly_chart(fig, use_container_width=True)

# ─── RETURNS DISTRIBUTION ─────────────────────────────────────────────────────
st.markdown('<p class="section-header">Returns Distribution</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=60,
        marker_color="#4f6ef7",
        opacity=0.8,
        name="Daily Returns"
    ))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#8b8fa8")
    fig_hist.update_layout(title="Daily Return Distribution (%)", height=320, **PLOT_THEME)
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol,
        fill='tozeroy', fillcolor='rgba(79,110,247,0.15)',
        line=dict(color="#4f6ef7", width=1.5),
        name="30-day Rolling Volatility"
    ))
    fig_vol.update_layout(title="Rolling 30-day Annualised Volatility (%)", height=320, **PLOT_THEME)
    st.plotly_chart(fig_vol, use_container_width=True)

# ─── RAW DATA TABLE ───────────────────────────────────────────────────────────
with st.expander("📋 Raw Data (last 30 rows)"):
    display_df = data.tail(30).copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    st.dataframe(display_df.style.format("{:.2f}"), use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="footer-note">Data sourced from Yahoo Finance via yfinance. '
    'This tool is for informational purposes only and does not constitute financial advice.</p>',
    unsafe_allow_html=True
)
