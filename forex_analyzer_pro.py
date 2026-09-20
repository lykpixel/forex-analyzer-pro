"""
Forex Analyzer Pro v2 — single-file Streamlit dashboard

Install:
    pip install streamlit pandas numpy plotly yfinance

Run:
    streamlit run forex_analyzer_pro.py

Notes:
- Market data: Yahoo Finance via yfinance.
- Analysis only. No broker orders are sent.
- Signals are rule-based and are not financial advice.
- Intraday Yahoo Finance data has provider-specific history limits.
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Forex Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "EUR/GBP": "EURGBP=X",
    "XAU/USD (Gold)": "GC=F",
    "BTC/USD": "BTC-USD",
}

TIMEFRAMES = {
    "15 นาที": "15m",
    "30 นาที": "30m",
    "1 ชั่วโมง": "1h",
    "4 ชั่วโมง": "4h",
    "1 วัน": "1d",
}

PERIODS = {
    "1 เดือน": "1mo",
    "3 เดือน": "3mo",
    "6 เดือน": "6mo",
    "1 ปี": "1y",
    "2 ปี": "2y",
    "5 ปี": "5y",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .signal {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.25);
        text-align: center;
        font-weight: 700;
        font-size: 24px;
    }
    .small {font-size: 12px; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Normalize timezone for display/backtesting.
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    return df


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Build 4H candles from hourly data when provider supports 1h."""
    if df.empty:
        return df

    out = df.resample("4h", label="right", closed="right").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return out.dropna()


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    df: pd.DataFrame,
    ema_fast=20,
    ema_slow=50,
    ema_trend=200,
    rsi_period=14,
    atr_period=14,
) -> pd.DataFrame:
    x = df.copy()

    x[f"EMA{ema_fast}"] = x["Close"].ewm(span=ema_fast, adjust=False).mean()
    x[f"EMA{ema_slow}"] = x["Close"].ewm(span=ema_slow, adjust=False).mean()
    x[f"EMA{ema_trend}"] = x["Close"].ewm(span=ema_trend, adjust=False).mean()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))
    x["RSI"] = x["RSI"].fillna(50)

    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_Signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_Hist"] = x["MACD"] - x["MACD_Signal"]

    prev = x["Close"].shift(1)
    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - prev).abs(),
            (x["Low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["ATR"] = tr.rolling(atr_period).mean()
    x["ATR_Pct"] = x["ATR"] / x["Close"] * 100

    # 20-bar dynamic levels.
    x["Support"] = x["Low"].rolling(20).min()
    x["Resistance"] = x["High"].rolling(20).max()

    # Volume average where volume exists.
    x["Volume_MA"] = x["Volume"].rolling(20).mean()

    return x.dropna()


# ============================================================
# SIGNAL ENGINE
# ============================================================

def signal_for_row(row, ema_fast=20, ema_slow=50, ema_trend=200):
    ef = row[f"EMA{ema_fast}"]
    es = row[f"EMA{ema_slow}"]
    et = row[f"EMA{ema_trend}"]

    score = 0
    reasons = []

    if row["Close"] > ef:
        score += 1
        reasons.append("ราคาเหนือ EMA เร็ว")
    else:
        score -= 1
        reasons.append("ราคาต่ำกว่า EMA เร็ว")

    if ef > es:
        score += 1
        reasons.append("EMA เร็ว > EMA ช้า")
    else:
        score -= 1
        reasons.append("EMA เร็ว < EMA ช้า")

    if row["Close"] > et:
        score += 1
        reasons.append("ราคาเหนือ EMA Trend")
    else:
        score -= 1
        reasons.append("ราคาต่ำกว่า EMA Trend")

    if row["RSI"] >= 50:
        score += 1
        reasons.append("RSI > 50")
    else:
        score -= 1
        reasons.append("RSI < 50")

    if row["MACD"] > row["MACD_Signal"]:
        score += 1
        reasons.append("MACD bullish")
    else:
        score -= 1
        reasons.append("MACD bearish")

    if score >= 3:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    confidence = min(100, int(50 + abs(score) * 10))

    return signal, score, confidence, reasons


def multi_timeframe_bias(symbol: str, selected_period: str):
    """Fetch daily and hourly context for a simple MTF bias."""
    results = {}

    for label, interval in [("H1", "1h"), ("D1", "1d")]:
        try:
            d = download_data(symbol, selected_period, interval)
            if len(d) < 220:
                results[label] = ("N/A", 0)
                continue

            d = add_indicators(d)
            row = d.iloc[-1]
            sig, score, conf, _ = signal_for_row(row)
            results[label] = (sig, score)
        except Exception:
            results[label] = ("N/A", 0)

    return results


# ============================================================
# BACKTEST
# ============================================================

def backtest(df: pd.DataFrame, ema_fast, ema_slow, ema_trend):
    x = df.copy()

    sigs = []
    scores = []

    for _, row in x.iterrows():
        s, sc, _, _ = signal_for_row(row, ema_fast, ema_slow, ema_trend)
        sigs.append(s)
        scores.append(sc)

    x["Signal"] = sigs
    x["Score"] = scores

    # Position is based on prior candle signal to avoid look-ahead.
    raw_pos = x["Signal"].map({"BUY": 1, "SELL": -1, "NEUTRAL": 0})
    x["Position"] = raw_pos.replace(0, np.nan).ffill().fillna(0)

    x["Market_Return"] = x["Close"].pct_change().fillna(0)
    x["Strategy_Return"] = x["Position"].shift(1).fillna(0) * x["Market_Return"]

    x["Equity"] = (1 + x["Strategy_Return"]).cumprod()

    total_return = float(x["Equity"].iloc[-1] - 1)

    running_max = x["Equity"].cummax()
    dd = x["Equity"] / running_max - 1
    max_dd = float(dd.min())

    # Annualized Sharpe using observed bars. This is an approximation.
    sr_std = x["Strategy_Return"].std()
    sharpe = (
        float(x["Strategy_Return"].mean() / sr_std * np.sqrt(252))
        if sr_std and np.isfinite(sr_std)
        else 0.0
    )

    changes = x["Position"].diff().fillna(0)
    entries = int((changes != 0).sum())

    return x, total_return, max_dd, sharpe, entries


# ============================================================
# RISK
# ============================================================

def pip_size(asset_name):
    if "JPY" in asset_name:
        return 0.01
    if "Gold" in asset_name:
        return 0.1
    if "BTC" in asset_name:
        return 1.0
    return 0.0001


def risk_plan(price, atr, signal, account, risk_pct, rr, asset_name):
    pips = pip_size(asset_name)

    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.005

    if signal == "BUY":
        entry = price
        sl = price - atr
        tp = price + atr * rr
    elif signal == "SELL":
        entry = price
        sl = price + atr
        tp = price - atr * rr
    else:
        entry = price
        sl = price - atr
        tp = price + atr * rr

    risk_cash = account * risk_pct / 100
    stop_distance = abs(entry - sl)
    units = risk_cash / stop_distance if stop_distance else 0
    stop_pips = stop_distance / pips

    return entry, sl, tp, risk_cash, units, stop_pips


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📈 Forex Analyzer Pro")

asset_name = st.sidebar.selectbox("สินทรัพย์", list(ASSETS.keys()), index=0)
symbol = ASSETS[asset_name]

tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=2)
interval = TIMEFRAMES[tf_label]

period_label = st.sidebar.selectbox("ช่วงข้อมูล", list(PERIODS.keys()), index=2)
period = PERIODS[period_label]

st.sidebar.divider()
st.sidebar.subheader("Indicators")

ema_fast = st.sidebar.number_input("EMA Fast", 2, 100, 20)
ema_slow = st.sidebar.number_input("EMA Slow", 3, 200, 50)
ema_trend = st.sidebar.number_input("EMA Trend", 10, 500, 200)
rsi_period = st.sidebar.number_input("RSI Period", 2, 50, 14)
atr_period = st.sidebar.number_input("ATR Period", 2, 50, 14)

st.sidebar.divider()
st.sidebar.subheader("Risk Management")

account = st.sidebar.number_input(
    "Account size",
    min_value=100.0,
    value=1000.0,
    step=100.0,
)

risk_pct = st.sidebar.slider(
    "Risk per trade (%)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

rr = st.sidebar.slider(
    "Risk / Reward",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
)

st.sidebar.divider()

refresh = st.sidebar.button("🔄 Refresh data", use_container_width=True)
if refresh:
    st.cache_data.clear()
    st.rerun()


# ============================================================
# HEADER / LOAD
# ============================================================

st.title("📊 Forex Analyzer Pro")
st.caption(
    f"{asset_name} • {tf_label} • Data source: Yahoo Finance • "
    f"อัปเดตหน้า Dashboard เมื่อโหลดข้อมูล"
)

with st.spinner("กำลังโหลดและคำนวณ indicators..."):
    raw = download_data(symbol, period, interval)

# Yahoo Finance often doesn't provide 4H directly.
if interval == "4h":
    hourly = download_data(symbol, period, "1h")
    raw = resample_4h(hourly)

if raw.empty:
    st.error(
        "ไม่พบข้อมูลจากผู้ให้บริการข้อมูลในขณะนี้ "
        "ลองเปลี่ยน Timeframe/ช่วงข้อมูล แล้วกด Refresh"
    )
    st.stop()

data = add_indicators(
    raw,
    ema_fast,
    ema_slow,
    ema_trend,
    rsi_period,
    atr_period,
)

if len(data) < 30:
    st.error("ข้อมูลไม่เพียงพอสำหรับคำนวณ indicators")
    st.stop()

latest = data.iloc[-1]
signal, score, confidence, reasons = signal_for_row(
    latest, ema_fast, ema_slow, ema_trend
)

# MTF
mtf = multi_timeframe_bias(symbol, period)


# ============================================================
# TOP CARDS
# ============================================================

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric("ราคา", f"{latest['Close']:.5f}")
m2.metric("Signal", signal)
m3.metric("Score", f"{score:+d}/5")
m4.metric("Confidence", f"{confidence}%")
m5.metric("RSI", f"{latest['RSI']:.1f}")
m6.metric("ATR", f"{latest['ATR']:.5f}")

st.divider()

# ============================================================
# MARKET OVERVIEW
# ============================================================

left, right = st.columns([1.5, 1])

with left:
    st.subheader("🧭 Market Overview")

    ef = latest[f"EMA{ema_fast}"]
    es = latest[f"EMA{ema_slow}"]
    et = latest[f"EMA{ema_trend}"]

    bullish_votes = sum(
        [
            latest["Close"] > ef,
            ef > es,
            latest["Close"] > et,
            latest["RSI"] >= 50,
            latest["MACD"] > latest["MACD_Signal"],
        ]
    )

    if bullish_votes >= 4:
        trend = "Bullish"
    elif bullish_votes <= 1:
        trend = "Bearish"
    else:
        trend = "Sideways"

    a, b, c = st.columns(3)
    a.metric("Trend", trend)
    b.metric("Bullish votes", f"{bullish_votes}/5")
    c.metric("Volatility", f"{latest['ATR_Pct']:.2f}% ATR")

    st.write("**เหตุผลของสัญญาณ**")
    for reason in reasons:
        st.write("• " + reason)

with right:
    st.subheader("🌐 Multi-Timeframe")

    for label in ["H1", "D1"]:
        s, sc = mtf[label]
        st.metric(label, s, f"{sc:+d}" if s != "N/A" else "")

    st.caption(
        "MTF ใช้เป็นบริบทประกอบ ไม่ใช่การยืนยันผลลัพธ์ในอนาคต"
    )


# ============================================================
# PRICE CHART
# ============================================================

st.subheader("📈 Price / Indicators")

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.035,
    row_heights=[0.58, 0.21, 0.21],
)

fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data[f"EMA{ema_fast}"],
        name=f"EMA {ema_fast}",
        line=dict(width=1.5),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data[f"EMA{ema_slow}"],
        name=f"EMA {ema_slow}",
        line=dict(width=1.5),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data[f"EMA{ema_trend}"],
        name=f"EMA {ema_trend}",
        line=dict(width=1.5),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Support"],
        name="Support",
        line=dict(dash="dot"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Resistance"],
        name="Resistance",
        line=dict(dash="dot"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["RSI"],
        name="RSI",
    ),
    row=2,
    col=1,
)

fig.add_hline(y=70, line_dash="dot", row=2, col=1)
fig.add_hline(y=50, line_dash="dot", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", row=2, col=1)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MACD"],
        name="MACD",
    ),
    row=3,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MACD_Signal"],
        name="MACD Signal",
    ),
    row=3,
    col=1,
)

fig.update_layout(
    height=850,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h"),
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RISK PLAN
# ============================================================

st.subheader("🛡️ Risk Management")

entry, sl, tp, risk_cash, units, stop_pips = risk_plan(
    float(latest["Close"]),
    float(latest["ATR"]),
    signal,
    account,
    risk_pct,
    rr,
    asset_name,
)

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Entry", f"{entry:.5f}")
r2.metric("Stop Loss", f"{sl:.5f}")
r3.metric("Take Profit", f"{tp:.5f}")
r4.metric("Risk", f"{risk_cash:.2f}")
r5.metric("SL distance", f"{stop_pips:.1f} pips")

st.caption(
    f"Position size เชิงคณิตศาสตร์ประมาณ {units:,.2f} units "
    f"จากความเสี่ยง {risk_pct:.1f}% ของบัญชี. "
    "Lot จริงต้องคำนวณตาม contract size และ pip value ของโบรกเกอร์"
)


# ============================================================
# BACKTEST
# ============================================================

st.subheader("🧪 Backtest")

bt, total_return, max_dd, sharpe, entries = backtest(
    data, ema_fast, ema_slow, ema_trend
)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Strategy Return", f"{total_return * 100:.2f}%")
b2.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
b3.metric("Sharpe (approx.)", f"{sharpe:.2f}")
b4.metric("Position changes", f"{entries}")

bt_fig = go.Figure()
bt_fig.add_trace(
    go.Scatter(
        x=bt.index,
        y=bt["Equity"],
        mode="lines",
        name="Equity",
    )
)
bt_fig.update_layout(
    height=350,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(bt_fig, use_container_width=True)


# ============================================================
# SIGNAL HISTORY
# ============================================================

with st.expander("📋 Signal History / Raw Data"):
    view = bt.copy()
    view["Signal"] = view["Signal"].astype(str)

    cols = [
        "Open", "High", "Low", "Close",
        f"EMA{ema_fast}",
        f"EMA{ema_slow}",
        f"EMA{ema_trend}",
        "RSI",
        "MACD",
        "MACD_Signal",
        "ATR",
        "Support",
        "Resistance",
        "Signal",
        "Score",
    ]

    cols = [c for c in cols if c in view.columns]

    st.dataframe(
        view[cols].tail(100).sort_index(ascending=False),
        use_container_width=True,
    )

    csv = view[cols].to_csv(index=True).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="forex_analyzer_history.csv",
        mime="text/csv",
    )


# ============================================================
# STATUS / DISCLAIMERS
# ============================================================

st.divider()

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

st.caption(
    f"Last dashboard calculation: {now} • "
    "ข้อมูลและราคาอาจล่าช้า/มีข้อจำกัดตามผู้ให้บริการข้อมูล"
)

st.warning(
    "⚠️ เครื่องมือนี้ใช้เพื่อการวิเคราะห์และการทดสอบกลยุทธ์เท่านั้น "
    "ไม่ได้ส่งคำสั่งซื้อขายจริง และผล Backtest ไม่ได้ยืนยันผลลัพธ์ในอนาคต. "
    "ตรวจสอบ spread, slippage, commission, contract size และ pip value "
    "กับโบรกเกอร์ก่อนนำผลไปใช้งานจริง."
)
