"""
Forex Analyzer Pro v3 — single-file Streamlit dashboard

Improvements in v3:
- Stronger asset/data synchronization to prevent stale chart data.
- Versioned data cache key so old cached datasets are not reused after update.
- Explicit validation that the selected asset's latest price matches the plotted data.
- Chart title and data-status line show the selected asset and Yahoo ticker.
- 4H data is built from H1 data with the same asset-validation path.
- Safer numeric cleaning for Yahoo Finance responses.
- Refresh button clears the data cache.

Install:
    pip install streamlit pandas numpy plotly yfinance

Run:
    streamlit run forex_analyzer_pro_v3.py

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

APP_VERSION = "v3"
DATA_CACHE_VERSION = "forex-analyzer-v3-data"

st.set_page_config(
    page_title="Forex Analyzer Pro v3",
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
    .small {font-size: 12px; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_data(
    symbol: str,
    period: str,
    interval: str,
    cache_version: str = DATA_CACHE_VERSION,
) -> pd.DataFrame:
    """Download one symbol only and normalize the Yahoo response.

    cache_version is deliberately part of the cache key so a new app version
    cannot accidentally reuse a dataframe cached by an older version.
    """
    del cache_version  # only used to version the Streamlit cache key

    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Yahoo may return MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the OHLCV level. For a one-ticker response this safely
        # collapses (field, ticker) -> field.
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        elif "Close" in df.columns.get_level_values(-1):
            df.columns = df.columns.get_level_values(-1)
        else:
            df.columns = [str(c[-1]) for c in df.columns]

    df.columns = [str(c).strip().title() for c in df.columns]

    required = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    if df.empty:
        return pd.DataFrame()

    # Keep volume usable even when a provider returns NaN/None.
    df["Volume"] = df["Volume"].fillna(0.0)

    # Remove duplicate timestamps and sort chronologically.
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Normalize timezone for display/backtesting.
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    # Basic OHLC integrity checks.
    bad = (
        (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
        | (df["High"] < df["Low"])
    )
    df = df.loc[~bad].copy()

    if df.empty:
        return pd.DataFrame()

    df.attrs["symbol"] = symbol
    df.attrs["cache_version"] = DATA_CACHE_VERSION
    return df


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Build 4H candles from hourly data."""
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
    ).dropna(subset=["Open", "High", "Low", "Close"])

    out.attrs["symbol"] = df.attrs.get("symbol", "")
    out.attrs["cache_version"] = DATA_CACHE_VERSION
    return out


def validate_asset_data(df: pd.DataFrame, asset_name: str, symbol: str) -> tuple[bool, str]:
    """Check that the dataframe is plausible for the selected asset."""
    if df.empty:
        return False, "ไม่พบข้อมูล"

    stored_symbol = df.attrs.get("symbol", symbol)
    if stored_symbol != symbol:
        return False, f"ข้อมูลไม่ตรงสินทรัพย์: ได้ {stored_symbol} แต่เลือก {symbol}"

    last_close = float(df["Close"].iloc[-1])
    if not np.isfinite(last_close) or last_close <= 0:
        return False, "ราคาล่าสุดไม่ถูกต้อง"

    # Sanity checks designed to catch cross-asset/stale-data mixups.
    if asset_name == "BTC/USD" and last_close < 1000:
        return False, f"BTC/USD แต่ราคาที่ได้ {last_close:.5f} ต่ำผิดปกติ — อาจเป็นข้อมูลค้าง"
    if asset_name != "BTC/USD" and "Gold" not in asset_name and last_close > 10000:
        return False, f"{asset_name} แต่ราคาที่ได้ {last_close:.5f} สูงผิดปกติ — อาจเป็นข้อมูลค้าง"

    return True, f"{asset_name} • Yahoo: {symbol} • ล่าสุด {last_close:.8g}"


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
    x.attrs.update(df.attrs)

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

    x["Support"] = x["Low"].rolling(20).min()
    x["Resistance"] = x["High"].rolling(20).max()
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
    results = {}
    for label, interval in [("H1", "1h"), ("D1", "1d")]:
        try:
            d = download_data(symbol, selected_period, interval)
            if len(d) < 220:
                results[label] = ("N/A", 0)
                continue
            d = add_indicators(d)
            row = d.iloc[-1]
            sig, score, _, _ = signal_for_row(row)
            results[label] = (sig, score)
        except Exception:
            results[label] = ("N/A", 0)
    return results


# ============================================================
# BACKTEST
# ============================================================

def backtest(df: pd.DataFrame, ema_fast, ema_slow, ema_trend):
    x = df.copy()
    sigs, scores = [], []

    for _, row in x.iterrows():
        s, sc, _, _ = signal_for_row(row, ema_fast, ema_slow, ema_trend)
        sigs.append(s)
        scores.append(sc)

    x["Signal"] = sigs
    x["Score"] = scores

    raw_pos = x["Signal"].map({"BUY": 1, "SELL": -1, "NEUTRAL": 0})
    x["Position"] = raw_pos.replace(0, np.nan).ffill().fillna(0)

    x["Market_Return"] = x["Close"].pct_change().fillna(0)
    x["Strategy_Return"] = x["Position"].shift(1).fillna(0) * x["Market_Return"]
    x["Equity"] = (1 + x["Strategy_Return"]).cumprod()

    total_return = float(x["Equity"].iloc[-1] - 1)
    running_max = x["Equity"].cummax()
    dd = x["Equity"] / running_max - 1
    max_dd = float(dd.min())

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
        entry, sl, tp = price, price - atr, price + atr * rr
    elif signal == "SELL":
        entry, sl, tp = price, price + atr, price - atr * rr
    else:
        entry, sl, tp = price, price - atr, price + atr * rr

    risk_cash = account * risk_pct / 100
    stop_distance = abs(entry - sl)
    units = risk_cash / stop_distance if stop_distance else 0
    stop_pips = stop_distance / pips
    return entry, sl, tp, risk_cash, units, stop_pips


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(f"📈 Forex Analyzer Pro {APP_VERSION}")

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

account = st.sidebar.number_input("Account size", min_value=100.0, value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
rr = st.sidebar.slider("Risk / Reward", 0.5, 5.0, 2.0, 0.5)

st.sidebar.divider()
refresh = st.sidebar.button("🔄 Refresh data", use_container_width=True)
if refresh:
    st.cache_data.clear()
    st.rerun()


# ============================================================
# HEADER / LOAD
# ============================================================

st.title(f"📊 Forex Analyzer Pro {APP_VERSION}")
st.caption(
    f"{asset_name} • {tf_label} • Yahoo Finance: {symbol} • "
    "เปลี่ยนสินทรัพย์/Timeframe แล้วระบบจะโหลดชุดข้อมูลของสินทรัพย์นั้นโดยตรง"
)

with st.spinner("กำลังโหลดและตรวจสอบข้อมูล..."):

    # Yahoo Finance จำกัดข้อมูลย้อนหลังสำหรับ Intraday
    if interval in ("15m", "30m", "1h"):
        download_period = "60d"
    else:
        download_period = period

    if interval == "4h":
        hourly = download_data(symbol, download_period, "1h")
        raw = resample_4h(hourly)
    else:
        raw = download_data(symbol, download_period, interval)

ok, status_text = validate_asset_data(raw, asset_name, symbol)
if not ok:
    st.error(f"ตรวจสอบข้อมูลไม่ผ่าน: {status_text}")
    st.info("ลองกด 🔄 Refresh data เพื่อดึงข้อมูลชุดใหม่")
    st.stop()

st.success(f"✅ Data integrity OK • {status_text}")

data = add_indicators(raw, ema_fast, ema_slow, ema_trend, rsi_period, atr_period)

if len(data) < 30:
    st.error("ข้อมูลไม่เพียงพอสำหรับคำนวณ indicators")
    st.stop()

latest = data.iloc[-1]

# Final consistency check after indicators.
latest_close = float(latest["Close"])
if asset_name == "BTC/USD" and latest_close < 1000:
    st.error("หยุดการวิเคราะห์: BTC/USD แต่ราคากราฟไม่สอดคล้องกับ BTC/USD")
    st.stop()

signal, score, confidence, reasons = signal_for_row(latest, ema_fast, ema_slow, ema_trend)
mtf = multi_timeframe_bias(symbol, period)


# ============================================================
# TOP CARDS
# ============================================================

# อ่านราคาแบบคั่นหลักพันและทศนิยม 2 ตำแหน่ง
price_display = f"{latest_close:,.2f}"

# Signal ให้เห็นชัดบนมือถือ
signal_color = "#16a34a" if signal == "BUY" else "#dc2626" if signal == "SELL" else "#6b7280"
signal_icon = "▲" if signal == "BUY" else "▼" if signal == "SELL" else "●"

st.markdown(
    f"""
    <div style="
        display:flex;
        gap:12px;
        align-items:stretch;
        flex-wrap:wrap;
        margin-bottom:12px;
    ">
      <div style="
          flex:1 1 220px;
          padding:16px 18px;
          border:1px solid #e5e7eb;
          border-radius:14px;
          background:#ffffff;
      ">
        <div style="font-size:0.95rem;color:#6b7280;">ราคา</div>
        <div style="font-size:2rem;font-weight:750;line-height:1.2;">{price_display}</div>
      </div>
      <div style="
          flex:1 1 180px;
          padding:14px 18px;
          border-radius:14px;
          background:{signal_color};
          color:white;
          text-align:center;
      ">
        <div style="font-size:0.95rem;font-weight:600;opacity:0.95;">SIGNAL</div>
        <div style="font-size:2rem;font-weight:800;line-height:1.2;">
          {signal_icon} {signal}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Score", f"{score:+d}/5")
m2.metric("Confidence", f"{confidence}%")
m3.metric("RSI", f"{latest['RSI']:.1f}")
m4.metric("ATR", f"{latest['ATR']:,.2f}")

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

    trend = "Bullish" if bullish_votes >= 4 else "Bearish" if bullish_votes <= 1 else "Sideways"

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
    st.caption("MTF ใช้เป็นบริบทประกอบ ไม่ใช่การยืนยันผลลัพธ์ในอนาคต")


# ============================================================
# PRICE CHART
# ============================================================

st.subheader(f"📈 Price / Indicators — {asset_name} ({symbol})")

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
        name=f"Price • {asset_name}",
    ),
    row=1,
    col=1,
)

for col, name in [
    (f"EMA{ema_fast}", f"EMA {ema_fast}"),
    (f"EMA{ema_slow}", f"EMA {ema_slow}"),
    (f"EMA{ema_trend}", f"EMA {ema_trend}"),
    ("Support", "Support"),
    ("Resistance", "Resistance"),
]:
    dash = "dot" if col in ["Support", "Resistance"] else "solid"
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[col],
            name=name,
            line=dict(width=1.3, dash=dash),
        ),
        row=1,
        col=1,
    )

fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", row=2, col=1)
fig.add_hline(y=50, line_dash="dot", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", row=2, col=1)

fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="MACD Signal"), row=3, col=1)

fig.update_layout(
    title=f"{asset_name} • {tf_label} • Yahoo Finance {symbol} • Last {latest_close:.8g}",
    height=850,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(orientation="h"),
)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig.update_yaxes(title_text="MACD", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RISK PLAN
# ============================================================

st.subheader("🛡️ Risk Management")
entry, sl, tp, risk_cash, units, stop_pips = risk_plan(
    latest_close,
    float(latest["ATR"]),
    signal,
    account,
    risk_pct,
    rr,
    asset_name,
)

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Entry", f"{entry:.8g}")
r2.metric("Stop Loss", f"{sl:.8g}")
r3.metric("Take Profit", f"{tp:.8g}")
r4.metric("Risk", f"{risk_cash:.2f}")
r5.metric("SL distance", f"{stop_pips:.1f} pips")

st.caption(
    f"Position size เชิงคณิตศาสตร์ประมาณ {units:,.4f} units จากความเสี่ยง {risk_pct:.1f}% ของบัญชี. "
    "Lot จริงต้องคำนวณตาม contract size และ pip value ของโบรกเกอร์"
)


# ============================================================
# BACKTEST
# ============================================================

st.subheader("🧪 Backtest")
bt, total_return, max_dd, sharpe, entries = backtest(data, ema_fast, ema_slow, ema_trend)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Strategy Return", f"{total_return * 100:.2f}%")
b2.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
b3.metric("Sharpe (approx.)", f"{sharpe:.2f}")
b4.metric("Position changes", f"{entries}")

bt_fig = go.Figure()
bt_fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity"], mode="lines", name="Equity"))
bt_fig.update_layout(height=350, hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(bt_fig, use_container_width=True)


# ============================================================
# SIGNAL HISTORY
# ============================================================

with st.expander("📋 Signal History / Raw Data"):
    view = bt.copy()
    view["Signal"] = view["Signal"].astype(str)

    cols = [
        "Open", "High", "Low", "Close",
        f"EMA{ema_fast}", f"EMA{ema_slow}", f"EMA{ema_trend}",
        "RSI", "MACD", "MACD_Signal", "ATR", "Support", "Resistance",
        "Signal", "Score",
    ]
    cols = [c for c in cols if c in view.columns]

    table_data = view[cols].tail(100).sort_index(ascending=False)

    # จัดรูปแบบตัวเลขในตารางให้เป็น 117,070.43 แทน 117070.4297
    number_columns = [
        "Open", "High", "Low", "Close",
        f"EMA{ema_fast}", f"EMA{ema_slow}", f"EMA{ema_trend}",
        "ATR", "Support", "Resistance",
    ]
    number_columns = [c for c in number_columns if c in table_data.columns]

    column_config = {
        c: st.column_config.NumberColumn(c, format="%,.2f")
        for c in number_columns
    }
    if "RSI" in table_data.columns:
        column_config["RSI"] = st.column_config.NumberColumn("RSI", format="%.1f")
    if "Score" in table_data.columns:
        column_config["Score"] = st.column_config.NumberColumn("Score", format="%+d")

    st.dataframe(
        table_data,
        use_container_width=True,
        column_config=column_config,
    )

    csv = view[cols].to_csv(index=True).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=f"forex_analyzer_{asset_name.replace('/', '_').replace(' ', '_')}_{APP_VERSION}.csv",
        mime="text/csv",
    )


# ============================================================
# STATUS / DISCLAIMERS
# ============================================================

st.divider()
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.caption(
    f"Forex Analyzer Pro {APP_VERSION} • Last dashboard calculation: {now} • "
    f"Asset: {asset_name} • Yahoo ticker: {symbol}"
)

st.warning(
    "⚠️ เครื่องมือนี้ใช้เพื่อการวิเคราะห์และการทดสอบกลยุทธ์เท่านั้น "
    "ไม่ได้ส่งคำสั่งซื้อขายจริง และผล Backtest ไม่ได้ยืนยันผลลัพธ์ในอนาคต. "
    "ตรวจสอบ spread, slippage, commission, contract size และ pip value "
    "กับโบรกเกอร์ก่อนนำผลไปใช้งานจริง."
)

# ============================================================
# M30 V3 - SIGNAL ENGINE
# Step 1: Trend + Market Structure
# ============================================================

import pandas as pd
import numpy as np


def calculate_m30_signal(df):
    """
    วิเคราะห์ Trend และ Market Structure สำหรับ M30
    คืนค่าเป็น dictionary เพื่อใช้ต่อกับระบบ Signal
    """

    if df is None or df.empty:
        return {
            "signal": "NO DATA",
            "trend": "UNKNOWN",
            "structure": "UNKNOWN",
            "score": 0
        }

    data = df.copy()

    # ตรวจสอบชื่อคอลัมน์
    data.columns = [str(c).lower() for c in data.columns]

    required = ["open", "high", "low", "close"]

    if not all(col in data.columns for col in required):
        return {
            "signal": "ERROR",
            "trend": "UNKNOWN",
            "structure": "UNKNOWN",
            "score": 0
        }

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
    data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()

    last = data.iloc[-1]

    if last["close"] > last["ema20"] and last["ema20"] > last["ema50"]:
        trend = "BULLISH"

    elif last["close"] < last["ema20"] and last["ema20"] < last["ema50"]:
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"

    # --------------------------------------------------------
    # Market Structure
    # --------------------------------------------------------

    lookback = min(10, len(data))

    recent = data.tail(lookback)

    recent_high = recent["high"].max()
    recent_low = recent["low"].min()

    previous = data.iloc[-2] if len(data) >= 2 else last

    if last["close"] > recent_high:
        structure = "BREAKOUT_UP"

    elif last["close"] < recent_low:
        structure = "BREAKOUT_DOWN"

    elif last["close"] > previous["high"]:
        structure = "HIGHER_HIGH"

    elif last["close"] < previous["low"]:
        structure = "LOWER_LOW"

    else:
        structure = "RANGE"

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = 0

    if trend == "BULLISH":
        score += 40

    elif trend == "BEARISH":
        score -= 40

    if structure in ["BREAKOUT_UP", "HIGHER_HIGH"]:
        score += 30

    elif structure in ["BREAKOUT_DOWN", "LOWER_LOW"]:
        score -= 30

    # --------------------------------------------------------
    # Final Signal
    # --------------------------------------------------------

    if score >= 50:
        signal = "LONG"

    elif score <= -50:
        signal = "SHORT"

    else:
        signal = "WAIT"

    return {
        "signal": signal,
        "trend": trend,
        "structure": structure,
        "score": score
    }
