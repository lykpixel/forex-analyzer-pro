import pandas as pd


def calculate_m30_signal(df):
    """
    M30 V3 Signal Engine

    Architecture:
    Trend
    Structure
    Momentum
    Momentum Strength
    Confirmation
    Final Signal
    """

    if df is None or df.empty:
        return {
            "signal": "NO DATA",
            "trend": "UNKNOWN",
            "structure": "UNKNOWN",
            "momentum": "UNKNOWN",
            "momentum_strength": "UNKNOWN",
            "confirmation": "NO",
            "score": 0,
            "rsi": 0.0,
            "price_momentum": 0.0,
        }

    data = df.copy()

    data.columns = [
        str(c).lower()
        for c in data.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    if not all(
        col in data.columns
        for col in required
    ):
        return {
            "signal": "ERROR",
            "trend": "UNKNOWN",
            "structure": "UNKNOWN",
            "momentum": "UNKNOWN",
            "momentum_strength": "UNKNOWN",
            "confirmation": "NO",
            "score": 0,
            "rsi": 0.0,
            "price_momentum": 0.0,
        }

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    data["ema20"] = data["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    data["ema50"] = data["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    last = data.iloc[-1]

    if (
        last["close"] > last["ema20"]
        and last["ema20"] > last["ema50"]
    ):
        trend = "BULLISH"

    elif (
        last["close"] < last["ema20"]
        and last["ema20"] < last["ema50"]
    ):
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

    previous = (
        data.iloc[-2]
        if len(data) >= 2
        else last
    )

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
    # RSI
    # --------------------------------------------------------

    delta = data["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        float("nan")
    )

    data["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # Price Momentum
    # --------------------------------------------------------

    data["momentum"] = (
        data["close"].pct_change(5) * 100
    )

    last = data.iloc[-1]

    rsi = last["rsi"]
    price_momentum = last["momentum"]

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    if (
        pd.isna(rsi)
        or pd.isna(price_momentum)
    ):
        momentum = "NEUTRAL"

    elif (
        rsi >= 55
        and price_momentum > 0
    ):
        momentum = "BULLISH"

    elif (
        rsi <= 45
        and price_momentum < 0
    ):
        momentum = "BEARISH"

    else:
        momentum = "NEUTRAL"

    # --------------------------------------------------------
    # Momentum Strength
    # --------------------------------------------------------

    if pd.isna(price_momentum):

        momentum_strength = "UNKNOWN"

    elif abs(price_momentum) >= 0.15:

        momentum_strength = "STRONG"

    elif abs(price_momentum) >= 0.05:

        momentum_strength = "MODERATE"

    else:

        momentum_strength = "WEAK"

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = 0

    if trend == "BULLISH":
        score += 40

    elif trend == "BEARISH":
        score -= 40

    if structure in [
        "BREAKOUT_UP",
        "HIGHER_HIGH",
    ]:
        score += 30

    elif structure in [
        "BREAKOUT_DOWN",
        "LOWER_LOW",
    ]:
        score -= 30

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    bullish_structure = structure in [
        "BREAKOUT_UP",
        "HIGHER_HIGH",
    ]

    bearish_structure = structure in [
        "BREAKOUT_DOWN",
        "LOWER_LOW",
    ]

    bullish_confirmation = (
        trend == "BULLISH"
        and bullish_structure
        and momentum == "BULLISH"
    )

    bearish_confirmation = (
        trend == "BEARISH"
        and bearish_structure
        and momentum == "BEARISH"
    )

    if bullish_confirmation:

        confirmation = "CONFIRMED LONG"

    elif bearish_confirmation:

        confirmation = "CONFIRMED SHORT"

    else:

        confirmation = "NOT CONFIRMED"

    # --------------------------------------------------------
    # Final Signal
    # --------------------------------------------------------

    if (
        bullish_confirmation
        and score >= 50
    ):
        signal = "LONG"

    elif (
        bearish_confirmation
        and score <= -50
    ):
        signal = "SHORT"

    else:
        signal = "WAIT"

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "signal": signal,
        "trend": trend,
        "structure": structure,
        "momentum": momentum,
        "momentum_strength": momentum_strength,
        "confirmation": confirmation,
        "score": score,
        "rsi": (
            float(rsi)
            if pd.notna(rsi)
            else 0.0
        ),
        "price_momentum": (
            float(price_momentum)
            if pd.notna(price_momentum)
            else 0.0
        ),
    }
