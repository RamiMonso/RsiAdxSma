# backtester_streamlit.py
# Streamlit backtester with extended indicators + grid search using `ta` library
# Robust indicator computation (safe fallbacks) to avoid ta runtime errors
# User-selectable indicator participation in strategy (entry/exit),
# MACD/Stochastic/ATR participation, thresholds, Grid Search.
# Dependencies: pip install streamlit yfinance pandas numpy plotly ta

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from itertools import product

# TA indicators (used when available)
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="Indicator Backtester — Robust", layout="wide")

# --------------------- Utilities / Safe helpers ---------------------

def _flatten_columns(df):
    """Flatten MultiIndex columns to single strings and return a copy of df."""
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            cols.append('_'.join([str(x) for x in c if x is not None and str(x) != '']))
        else:
            cols.append(str(c))
    df2 = df.copy()
    df2.columns = cols
    return df2

def safe_to_series(s):
    """Ensure the input is a 1D pandas Series (robust to dataframes with various close column names)."""
    import pandas as pd
    import numpy as np

    if isinstance(s, pd.Series):
        return s.copy()

    if isinstance(s, pd.DataFrame):
        df = _flatten_columns(s)
        # prefer exact 'Close' then 'Adj Close' then any column containing 'close'
        candidates = [c for c in df.columns if c.lower() == 'close']
        if not candidates:
            candidates = [c for c in df.columns if c.lower() == 'adj close' or 'adjclose' in c.lower()]
        if not candidates:
            candidates = [c for c in df.columns if 'close' in c.lower()]
        if candidates:
            return df[candidates[0]].copy()
        # if only one column present, return it
        if df.shape[1] == 1:
            return df.iloc[:, 0].copy()
        raise ValueError(f"safe_to_series(): could not find a close-like column in DataFrame. Columns: {list(df.columns)}")

    # numpy array or list
    if isinstance(s, (list, np.ndarray)):
        return pd.Series(s)

    raise ValueError(f"Unsupported data type for safe_to_series(): {type(s)}")


def safe_rsi(close, window=14):
    close = safe_to_series(close)
    try:
        rsi = RSIIndicator(close=close, window=int(window)).rsi()
        rsi = pd.Series(rsi, index=close.index)
        return rsi
    except Exception:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)


def safe_adx(high, low, close, window=14):
    high = safe_to_series(high)
    low = safe_to_series(low)
    close = safe_to_series(close)
    try:
        adx = ADXIndicator(high=high, low=low, close=close, window=int(window)).adx()
        adx = pd.Series(adx, index=close.index)
        return adx
    except Exception:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean() / atr.replace(0, np.nan))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.ewm(alpha=1/window, adjust=False).mean()
        return adx.fillna(0)


def safe_ema(close, window=50):
    close = safe_to_series(close)
    try:
        ema = EMAIndicator(close=close, window=int(window)).ema_indicator()
        ema = pd.Series(ema, index=close.index)
        return ema
    except Exception:
        return close.ewm(span=window, adjust=False).mean()


def safe_macd(close):
    close = safe_to_series(close)
    try:
        macd = MACD(close=close)
        macd_line = pd.Series(macd.macd(), index=close.index)
        macd_sig = pd.Series(macd.macd_signal(), index=close.index)
        return macd_line, macd_sig
    except Exception:
        fast = close.ewm(span=12, adjust=False).mean()
        slow = close.ewm(span=26, adjust=False).mean()
        macd_line = fast - slow
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line, macd_sig


def safe_stochastic(high, low
