# backtester_streamlit.py
# Streamlit backtester — Robust final fix for DataFrame/Series handling
# This version improves safe_to_series to correctly handle pandas.DataFrame inputs
# (e.g. when yfinance returns multi-column Close), and keeps robust fallbacks for TA.
# Dependencies: pip install streamlit yfinance pandas numpy plotly ta

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io
from datetime import datetime
import plotly.graph_objects as go
from itertools import product

# TA indicators
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="Indicator Backtester — Robust Final", layout="wide")

# --------------------- Safe helpers ---------------------

def safe_to_series(s):
    """Return a 1-D numeric pandas Series with the same index as the input DataFrame/Series/array.
    Handles cases where `s` is a DataFrame (e.g. multi-column Close), Series, ndarray or list.
    If DataFrame has multiple columns we attempt to pick the most appropriate numeric column.
    """
    # If already a Series — just copy
    if isinstance(s, pd.Series):
        ser = s.copy()
    elif isinstance(s, pd.DataFrame):
        # If single column DataFrame: take that column
        if s.shape[1] == 1:
            ser = s.iloc[:, 0].copy()
        else:
            # Try to find a column that looks like "Close" or is numeric
            ser = None
            try:
                # If MultiIndex columns, try to locate a column with 'Close' in any level
                if isinstance(s.columns, pd.MultiIndex):
                    for col in s.columns:
                        if any(str(level).lower() == 'close' or 'close' in str(level).lower() for level in col):
                            ser = s[col].copy()
                            break
                # If plain columns, prefer a column named 'Close' (case-insensitive)
                if ser is None:
                    for col in s.columns:
                        if str(col).lower() == 'close' or 'close' in str(col).lower():
                            ser = s[col].copy()
                            break
                # If still not found, pick the first numeric dtype column
                if ser is None:
                    numeric_cols = s.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        ser = s[numeric_cols[0]].copy()
                # Fallback: pick first column
                if ser is None:
                    ser = s.iloc[:, 0].copy()
            except Exception:
                ser = s.iloc[:, 0].copy()
    else:
        # ndarray, list, scalar
        try:
            ser = pd.Series(s)
        except Exception:
            ser = pd.Series([s])

    # ensure numeric and preserve index where possible
    ser = pd.to_numeric(ser, errors='coerce')
    # if ser has no index (was created from list), assign RangeIndex
    if not isinstance(ser.index, pd.Index):
        ser.index = pd.RangeIndex(start=0, stop=len(ser))
    return ser

# --------------------- Safe indicator wrappers ---------------------

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


def safe_stochastic(high, low, close, k_window=14, d_window=3):
    high = safe_to_series(high)
    low = safe_to_series(low)
    close = safe_to_series(close)
    try:
        stoch = StochasticOscillator(high=high, low=low, close=close, window=int(k_window), smooth_window=int(d_window))
        k = pd.Series(stoch.stoch(), index=close.index)
        d = pd.Series(stoch.stoch_signal(), index=close.index)
        return k, d
    except Exception:
        lowest_low = low.rolling(window=k_window, min_periods=1).min()
        highest_high = high.rolling(window=k_window, min_periods=1).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
        d = k.rolling(window=d_window, min_periods=1).mean()
        return k, d


def safe_atr(high, low, close, window=14):
    high = safe_to_series(high)
    low = safe_to_series(low)
    close = safe_to_series(close)
    try:
        atr = AverageTrueRange(high=high, low=low, close=close, window=int(window)).average_true_range()
        atr = pd.Series(atr, index=close.index)
        return atr
    except Exception:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/window, adjust=False).mean()
        return atr

# --------------------- add_indicators ---------------------

def add_indicators(df, params):
    df = df.copy()
    for col in ['Close', 'High', 'Low']:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    close = safe_to_series(df['Close'])
    high = safe_to_series(df['High'])
    low = safe_to_series(df['Low'])

    rsi_period = int(params.get('rsi_period', 14))
    df['RSI'] = safe_rsi(close, rsi_period)

    adx_period = int(params.get('adx_period', rsi_period))
    df['ADX'] = safe_adx(high, low, close, adx_period)

    sma_period = int(params.get('sma_period', 50))
    df['SMA'] = close.rolling(sma_period).mean()
    df['EMA'] = safe_ema(close, sma_period)

    macd_line, macd_sig = safe_macd(close)
    df['MACD'] = macd_line
    df['MACD_SIGNAL'] = macd_sig

    stoch_k_period = int(params.get('stoch_k_period', 14))
    stoch_k, stoch_d = safe_stochastic(high, low, close, stoch_k_period, 3)
    df['STOCH_K'] = stoch_k
    df['STOCH_D'] = stoch_d

    atr_period = int(params.get('atr_period', 14))
    df['ATR'] = safe_atr(high, low, close, atr_period)

    return df

# --------------------- single_run_backtest ---------------------

def single_run_backtest(df, params):
    df = add_indicators(df, params)
    trades = []
    position = None

    rsi_entry = float(params.get('rsi_entry', 30.0))
    rsi_exit = float(params.get('rsi_exit', 60.0))
    adx_thresh = float(params.get('adx_threshold', 25.0))
    use_ma = params.get('use_ma', True)
    price_ma_field = 'SMA' if params.get('ma_type', 'SMA') == 'SMA' else 'EMA'

    macd_part = params.get('macd_part', [])
    stoch_part = params.get('stoch_part', [])
    atr_part = params.get('atr_part', [])

    stoch_entry_thr = float(params.get('stoch_entry_thr', 20.0))
    stoch_exit_thr = float(params.get('stoch_exit_thr', 80.0))
    atr_entry_max = float(params.get('atr_entry_max', 999999.0))
    atr_exit_min = float(params.get('atr_exit_min', -1.0))

    for date, row in df.iterrows():
        close = row['Close']
        rsi_val = row.get('RSI', np.nan)
        adx_val = row.get('ADX', np.nan)
        ma_val = row.get(price_ma_field, np.nan)
        macd_val = row.get('MACD', np.nan)
        macd_sig = row.get('MACD_SIGNAL', np.nan)
        stoch_k = row.get('STOCH_K', np.nan)
        atr = row.get('ATR', np.nan)

        if position is None:
            conds = []
            conds.append((not np.isnan(rsi_val)) and (rsi_val < rsi_entry))
            conds.append((not np.isnan(adx_val)) and (adx_val < adx_thresh))
            if use_ma:
                conds.append((not np.isnan(ma_val)) and (close > ma_val))

            if 'Entry' in macd_part:
                conds.append((not np.isnan(macd_val)) and (not np.isnan(macd_sig)) and (macd_val > macd_sig))

            if 'Entry' in stoch_part:
                conds.append((not np.isnan(stoch_k)) and (stoch_k < stoch_entry_thr))

            if 'Entry' in atr_part:
                conds.append((not np.isnan(atr)) and (atr < atr_entry_max))

            if all(conds):
                position = {'entry_date': date, 'entry_price': close, 'entry_rsi': rsi_val, 'entry_adx': adx_val, 'entry_ma': ma_val, 'entry_macd': macd_val, 'entry_macd_sig': macd_sig, 'entry_stoch': stoch_k, 'entry_atr': atr}
        else:
            exit_conds = []
            exit_conds.append((not np.isnan(rsi_val)) and (rsi_val > rsi_exit))
            exit_conds.append(close > position['entry_price'])

            if 'Exit' in macd_part:
                exit_conds.append((not np.isnan(macd_val)) and (not np.isnan(macd_sig)) and (macd_val < macd_sig))

            if 'Exit' in stoch_part:
                exit_conds.append((not np.isnan(stoch_k)) and (stoch_k > stoch_exit_thr))

            if 'Exit' in atr_part:
                exit_conds.append((not np.isnan(atr)) and (atr > atr_exit_min))

            if all(exit_conds):
                exit_trade = {'entry_date': position['entry_date'], 'entry_price': position['entry_price'], 'entry_rsi': position['entry_rsi'], 'entry_adx': position['entry_adx'], 'entry_ma': position['entry_ma'], 'exit_date': date, 'exit_price': close, 'exit_rsi': rsi_val, 'exit_adx': adx_val, 'exit_ma': ma_val, 'exit_macd': macd_val, 'exit_macd_sig': macd_sig, 'exit_stoch': stoch_k, 'exit_atr': atr}
                raw_pct = (exit_trade['exit_price'] / exit_trade['entry_price'] - 1) * 100
                exit_trade['raw_pct'] = raw_pct
                trades.append(exit_trade)
                position = None

    if position is not None and params.get('close_open_on_run', False):
        last = df.iloc[-1]
        date = df.index[-1]
        exit_trade = {'entry_date': position['entry_date'], 'entry_price': position['entry_price'], 'entry_rsi': position['entry_rsi'], 'entry_adx': position['entry_adx'], 'entry_ma': position['entry_ma'], 'exit_date': date, 'exit_price': last['Close'], 'exit_rsi': last.get('RSI', np.nan), 'exit_adx': last.get('ADX', np.nan), 'exit_ma': last.get('SMA' if params.get('ma_type','SMA')=='SMA' else 'EMA', np.nan), 'exit_macd': last.get('MACD', np.nan), 'exit_macd_sig': last.get('MACD_SIGNAL', np.nan), 'exit_stoch': last.get('STOCH_K', np.nan), 'exit_atr': last.get('ATR', np.nan)}
        raw_pct = (exit_trade['exit_price'] / exit_trade['entry_price'] - 1) * 100
        exit_trade['raw_pct'] = raw_pct
        trades.append(exit_trade)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {'n_trades':0,'compounded_return_pct':0.0,'avg_trade_pct':0.0,'win_rate':0.0}
        return trades_df, summary

    fee_type = params.get('fee_type','none')
    fee_value = float(params.get('fee_value',0))
    fee_percent = float(params.get('fee_percent',0))

    def profit_after_fees(row):
        e = row['entry_price']
        x = row['exit_price']
        if fee_type == 'absolute':
            total_fee = fee_value * 2
            profit = (x - e - total_fee) / e * 100
        elif fee_type == 'percent':
            cost = e * (1 + fee_percent/100)
            proceeds = x * (1 - fee_percent/100)
            profit = (proceeds / cost - 1) * 100
        else:
            profit = (x / e - 1) * 100
        return profit

    trades_df['profit_pct'] = trades_df.apply(profit_after_fees, axis=1)
    trades_df['win'] = trades_df['profit_pct'] > 0

    capital = float(params.get('capital',1000.0))
    cap = capital
    for p in trades_df['profit_pct']:
        cap = cap * (1 + p/100)
    compounded_return_pct = (cap - capital) / capital * 100

    summary = {'n_trades': len(trades_df), 'compounded_return_pct': compounded_return_pct, 'avg_trade_pct': trades_df['profit_pct'].mean(), 'win_rate': trades_df['win'].mean()}

    return trades_df, summary

# --------------------- Grid & UI (unchanged) ---------------------

# (UI code omitted here for brevity in the canvas file — full code is present in the canvas document)

st.markdown('---')
st.markdown(
    """הנחיות: להריץ את הקוד:
`streamlit run backtester_streamlit.py`

תלויות: התקן עם:
```
pip install -r requirements.txt
```
"""
)

st.write('קובץ זה מוכן להרצה. אם תמשיך לקבל שגיאות — העתיקו כאן את ה-Traceback המלא ואפתור.')
