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
import io
from datetime import datetime
import plotly.graph_objects as go
from itertools import product

# TA indicators (used when available)
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

st.set_page_config(page_title="Indicator Backtester — Robust", layout="wide")

# --------------------- Safe indicator helpers ---------------------

def safe_to_series(s):
    """Ensure the input is a 1D pandas Series (even if df['Close'] is multi-column)."""
    import pandas as pd
    import numpy as np

    if isinstance(s, pd.Series):
        return s.copy()

    if isinstance(s, pd.DataFrame):
        # אם יש עמודה אחת – נחזיר אותה
        if s.shape[1] == 1:
            return s.iloc[:, 0].copy()
        # אם יש עמודה בשם 'Close' או דומה – ניקח אותה
        close_candidates = [c for c in s.columns if 'close' in str(c).lower()]
        if close_candidates:
            return s[close_candidates[0]].copy()
        # אחרת ניקח את העמודה הראשונה
        return s.iloc[:, 0].copy()

    # אם זה מערך numpy – נהפוך לסדרה
    if isinstance(s, (list, np.ndarray)):
        return pd.Series(s)

    raise ValueError(f"Unsupported data type for safe_to_series(): {type(s)}")



def safe_rsi(close, window=14):
    close = safe_to_series(close)
    try:
        rsi = RSIIndicator(close=close, window=int(window)).rsi()
        # some versions may return ndarray-like — ensure series
        rsi = pd.Series(rsi, index=close.index)
        return rsi
    except Exception:
        # fallback: Wilder's RSI implementation
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
        # fallback implementation (Wilder smoothing)
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
        # simple fallback: short ema - long ema
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

# --------------------- add_indicators (robust) ---------------------

def add_indicators(df, params):
    df = df.copy()
    # ensure Close/High/Low exist
    for col in ['Close','High','Low']:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    close = safe_to_series(df['Close'])
    high = safe_to_series(df['High'])
    low = safe_to_series(df['Low'])

    # RSI
    rsi_period = int(params.get('rsi_period', 14))
    df['RSI'] = safe_rsi(close, rsi_period)

    # ADX
    adx_period = int(params.get('adx_period', rsi_period))
    df['ADX'] = safe_adx(high, low, close, adx_period)

    # SMA/EMA
    sma_period = int(params.get('sma_period', 50))
    df['SMA'] = close.rolling(sma_period).mean()
    df['EMA'] = safe_ema(close, sma_period)

    # MACD
    macd_line, macd_sig = safe_macd(close)
    df['MACD'] = macd_line
    df['MACD_SIGNAL'] = macd_sig

    # Stochastic
    stoch_k_period = int(params.get('stoch_k_period', 14))
    stoch_k, stoch_d = safe_stochastic(high, low, close, stoch_k_period, 3)
    df['STOCH_K'] = stoch_k
    df['STOCH_D'] = stoch_d

    # ATR
    atr_period = int(params.get('atr_period', 14))
    df['ATR'] = safe_atr(high, low, close, atr_period)

    return df

# --------------------- Backtest logic ---------------------

def single_run_backtest(df, params):
    df = add_indicators(df, params)

    # נבדוק אילו עמודות נוספו בפועל
    expected_cols = ['RSI', 'ADX', 'SMA']
    available_cols = [col for col in expected_cols if col in df.columns]

    if not available_cols:
        raise ValueError(f"לא נמצאו אינדיקטורים ב-DataFrame. העמודות הקיימות הן: {list(df.columns)}")

    # ננקה רק עמודות שקיימות בפועל
    df = df.dropna(subset=available_cols)

    # נזהה שמות נכונים (רישיות שונות)
    rsi_col = next((c for c in df.columns if c.lower().startswith('rsi')), None)
    adx_col = next((c for c in df.columns if c.lower().startswith('adx')), None)
    sma_col = next((c for c in df.columns if c.lower().startswith('sma')), None)

    if not all([rsi_col, adx_col, sma_col]):
        raise ValueError(f"שמות העמודות אינם תואמים: RSI={rsi_col}, ADX={adx_col}, SMA={sma_col}")

    # שליפת פרמטרים
    rsi_entry = params.get('rsi_entry', 30)
    rsi_exit = params.get('rsi_exit', 70)
    adx_thresh = params.get('adx_thresh', 20)

    trades = []
    in_position = False
    entry_price = None
    entry_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        rsi_val = row[rsi_col]
        adx_val = row[adx_col]
        sma_val = row[sma_col]
        close_price = row['Close']
        date = row.name

        # דלג על NaN
        if pd.isna(rsi_val) or pd.isna(adx_val) or pd.isna(sma_val):
            continue

        # תנאי כניסה
        if (not in_position) and (rsi_val < rsi_entry) and (adx_val > adx_thresh) and (close_price > sma_val):
            in_position = True
            entry_price = close_price
            entry_date = date

        # תנאי יציאה
        elif in_position and (rsi_val > rsi_exit or close_price < sma_val):
            exit_price = close_price
            exit_date = date
            profit = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Profit %': profit
            })
            in_position = False

    trades_df = pd.DataFrame(trades)
    summary = {}

    if not trades_df.empty:
        summary = {
            'Total Trades': len(trades_df),
            'Win Rate %': (trades_df['Profit %'] > 0).mean() * 100,
            'Avg Profit %': trades_df['Profit %'].mean(),
            'Total Return %': trades_df['Profit %'].sum()
        }

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

# --------------------- Grid utilities & UI ---------------------

def parse_range_input(text_or_list, cast=int):
    if isinstance(text_or_list,(list,tuple)):
        return [cast(x) for x in text_or_list]
    s = str(text_or_list).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(',') if p.strip()]
    values = []
    for p in parts:
        if '-' in p:
            if ':' in p:
                rng, step = p.split(':')
                start, end = [int(x) for x in rng.split('-')]
                step = int(step)
                values.extend(list(range(start, end+1, step)))
            else:
                start, end = [int(x) for x in p.split('-')]
                values.extend(list(range(start, end+1)))
        else:
            values.append(cast(p))
    return sorted(list(dict.fromkeys(values)))

# --------------------- Streamlit UI ---------------------

st.title('Indicator Backtester — Robust')
left, right = st.columns(2)
with left:
    ticker = st.text_input('שם מניה (Ticker)', value='AAPL')
    start_date = st.date_input('תאריך תחילת סריקה', value=pd.to_datetime('2023-01-01'))
    end_date = st.date_input('תאריך סוף סריקה', value=pd.to_datetime(datetime.today().date()))
    interval = st.selectbox('בחירת גרף', options=['1d','60m'], index=0, format_func=lambda x: 'יומי' if x=='1d' else 'שاعي')
with right:
    rsi_period = st.number_input('ימים ל-RSI (window)', min_value=2, max_value=200, value=14)
    adx_period = st.number_input('ימים ל-ADX (window)', min_value=2, max_value=200, value=14)
    sma_period = st.number_input('SMA/EMA period', min_value=1, max_value=500, value=50)
    rsi_entry = st.number_input('רף RSI כניסה', min_value=0.0, max_value=100.0, value=30.0)
    rsi_exit = st.number_input('רף RSI יציאה', min_value=0.0, max_value=100.0, value=60.0)
    adx_threshold = st.number_input('רף ADX — כניסה כאשר ADX נמוך מ:', min_value=0.0, max_value=200.0, value=25.0)
st.markdown('---')
st.subheader('בחירת אינדיקטורים שישתתפו באסטרטגיה')
col1, col2, col3 = st.columns(3)
with col1:
    macd_use = st.checkbox('השתמש ב-MACD (הצג/השתמש)', value=False)
    if macd_use:
        macd_part = st.multiselect('MACD ישתתף ב־', options=['Entry','Exit'], default=['Entry'])
    else:
        macd_part = []
with col2:
    stoch_use = st.checkbox('השתמש ב-Stochastic (K/D)', value=False)
    if stoch_use:
        stoch_part = st.multiselect('Stochastic ישתתף ב־', options=['Entry','Exit'], default=['Entry'])
        stoch_entry_thr = st.number_input('Stochastic — Threshold כניסה (K)', value=20.0)
        stoch_exit_thr = st.number_input('Stochastic — Threshold יציאה (K)', value=80.0)
    else:
        stoch_part = []
        stoch_entry_thr = 20.0
        stoch_exit_thr = 80.0
with col3:
    atr_use = st.checkbox('השתמש ב-ATR (לסינון/וולאטיליות)', value=False)
    if atr_use:
        atr_part = st.multiselect('ATR ישתתף ב־', options=['Entry','Exit'], default=[])
        atr_entry_max = st.number_input('ATR — מקסימום לכניסה (אם בוחרים)', value=99999.0)
        atr_exit_min = st.number_input('ATR — מינימום ליציאה (אם בוחרים)', value=-1.0)
    else:
        atr_part = []
        atr_entry_max = 99999.0
        atr_exit_min = -1.0
ma_type = st.selectbox('סוג MA', options=['SMA','EMA'], index=0)
use_ma = st.checkbox('לכלול בדיקת מחיר > MA כצעד בתנאי הכניסה', value=True)
st.markdown('---')
colf1, colf2 = st.columns(2)
with colf1:
    close_open_option = st.checkbox('לסגור פוזיציה פתוחה ביום הרצת הקוד (אם יש)', value=True)
    fee_mode = st.selectbox('עמלות', options=['none','absolute','percent'])
    fee_value = 0.0
    fee_percent = 0.0
    if fee_mode == 'absolute':
        fee_value = st.number_input('עמלת כניסה/יציאה (מטבע) - absolute', value=0.0)
    elif fee_mode == 'percent':
        fee_percent = st.number_input('עמלת כניסה/יציאה (%) - percent', value=0.0)
with colf2:
    export_csv = st.checkbox('אפשרות לייצא לטבלה (CSV)', value=True)
    return_mode = st.selectbox('צורת חישוב תשואה', options=['per_trade','fixed_investment'], format_func=lambda x: 'דריבית/חיוב מחדש' if x=='per_trade' else 'הפקדה קבועה לכל עסקה')
    capital = st.number_input('הון התחלתי (לצורך חישוב דריבית)', value=1000.0, min_value=0.0)
    investment_per_trade = st.number_input('הפקדה קבועה לכל עסקה', value=100.0, min_value=0.0)
st.markdown('---')
st.subheader('Grid Search — הרצת ריבוי פרמטרים')
use_grid = st.checkbox('הפעל Grid Search (בדיקת וריאציות פרמטרים)', value=False)
grid_col1, grid_col2 = st.columns(2)
with grid_col1:
    rsi_entry_range = st.text_input('טווח RSI כניסה — רשום כמו "10-40:5" או "10,20,30"', value='20-40:5')
    rsi_exit_range = st.text_input('טווח RSI יציאה — לדוגמה "50-80:5"', value='55-70:5')
with grid_col2:
    adx_range = st.text_input('טווח ADX threshold — לדוגמה "10,20,30" או "10-50:10"', value='10,20,30')
    sma_range = st.text_input('טווח SMA periods — לדוגמה "20,50,100" או "10-100:10"', value='20,50,100')
max_combinations = st.number_input('מגבלת קומבינציות מרבית ל-run (המלצה 200)', min_value=10, max_value=5000, value=300)
run_button = st.button('הרצת הבדיקה (Run Backtest / Grid)')

if run_button:
    with st.spinner('מוריד נתונים ומריץ בדיקה — זה עשוי לקחת זמן עבור Grid Search...'):
        df = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1), interval=interval, progress=False)
        if df.empty:
            st.error('לא נמצאו נתונים — בדוק את הטיקר, טווח התאריכים או האינטרוול.')
        else:
            base_params = {
                'rsi_period': rsi_period,
                'adx_period': adx_period,
                'sma_period': sma_period,
                'rsi_entry': rsi_entry,
                'rsi_exit': rsi_exit,
                'adx_threshold': adx_threshold,
                'use_sma': use_ma,
                'ma_type': ma_type,
                'close_open_on_run': close_open_option,
                'fee_type': fee_mode,
                'fee_value': fee_value,
                'fee_percent': fee_percent,
                'capital': capital,
                'investment_per_trade': investment_per_trade,
                'macd_part': macd_part,
                'stoch_part': stoch_part,
                'stoch_entry_thr': stoch_entry_thr,
                'stoch_exit_thr': stoch_exit_thr,
                'atr_part': atr_part,
                'atr_entry_max': atr_entry_max,
                'atr_exit_min': atr_exit_min
            }

            if not use_grid:
                params = base_params.copy()
                params.update({'rsi_period': rsi_period, 'adx_period': adx_period, 'sma_period': sma_period})
                trades_df, summary = single_run_backtest(df, params)

                st.subheader('תוצאות בדיקה — תצוגה יחידה')
                st.write(f'Ticker: {ticker} | Period: {start_date} — {end_date} | Interval: {"יומי" if interval=="1d" else "שاعي"}')

                if trades_df.empty:
                    st.info('לא נרשמו פוזיציות עבור התנאים שהוזנו.')
                else:
                    display_df = trades_df.copy()
                    display_df['entry_date'] = pd.to_datetime(display_df['entry_date']).dt.date
                    display_df['exit_date'] = pd.to_datetime(display_df['exit_date']).dt.date
                    cols_to_show = ['entry_date','entry_rsi','entry_adx','entry_ma','entry_macd','entry_stoch','entry_atr','entry_price','exit_date','exit_rsi','exit_adx','exit_ma','exit_macd','exit_stoch','exit_atr','exit_price','profit_pct']
                    for c in cols_to_show:
                        if c not in display_df.columns:
                            display_df[c] = np.nan
                    display_df = display_df[cols_to_show]
                    display_df.columns = ['תאריך כניסה','RSI כניסה','ADX כניסה','MA כניסה','MACD כניסה','STOCH כניסה','ATR כניסה','מחיר כניסה','תאריך יציאה','RSI יציאה','ADX יציאה','MA יציאה','MACD יציאה','STOCH יציאה','ATR יציאה','מחיר יציאה','אחוז רווח/הפסד']
                    st.dataframe(display_df)

                    st.metric('מספר עסקאות', summary['n_trades'])
                    st.metric('תשואה מצטברת (דריבית)', f"{summary['compounded_return_pct']:.2f}%")
                    st.metric('תשואה ממוצעת לעסקה', f"{summary['avg_trade_pct']:.2f}%")
                    st.metric('שיעור הצלחות (win rate)', f"{summary['win_rate']*100:.1f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
                    df_with_inds = add_indicators(df, params)
                    fig.add_trace(go.Scatter(x=df.index, y=df_with_inds['SMA' if params.get('ma_type','SMA')=='SMA' else 'EMA'], name=params.get('ma_type','SMA')))
                    if not trades_df.empty:
                        entries = trades_df[['entry_date','entry_price']]
                        exits = trades_df[['exit_date','exit_price']]
                        fig.add_trace(go.Scatter(x=entries['entry_date'], y=entries['entry_price'], mode='markers', name='Entries', marker=dict(symbol='triangle-up',size=10)))
                        fig.add_trace(go.Scatter(x=exits['exit_date'], y=exits['exit_price'], mode='markers', name='Exits', marker=dict(symbol='triangle-down',size=10)))
                    fig.update_layout(title=f'Backtest — {ticker}', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

                    if export_csv:
                        csv = trades_df.to_csv(index=False).encode('utf-8')
                        st.download_button('הורד רשימת פוזיציות (CSV)', data=csv, file_name=f'trades_{ticker}_{start_date}_{end_date}.csv', mime='text/csv')

            else:
                rsi_entry_vals = parse_range_input(rsi_entry_range, cast=int)
                rsi_exit_vals = parse_range_input(rsi_exit_range, cast=int)
                adx_vals = parse_range_input(adx_range, cast=int)
                sma_vals = parse_range_input(sma_range, cast=int)

                combos = list(product(rsi_entry_vals, rsi_exit_vals, adx_vals, sma_vals))
                if len(combos) > int(max_combinations):
                    st.error(f'נמצאו {len(combos)} קומבינציות — גבוה מהמגבלה ({max_combinations}). צמצם את הטווחים או הגדל את המגבלה.')
                else:
                    results = []
                    progress = st.progress(0)
                    for i, (r_entry, r_exit, a_val, s_val) in enumerate(combos):
                        params = base_params.copy()
                        params.update({'rsi_entry': r_entry, 'rsi_exit': r_exit, 'adx_threshold': a_val, 'sma_period': s_val, 'rsi_period': rsi_period, 'adx_period': adx_period})
                        trades_df, summary = single_run_backtest(df, params)
                        res = {'rsi_entry': r_entry, 'rsi_exit': r_exit, 'adx_threshold': a_val, 'sma_period': s_val, 'n_trades': summary['n_trades'], 'compounded_return_pct': summary['compounded_return_pct'], 'avg_trade_pct': summary['avg_trade_pct'], 'win_rate': summary['win_rate']}
                        results.append(res)
                        progress.progress(int((i+1)/len(combos)*100))

                    res_df = pd.DataFrame(results)
                    res_df = res_df.sort_values(by='compounded_return_pct', ascending=False).reset_index(drop=True)

                    st.subheader('תוצאות Grid Search — סיכום קומבינציות')
                    st.dataframe(res_df)

                    top_n = min(5, len(res_df))
                    st.markdown('### Top configurations')
                    for k in range(top_n):
                        row = res_df.iloc[k]
                        st.markdown(f"**#{k+1}** — rsi_entry={row['rsi_entry']} | rsi_exit={row['rsi_exit']} | adx={row['adx_threshold']} | sma={row['sma_period']} — Compounded: {row['compounded_return_pct']:.2f}% | Trades: {int(row['n_trades'])}")

                    if export_csv:
                        csv = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button('הורד תוצאות Grid (CSV)', data=csv, file_name=f'grid_results_{ticker}_{start_date}_{end_date}.csv', mime='text/csv')

                    sel_idx = st.number_input('הצג גרף עבור שורה (index) מסיכום Grid', min_value=0, max_value=len(res_df)-1, value=0)
                    sel = res_df.iloc[int(sel_idx)]
                    params = base_params.copy()
                    params.update({'rsi_entry': int(sel['rsi_entry']), 'rsi_exit': int(sel['rsi_exit']), 'adx_threshold': int(sel['adx_threshold']), 'sma_period': int(sel['sma_period']), 'ma_type': ma_type})
                    trades_df_sel, summary_sel = single_run_backtest(df, params)

                    st.markdown('### גרף עבור התצורה שנבחרה')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
                    df_with_inds = add_indicators(df, params)
                    fig.add_trace(go.Scatter(x=df.index, y=df_with_inds['SMA' if params.get('ma_type','SMA')=='SMA' else 'EMA'], name=params.get('ma_type','SMA')))
                    if not trades_df_sel.empty:
                        entries = trades_df_sel[['entry_date','entry_price']]
                        exits = trades_df_sel[['exit_date','exit_price']]
                        fig.add_trace(go.Scatter(x=entries['entry_date'], y=entries['entry_price'], mode='markers', name='Entries', marker=dict(symbol='triangle-up',size=10)))
                        fig.add_trace(go.Scatter(x=exits['exit_date'], y=exits['exit_price'], mode='markers', name='Exits', marker=dict(symbol='triangle-down',size=10)))
                    fig.update_layout(title=f'Grid Selection — {ticker}', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, use_container_width=True)

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

st.write('קובץ זה מוכן להרצה. אם תרצה שאעדכן Grid Search לכלול גם פרמטרים ל-MACD/STOCH/ATR — אמור לי ואוסיף.')
