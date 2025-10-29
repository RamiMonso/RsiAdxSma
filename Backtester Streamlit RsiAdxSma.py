# streamlit_backtester_rsi_adx.py
# Backtester RSI + ADX + SMA — גרסה מתוקנת לטיפול בערכים סדרתיים ועמודי NaN
# שמור כקובץ והריץ: streamlit run streamlit_backtester_rsi_adx.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta

# matplotlib optional
MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    MATPLOTLIB_AVAILABLE = False

ALT_AVAILABLE = True
try:
    import altair as alt
except Exception:
    ALT_AVAILABLE = False

st.set_page_config(page_title='Backtester — RSI+ADX+SMA', layout='wide')
st.title('Backtester — בדיקת אסטרטגיות RSI + ADX + SMA')
st.caption('נתוני Adjusted Close מ-Yahoo Finance | חימום אינדיקטורים: 250 ימי מסחר')

# ---------------------- פונקציות אינדיקטורים (פנימיות) ----------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    return rsi

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (smooth_plus / atr)
    minus_di = 100 * (smooth_minus / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    adx = adx.replace([np.inf, -np.inf], np.nan)
    return adx

# ---------------------- עזרי מערכת ----------------------
def scalarize(v):
    """
    ההופך ערך אפשרי שהוא Series/ndarray/list לסקלר בטוח.
    אם v סדרה — מחזיר את האיבר הראשון; אם ריקה/שגיאה — np.nan.
    """
    try:
        if isinstance(v, pd.Series) or isinstance(v, np.ndarray) or isinstance(v, list):
            if len(v) > 0:
                return v[0]
            else:
                return np.nan
    except Exception:
        return np.nan
    return v

def download_data(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str, warmup_days: int = 250) -> pd.DataFrame:
    start_fetch = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
    end_fetch = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    df = yf.download(ticker, start=start_fetch.strftime('%Y-%m-%d'), end=end_fetch.strftime('%Y-%m-%d'),
                     interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            cols_lower = [c.lower() for c in df.columns]
            if 'close' in cols_lower:
                # תיקון שמות עמודות אם נדרש
                df.columns = [c.capitalize() if c.lower()=='close' else c for c in df.columns]
            else:
                return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
    return df

def calc_commission(value: float, commission_type: str, commission_value: float) -> float:
    if commission_value == 0:
        return 0.0
    if commission_type == 'אחוז':
        return value * (commission_value / 100.0)
    else:
        return commission_value

# ---------------------- UI: הגדרות משתמש ----------------------
with st.sidebar.form('settings'):
    st.header('הגדרות בדיקה')
    tickers_input = st.text_input('טיקר (או רשימת טיקרים מופרדים בפסיקים)', value='AAPL')
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    timeframe = st.selectbox('פרק זמן', options=['1d', '1h'], index=0, format_func=lambda x: 'יומי' if x=='1d' else 'שעתי')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('תאריך התחלה', value=(datetime.today() - timedelta(days=365)).date())
    with col2:
        end_date = st.date_input('תאריך סוף', value=datetime.today().date())

    st.subheader('אינדיקטורים')
    rsi_period = st.number_input('RSI — מספר ימים', min_value=2, max_value=200, value=14)
    adx_period = st.number_input('ADX — מספר ימים', min_value=2, max_value=200, value=14)
    sma_period = st.number_input('SMA — מספר ימים', min_value=2, max_value=500, value=50)

    rsi_entry_thresh = st.number_input('סף RSI לכניסה (≤)', min_value=1, max_value=100, value=30)
    rsi_exit_thresh = st.number_input('סף RSI ליציאה (≥)', min_value=1, max_value=100, value=50)
    adx_thresh = st.number_input('סף ADX לכניסה (≤)', min_value=1, max_value=100, value=25)

    include_rsi = st.checkbox('להשתמש ב-RSI?', value=True)
    include_adx = st.checkbox('להשתמש ב-ADX?', value=True)
    include_sma = st.checkbox('להשתמש ב-SMA?', value=True)
    check_not_strong_down = st.checkbox('לבדוק שהטרנד לא חזק מטה (SMA לא יורד)?', value=True)

    st.subheader('מסחר & מימון')
    fractional_shares = st.checkbox('לאפשר רכישת שברי מניה?', value=True)
    capital = st.number_input('הון התחלתי', min_value=1.0, value=10000.0, step=100.0)

    invest_mode = st.radio('שיטת השקעה לכל עסקה', options=['סכום קבוע לכל עסקה', 'הון ראשוני + ריבית דריבית'], index=0)
    fixed_invest_amount = st.number_input('סכום להשקעה בכל עסקה (כאשר בחרת סכום קבוע)', min_value=1.0, value=1000.0, step=100.0)

    st.subheader('עמלות')
    commission_type = st.selectbox('סוג עמלה', options=['אחוז', 'סכום'], index=0)
    commission_value = st.number_input('ערך עמלה (לדוגמה: 0.1 עבור 0.1% או 2 עבור 2 ש"ח)', min_value=0.0, value=0.1)

    exec_mode = st.radio('מתי לבצע ביצועים', options=['ביום האותן', 'ביום המסחר הבא'], index=0)
    execute_next_day = (exec_mode == 'ביום המסחר הבא')

    close_open_at_run = st.checkbox('לסגור פוזיציה פתוחה לפי מחיר יום ההרצה (אם קיימת)', value=True)

    st.subheader('ייצוא')
    enable_excel = st.checkbox('ייצוא ל-Excel', value=True)
    enable_pdf = st.checkbox('ייצוא ל-PDF (דורש matplotlib)', value=True)
    enable_png = st.checkbox('הורדת גרף PNG (דורש matplotlib)', value=True)

    warmup_days = 250
    st.caption(f'חימום אינדיקטורים: {warmup_days} ימי מסחר (משמש לחישובים בלבד)')

    submit = st.form_submit_button('הרץ את הבדיקה')

if not MATPLOTLIB_AVAILABLE:
    if enable_pdf or enable_png:
        st.sidebar.warning('matplotlib לא מותקן — הורדת PNG/PDF מושבתת עד להתקנת matplotlib.')
        enable_pdf = False
        enable_png = False

# ---------------------- לוגיקת backtest ----------------------
if submit:
    if not tickers:
        st.error('אין טיקרים. הזן טיקר/ים תקינים.')
    else:
        overall_results = {}
        bh_results = {}

        for ticker in tickers:
            with st.spinner(f'מוריד נתונים עבור {ticker}...'):
                raw = download_data(ticker, pd.to_datetime(start_date), pd.to_datetime(end_date), timeframe, warmup_days)
                if raw.empty:
                    st.error(f'לא נמצאו נתונים עבור {ticker} בטווח ובטיימפריים שנבחרו.')
                    continue

                df = raw.copy()
                if include_rsi:
                    df['RSI'] = compute_rsi(df['Close'], period=rsi_period)
                else:
                    df['RSI'] = np.nan
                if include_adx:
                    df['ADX'] = compute_adx(df['High'], df['Low'], df['Close'], period=adx_period)
                else:
                    df['ADX'] = np.nan
                if include_sma:
                    df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period, min_periods=1).mean()
                else:
                    df[f'SMA_{sma_period}'] = np.nan

                try:
                    scan_df = df.loc[str(start_date):str(end_date)].copy()
                except Exception:
                    scan_df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))].copy()

                if scan_df.empty:
                    st.warning(f'לא נותרו נתונים בסקירה עבור {ticker} לאחר החימום ו/או לפי טווח התאריכים.')
                    continue

                idx = list(scan_df.index)
                trades = []
                in_position = False
                entry = None
                equity = capital
                cumulative_equity = []

                bh_start_price = scan_df['Close'].iloc[0]
                bh_end_price = scan_df['Close'].iloc[-1]

                for i, current_date in enumerate(idx):
                    row = scan_df.loc[current_date]
                    price = scalarize(row.get('Close', np.nan))
                    rsi_v = scalarize(row.get('RSI', np.nan))
                    adx_v = scalarize(row.get('ADX', np.nan))
                    sma_v = scalarize(row.get(f'SMA_{sma_period}', np.nan))

                    # כניסה — בונים תנאים בצורה בטוחה
                    conds = []
                    if include_rsi:
                        if pd.isna(rsi_v):
                            conds.append(False)
                        else:
                            try:
                                conds.append(float(rsi_v) <= float(rsi_entry_thresh))
                            except Exception:
                                conds.append(False)
                    if include_adx:
                        if pd.isna(adx_v):
                            conds.append(False)
                        else:
                            try:
                                conds.append(float(adx_v) <= float(adx_thresh))
                            except Exception:
                                conds.append(False)
                    if include_sma:
                        if pd.isna(sma_v) or pd.isna(price):
                            conds.append(False)
                        else:
                            try:
                                conds.append(float(price) > float(sma_v))
                            except Exception:
                                conds.append(False)

                    # בדיקת 방향 SMA (לא חזק מטה) אם נבחר
                    if check_not_strong_down and include_sma:
                        prev_idx = i - 1
                        if prev_idx >= 0:
                            prev_sma_raw = scan_df.iloc[prev_idx].get(f'SMA_{sma_period}', np.nan)
                            prev_sma = scalarize(prev_sma_raw)
                            if pd.isna(prev_sma) or pd.isna(sma_v):
                                conds.append(False)  # אם אין מידע קודם — לא נכנס
                            else:
                                try:
                                    conds.append(float(sma_v) >= float(prev_sma))
                                except Exception:
                                    conds.append(False)

                    entry_cond = all(conds) if len(conds) > 0 else True

                    if (not in_position) and entry_cond:
                        exec_i = i + 1 if execute_next_day else i
                        if exec_i < len(idx):
                            exec_date = idx[exec_i]
                            exec_price = scalarize(scan_df.loc[exec_date, 'Close'])

                            if invest_mode == 'סכום קבוע לכל עסקה':
                                invest_amount = fixed_invest_amount
                            else:
                                invest_amount = equity if equity > 0 else fixed_invest_amount

                            quantity = invest_amount / exec_price if exec_price and exec_price>0 else 0
                            if not fractional_shares:
                                quantity = float(np.floor(quantity)) if quantity>0 else 0
                                if quantity <= 0:
                                    continue

                            entry_comm = calc_commission(invest_amount, commission_type, commission_value)
                            equity -= entry_comm

                            entry = {
                                'entry_idx': exec_i,
                                'entry_date': exec_date,
                                'entry_price': exec_price,
                                'entry_RSI': scalarize(scan_df.loc[exec_date].get('RSI', np.nan)),
                                'entry_ADX': scalarize(scan_df.loc[exec_date].get('ADX', np.nan)),
                                'entry_SMA': scalarize(scan_df.loc[exec_date].get(f'SMA_{sma_period}', np.nan)),
                                'quantity': quantity,
                                'invest_amount': invest_amount,
                                'entry_commission': entry_comm
                            }
                            in_position = True

                    # יציאה — נבדוק תנאי RSI ונבדוק מחיר יחסית לכניסה
                    if in_position:
                        exit_ok = True
                        if include_rsi:
                            cur_rsi = rsi_v
                            if pd.isna(cur_rsi):
                                exit_ok = False
                            else:
                                try:
                                    exit_ok = float(cur_rsi) >= float(rsi_exit_thresh)
                                except Exception:
                                    exit_ok = False
                        # אם תנאי היציאה מתקיים — בוצע מהלך להעריך ביצוע
                        if exit_ok:
                            exec
