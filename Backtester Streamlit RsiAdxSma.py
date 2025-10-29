# streamlit_backtester_rsi_adx.py
# Backtester RSI + ADX + SMA — כולל בדיקה לאחר התקופה עד יום הריצה וסימון סגירות מיוחדות
# שמור והרץ: streamlit run streamlit_backtester_rsi_adx.py

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

# altair fallback
ALT_AVAILABLE = True
try:
    import altair as alt
except Exception:
    ALT_AVAILABLE = False

st.set_page_config(page_title='Backtester — RSI+ADX+SMA', layout='wide')
st.title('Backtester — בדיקת אסטרטגיות RSI + ADX + SMA')
st.caption('נתוני Adjusted Close מ-Yahoo Finance | חימום אינדיקטורים: 250 ימי מסחר')

# ---------------------- פונקציות עזר ----------------------
def to_scalar(x):
    if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, (np.ndarray, pd.Series)):
        return x
    if x is None:
        return np.nan
    if isinstance(x, pd.Series):
        if x.size == 0:
            return np.nan
        if x.size == 1:
            try:
                return x.iloc[0]
            except Exception:
                try:
                    return x.values[0]
                except Exception:
                    return np.nan
        else:
            return np.nan
    if isinstance(x, (np.ndarray, list, tuple)):
        try:
            arr = np.asarray(x)
            if arr.size == 0:
                return np.nan
            if arr.size == 1:
                return arr.reshape(-1)[0].item()
            return np.nan
        except Exception:
            return np.nan
    try:
        return float(x)
    except Exception:
        try:
            return x.item()
        except Exception:
            return np.nan

# ---------------------- אינדיקטורים (Wilder style) ----------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    return rsi

def compute_adx(high, low, close, period: int = 14) -> pd.Series:
    if isinstance(high, pd.DataFrame):
        if high.shape[1] == 1:
            high = high.iloc[:, 0]
        else:
            high = high.select_dtypes(include=[np.number]).iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        if low.shape[1] == 1:
            low = low.iloc[:, 0]
        else:
            low = low.select_dtypes(include=[np.number]).iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            close = close.select_dtypes(include=[np.number]).iloc[:, 0]

    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    atr_safe = atr.replace(0, np.nan)
    plus_di = 100 * (smooth_plus / atr_safe)
    minus_di = 100 * (smooth_minus / atr_safe)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom

    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    adx = adx.replace([np.inf, -np.inf], np.nan)
    adx.index = close.index
    return adx

# ---------------------- עזרי מערכת ----------------------
def download_data(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str, warmup_days: int = 250) -> pd.DataFrame:
    """
    הורדת נתונים: מורידים עד יום הריצה (כדי לאפשר בדיקה של ימים לאחר end_date).
    החימום עדיין נעשה לפני start_date.
    """
    start_fetch = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
    # הורד עד היום (כולל), כדי שנוכל לבדוק ימים אחרי התאריך שסומנה ע"י המשתמש
    today_date = pd.to_datetime(datetime.today().date())
    end_fetch = max(pd.to_datetime(end_date), today_date) + pd.Timedelta(days=1)
    df = yf.download(ticker,
                     start=start_fetch.strftime('%Y-%m-%d'),
                     end=end_fetch.strftime('%Y-%m-%d'),
                     interval=interval,
                     progress=False,
                     auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            cols_lower = [c.lower() for c in df.columns]
            if 'close' in cols_lower:
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

# ---------------------- UI — הגדרות משתמש ----------------------
with st.sidebar.form('settings'):
    st.header('הגדרות בדיקה')
    tickers_input = st.text_input('טיקר (או רשימת טיקרים, מופרדים בפסיקים)', value='AAPL')
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    timeframe = st.selectbox('פרק זמן', options=['1d', '1h'], index=0, format_func=lambda x: 'יומי' if x=='1d' else 'שעתי')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('תאריך התחלה', value=(datetime.today() - timedelta(days=365)).date())
    with col2:
        end_date = st.date_input('תאריך סוף', value=datetime.today().date())

    st.subheader('אינדיקטורים')
    rsi_period = st.number_input('RSI — מספר תקופות', min_value=2, max_value=200, value=14)
    adx_period = st.number_input('ADX — מספר תקופות', min_value=2, max_value=200, value=14)
    sma_period = st.number_input('SMA — מספר תקופות', min_value=2, max_value=500, value=200)

    rsi_entry_thresh = st.number_input('סף RSI לכניסה (≤)', min_value=1, max_value=100, value=40)
    rsi_exit_thresh = st.number_input('סף RSI ליציאה (≥)', min_value=1, max_value=100, value=60)
    adx_thresh = st.number_input('סף ADX לכניסה (≤)', min_value=1, max_value=100, value=25)

    include_rsi = st.checkbox('להשתמש ב-RSI?', value=True)
    include_adx = st.checkbox('להשתמש ב-ADX?', value=True)
    include_sma = st.checkbox('להשתמש ב-SMA?', value=True)
    check_not_strong_down = st.checkbox('לבדוק שה-SMA לא יורד לעומת יום קודם?', value=false)

    st.subheader('מסחר & מימון')
    fractional_shares = st.checkbox('לאפשר רכישת שברי מניה?', value=True)
    capital = st.number_input('הון התחלתי', min_value=1.0, value=10000.0, step=100.0)

    invest_mode = st.radio('שיטת השקעה לכל עסקה', options=['סכום קבוע לכל עסקה', 'הון ראשוני + ריבית דריבית'], index=0)
    fixed_invest_amount = st.number_input('סכום להשקעה בכל עסקה (כשבחרת סכום קבוע)', min_value=1.0, value=1000.0, step=100.0)

    st.subheader('עמלות')
    commission_type = st.selectbox('סוג עמלה', options=['אחוז', 'סכום'], index=0)
    commission_value = st.number_input('ערך עמלה (למשל: 0.1 עבור 0.1% או 2 עבור 2 ש"ח)', min_value=0.0, value=0)

    exec_mode = st.radio('מתי לבצע ביצוע כאשר התנאי מתקיים', options=['ביום הסגירה', 'ביום המסחר הבא'], index=0)
    execute_next_day = (exec_mode == 'ביום המסחר הבא')

    close_open_at_run = st.checkbox('לסגור פוזיציה פתוחה לפי מחיר יום ההרצה (אם קיימת)', value=True)

    st.subheader('ייצוא / ויזואליזציה')
    enable_excel = st.checkbox('ייצוא ל-Excel', value=True)
    enable_pdf = st.checkbox('ייצוא ל-PDF (דורש matplotlib)', value=True)
    enable_png = st.checkbox('הורדת גרף PNG (דורש matplotlib)', value=True)

    warmup_days = 250
    st.caption(f'חימום אינדיקטורים: {warmup_days} ימי מסחר (משמש לחישובים בלבד)')

    submit = st.form_submit_button('הרץ את הבדיקה')

# השבת הורדות אם matplotlib חסר
if not MATPLOTLIB_AVAILABLE:
    if enable_pdf or enable_png:
        st.sidebar.warning('matplotlib לא מותקן — אפשרויות הורדת PNG/PDF מושבתות עד להתקנה.')
        enable_pdf = False
        enable_png = False

# ---------------------- לוגיקת הבדיקה ----------------------
if submit:
    if not tickers:
        st.error('אין טיקרים. אנא הזן טיקר/ים תקינים.')
    else:
        results_by_ticker = {}
        bh_by_ticker = {}

        today_date = pd.to_datetime(datetime.today().date())

        for ticker in tickers:
            with st.spinner(f'מוריד נתונים ועורך חישובים עבור {ticker}...'):
                # הורדה — שים לב: download_data יוריד גם ימים אחרי end_date עד היום
                raw = download_data(ticker, pd.to_datetime(start_date), pd.to_datetime(end_date), timeframe, warmup_days)
                if raw.empty:
                    st.error(f'לא נמצאו נתונים עבור {ticker} — בדוק טווח ותדירות.')
                    continue

                df = raw.copy()
                initial_capital = float(capital)

                # חישוב אינדיקטורים
                df['RSI'] = compute_rsi(df['Close'], period=rsi_period) if include_rsi else np.nan
                df['ADX'] = compute_adx(df['High'], df['Low'], df['Close'], period=adx_period) if include_adx else np.nan
                df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period, min_periods=1).mean() if include_sma else np.nan

                # טווח הסריקה העיקרי (לכניסות)
                try:
                    scan_df = df.loc[str(start_date):str(end_date)].copy()
                except Exception:
                    scan_df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))].copy()

                if scan_df.empty:
                    st.warning(f'לא נותרו נתונים בסקירה עבור {ticker} (לאחר חימום/טווח).')
                    continue

                # רשימת אינדקס מלאה (כדי לאתר מיקומים אחרי end_date)
                full_idx = list(df.index)
                # last index inside scan_df (נשתמש בו כנקודת התחלה לבדיקות לאחר התקופה)
                scan_idx_list = list(scan_df.index)
                last_scan_index = scan_idx_list[-1]

                # מציאת המיקום הספציפי האחרון ב-full_idx (למקרה של כפילויות — ניקח את ההופעה האחרונה)
                positions = [i for i, v in enumerate(full_idx) if v == last_scan_index]
                if positions:
                    last_scan_pos_in_full = positions[-1]
                else:
                    # fallback — השתמש באורך של scan_df כנקודת התחלה יחסית
                    last_scan_pos_in_full = len(full_idx) - 1 - (len(df) - len(scan_df))

                idx = scan_idx_list
                n = len(idx)
                trades = []
                in_position = False
                entry = None
                equity = initial_capital
                cumulative_equity = []

                bh_start_price = to_scalar(scan_df['Close'].iloc[0])
                bh_end_price = to_scalar(scan_df['Close'].iloc[-1])

                # --- לולאת בדיקה בתוך טווח המשתמש (כניסות/יציאות רגילות) ---
                for i in range(n):
                    row = scan_df.iloc[i]
                    price = to_scalar(row.get('Close', np.nan))
                    rsi_v = to_scalar(row.get('RSI', np.nan))
                    adx_v = to_scalar(row.get('ADX', np.nan))
                    sma_v = to_scalar(row.get(f'SMA_{sma_period}', np.nan))

                    # תנאי כניסה
                    entry_cond = True
                    if include_rsi:
                        entry_cond = entry_cond and (not pd.isna(rsi_v)) and (float(rsi_v) <= float(rsi_entry_thresh))
                    if include_adx:
                        entry_cond = entry_cond and (not pd.isna(adx_v)) and (float(adx_v) <= float(adx_thresh))
                    if include_sma:
                        entry_cond = entry_cond and (not pd.isna(sma_v)) and (price > float(sma_v))
                    if check_not_strong_down and include_sma:
                        prev_i = i - 1
                        if prev_i >= 0:
                            prev_sma = to_scalar(scan_df.iloc[prev_i].get(f'SMA_{sma_period}', np.nan))
                            if (not pd.isna(prev_sma)) and (not pd.isna(sma_v)):
                                entry_cond = entry_cond and (float(sma_v) >= float(prev_sma))

                    # כניסה לפוזיציה
                    if (not in_position) and entry_cond:
                        exec_i = i + 1 if execute_next_day else i
                        if exec_i < n:
                            exec_row = scan_df.iloc[exec_i]
                            exec_price = to_scalar(exec_row.get('Close', np.nan))
                            if exec_price is np.nan or pd.isna(exec_price):
                                continue

                            if invest_mode == 'סכום קבוע לכל עסקה':
                                invest_amount = float(fixed_invest_amount)
                            else:
                                invest_amount = float(equity) if equity > 0 else float(fixed_invest_amount)

                            quantity = invest_amount / float(exec_price) if float(exec_price) != 0 else 0.0
                            if not fractional_shares:
                                quantity = float(np.floor(quantity))
                                if quantity <= 0:
                                    continue

                            entry_comm = calc_commission(invest_amount, commission_type, commission_value)
                            equity -= entry_comm

                            entry = {
                                'entry_pos_full': last_scan_pos_in_full - (n - 1 - exec_i),  # approximate full position
                                'entry_idx': exec_i,
                                'entry_date': idx[exec_i],
                                'entry_price': float(exec_price),
                                'entry_RSI': to_scalar(exec_row.get('RSI', np.nan)),
                                'entry_ADX': to_scalar(exec_row.get('ADX', np.nan)),
                                'entry_SMA': to_scalar(exec_row.get(f'SMA_{sma_period}', np.nan)),
                                'quantity': quantity,
                                'invest_amount': invest_amount,
                                'entry_commission': entry_comm
                            }
                            in_position = True

                    # יציאה בתוך הטווח
                    if in_position and (entry is not None):
                        exit_cond = True
                        if include_rsi:
                            cur_rsi = to_scalar(row.get('RSI', np.nan))
                            exit_cond = exit_cond and (not pd.isna(cur_rsi)) and (float(cur_rsi) >= float(rsi_exit_thresh))

                        if exit_cond:
                            exec_i = i + 1 if execute_next_day else i
                            if exec_i < n:
                                exit_row = scan_df.iloc[exec_i]
                                exit_price = to_scalar(exit_row.get('Close', np.nan))
                                if pd.isna(exit_price):
                                    pass
                                else:
                                    exit_price = float(exit_price)
                                    if exit_price > entry['entry_price']:
                                        gross_pl = (exit_price - entry['entry_price']) * entry['quantity']
                                        exit_comm = calc_commission(exit_price * entry['quantity'], commission_type, commission_value)
                                        net_pl = gross_pl - exit_comm

                                        if invest_mode == 'סכום קבוע לכל עסקה':
                                            equity += entry['invest_amount'] + net_pl
                                        else:
                                            equity = equity + net_pl

                                        exit_dt = pd.to_datetime(idx[exec_i])
                                        entry_dt = pd.to_datetime(entry['entry_date'])
                                        duration_days = (exit_dt.normalize() - entry_dt.normalize()).days + 1
                                        try:
                                            duration_days = int(duration_days)
                                            if duration_days < 1:
                                                duration_days = 1
                                        except Exception:
                                            duration_days = np.nan

                                        trades.append({
                                            'entry_date': pd.to_datetime(entry['entry_date']),
                                            'entry_price': entry['entry_price'],
                                            'entry_RSI': entry.get('entry_RSI', np.nan),
                                            'entry_ADX': entry.get('entry_ADX', np.nan),
                                            'entry_SMA': entry.get('entry_SMA', np.nan),
                                            'exit_date': pd.to_datetime(idx[exec_i]),
                                            'exit_price': exit_price,
                                            'exit_RSI': to_scalar(exit_row.get('RSI', np.nan)),
                                            'exit_ADX': to_scalar(exit_row.get('ADX', np.nan)),
                                            'exit_SMA': to_scalar(exit_row.get(f'SMA_{sma_period}', np.nan)),
                                            'quantity': entry['quantity'],
                                            'gross_PL': gross_pl,
                                            'entry_commission': entry['entry_commission'],
                                            'exit_commission': exit_comm,
                                            'net_PL': net_pl,
                                            'pnl_pct': (net_pl / (entry['invest_amount'] if entry['invest_amount']>0 else 1)) * 100,
                                            'duration_days': duration_days,
                                            'closed_after_period': False,
                                            'closed_at_run': False
                                        })

                                        in_position = False
                                        entry = None

                    cumulative_equity.append(equity)

                # --- אם נשארה פוזיציה פתוחה בסוף התקופה - נבדוק ימים אחרי התקופה עד יום הריצה ---
                if in_position and entry is not None:
                    closed_during_extension = False
                    # נתחיל מהמיקום הבא ב-full_idx אחרי last_scan_pos_in_full
                    start_pos = last_scan_pos_in_full + 1
                    for pos in range(start_pos, len(full_idx)):
                        # נעבור על הימים/תצפיות אחרי תום הטווח ועד היום (כולל)
                        row_future = df.iloc[pos]
                        # אם התאריך גדול מיום הריצה — נפסיק
                        if pd.to_datetime(full_idx[pos]).normalize() > today_date:
                            break

                        # בדיקת תנאי יציאה על אותו יום
                        rsi_v_f = to_scalar(row_future.get('RSI', np.nan))
                        adx_v_f = to_scalar(row_future.get('ADX', np.nan))
                        sma_v_f = to_scalar(row_future.get(f'SMA_{sma_period}', np.nan))
                        price_f = to_scalar(row_future.get('Close', np.nan))

                        exit_cond_future = True
                        if include_rsi:
                            exit_cond_future = exit_cond_future and (not pd.isna(rsi_v_f)) and (float(rsi_v_f) >= float(rsi_exit_thresh))
                        # (בדרישות המקוריות יציאה אינה תלויה ב-ADX/SMA אלא רק ב-RSI ובמחיר גבוה ממחיר הכניסה;
                        # נשמור ריצ'ק עליה — כלומר נחייב גם התניה שמחיר היום יהיה גבוה ממחיר הכניסה)
                        if include_adx:
                            exit_cond_future = exit_cond_future and (not pd.isna(adx_v_f)) and (float(adx_v_f) <= float(adx_thresh))
                        if include_sma:
                            exit_cond_future = exit_cond_future and (not pd.isna(sma_v_f)) and (price_f > float(sma_v_f))

                        if exit_cond_future:
                            # אם המשתמש בחר לבצע יום הבא — נספור את היציאה ביום הבא, אחרת באותו יום
                            exec_pos = pos + 1 if execute_next_day else pos
                            # אם יש מיקום ביצוע זמין — נדאג להוציא שם. אחרת, נקבל את המחיר האחרון הזמין.
                            if exec_pos < len(full_idx):
                                exec_row = df.iloc[exec_pos]
                                exit_price = to_scalar(exec_row.get('Close', np.nan))
                                exit_dt = pd.to_datetime(full_idx[exec_pos])
                            else:
                                # אין שורה ליום הבא — קח את המחיר האחרון הזמין
                                exec_row = df.iloc[-1]
                                exit_price = to_scalar(exec_row.get('Close', np.nan))
                                exit_dt = pd.to_datetime(full_idx[-1])

                            if pd.isna(exit_price):
                                # לא ניתן לסגור אם אין מחיר ביצוע
                                continue
                            exit_price = float(exit_price)
                            # נדרוש מחיר יציאה גבוה ממחיר כניסה (כמו בהגדרה)
                            if exit_price > entry['entry_price']:
                                gross_pl = (exit_price - entry['entry_price']) * entry['quantity']
                                exit_comm = calc_commission(exit_price * entry['quantity'], commission_type, commission_value)
                                net_pl = gross_pl - exit_comm

                                if invest_mode == 'סכום קבוע לכל עסקה':
                                    equity += entry['invest_amount'] + net_pl
                                else:
                                    equity = equity + net_pl

                                # חשב משך (מיום הכניסה עד יום היציאה בפועל)
                                entry_dt = pd.to_datetime(entry['entry_date'])
                                duration_days = (pd.to_datetime(exit_dt).normalize() - entry_dt.normalize()).days + 1
                                try:
                                    duration_days = int(duration_days)
                                    if duration_days < 1:
                                        duration_days = 1
                                except Exception:
                                    duration_days = np.nan

                                trades.append({
                                    'entry_date': pd.to_datetime(entry['entry_date']),
                                    'entry_price': entry['entry_price'],
                                    'entry_RSI': entry.get('entry_RSI', np.nan),
                                    'entry_ADX': entry.get('entry_ADX', np.nan),
                                    'entry_SMA': entry.get('entry_SMA', np.nan),
                                    'exit_date': pd.to_datetime(exit_dt),
                                    'exit_price': exit_price,
                                    'exit_RSI': to_scalar(exec_row.get('RSI', np.nan)),
                                    'exit_ADX': to_scalar(exec_row.get('ADX', np.nan)),
                                    'exit_SMA': to_scalar(exec_row.get(f'SMA_{sma_period}', np.nan)),
                                    'quantity': entry['quantity'],
                                    'gross_PL': gross_pl,
                                    'entry_commission': entry['entry_commission'],
                                    'exit_commission': exit_comm,
                                    'net_PL': net_pl,
                                    'pnl_pct': (net_pl / (entry['invest_amount'] if entry['invest_amount']>0 else 1)) * 100,
                                    'duration_days': duration_days,
                                    'closed_after_period': True,
                                    'closed_at_run': False,
                                    'note': f'סגירה לאחר התקופה ב-{pd.to_datetime(exit_dt).date()}'
                                })

                                closed_during_extension = True
                                in_position = False
                                entry = None
                                break
                            else:
                                # מחיר יציאה אינו גבוה ממחיר כניסה -> לא סוגרים כאן
                                continue

                    # אם לא נסגר במהלך ההארכה עד יום הריצה — נסגור לפי מחיר יום הריצה (לדוח)
                    if in_position and entry is not None:
                        # ניקח את המחיר האחרון הזמין ב-df (שאמור להיות עד היום)
                        last_price = to_scalar(df['Close'].iloc[-1])
                        if not pd.isna(last_price):
                            last_price = float(last_price)
                            gross_pl = (last_price - entry['entry_price']) * entry['quantity']
                            exit_comm = calc_commission(last_price * entry['quantity'], commission_type, commission_value)
                            net_pl = gross_pl - exit_comm

                            if invest_mode == 'סכום קבוע לכל עסקה':
                                equity += entry['invest_amount'] + net_pl
                            else:
                                equity = equity + net_pl

                            exit_dt = pd.to_datetime(df.index[-1])
                            entry_dt = pd.to_datetime(entry['entry_date'])
                            duration_days = (exit_dt.normalize() - entry_dt.normalize()).days + 1
                            try:
                                duration_days = int(duration_days)
                                if duration_days < 1:
                                    duration_days = 1
                            except Exception:
                                duration_days = np.nan

                            trades.append({
                                'entry_date': pd.to_datetime(entry['entry_date']),
                                'entry_price': entry['entry_price'],
                                'entry_RSI': entry.get('entry_RSI', np.nan),
                                'entry_ADX': entry.get('entry_ADX', np.nan),
                                'entry_SMA': entry.get('entry_SMA', np.nan),
                                'exit_date': pd.to_datetime(exit_dt),
                                'exit_price': last_price,
                                'exit_RSI': to_scalar(df['RSI'].iloc[-1]) if 'RSI' in df.columns else np.nan,
                                'exit_ADX': to_scalar(df['ADX'].iloc[-1]) if 'ADX' in df.columns else np.nan,
                                'exit_SMA': to_scalar(df[f'SMA_{sma_period}'].iloc[-1]) if f'SMA_{sma_period}' in df.columns else np.nan,
                                'quantity': entry['quantity'],
                                'gross_PL': gross_pl,
                                'entry_commission': entry['entry_commission'],
                                'exit_commission': exit_comm,
                                'net_PL': net_pl,
                                'pnl_pct': (net_pl / (entry['invest_amount'] if entry['invest_amount']>0 else 1)) * 100,
                                'duration_days': duration_days,
                                'closed_after_period': False,
                                'closed_at_run': True,
                                'note': 'סגירה לפי יום הריצה (לא נמצאה יציאה תקנית)'
                            })

                            in_position = False
                            entry = None

                # --- Buy&Hold חישוב מבוסס initial_capital ---
                bh_shares = initial_capital / bh_start_price if (bh_start_price and bh_start_price > 0) else 0.0
                bh_gross = (bh_end_price - bh_start_price) * bh_shares if (bh_start_price and bh_end_price) else 0.0
                bh_entry_comm = calc_commission(initial_capital, commission_type, commission_value)
                bh_exit_comm = calc_commission(bh_end_price * bh_shares, commission_type, commission_value) if bh_shares > 0 else 0.0
                bh_net = bh_gross - (bh_entry_comm + bh_exit_comm)
                bh_pct = (bh_net / initial_capital) * 100 if initial_capital > 0 else 0.0

                trades_df = pd.DataFrame(trades)
                total_net = trades_df['net_PL'].sum() if not trades_df.empty else 0.0
                total_gross = trades_df['gross_PL'].sum() if not trades_df.empty else 0.0
                total_comm = (trades_df['entry_commission'].sum() + trades_df['exit_commission'].sum()) if not trades_df.empty else 0.0

                results_by_ticker[ticker] = {
                    'trades_df': trades_df,
                    'total_net': total_net,
                    'total_gross': total_gross,
                    'total_commissions': total_comm,
                    'final_equity': equity,
                    'cumulative_equity': cumulative_equity,
                    'price_df': df  # שמרנו את כל ה־df (כולל ימים אחרי end_date) כדי לצייר סימונים
                }
                bh_by_ticker[ticker] = {
                    'bh_start_price': bh_start_price,
                    'bh_end_price': bh_end_price,
                    'bh_net': bh_net,
                    'bh_pct': bh_pct
                }

        # ---------- הצגת תוצאות ---------- #
        for ticker, res in results_by_ticker.items():
            st.header(f'תוצאות עבור {ticker}')
            trades_df = res['trades_df']

            if trades_df.empty:
                st.info('לא נרשמו פוזיציות במהלך התקופה.')
            else:
                st.subheader('טבלת פוזיציות')
                df_show = trades_df.copy()
                if not df_show.empty:
                    df_show['entry_date'] = pd.to_datetime(df_show['entry_date'])
                    df_show['exit_date'] = pd.to_datetime(df_show['exit_date'])
                st.dataframe(df_show)

                st.markdown('**סיכום ביצועים**')
                total_trades = len(trades_df)
                wins = int((trades_df['net_PL'] > 0).sum()) if total_trades>0 else 0
                win_rate = (wins / total_trades) * 100 if total_trades>0 else 0.0

                final_equity = float(res.get('final_equity', 0.0))
                initial_cap = float(capital)
                total_return_pct = ((final_equity - initial_cap) / initial_cap * 100) if initial_cap != 0 else 0.0

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric('סה״כ רווח נקי', f"{res['total_net']:.2f}")
                c2.metric('סה״כ רווח ברוטו', f"{res['total_gross']:.2f}")
                c3.metric('סה״כ עמלות', f"{res['total_commissions']:.2f}")
                c4.metric('אחוזי ביצוע (Win Rate)', f"{win_rate:.2f}%")
                c5.metric('תשואה כוללת (%)', f"{total_return_pct:.2f}%")

            st.subheader('גרף מחיר — כניסות ויציאות (כולל סגירות לאחר התקופה)')
            price_df = res['price_df'].reset_index().rename(columns={'index': 'Date'})
            price_df['Date'] = pd.to_datetime(price_df['Date'])

            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(price_df['Date'], price_df['Close'], label='מחיר (Adjusted Close)')
                sma_col = f'SMA_{sma_period}'
                if sma_col in price_df.columns:
                    ax.plot(price_df['Date'], price_df[sma_col], label=f'SMA {sma_period}')
                if not trades_df.empty:
                    for _, t in trades_df.iterrows():
                        try:
                            # כניסה - תמיד חץ ירוק
                            ax.scatter(t['entry_date'], t['entry_price'], marker='^', s=80, facecolors='none', edgecolors='green', linewidths=2, zorder=5)
                            # יציאה - תלוי סוג סגירה
                            if t.get('closed_after_period', False):
                                # סגירה לאחר התקופה — סמן 'x' כחול
                                ax.scatter(t['exit_date'], t['exit_price'], marker='x', s=100, color='blue', zorder=6)
                            elif t.get('closed_at_run', False):
                                # סגירה לפי יום הריצה — סמן ריבוע כתום
                                ax.scatter(t['exit_date'], t['exit_price'], marker='s', s=100, color='orange', edgecolors='black', zorder=6)
                            else:
                                # יציאה רגילה - חץ אדום
                                ax.scatter(t['exit_date'], t['exit_price'], marker='v', s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
                        except Exception:
                            pass
                ax.set_title(f'{ticker} — Price with entries/exits')
                ax.legend()
                st.pyplot(fig)

                if enable_png and not trades_df.empty:
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(label='הורד גרף PNG', data=buf, file_name=f'{ticker}_chart.png', mime='image/png')

                if enable_pdf and not trades_df.empty:
                    pdf_bytes = BytesIO()
                    with PdfPages(pdf_bytes) as pdf:
                        pdf.savefig(fig)
                        fig_table, ax_table = plt.subplots(figsize=(12, 6))
                        ax_table.axis('off')
                        table = ax_table.table(cellText=df_show.round(6).values, colLabels=df_show.columns, loc='center')
                        table.auto_set_font_size(False)
                        table.set_fontsize(8)
                        table.scale(1, 1.5)
                        pdf.savefig(fig_table)
                        plt.close(fig_table)
                    pdf_bytes.seek(0)
                    st.download_button(label='הורד דוח PDF', data=pdf_bytes, file_name=f'{ticker}_report.pdf', mime='application/pdf')

            else:
                if ALT_AVAILABLE:
                    base = alt.Chart(price_df).encode(x='Date:T')
                    line = base.mark_line().encode(y='Close:Q', tooltip=['Date:T', 'Close:Q'])
                    charts = [line]
                    if sma_col in price_df.columns:
                        sma_df = price_df[['Date', sma_col]].rename(columns={sma_col: 'SMA'})
                        charts.append(alt.Chart(sma_df).mark_line(strokeDash=[4, 2]).encode(x='Date:T', y='SMA:Q'))
                    chart = alt.layer(*charts).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.line_chart(price_df.set_index('Date')['Close'])

                if not MATPLOTLIB_AVAILABLE:
                    st.info('matplotlib לא מותקן — הורדות PNG/PDF אינן זמינות. התקן matplotlib כדי לאפשר זאת.')

            if enable_excel and not trades_df.empty:
                out = BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                    df_show.to_excel(writer, sheet_name='Trades', index=False)
                    res['price_df'].to_excel(writer, sheet_name='PriceData')
                out.seek(0)
                st.download_button(label='הורד דוח Excel (.xlsx)', data=out, file_name=f'{ticker}_backtest.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            if ticker in bh_by_ticker:
                b = bh_by_ticker[ticker]
                st.subheader('השוואה ל-Buy & Hold')
                cb1, cb2, cb3 = st.columns(3)
                cb1.metric('BH רווח נקי', f"{b['bh_net']:.2f}")
                cb2.metric('BH תשואה (%)', f"{b['bh_pct']:.2f}%")
                cb3.write('---')

        st.success('הרצה הושלמה.')

st.markdown('''
**הערות חשובות:**  
- הקוד ממשיך לבדוק ימים אחרי תאריכי הסריקה שציינת (עד ליום הריצה). אם התנאים ליציאה מתקיימים בכל אחד מהימים הללו — נסגור שם ונחשב רווח/הפסד.  
- אם לא נמצאה יציאה עד יום הריצה — הקוד יחשב וידווח P/L לפי שווי המניה ביום הריצה ויתייג את הסגירה כ׳סגירה לפי יום הריצה׳.  
- בגרף: יציאות לאחר התקופה יסומנו ב־X כחול; יציאות לפי יום הריצה יסומנו בריבוע כתום; יציאות רגילות ישארו חץ אדום.  
- לא שיניתי שום לוגיקה אחרת — רק הוספתי את ההתנהגות לתיעוד/סגירה לאחר התקופה כפי שביקשת.
''')
