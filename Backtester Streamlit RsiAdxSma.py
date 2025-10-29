# streamlit_backtester_rsi_adx.py
# Backtester RSI + ADX + SMA — ממשק עברי, חישובים פנימיים, שימוש ב-Adjusted Close
# שמור כקובץ והריץ: streamlit run streamlit_backtester_rsi_adx.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta

# matplotlib optional (לעדיפות הורדת PNG/PDF). אם לא קיים, נזהיר ונשתמש ב-altair/line_chart
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

# ---------------------- פונקציות אינדיקטורים (פנימיות, ללא pandas_ta) ----------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI לפי Wilder (EMA smoothing). מחזיר סדרת RSI (NaN לתחילת הסדרה לפי חימום).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing approximation (ewm with alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    # במקום למלא 0, נשאיר NaN במקומות where avg_loss==0 initially to avoid false signals
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    return rsi

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ADX (Wilder). מחזיר סדרת ADX.
    """
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
def download_data(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str, warmup_days: int = 250) -> pd.DataFrame:
    """
    מוריד נתונים מ-yfinance עם auto_adjust=True (Adjusted Close נתון כ-'Close').
    מוסיף חימום של warmup_days לפני start_date.
    מחזיר DataFrame עם אינדקס datetime.
    """
    start_fetch = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
    end_fetch = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # include end day
    df = yf.download(ticker, start=start_fetch.strftime('%Y-%m-%d'), end=end_fetch.strftime('%Y-%m-%d'),
                     interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    # וודא שיש עמודת Close (auto_adjust=True בדרך כלל ממפה ל'Close'); אם אין — נסו 'Adj Close'
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            # נסיון להתאים שמות אחרים
            cols = [c.lower() for c in df.columns]
            if 'close' in cols:
                df.columns = [c.capitalize() if c.lower()=='close' else c for c in df.columns]
            else:
                return pd.DataFrame()
    # וודא אינדקס מסוג DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
    return df

def calc_commission(value: float, commission_type: str, commission_value: float) -> float:
    """
    מחשב עמלה: commission_type = 'אחוז' או 'סכום'
    commission_value: אם אחוז — הזן ערך כמו 0.1 עבור 0.1% (ולא 0.001).
    """
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
    st.caption(f'חימום אינדיקטורים: {warmup_days} ימי מסחר (מוסתרים מהמשתמש אך משמשים לחישוב מדויק)')

    submit = st.form_submit_button('הרץ את הבדיקה')

# אם matplotlib חסר — ננטרל אפשרויות שלא יעבדו
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

                # חישוב אינדיקטורים
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

                # טווח לסריקה — החל מ start_date ועד end_date (כלול)
                try:
                    scan_df = df.loc[str(start_date):str(end_date)].copy()
                except Exception:
                    scan_df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))].copy()

                if scan_df.empty:
                    st.warning(f'לא נותרו נתונים בסקירה עבור {ticker} לאחר החימום ו/או לפי טווח התאריכים.')
                    continue

                # המרה לרשימת אינדקס לניווט לפי מיקום
                idx = list(scan_df.index)
                trades = []
                in_position = False
                entry = None
                equity = capital
                cumulative_equity = []

                # BUY & HOLD baseline
                bh_start_price = scan_df['Close'].iloc[0]
                bh_end_price = scan_df['Close'].iloc[-1]

                for i, current_date in enumerate(idx):
                    row = scan_df.loc[current_date]
                    price = row['Close']
                    rsi_v = row.get('RSI', np.nan)
                    adx_v = row.get('ADX', np.nan)
                    sma_v = row.get(f'SMA_{sma_period}', np.nan)

                    # תנאי כניסה
                    entry_cond = True
                    if include_rsi:
                        entry_cond = entry_cond and (not np.isnan(rsi_v)) and (rsi_v <= rsi_entry_thresh)
                    if include_adx:
                        entry_cond = entry_cond and (not np.isnan(adx_v)) and (adx_v <= adx_thresh)
                    if include_sma:
                        entry_cond = entry_cond and (not np.isnan(sma_v)) and (price > sma_v)
                    if check_not_strong_down and include_sma:
                        prev_idx = i - 1
                        if prev_idx >= 0:
                            prev_sma = scan_df.iloc[prev_idx].get(f'SMA_{sma_period}', np.nan)
                            if not np.isnan(prev_sma) and not np.isnan(sma_v):
                                entry_cond = entry_cond and (sma_v >= prev_sma)

                    # כניסה
                    if (not in_position) and entry_cond:
                        exec_i = i+1 if execute_next_day else i
                        if exec_i < len(idx):
                            exec_date = idx[exec_i]
                            exec_price = scan_df.loc[exec_date, 'Close']

                            # קביעת סכום השקעה וכמות
                            if invest_mode == 'סכום קבוע לכל עסקה':
                                invest_amount = fixed_invest_amount
                            else:
                                invest_amount = equity if equity > 0 else fixed_invest_amount

                            quantity = invest_amount / exec_price
                            if not fractional_shares:
                                quantity = float(np.floor(quantity))
                                if quantity <= 0:
                                    # לא ניתן לקנות מניה שלמה בסכום זה
                                    continue

                            entry_comm = calc_commission(invest_amount, commission_type, commission_value)
                            # נחסר עמלת כניסה מההון (כפי שביקשת)
                            equity -= entry_comm

                            entry = {
                                'entry_idx': exec_i,
                                'entry_date': exec_date,
                                'entry_price': exec_price,
                                'entry_RSI': scan_df.loc[exec_date].get('RSI', np.nan),
                                'entry_ADX': scan_df.loc[exec_date].get('ADX', np.nan),
                                'entry_SMA': scan_df.loc[exec_date].get(f'SMA_{sma_period}', np.nan),
                                'quantity': quantity,
                                'invest_amount': invest_amount,
                                'entry_commission': entry_comm
                            }
                            in_position = True

                    # יציאה
                    if in_position:
                        exit_cond = True
                        if include_rsi:
                            # בודקים את ה-RSI של השורה הנוכחית (התניה לפי דרישתך)
                            exit_cond = exit_cond and (not np.isnan(row.get('RSI', np.nan))) and (row.get('RSI', np.nan) >= rsi_exit_thresh)

                        if exit_cond:
                            exec_i = i+1 if execute_next_day else i
                            if exec_i < len(idx):
                                exit_date = idx[exec_i]
                                exit_price = scan_df.loc[exit_date, 'Close']

                                # תנאי נוסף: מחיר יציאה גבוה ממחיר כניסה
                                if exit_price > entry['entry_price']:
                                    gross_pl = (exit_price - entry['entry_price']) * entry['quantity']
                                    exit_comm = calc_commission(exit_price * entry['quantity'], commission_type, commission_value)
                                    net_pl = gross_pl - exit_comm

                                    # עדכון equity לפי מצב מימון
                                    if invest_mode == 'סכום קבוע לכל עסקה':
                                        equity += entry['invest_amount'] + net_pl
                                    else:
                                        equity = equity + net_pl

                                    trades.append({
                                        'entry_date': pd.to_datetime(entry['entry_date']),
                                        'entry_price': entry['entry_price'],
                                        'entry_RSI': entry.get('entry_RSI', np.nan),
                                        'entry_ADX': entry.get('entry_ADX', np.nan),
                                        'entry_SMA': entry.get('entry_SMA', np.nan),
                                        'exit_date': pd.to_datetime(exit_date),
                                        'exit_price': exit_price,
                                        'exit_RSI': scan_df.loc[exit_date].get('RSI', np.nan),
                                        'exit_ADX': scan_df.loc[exit_date].get('ADX', np.nan),
                                        'exit_SMA': scan_df.loc[exit_date].get(f'SMA_{sma_period}', np.nan),
                                        'quantity': entry['quantity'],
                                        'gross_PL': gross_pl,
                                        'entry_commission': entry['entry_commission'],
                                        'exit_commission': exit_comm,
                                        'net_PL': net_pl,
                                        'pnl_pct': (net_pl / (entry['invest_amount'] if entry['invest_amount']>0 else 1)) * 100
                                    })

                                    in_position = False
                                    entry = None

                    cumulative_equity.append(equity)

                # טיפול בפוזיציה פתוחה בתום התקופה
                if in_position and entry is not None:
                    last_idx = idx[-1]
                    last_price = scan_df.loc[last_idx, 'Close']
                    if close_open_at_run:
                        gross_pl = (last_price - entry['entry_price']) * entry['quantity']
                        exit_comm = calc_commission(last_price * entry['quantity'], commission_type, commission_value)
                        net_pl = gross_pl - exit_comm
                        if invest_mode == 'סכום קבוע לכל עסקה':
                            equity += entry['invest_amount'] + net_pl
                        else:
                            equity = equity + net_pl

                        trades.append({
                            'entry_date': pd.to_datetime(entry['entry_date']),
                            'entry_price': entry['entry_price'],
                            'entry_RSI': entry.get('entry_RSI', np.nan),
                            'entry_ADX': entry.get('entry_ADX', np.nan),
                            'entry_SMA': entry.get('entry_SMA', np.nan),
                            'exit_date': pd.to_datetime(last_idx),
                            'exit_price': last_price,
                            'exit_RSI': scan_df.loc[last_idx].get('RSI', np.nan),
                            'exit_ADX': scan_df.loc[last_idx].get('ADX', np.nan),
                            'exit_SMA': scan_df.loc[last_idx].get(f'SMA_{sma_period}', np.nan),
                            'quantity': entry['quantity'],
                            'gross_PL': gross_pl,
                            'entry_commission': entry['entry_commission'],
                            'exit_commission': exit_comm,
                            'net_PL': net_pl,
                            'pnl_pct': (net_pl / (entry['invest_amount'] if entry['invest_amount']>0 else 1)) * 100,
                            'note': 'סגירה לפי יום הריצה (פוזיציה פתוחה)'
                        })
                        in_position = False
                        entry = None

                # חישוב BUY & HOLD (אותו הון התחלתי, שברי מניה מותרים)
                bh_shares = capital / bh_start_price if bh_start_price and bh_start_price>0 else 0
                bh_gross = (bh_end_price - bh_start_price) * bh_shares if bh_start_price and bh_end_price else 0
                bh_entry_comm = calc_commission(capital, commission_type, commission_value)
                bh_exit_comm = calc_commission(bh_end_price * bh_shares, commission_type, commission_value) if bh_shares>0 else 0
                bh_net = bh_gross - (bh_entry_comm + bh_exit_comm)
                bh_pct = (bh_net / capital) * 100 if capital>0 else 0

                trades_df = pd.DataFrame(trades)
                total_net = trades_df['net_PL'].sum() if not trades_df.empty else 0.0
                total_gross = trades_df['gross_PL'].sum() if not trades_df.empty else 0.0
                total_comm = (trades_df['entry_commission'].sum() + trades_df['exit_commission'].sum()) if not trades_df.empty else 0.0

                overall_results[ticker] = {
                    'trades_df': trades_df,
                    'total_net': total_net,
                    'total_gross': total_gross,
                    'total_commissions': total_comm,
                    'final_equity': equity,
                    'cumulative_equity': cumulative_equity,
                    'price_df': scan_df
                }
                bh_results[ticker] = {
                    'bh_start_price': bh_start_price,
                    'bh_end_price': bh_end_price,
                    'bh_net': bh_net,
                    'bh_pct': bh_pct
                }

        # ---------- הצגת תוצאות ---------- #
        for ticker, res in overall_results.items():
            st.header(f'תוצאות עבור {ticker}')
            trades_df = res['trades_df']

            if trades_df.empty:
                st.info('לא נרשמו פוזיציות במהלך התקופה.')
            else:
                st.subheader('טבלת פוזיציות')
                # עיגון תאריכים להצגה נוחה
                df_show = trades_df.copy()
                df_show['entry_date'] = pd.to_datetime(df_show['entry_date'])
                df_show['exit_date'] = pd.to_datetime(df_show['exit_date'])
                st.dataframe(df_show)

                st.markdown('**סיכום ביצועים**')
                c1, c2, c3 = st.columns(3)
                c1.metric('סה״כ רווח נקי', f"{res['total_net']:.2f}")
                c2.metric('סה״כ רווח ברוטו', f"{res['total_gross']:.2f}")
                c3.metric('סה״כ עמלות', f"{res['total_commissions']:.2f}")

            # גרף מחיר + SMA + סימוני כניסה/יציאה
            st.subheader('גרף מחיר — כניסות ויציאות')
            price_df = res['price_df'].reset_index().rename(columns={'index':'Date'})
            price_df['Date'] = pd.to_datetime(price_df['Date'])

            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(12,5))
                ax.plot(price_df['Date'], price_df['Close'], label='מחיר (Adjusted Close)')
                sma_col = f'SMA_{sma_period}'
                if sma_col in price_df.columns:
                    ax.plot(price_df['Date'], price_df[sma_col], label=f'SMA {sma_period}')
                if not trades_df.empty:
                    for _, t in trades_df.iterrows():
                        try:
                            ax.scatter(t['entry_date'], t['entry_price'], marker='^', s=80, c='green', zorder=5)
                            ax.scatter(t['exit_date'], t['exit_price'], marker='v', s=80, c='red', zorder=5)
                        except Exception:
                            pass
                ax.set_title(f'{ticker} — Price with entries/exits')
                ax.legend()
                st.pyplot(fig)

                # הורדת PNG
                if enable_png and not trades_df.empty:
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(label='הורד גרף PNG', data=buf, file_name=f'{ticker}_chart.png', mime='image/png')

                # PDF: גרף + טבלה
                if enable_pdf and not trades_df.empty:
                    pdf_bytes = BytesIO()
                    with PdfPages(pdf_bytes) as pdf:
                        pdf.savefig(fig)
                        fig_table, ax_table = plt.subplots(figsize=(12,6))
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
                # altair fallback
                if ALT_AVAILABLE:
                    base = alt.Chart(price_df).encode(x='Date:T')
                    line = base.mark_line().encode(y='Close:Q', tooltip=['Date:T','Close:Q'])
                    charts = [line]
                    sma_col = f'SMA_{sma_period}'
                    if sma_col in price_df.columns:
                        sma_df = price_df[['Date', sma_col]].rename(columns={sma_col:'SMA'})
                        charts.append(alt.Chart(sma_df).mark_line(strokeDash=[4,2]).encode(x='Date:T', y='SMA:Q'))
                    chart = alt.layer(*charts).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.line_chart(price_df.set_index('Date')['Close'])

                if not MATPLOTLIB_AVAILABLE:
                    st.info('matplotlib לא מותקן — הורדות PNG/PDF אינן זמינות. התקן matplotlib כדי להפעיל אפשרות זו.')

            # Excel export
            if enable_excel and not trades_df.empty:
                out = BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                    df_show.to_excel(writer, sheet_name='Trades', index=False)
                    res['price_df'].to_excel(writer, sheet_name='PriceData')
                out.seek(0)
                st.download_button(label='הורד דוח Excel (.xlsx)', data=out, file_name=f'{ticker}_backtest.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            # Buy&Hold comparison
            if ticker in bh_results:
                b = bh_results[ticker]
                st.subheader('השוואה ל-Buy & Hold')
                cb1, cb2, cb3 = st.columns(3)
                cb1.metric('BH רווח נקי', f"{b['bh_net']:.2f}")
                cb2.metric('BH תשואה (%)', f"{b['bh_pct']:.2f}%")
                cb3.write('---')

        st.success('הרצה הושלמה.')

st.markdown(''' 
**הערות חשובות:**  
- כל המחירים והמחשובים מבוססים על Adjusted Close (auto_adjust=True ב-yfinance).  
- חימום אינדיקטורים: 250 יום לפני תאריך ההתחלה (נדרש ליציבות חישובית).  
- אם תרצה להפעיל גרסאות מתקדמות יותר של דוחות/מדדים (Sharpe, drawdown, etc.) — אוכל להוסיף.  
''')
