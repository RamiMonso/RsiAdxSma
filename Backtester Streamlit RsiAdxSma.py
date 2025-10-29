# streamlit_backtester_rsi_adx.py
# אפליקציית Streamlit לבחינת אסטרטגיות RSI + ADX + SMA
# משולבת חישובי RSI/ADX פנימיים — אין תלות ב-pandas_ta
# הוראות: שמור כקובץ והריץ `streamlit run streamlit_backtester_rsi_adx.py`

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(page_title='Backtester RSI+ADX+SMA', layout='wide')

# ----------------------- פונקציות אינדיקטורים פנימיות -----------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI לפי Wilder (EMA smoothing approximation)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing using ewm with alpha=1/period and adjust=False
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0)
    return rsi

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ADX approximation using Wilder smoothing:
    returns ADX series (0..100)
    """
    # True Range (TR)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Wilder smoothing (approx via ewm with alpha=1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (smooth_plus / atr)
    minus_di = 100 * (smooth_minus / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    adx = adx.fillna(0)
    return adx

# ----------------------- עזרי מערכת -----------------------
def download_price_data(ticker, start_date, end_date, interval, warmup_days):
    start_fetch = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
    # auto_adjust=True כדי לקבל Adjusted Close דרך 'Close'
    df = yf.download(ticker,
                     start=start_fetch.strftime('%Y-%m-%d'),
                     end=(pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                     interval=interval,
                     progress=False,
                     auto_adjust=True)
    if df is None:
        return pd.DataFrame()
    df = df.dropna(how='all')
    return df

def calc_commission(value, commission_type, commission_value):
    if commission_value == 0:
        return 0.0
    if commission_type == 'אחוז מכל עסקה':
        # commission_value is percent (e.g. 0.1 means 0.1%)
        return value * (commission_value / 100.0)
    else:
        return commission_value

# ----------------------- ממשק משתמש -----------------------
st.title('Backtester — בדיקת אסטרטגיות RSI + ADX + SMA')
st.caption('נתוני Adjusted Close מ-Yahoo Finance | חימום אינדיקטורים: 250 ימי מסחר')

with st.sidebar.form('settings'):
    st.header('הגדרות')
    tickers_input = st.text_input('הזן טיקר או רשימת טיקרים (מופרדים בפסיקים)', value='AAPL')
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    timeframe = st.selectbox('בחירת פרק זמן', options=['1d', '1h'], index=0,
                             format_func=lambda x: 'יומי' if x == '1d' else 'שעתי')

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

    st.subheader('כללי מסחר')
    fractional_shares = st.checkbox('לאפשר רכישת שברי מניה?', value=True)
    capital = st.number_input('הון התחלתי', min_value=1.0, value=10000.0, step=100.0)

    invest_mode = st.radio('שיטת השקעה לכל עסקה', options=['סכום קבוע לכל עסקה', 'הון ראשוני + ריבית דריבית'], index=0)
    fixed_invest_amount = st.number_input('סכום להשקעה בכל עסקה (כאשר בחרת סכום קבוע)', min_value=1.0, value=1000.0, step=100.0)

    st.subheader('עמלות')
    commission_type = st.selectbox('סוג עמלה', options=['אחוז מכל עסקה', 'סכום קבוע'], index=0)
    commission_value = st.number_input('ערך עמלה (לדוגמה: 0.1 עבור 0.1% או 2 עבור 2 ש"ח)', min_value=0.0, value=0.1)

    exec_mode = st.radio('מתי לבצע ביצוע (כניסה/יציאה) כאשר התנאי מתקיים', options=['ביום האותן', 'ביום המסחר הבא'], index=0)
    execute_next_day = (exec_mode == 'ביום המסחר הבא')

    close_open_at_run = st.checkbox('לסגור פוזיציה פתוחה לפי מחיר יום ההרצה (אם קיימת)', value=True)

    enable_excel = st.checkbox('אפשרות ייצוא ל-Excel', value=True)
    enable_pdf = st.checkbox('אפשרות ייצוא ל-PDF', value=True)

    warmup_days = 250
    st.caption(f'חימום אינדיקטורים: {warmup_days} ימי מסחר')

    submit = st.form_submit_button('הרץ את הבדיקה')

# ----------------------- לוגיקת הבדיקה -----------------------
if submit:
    if not tickers:
        st.error('לא הוזן טיקר חוקי.')
    else:
        results_all = {}
        bh_comparison = {}

        for ticker in tickers:
            with st.spinner(f'מוריד נתונים עבור {ticker}...'):
                df = download_price_data(ticker, start_date, end_date, timeframe, warmup_days)
                if df.empty:
                    st.error(f'לא נמצאו נתונים עבור {ticker}. בדוק טיקר / טיימפריים.')
                    continue

                # חישוב אינדיקטורים
                df = df.copy()
                if include_rsi:
                    df['RSI'] = compute_rsi(df['Close'], period=rsi_period)
                else:
                    df['RSI'] = np.nan

                if include_adx:
                    df['ADX'] = compute_adx(df['High'], df['Low'], df['Close'], period=adx_period)
                else:
                    df['ADX'] = np.nan

                if include_sma:
                    df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()
                else:
                    df[f'SMA_{sma_period}'] = np.nan

                # טווח סריקה (מתחיל ב-start_date)
                scan_df = df.loc[str(start_date):str(end_date)].copy()
                scan_df = scan_df.dropna(subset=['Close'])
                if scan_df.empty:
                    st.warning(f'לא נמצאו ימים בסקירה עבור {ticker} לאחר חימום.')
                    continue

                trades = []
                in_position = False
                entry = {}
                equity = capital
                cumulative_equity = []
                idx_list = list(scan_df.index)

                # Buy & Hold start and end
                bh_start_price = None
                bh_end_price = None
                if len(idx_list) > 0:
                    bh_start_price = scan_df.iloc[0]['Close']
                    bh_end_price = scan_df.iloc[-1]['Close']

                for i, current_date in enumerate(idx_list):
                    row = scan_df.loc[current_date]
                    price = row['Close']
                    rsi_v = row.get('RSI', np.nan)
                    adx_v = row.get('ADX', np.nan)
                    sma_v = row.get(f'SMA_{sma_period}', np.nan)

                    # ENTRY conditions
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

                    if (not in_position) and entry_cond:
                        exec_i = i + 1 if execute_next_day else i
                        if exec_i < len(idx_list):
                            exec_date = idx_list[exec_i]
                            exec_price = scan_df.loc[exec_date, 'Close']

                            if invest_mode == 'סכום קבוע לכל עסקה':
                                invest_amount = fixed_invest_amount
                            else:
                                invest_amount = equity

                            quantity = invest_amount / exec_price
                            if not fractional_shares:
                                quantity = np.floor(quantity)
                                if quantity <= 0:
                                    # לא ניתן לקנות מניה שלמה במחיר זה
                                    continue

                            entry_comm = calc_commission(invest_amount, commission_type, commission_value)
                            equity -= entry_comm

                            entry = {
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

                    # EXIT logic when in position
                    if in_position:
                        exit_cond = True
                        if include_rsi:
                            exit_cond = exit_cond and (not np.isnan(row.get('RSI', np.nan))) and (row.get('RSI', np.nan) >= rsi_exit_thresh)

                        if exit_cond:
                            exec_i = i + 1 if execute_next_day else i
                            if exec_i < len(idx_list):
                                exit_date = idx_list[exec_i]
                                exit_price = scan_df.loc[exit_date, 'Close']

                                if exit_price > entry['entry_price']:
                                    gross = (exit_price - entry['entry_price']) * entry['quantity']
                                    exit_comm = calc_commission(exit_price * entry['quantity'], commission_type, commission_value)
                                    net = gross - exit_comm

                                    if invest_mode == 'סכום קבוע לכל עסקה':
                                        equity += entry['invest_amount'] + net
                                    else:
                                        equity = equity + net

                                    trades.append({
                                        'entry_date': pd.to_datetime(entry['entry_date']).strftime('%Y-%m-%d %H:%M:%S'),
                                        'entry_price': entry['entry_price'],
                                        'entry_RSI': entry.get('entry_RSI', np.nan),
                                        'entry_ADX': entry.get('entry_ADX', np.nan),
                                        'entry_SMA': entry.get('entry_SMA', np.nan),
                                        'exit_date': pd.to_datetime(exit_date).strftime('%Y-%m-%d %H:%M:%S'),
                                        'exit_price': exit_price,
                                        'exit_RSI': scan_df.loc[exit_date].get('RSI', np.nan),
                                        'exit_ADX': scan_df.loc[exit_date].get('ADX', np.nan),
                                        'exit_SMA': scan_df.loc[exit_date].get(f'SMA_{sma_period}', np.nan),
                                        'quantity': entry['quantity'],
                                        'gross_PL': gross,
                                        'entry_commission': entry['entry_commission'],
                                        'exit_commission': exit_comm,
                                        'net_PL': net,
                                        'pnl_pct': (net / (entry['invest_amount'] if entry['invest_amount'] > 0 else 1)) * 100
                                    })

                                    in_position = False
                                    entry = {}

                    cumulative_equity.append(equity)

                # טיפול בפוזיציה פתוחה בסוף הסקירה
                if in_position:
                    last_date = idx_list[-1]
                    last_price = scan_df.loc[last_date, 'Close']
                    if close_open_at_run:
                        exit_price = last_price
                        gross = (exit_price - entry['entry_price']) * entry['quantity']
                        exit_comm = calc_commission(exit_price * entry['quantity'], commission_type, commission_value)
                        net = gross - exit_comm

                        if invest_mode == 'סכום קבוע לכל עסקה':
                            equity += entry['invest_amount'] + net
                        else:
                            equity = equity + net

                        trades.append({
                            'entry_date': pd.to_datetime(entry['entry_date']).strftime('%Y-%m-%d %H:%M:%S'),
                            'entry_price': entry['entry_price'],
                            'entry_RSI': entry.get('entry_RSI', np.nan),
                            'entry_ADX': entry.get('entry_ADX', np.nan),
                            'entry_SMA': entry.get('entry_SMA', np.nan),
                            'exit_date': pd.to_datetime(last_date).strftime('%Y-%m-%d %H:%M:%S'),
                            'exit_price': exit_price,
                            'exit_RSI': scan_df.loc[last_date].get('RSI', np.nan),
                            'exit_ADX': scan_df.loc[last_date].get('ADX', np.nan),
                            'exit_SMA': scan_df.loc[last_date].get(f'SMA_{sma_period}', np.nan),
                            'quantity': entry['quantity'],
                            'gross_PL': gross,
                            'entry_commission': entry['entry_commission'],
                            'exit_commission': exit_comm,
                            'net_PL': net,
                            'pnl_pct': (net / (entry['invest_amount'] if entry['invest_amount'] > 0 else 1)) * 100,
                            'note': 'סגירה לפי יום הריצה (פוזיציה פתוחה)'
                        })
                        in_position = False

                # Buy & Hold comparison
                if bh_start_price is not None:
                    bh_shares = capital / bh_start_price
                    bh_gross = (bh_end_price - bh_start_price) * bh_shares
                    bh_entry_comm = calc_commission(capital, commission_type, commission_value)
                    bh_exit_comm = calc_commission(bh_end_price * bh_shares, commission_type, commission_value)
                    bh_net = bh_gross - (bh_entry_comm + bh_exit_comm)
                    bh_comparison[ticker] = {
                        'bh_start_price': bh_start_price,
                        'bh_end_price': bh_end_price,
                        'bh_net_PL': bh_net,
                        'bh_pct_net': (bh_net / capital) * 100
                    }

                trades_df = pd.DataFrame(trades)
                total_net = trades_df['net_PL'].sum() if not trades_df.empty else 0.0
                total_gross = trades_df['gross_PL'].sum() if not trades_df.empty else 0.0
                total_comm = (trades_df['entry_commission'].sum() + trades_df['exit_commission'].sum()) if not trades_df.empty else 0.0

                results_all[ticker] = {
                    'trades_df': trades_df,
                    'total_net': total_net,
                    'total_gross': total_gross,
                    'total_commissions': total_comm,
                    'final_equity': equity,
                    'cumulative_equity': cumulative_equity,
                    'price_df': scan_df
                }

        # ----------------------- הצגת תוצאות -----------------------
        for ticker, res in results_all.items():
            st.header(f'תוצאות עבור {ticker}')
            trades_df = res['trades_df']

            if trades_df.empty:
                st.info('לא נרשמו פוזיציות במהלך התקופה.')
            else:
                st.subheader('טבלת פוזיציות')
                st.dataframe(trades_df)

                st.markdown('**סיכום ביצועים**')
                col1, col2, col3 = st.columns(3)
                col1.metric('סה"כ רווח נקי', f"{res['total_net']:.2f}")
                col2.metric('סה"כ רווח ברוטו', f"{res['total_gross']:.2f}")
                col3.metric('סה"כ עמלות ששולמו', f"{res['total_commissions']:.2f}")

            st.subheader('גרף מחיר — כניסות ויציאות')
            price_df = res['price_df']
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(price_df.index, price_df['Close'], label='מחיר (Adjusted Close)')
            if f'SMA_{sma_period}' in price_df.columns:
                ax.plot(price_df.index, price_df[f'SMA_{sma_period}'], label=f'SMA {sma_period}')

            if not trades_df.empty:
                for _, t in trades_df.iterrows():
                    try:
                        ed = pd.to_datetime(t['entry_date'])
                        xd = pd.to_datetime(t['exit_date'])
                        ax.scatter(ed, t['entry_price'], marker='^', s=100)
                        ax.scatter(xd, t['exit_price'], marker='v', s=100)
                    except Exception:
                        pass

            ax.set_title(f'{ticker} — Price with entries/exits')
            ax.legend()
            st.pyplot(fig)

            # הורדות
            if enable_excel and not trades_df.empty:
                towrite = BytesIO()
                with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                    res['price_df'].to_excel(writer, sheet_name='PriceData')
                towrite.seek(0)
                st.download_button(label='הורד דוח Excel (.xlsx)', data=towrite, file_name=f'{ticker}_backtest.xlsx',
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            if enable_pdf and not trades_df.empty:
                pdf_bytes = BytesIO()
                with PdfPages(pdf_bytes) as pdf:
                    pdf.savefig(fig)
                    # טבלה כתרשים matplotlib
                    fig_table, ax_table = plt.subplots(figsize=(12, 6))
                    ax_table.axis('off')
                    tbl = ax_table.table(cellText=trades_df.round(4).values, colLabels=trades_df.columns, loc='center')
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(8)
                    tbl.scale(1, 1.5)
                    pdf.savefig(fig_table)
                    plt.close(fig_table)
                pdf_bytes.seek(0)
                st.download_button(label='הורד דוח PDF', data=pdf_bytes, file_name=f'{ticker}_report.pdf', mime='application/pdf')

            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.download_button(label='הורד גרף PNG', data=buf, file_name=f'{ticker}_chart.png', mime='image/png')

            if ticker in bh_comparison:
                st.subheader('השוואה ל-Buy & Hold')
                b = bh_comparison[ticker]
                c1, c2, c3 = st.columns(3)
                c1.metric('BH רווח נקי', f"{b['bh_net_PL']:.2f}")
                c2.metric('BH תשואה נקו (%)', f"{b['bh_pct_net']:.2f}%")
                c3.write('---')

        st.success('הרצה הושלמה.')

st.markdown('''
### הערות
- הקוד מחשב RSI ו-ADX פנימית (אין צורך ב-pandas_ta).
- נתונים אינטרדיים (שעתי) מה-Yahoo מוגבלים לזמני טווח היסטוריים — אם לא מוצאים נתונים שעתי נסה טווח קצר יותר.
- חימום אינדיקטורים: 250 יום לפני התאריך התחלה לחישוב יציב.
''')
