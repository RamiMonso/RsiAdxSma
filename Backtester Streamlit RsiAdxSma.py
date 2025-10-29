# streamlit_backtester_rsi_adx.py
# אפליקציית Streamlit לבחינת אסטרטגיות כניסה/יציאה על בסיס RSI, ADX ו-SMA
# שפת ממשק: עברית
# הנחיות: שמור כקובץ app.py או streamlit_backtester_rsi_adx.py והרץ: `streamlit run streamlit_backtester_rsi_adx.py`

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(page_title='Backtester RSI+ADX+SMA', layout='wide')

# ---------- עיצוב כותרת ----------
st.title('Backtester — בדיקת אסטרטגיות RSI + ADX + SMA')
st.caption('נתוני Adjusted Close מ-Yahoo Finance | חימום אינדיקטורים: 250 ימי מסחר')

# ---------- סיידבר: הגדרות משתמש ----------
with st.sidebar.form('settings'):
    st.header('הגדרות סקריפט')

    tickers_input = st.text_input('הזן טיקר או רשימת טיקרים (מופרדים בפסיקים)', value='AAPL')
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    timeframe = st.selectbox('בחירת פרק זמן', options=['1d', '1h'], index=0, format_func=lambda x: 'יומי' if x=='1d' else 'שעתי')

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('תאריך התחלה', value=(datetime.today() - timedelta(days=365)).date())
    with col2:
        end_date = st.date_input('תאריך סוף', value=datetime.today().date())

    # אינדיקטורים
    st.subheader('הגדרות אינדיקטורים')
    rsi_period = st.number_input('RSI — מספר ימים', min_value=2, max_value=200, value=14)
    adx_period = st.number_input('ADX — מספר ימים', min_value=2, max_value=200, value=14)
    sma_period = st.number_input('SMA — מספר ימים', min_value=2, max_value=500, value=50)

    rsi_entry_thresh = st.number_input('סף RSI לכניסה (≤)', min_value=1, max_value=100, value=30)
    rsi_exit_thresh = st.number_input('סף RSI ליציאה (≥)', min_value=1, max_value=100, value=50)
    adx_thresh = st.number_input('סף ADX לכניסה (≤)', min_value=1, max_value=100, value=25)

    # אילוצים לכניסה/יציאה
    include_rsi = st.checkbox('להשתמש ב‑RSI?', value=True)
    include_adx = st.checkbox('להשתמש ב‑ADX?', value=True)
    include_sma = st.checkbox('להשתמש ב‑SMA?', value=True)
    check_not_strong_down = st.checkbox('לבדוק שהטרנד לא חזק מטה (SMA לא יורד)?', value=True)

    st.subheader('כללי מסחר & מימון')
    fractional_shares = st.checkbox('לאפשר רכישת שברי מניה?', value=True)
    capital = st.number_input('הון התחלתי (ברירת מחדל)', min_value=1.0, value=10000.0, step=100.0)

    invest_mode = st.radio('שיטת השקעה לכל עסקה', options=['סכום קבוע לכל עסקה', 'הון ראשוני + ריבית דריבית'], index=0)
    fixed_invest_amount = st.number_input('סכום להשקעה בכל עסקה (כאשר בחרת סכום קבוע)', min_value=1.0, value=1000.0, step=100.0)

    # עמלות
    st.subheader('עמלות')
    commission_type = st.selectbox('סוג עמלה', options=['אחוז מכל עסקה', 'סכום קבוע'], index=0)
    commission_value = st.number_input('ערך עמלה (לדוגמה: 0.1 עבור 0.1% או 2 עבור 2 ש"ח)', min_value=0.0, value=0.1)

    # בחירת ביצוע - כניסה/יציאה ביום אותן או מחר
    exec_mode = st.radio('מתי לבצע ביצוע (כניסה/יציאה) כאשר התנאי מתקיים', options=['ביום האותן', 'ביום המסחר הבא'], index=0)
    execute_next_day = (exec_mode == 'ביום המסחר הבא')

    # טיפול בפוזיציה פתוחה בסיום
    close_open_at_run = st.checkbox('לסגור פוזיציה פתוחה לפי מחיר יום ההרצה (אם קיימת)', value=True)

    # אופציות יצוא
    st.subheader('ייצוא & ויזואליזציה')
    enable_excel = st.checkbox('אפשרות ייצוא ל‑Excel', value=True)
    enable_pdf = st.checkbox('אפשרות ייצוא ל‑PDF (טבלת דוח + גרף)', value=True)

    # חימום אינדיקטורים
    warmup_days = 250  # דרישתך — חימום קבוע
    st.caption(f'חימום אינדיקטורים: {warmup_days} ימי מסחר (מוסתר למשתמש)')

    submit = st.form_submit_button('הרץ את הבדיקה')

# ---------- פונקציות עזר ----------

def download_price_data(ticker, start_date, end_date, interval, warmup_days):
    # הורדה תכלול חימום: נוריד מהתאריך start_date פחות warmup_days
    start_fetch = pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
    # yfinance: להשתמש ב-auto_adjust=True כדי לקבל Close מותאם (Adjusted Close)
    df = yf.download(ticker, start=start_fetch.strftime('%Y-%m-%d'), end=(pd.to_datetime(end_date)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'), interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return df
    df = df.dropna(how='all')
    return df


def compute_indicators(df, rsi_period, adx_period, sma_period):
    df = df.copy()
    # RSI
    try:
        df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
    except Exception:
        df['RSI'] = df['Close'].rolling(rsi_period).apply(lambda x: np.nan)

    # ADX — pandas_ta מחזירה df עם טורים DI+/DI-/ADX
    try:
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=adx_period)
        # העמודה ADX_{len}
        adx_col = f'ADX_{adx_period}'
        if adx_col in adx_df.columns:
            df['ADX'] = adx_df[adx_col]
        else:
            df['ADX'] = adx_df['ADX']
    except Exception:
        df['ADX'] = np.nan

    # SMA על ה-Close (Adjusted Close ניתן כ-Close כאשר auto_adjust=True)
    df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()

    return df


def calc_commission(value, commission_type, commission_value):
    if commission_value == 0:
        return 0.0
    if commission_type == 'אחוז מכל עסקה':
        # commission_value given as percentage (e.g. 0.1 means 0.1%)
        return value * (commission_value / 100.0)
    else:
        # סכום קבוע
        return commission_value


# ---------- לוגיקת בחינה מרכזית ----------

if submit:
    # בדיקה מהירה על קלט
    if not tickers:
        st.error('אין טיקרים. אנא הזן טיקר/ים חוקיים.')
    else:
        results_all = {}
        bh_comparison = {}

        for ticker in tickers:
            with st.spinner(f'מוריד נתונים ועורך חישובים עבור {ticker} ...'):
                df = download_price_data(ticker, start_date, end_date, timeframe, warmup_days)
                if df.empty:
                    st.error(f'לא נמצאו נתונים עבור {ticker} בפרק זמן זה או השילוב של טיימפריים אינו נתמך על ידי Yahoo.')
                    continue

                df = compute_indicators(df, rsi_period, adx_period, sma_period)

                # הגדרת החלון שבו נרצה לסרוק — מ־start_date ועד end_date
                scan_df = df.loc[str(start_date):str(end_date)].copy()
                scan_df = scan_df.dropna(subset=['Close'])

                # מוודאים שיש מספיק נתונים (לאחר חימום)
                if scan_df.empty:
                    st.warning(f'לא נמצאו ימים לסריקה עבור {ticker} לאחר חימום.')
                    continue

                trades = []
                in_position = False
                entry = {}

                # Variables for capital management
                equity = capital
                cumulative_equity = []

                # Prepare for Buy & Hold
                # invest capital at first available day in scan
                bh_start_price = None
                bh_end_price = None

                # get index list for date navigation
                idx_list = list(scan_df.index)

                # find buy&hold start
                for d in idx_list:
                    bh_start_price = scan_df.loc[d, 'Close']
                    break

                # Loop through each day in scan_df
                for i, current_date in enumerate(idx_list):
                    row = scan_df.loc[current_date]

                    # read indicator values
                    price = row['Close']
                    rsi_v = row.get('RSI', np.nan)
                    adx_v = row.get('ADX', np.nan)
                    sma_v = row.get(f'SMA_{sma_period}', np.nan)

                    # ENTRY logic
                    entry_cond = True
                    if include_rsi:
                        entry_cond = entry_cond and (not np.isnan(rsi_v)) and (rsi_v <= rsi_entry_thresh)
                    if include_adx:
                        entry_cond = entry_cond and (not np.isnan(adx_v)) and (adx_v <= adx_thresh)
                    if include_sma:
                        entry_cond = entry_cond and (not np.isnan(sma_v)) and (price > sma_v)

                    if check_not_strong_down and include_sma:
                        # בדיקה שה-SMA לא יורד לעומת יום קודם (כללי פשוט)
                        prev_idx = i-1
                        if prev_idx >= 0:
                            prev_sma = scan_df.iloc[prev_idx].get(f'SMA_{sma_period}', np.nan)
                            if not np.isnan(prev_sma) and not np.isnan(sma_v):
                                entry_cond = entry_cond and (sma_v >= prev_sma)

                    # אם מתרחש תנאי כניסה ואין פוזיציה פתוחה
                    if (not in_position) and entry_cond:
                        # בחר יום ביצוע — היום או הבא
                        exec_i = i+1 if execute_next_day else i
                        if exec_i < len(idx_list):
                            exec_date = idx_list[exec_i]
                            exec_price = scan_df.loc[exec_date, 'Close']

                            # קביעת כמות השקעה
                            if invest_mode == 'סכום קבוע לכל עסקה':
                                invest_amount = fixed_invest_amount
                            else:
                                # compound: invest all equity
                                invest_amount = equity

                            quantity = invest_amount / exec_price
                            if not fractional_shares:
                                quantity = np.floor(quantity)
                                if quantity <= 0:
                                    # לא יכול לקנות מניה שלמה
                                    continue

                            entry_comm = calc_commission(invest_amount, commission_type, commission_value)
                            equity -= entry_comm  # יורד מההון

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

                    # EXIT logic — רק אם יש פוזיציה
                    if in_position:
                        exit_cond = True
                        # RSI condition
                        if include_rsi:
                            exit_cond = exit_cond and (not np.isnan(row.get('RSI', np.nan))) and (row.get('RSI', np.nan) >= rsi_exit_thresh)
                        # second condition: price exit > entry price (כפי שהוגדר)
                        # We'll evaluate on exec day price (today or next day depending on setting)

                        if exit_cond:
                            exec_i = i+1 if execute_next_day else i
                            if exec_i < len(idx_list):
                                exit_date = idx_list[exec_i]
                                exit_price = scan_df.loc[exit_date, 'Close']

                                # require price higher than entry price
                                if exit_price > entry['entry_price']:
                                    # סגירה
                                    gross = (exit_price - entry['entry_price']) * entry['quantity']
                                    exit_comm = calc_commission(exit_price * entry['quantity'], commission_type, commission_value)
                                    net = gross - exit_comm

                                    # עדכון equity בהתאם למצב בחישוב
                                    if invest_mode == 'סכום קבוע לכל עסקה':
                                        # לא משקיעים עוד מההון אלא משאירים הון כפי שהיה
                                        equity += entry['invest_amount'] + net
                                    else:
                                        # compound: נכנסנו עם equity והמשכנו עם כל הסכום עד עכשיו
                                        equity = equity + net

                                    trades.append({
                                        'entry_date': entry['entry_date'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(entry['entry_date'], pd.Timestamp) else str(entry['entry_date']),
                                        'entry_price': entry['entry_price'],
                                        'entry_RSI': entry.get('entry_RSI', np.nan),
                                        'entry_ADX': entry.get('entry_ADX', np.nan),
                                        'entry_SMA': entry.get('entry_SMA', np.nan),
                                        'exit_date': exit_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(exit_date, pd.Timestamp) else str(exit_date),
                                        'exit_price': exit_price,
                                        'exit_RSI': scan_df.loc[exit_date].get('RSI', np.nan),
                                        'exit_ADX': scan_df.loc[exit_date].get('ADX', np.nan),
                                        'exit_SMA': scan_df.loc[exit_date].get(f'SMA_{sma_period}', np.nan),
                                        'quantity': entry['quantity'],
                                        'gross_PL': gross,
                                        'entry_commission': entry['entry_commission'],
                                        'exit_commission': exit_comm,
                                        'net_PL': net,
                                        'pnl_pct': (net / (entry['invest_amount'] if entry['invest_amount']>0 else 1)) * 100
                                    })

                                    in_position = False
                                    entry = {}

                    # update cumulative equity tracking
                    cumulative_equity.append(equity)

                # אם נשארת פוזיציה פתוחה בסיום הסקירה
                if in_position:
                    last_date = idx_list[-1]
                    last_price = scan_df.loc[last_date, 'Close']
                    # אם המשתמש בחר לסגור לפי יום הריצה — נעשה זאת
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
                            'entry_date': entry['entry_date'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(entry['entry_date'], pd.Timestamp) else str(entry['entry_date']),
                            'entry_price': entry['entry_price'],
                            'entry_RSI': entry.get('entry_RSI', np.nan),
                            'entry_ADX': entry.get('entry_ADX', np.nan),
                            'entry_SMA': entry.get('entry_SMA', np.nan),
                            'exit_date': last_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_date, pd.Timestamp) else str(last_date),
                            'exit_price': exit_price,
                            'exit_RSI': scan_df.loc[last_date].get('RSI', np.nan),
                            'exit_ADX': scan_df.loc[last_date].get('ADX', np.nan),
                            'exit_SMA': scan_df.loc[last_date].get(f'SMA_{sma_period}', np.nan),
                            'quantity': entry['quantity'],
                            'gross_PL': gross,
                            'entry_commission': entry['entry_commission'],
                            'exit_commission': exit_comm,
                            'net_PL': net,
                            'pnl_pct': (net / (entry['invest_amount'] if entry['invest_amount']>0 else 1)) * 100,
                            'note': 'סגירה לפי יום הריצה (פוזיציה פתוחה בתום התקופה)'
                        })
                        in_position = False

                # Buy & Hold comparison
                if bh_start_price is not None:
                    # find last available price in scan_df
                    for d in reversed(idx_list):
                        bh_end_price = scan_df.loc[d, 'Close']
                        break

                    # compute shares bought with initial capital
                    bh_shares = capital / bh_start_price
                    bh_gross = (bh_end_price - bh_start_price) * bh_shares
                    # commissions: apply entry and exit
                    bh_entry_comm = calc_commission(capital, commission_type, commission_value)
                    bh_exit_comm = calc_commission(bh_end_price * bh_shares, commission_type, commission_value)
                    bh_net = bh_gross - (bh_entry_comm + bh_exit_comm)

                    bh_comparison[ticker] = {
                        'bh_start_price': bh_start_price,
                        'bh_end_price': bh_end_price,
                        'bh_gross_PL': bh_gross,
                        'bh_net_PL': bh_net,
                        'bh_pct_net': (bh_net / capital) * 100
                    }

                # יבוא תוצאות
                trades_df = pd.DataFrame(trades)
                total_net = trades_df['net_PL'].sum() if not trades_df.empty else 0.0
                total_gross = trades_df['gross_PL'].sum() if not trades_df.empty else 0.0
                total_comm = trades_df['entry_commission'].sum() + trades_df['exit_commission'].sum() if not trades_df.empty else 0.0

                results_all[ticker] = {
                    'trades_df': trades_df,
                    'total_net': total_net,
                    'total_gross': total_gross,
                    'total_commissions': total_comm,
                    'final_equity': equity,
                    'cumulative_equity': cumulative_equity,
                    'price_df': scan_df
                }

        # ---------- תצוגת תוצאות במסך ----------
        for ticker, res in results_all.items():
            st.header(f'תוצאות עבור {ticker}')
            trades_df = res['trades_df']

            if trades_df.empty:
                st.info('לא נרשמו פוזיציות במהלך התקופה.')
            else:
                # טבלה מפורטת
                st.subheader('טבלת פוזיציות')
                st.dataframe(trades_df)

                # סיכום מספרי
                st.markdown('**סיכום ביצועים**')
                col1, col2, col3 = st.columns(3)
                col1.metric('סה"כ רווח נקי', f"{res['total_net']:.2f}")
                col2.metric('סה"כ רווח ברוטו', f"{res['total_gross']:.2f}")
                col3.metric('סה"כ עמלות ששולמו', f"{res['total_commissions']:.2f}")

            # גרף מחיר עם סימוני כניסה/יציאה
            st.subheader('גרף מחיר — כניסות ויציאות')
            price_df = res['price_df']
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(price_df.index, price_df['Close'], label='מחיר (Adjusted Close)')
            if f'SMA_{sma_period}' in price_df.columns:
                ax.plot(price_df.index, price_df[f'SMA_{sma_period}'], label=f'SMA {sma_period}')

            # סימוני כניסה/יציאה
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

            # כפתורי הורדה: Excel / PDF / PNG
            # Excel
            if enable_excel and not trades_df.empty:
                towrite = BytesIO()
                with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                    # Price sheet
                    res['price_df'].to_excel(writer, sheet_name='PriceData')
                towrite.seek(0)
                st.download_button(label='הורד דוח Excel (.xlsx)', data=towrite, file_name=f'{ticker}_backtest.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            # PDF export — נשמור גרף + טבלה בדף PDF
            if enable_pdf and not trades_df.empty:
                pdf_bytes = BytesIO()
                with PdfPages(pdf_bytes) as pdf:
                    # גרף
                    pdf.savefig(fig)

                    # טבלה כשרטוט matplotlib
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

            # הורדת גרף PNG
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.download_button(label='הורד גרף PNG', data=buf, file_name=f'{ticker}_chart.png', mime='image/png')

            # השוואה ל-BUY & HOLD
            if ticker in bh_comparison:
                st.subheader('השוואה ל‑Buy & Hold')
                b = bh_comparison[ticker]
                col1, col2, col3 = st.columns(3)
                col1.metric('BH רווח נקי', f"{b['bh_net_PL']:.2f}")
                col2.metric('BH תשואה נקו (%)', f"{b['bh_pct_net']:.2f}%")
                col3.write('---')

        st.success('הרצה הושלמה.')

# ---------- הסברים נוספים למשתמש (מתחת לאפליקציה) ----------
st.markdown('''
### הערות חשובות על השימוש
- הנתונים נמשכים מ‑Yahoo Finance (בזמני שוק שונים ייתכנו מגבלות היסטוריות).
- כל החישובים מבוססי מחיר **Adjusted Close** (יושב על Close כאשר auto_adjust=True).
- החימום של 250 ימים משמש לחישוב אינדיקטורים — פוזיציות מדווחות רק בטווח שבחרת.
- תמיכה בזמני מסגרת: יומי (1d) ושעתי (1h) בלבד.
''')
