# backtester_streamlit.py
# Streamlit backtester with extended indicators + grid search using `ta` library
# Updated: user-selectable indicator participation in strategy (entry/exit),
# MACD/Stochastic/ATR participation, thresholds, and Grid Search.
# Fixed: proper multiline strings at the end to avoid SyntaxError.
# Dependencies: pip install streamlit yfinance pandas numpy plotly reportlab ta

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

st.set_page_config(page_title="Indicator Backtester — Extended", layout="wide")

# --------------------- Indicator helpers (using ta) ---------------------

def add_indicators(df, params):
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']

    # RSI
    rsi_period = int(params.get('rsi_period', 14))
    rsi_indicator = RSIIndicator(close=close, window=rsi_period)
    df['RSI'] = rsi_indicator.rsi()

    # ADX
    adx_period = int(params.get('adx_period', rsi_period))
    adx_indicator = ADXIndicator(high=high, low=low, close=close, window=adx_period)
    df['ADX'] = adx_indicator.adx()

    # SMA/EMA
    sma_period = int(params.get('sma_period', 50))
    df['SMA'] = close.rolling(sma_period).mean()
    df['EMA'] = EMAIndicator(close=close, window=sma_period).ema_indicator()

    # MACD
    macd = MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()

    # Stochastic (fast %K)
    stoch_k_period = int(params.get('stoch_k_period', 14))
    stoch = StochasticOscillator(high=high, low=low, close=close, window=stoch_k_period, smooth_window=3)
    df['STOCH_K'] = stoch.stoch()
    df['STOCH_D'] = stoch.stoch_signal()

    # ATR
    atr_period = int(params.get('atr_period', 14))
    atr = AverageTrueRange(high=high, low=low, close=close, window=atr_period)
    df['ATR'] = atr.average_true_range()

    return df

# --------------------- Backtest logic ---------------------

def single_run_backtest(df, params):
    """Run the strategy once on df given params.
    The 'params' dict controls which indicators participate in entry/exit and their thresholds.
    Returns trades_df and summary metrics.
    """
    df = add_indicators(df, params)
    trades = []
    position = None

    # Core thresholds
    rsi_entry = float(params.get('rsi_entry', 30.0))
    rsi_exit = float(params.get('rsi_exit', 60.0))
    adx_thresh = float(params.get('adx_threshold', 25.0))
    use_ma = params.get('use_ma', True)
    price_ma_field = 'SMA' if params.get('ma_type','SMA')=='SMA' else 'EMA'

    # Additional indicator participation settings
    macd_part = params.get('macd_part', [])  # list e.g. ['Entry','Exit']
    stoch_part = params.get('stoch_part', [])
    atr_part = params.get('atr_part', [])

    # Stochastic thresholds
    stoch_entry_thr = float(params.get('stoch_entry_thr', 20.0))
    stoch_exit_thr = float(params.get('stoch_exit_thr', 80.0))

    # ATR thresholds (absolute) for example usage: entry if ATR < atr_entry_max
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
            # RSI
            conds.append((not np.isnan(rsi_val)) and (rsi_val < rsi_entry))
            # ADX
            conds.append((not np.isnan(adx_val)) and (adx_val < adx_thresh))
            # MA price condition
            if use_ma:
                conds.append((not np.isnan(ma_val)) and (close > ma_val))

            # MACD participation for Entry: MACD > Signal
            if 'Entry' in macd_part:
                conds.append((not np.isnan(macd_val)) and (not np.isnan(macd_sig)) and (macd_val > macd_sig))

            # Stochastic participation for Entry: STOCH_K < threshold
            if 'Entry' in stoch_part:
                conds.append((not np.isnan(stoch_k)) and (stoch_k < stoch_entry_thr))

            # ATR participation for Entry: ATR < entry_max
            if 'Entry' in atr_part:
                conds.append((not np.isnan(atr)) and (atr < atr_entry_max))

            if all(conds):
                position = {
                    'entry_date': date,
                    'entry_price': close,
                    'entry_rsi': rsi_val,
                    'entry_adx': adx_val,
                    'entry_ma': ma_val,
                    'entry_macd': macd_val,
                    'entry_macd_sig': macd_sig,
                    'entry_stoch': stoch_k,
                    'entry_atr': atr
                }
        else:
            exit_conds = []
            # Core exit: RSI above exit threshold and price > entry price (per original logic)
            exit_conds.append((not np.isnan(rsi_val)) and (rsi_val > rsi_exit))
            exit_conds.append(close > position['entry_price'])

            # MACD participation for Exit: MACD < Signal
            if 'Exit' in macd_part:
                exit_conds.append((not np.isnan(macd_val)) and (not np.isnan(macd_sig)) and (macd_val < macd_sig))

            # Stochastic participation for Exit: STOCH_K > threshold
            if 'Exit' in stoch_part:
                exit_conds.append((not np.isnan(stoch_k)) and (stoch_k > stoch_exit_thr))

            # ATR participation for Exit: ATR > exit_min (example)
            if 'Exit' in atr_part:
                exit_conds.append((not np.isnan(atr)) and (atr > atr_exit_min))

            if all(exit_conds):
                exit_trade = {
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'entry_rsi': position['entry_rsi'],
                    'entry_adx': position['entry_adx'],
                    'entry_ma': position['entry_ma'],
                    'exit_date': date,
                    'exit_price': close,
                    'exit_rsi': rsi_val,
                    'exit_adx': adx_val,
                    'exit_ma': ma_val,
                    'exit_macd': macd_val,
                    'exit_macd_sig': macd_sig,
                    'exit_stoch': stoch_k,
                    'exit_atr': atr
                }
                raw_pct = (exit_trade['exit_price'] / exit_trade['entry_price'] - 1) * 100
                exit_trade['raw_pct'] = raw_pct
                trades.append(exit_trade)
                position = None

    # optionally close open position on run
    if position is not None and params.get('close_open_on_run', False):
        last = df.iloc[-1]
        date = df.index[-1]
        exit_trade = {
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'entry_rsi': position['entry_rsi'],
            'entry_adx': position['entry_adx'],
            'entry_ma': position['entry_ma'],
            'exit_date': date,
            'exit_price': last['Close'],
            'exit_rsi': last.get('RSI', np.nan),
            'exit_adx': last.get('ADX', np.nan),
            'exit_ma': last.get('SMA' if params.get('ma_type','SMA')=='SMA' else 'EMA', np.nan),
            'exit_macd': last.get('MACD', np.nan),
            'exit_macd_sig': last.get('MACD_SIGNAL', np.nan),
            'exit_stoch': last.get('STOCH_K', np.nan),
            'exit_atr': last.get('ATR', np.nan)
        }
        raw_pct = (exit_trade['exit_price'] / exit_trade['entry_price'] - 1) * 100
        exit_trade['raw_pct'] = raw_pct
        trades.append(exit_trade)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {
            'n_trades': 0,
            'compounded_return_pct': 0.0,
            'avg_trade_pct': 0.0,
            'win_rate': 0.0
        }
        return trades_df, summary

    # fees
    fee_type = params.get('fee_type', 'none')
    fee_value = float(params.get('fee_value', 0))
    fee_percent = float(params.get('fee_percent', 0))

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

    # compounded return assuming starting capital and reinvest
    capital = float(params.get('capital', 1000.0))
    cap = capital
    for p in trades_df['profit_pct']:
        cap = cap * (1 + p/100)
    compounded_return_pct = (cap - capital) / capital * 100

    summary = {
        'n_trades': len(trades_df),
        'compounded_return_pct': compounded_return_pct,
        'avg_trade_pct': trades_df['profit_pct'].mean(),
        'win_rate': trades_df['win'].mean()
    }

    return trades_df, summary

# --------------------- Grid search utilities ---------------------

def parse_range_input(text_or_list, cast=int):
    """Allow input as comma-separated list like '10,20,30' or range '10-50:10' meaning start-end:step"""
    if isinstance(text_or_list, (list, tuple)):
        return [cast(x) for x in text_or_list]
    s = str(text_or_list).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(',') if p.strip()]
    values = []
    for p in parts:
        if '-' in p:
            # maybe form start-end or start-end:step
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
    # unique sorted
    return sorted(list(dict.fromkeys(values)))

# --------------------- Streamlit UI ---------------------

st.title('Indicator Backtester — Extended (user-selectable indicator participation)')

# Inputs
left, right = st.columns(2)
with left:
    ticker = st.text_input('שם מניה (Ticker)', value='AAPL')
    start_date = st.date_input('תאריך תחילת סריקה', value=pd.to_datetime('2023-01-01'))
    end_date = st.date_input('תאריך סוף סריקה', value=pd.to_datetime(datetime.today().date()))
    interval = st.selectbox('בחירת גרף', options=['1d','60m'], index=0, format_func=lambda x: 'יומי' if x=='1d' else 'שاعي')

with right:
    # core params
    rsi_period = st.number_input('ימים ל-RSI (window)', min_value=2, max_value=200, value=14)
    adx_period = st.number_input('ימים ל-ADX (window)', min_value=2, max_value=200, value=14)
    sma_period = st.number_input('SMA/EMA period', min_value=1, max_value=500, value=50)
    rsi_entry = st.number_input('רף RSI כניסה', min_value=0.0, max_value=100.0, value=30.0)
    rsi_exit = st.number_input('רף RSI יציאה', min_value=0.0, max_value=100.0, value=60.0)
    adx_threshold = st.number_input('רף ADX — כניסה כאשר ADX נמוך מ:', min_value=0.0, max_value=200.0, value=25.0)

st.markdown('---')

# Additional indicators toggle + participation selection
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

# MA type and usage
ma_col1, ma_col2 = st.columns(2)
with ma_col1:
    ma_type = st.selectbox('סוג MA', options=['SMA','EMA'], index=0)
with ma_col2:
    use_ma = st.checkbox('לכלול בדיקת מחיר > MA כצעד בתנאי הכניסה', value=True)

st.markdown('---')

# Fees and options
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

# Grid search
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

# --------------------- Execution ---------------------
if run_button:
    with st.spinner('מוריד נתונים ומריץ בדיקה — זה עשוי לקחת זמן עבור Grid Search...'):
        # download data
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
                # additional indicator participation
                'macd_part': macd_part,
                'stoch_part': stoch_part,
                'stoch_entry_thr': stoch_entry_thr,
                'stoch_exit_thr': stoch_exit_thr,
                'atr_part': atr_part,
                'atr_entry_max': atr_entry_max,
                'atr_exit_min': atr_exit_min
            }

            # single run (no grid)
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
                    # show many columns including chosen indicators
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

                    # chart
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
                # GRID: parse ranges
                rsi_entry_vals = parse_range_input(rsi_entry_range, cast=int)
                rsi_exit_vals = parse_range_input(rsi_exit_range, cast=int)
                adx_vals = parse_range_input(adx_range, cast=int)
                sma_vals = parse_range_input(sma_range, cast=int)

                # build combinations
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
                        res = {
                            'rsi_entry': r_entry,
                            'rsi_exit': r_exit,
                            'adx_threshold': a_val,
                            'sma_period': s_val,
                            'n_trades': summary['n_trades'],
                            'compounded_return_pct': summary['compounded_return_pct'],
                            'avg_trade_pct': summary['avg_trade_pct'],
                            'win_rate': summary['win_rate']
                        }
                        results.append(res)
                        progress.progress(int((i+1)/len(combos)*100))

                    res_df = pd.DataFrame(results)
                    res_df = res_df.sort_values(by='compounded_return_pct', ascending=False).reset_index(drop=True)

                    st.subheader('תוצאות Grid Search — סיכום קומבינציות')
                    st.dataframe(res_df)

                    # top 5 plots
                    top_n = min(5, len(res_df))
                    st.markdown('### Top configurations')
                    for k in range(top_n):
                        row = res_df.iloc[k]
                        st.markdown(f"**#{k+1}** — rsi_entry={row['rsi_entry']} | rsi_exit={row['rsi_exit']} | adx={row['adx_threshold']} | sma={row['sma_period']} — Compounded: {row['compounded_return_pct']:.2f}% | Trades: {int(row['n_trades'])}")

                    if export_csv:
                        csv = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button('הורד תוצאות Grid (CSV)', data=csv, file_name=f'grid_results_{ticker}_{start_date}_{end_date}.csv', mime='text/csv')

                    # allow user to pick one of top combos to visualize
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

# --------------------- Footer / Instructions ---------------------
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
