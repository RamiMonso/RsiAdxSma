import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from io import BytesIO
import datetime

# ------------------------------------------------------------
# פונקציה עזר: הורדת נתונים ובדיקת עמודות
# ------------------------------------------------------------
def load_data(ticker, start, end, interval):
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
        if data.empty:
            st.error("❌ לא נמצאו נתונים לטיקר ולתאריכים שנבחרו.")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        # אם אין עמודת Close - נשתמש ב-Adj Close
        if "Close" not in data.columns:
            if "Adj Close" in data.columns:
                data["Close"] = data["Adj Close"]
            else:
                st.error("❌ לא נמצאה עמודת Close או Adj Close בנתונים.")
                return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"שגיאה בטעינת הנתונים: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------
# חישוב אינדיקטורים
# ------------------------------------------------------------
def calc_indicators(df, rsi_len, adx_len, sma_len):
    df["RSI"] = ta.rsi(df["Close"], length=rsi_len)
    df["ADX"] = ta.adx(df["High"], df["Low"], df["Close"], length=adx_len)["ADX_14"]
    df["SMA"] = ta.sma(df["Close"], length=sma_len)
    return df

# ------------------------------------------------------------
# בדיקת אסטרטגיה
# ------------------------------------------------------------
def backtest(df, rsi_entry, rsi_exit, adx_thresh, use_rsi, use_adx, use_sma,
             same_day_exit, fee, fee_type, compound):

    positions = []
    in_position = False
    entry_price = 0
    entry_date = None
    entry_rsi = None
    entry_adx = None
    entry_sma = None
    capital = 10000
    units = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # תנאי כניסה
        if not in_position:
            cond_rsi = (not use_rsi) or (row["RSI"] < rsi_entry)
            cond_adx = (not use_adx) or (row["ADX"] < adx_thresh)
            cond_sma = (not use_sma) or (row["Close"] > row["SMA"])

            if cond_rsi and cond_adx and cond_sma:
                in_position = True
                entry_price = row["Close"]
                entry_date = row["Date"]
                entry_rsi = row["RSI"]
                entry_adx = row["ADX"]
                entry_sma = row["SMA"]
                units = capital / entry_price  # רכישת שברי מניות
                # עמלה כניסה
                if fee_type == "אחוז":
                    capital -= capital * (fee / 100)
                else:
                    capital -= fee
        else:
            cond_exit_rsi = row["RSI"] > rsi_exit
            cond_exit_price = row["Close"] > entry_price

            if cond_exit_rsi and cond_exit_price:
                exit_idx = i if same_day_exit else min(i + 1, len(df) - 1)
                exit_row = df.iloc[exit_idx]
                exit_price = exit_row["Close"]
                profit = (exit_price - entry_price) * units
                # עמלה יציאה
                if fee_type == "אחוז":
                    profit -= capital * (fee / 100)
                else:
                    profit -= fee
                capital += profit
                positions.append({
                    "תאריך כניסה": entry_date,
                    "מחיר כניסה": entry_price,
                    "RSI כניסה": entry_rsi,
                    "ADX כניסה": entry_adx,
                    "SMA כניסה": entry_sma,
                    "תאריך יציאה": exit_row["Date"],
                    "מחיר יציאה": exit_price,
                    "RSI יציאה": exit_row["RSI"],
                    "ADX יציאה": exit_row["ADX"],
                    "SMA יציאה": exit_row["SMA"],
                    "רווח/הפסד %": (exit_price - entry_price) / entry_price * 100
                })
                in_position = False

    # אם יש עסקה פתוחה שלא נסגרה
    if in_position:
        last = df.iloc[-1]
        current_price = last["Close"]
        profit = (current_price - entry_price) * units
        positions.append({
            "תאריך כניסה": entry_date,
            "מחיר כניסה": entry_price,
            "RSI כניסה": entry_rsi,
            "ADX כניסה": entry_adx,
            "SMA כניסה": entry_sma,
            "תאריך יציאה": last["Date"],
            "מחיר יציאה": current_price,
            "RSI יציאה": last["RSI"],
            "ADX יציאה": last["ADX"],
            "SMA יציאה": last["SMA"],
            "רווח/הפסד %": (current_price - entry_price) / entry_price * 100
        })

    return pd.DataFrame(positions), capital

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="בודק אסטרטגיות RSI+ADX+SMA", layout="wide")

st.title("📊 בודק אסטרטגיות RSI + ADX + SMA")
st.sidebar.header("הגדרות")

ticker = st.sidebar.text_input("הזן טיקר (למשל AAPL):", "AAPL")
start_date = st.sidebar.date_input("תאריך התחלה", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("תאריך סיום", datetime.date.today())
interval = st.sidebar.selectbox("טווח זמן", ["1d", "1h"], index=0)

rsi_len = st.sidebar.number_input("ימי RSI", 5, 50, 14)
adx_len = st.sidebar.number_input("ימי ADX", 5, 50, 14)
sma_len = st.sidebar.number_input("ימי SMA", 5, 250, 50)

rsi_entry = st.sidebar.number_input("רף RSI לכניסה", 0, 100, 30)
rsi_exit = st.sidebar.number_input("רף RSI ליציאה", 0, 100, 70)
adx_thresh = st.sidebar.number_input("רף ADX לכניסה", 0, 100, 25)

st.sidebar.markdown("### אינדיקטורים פעילים:")
use_rsi = st.sidebar.checkbox("RSI", value=True)
use_adx = st.sidebar.checkbox("ADX", value=True)
use_sma = st.sidebar.checkbox("SMA", value=True)

same_day_exit = st.sidebar.checkbox("סגירה באותו יום", value=False)
fee_type = st.sidebar.selectbox("סוג עמלה", ["אחוז", "סכום"], index=0)
fee = st.sidebar.number_input("עמלה לכל פעולה", 0.0, 100.0, 0.1)
compound = st.sidebar.checkbox("חישוב כריבית דריבית", value=False)

if st.sidebar.button("הרץ אסטרטגיה 🚀"):
    with st.spinner("מוריד נתונים ומחשב..."):
        df = load_data(ticker, start_date - datetime.timedelta(days=250), end_date, interval)
        if not df.empty:
            df = calc_indicators(df, rsi_len, adx_len, sma_len)
            df = df.dropna(subset=["Close"], how="any")
            results, final_capital = backtest(df, rsi_entry, rsi_exit, adx_thresh,
                                              use_rsi, use_adx, use_sma,
                                              same_day_exit, fee, fee_type, compound)
            if not results.empty:
                st.success("✅ הסריקה הושלמה בהצלחה!")

                st.subheader("📅 טבלת עסקאות:")
                st.dataframe(results)

                st.markdown(f"**סה\"כ רווח/הפסד מצטבר:** {results['רווח/הפסד %'].sum():.2f}%")

                # גרף
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df["Date"], df["Close"], label="מחיר מניה")
                for _, r in results.iterrows():
                    ax.scatter(r["תאריך כניסה"], r["מחיר כניסה"], color="green", marker="^", s=100)
                    ax.scatter(r["תאריך יציאה"], r["מחיר יציאה"], color="red", marker="v", s=100)
                ax.legend()
                ax.set_title(f"גרף מחיר עם כניסות ויציאות ({ticker})")
                st.pyplot(fig)

                # ייצוא
                excel_buffer = BytesIO()
                results.to_excel(excel_buffer, index=False, engine='xlsxwriter')
                st.download_button("📥 הורד טבלה ל-Excel", data=excel_buffer.getvalue(),
                                   file_name=f"results_{ticker}.xlsx")

            else:
                st.warning("לא נמצאו עסקאות בהתאם לקריטריונים שהוגדרו.")
