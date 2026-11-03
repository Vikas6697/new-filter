import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="NSE Scanner", layout="wide")

NIFTY50 = """RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL ITC SBIN LT HINDUNILVR
KOTAKBANK AXISBANK BAJFINANCE ASIANPAINT MARUTI SUNPHARMA TITAN ULTRACEMCO NTPC
ONGC POWERGRID M&M TATAMOTORS TATASTEEL WIPRO HCLTECH TECHM ADANIENT ADANIPORTS
COALINDIA JSWSTEEL BAJAJFINSV NESTLEIND INDUSINDBK GRASIM CIPLA DRREDDY EICHERMOT
HEROMOTOCO BPCL BRITANNIA APOLLOHOSP DIVISLAB HDFCLIFE SBILIFE TATACONSUM
HINDALCO BAJAJ-AUTO SHRIRAMFIN TRENT""".split()


# ---------- indicators ----------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def add_indicators(df):
    c = df["Close"]
    df["rsi"] = rsi(c)
    df["ema20"], df["ema50"], df["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26)
    df["macd"], df["macd_sig"] = macd, ema(macd, 9)
    df["vol_avg"] = df["Volume"].rolling(20).mean()
    df["high52"] = df["High"].rolling(250, min_periods=50).max().shift(1)
    return df


# ---------- scan conditions (each returns True/False for the latest candle) ----------
CONDITIONS = {
    "RSI < 30 (oversold)": lambda d, p: d.rsi < 30,
    "RSI > 70 (overbought)": lambda d, p: d.rsi > 70,
    "Close > EMA200": lambda d, p: d.Close > d.ema200,
    "EMA20 crossed above EMA50": lambda d, p: d.ema20 > d.ema50 and p.ema20 <= p.ema50,
    "MACD bullish crossover": lambda d, p: d.macd > d.macd_sig and p.macd <= p.macd_sig,
    "Volume > 2x 20-day avg": lambda d, p: d.Volume > 2 * d.vol_avg,
    "52-week high breakout": lambda d, p: d.Close > d.high52,
}


@st.cache_data(ttl=300, show_spinner=False)
def load(symbols, interval, period):
    tickers = [s + ".NS" for s in symbols]
    return yf.download(tickers, period=period, interval=interval,
                       group_by="ticker", auto_adjust=True,
                       progress=False, threads=True)


def run_scan(symbols, chosen, interval, period):
    raw = load(tuple(symbols), interval, period)
    rows = []
    for s in symbols:
        try:
            df = raw[s + ".NS"].dropna()
            if len(df) < 60:
                continue
            df = add_indicators(df)
            d, p = df.iloc[-1], df.iloc[-2]
            if all(CONDITIONS[c](d, p) for c in chosen):
                rows.append({
                    "Symbol": s,
                    "Close": round(d.Close, 2),
                    "Chg %": round((d.Close / p.Close - 1) * 100, 2),
                    "RSI": round(d.rsi, 1),
                    "Vol x avg": round(d.Volume / d.vol_avg, 2),
                    "vs EMA200 %": round((d.Close / d.ema200 - 1) * 100, 2),
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def telegram(token, chat_id, text):
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": text}, timeout=10)
    return r.ok


# ---------- UI ----------
st.title("NSE Stock Scanner")

with st.sidebar:
    st.header("Universe")
    custom = st.text_area("Symbols (space/comma separated, no .NS)",
                          value=" ".join(NIFTY50), height=150)
    symbols = [x.strip().upper() for x in custom.replace(",", " ").split() if x.strip()]
    interval = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m"], index=0)
    period = {"1d": "2y", "1h": "6mo", "15m": "1mo", "5m": "1mo"}[interval]

    st.header("Telegram (optional)")
    tg_token = st.text_input("Bot token", type="password")
    tg_chat = st.text_input("Chat ID")

chosen = st.multiselect("Conditions (all must match)", list(CONDITIONS),
                        default=["Close > EMA200", "Volume > 2x 20-day avg"])

if st.button("Run scan", type="primary") and chosen:
    with st.spinner(f"Scanning {len(symbols)} stocks..."):
        res = run_scan(symbols, chosen, interval, period)
    st.session_state["res"] = res
    st.session_state["chosen"] = chosen

res = st.session_state.get("res")
if res is not None:
    st.subheader(f"{len(res)} matches")
    if not res.empty:
        st.dataframe(res.sort_values("Chg %", ascending=False),
                     use_container_width=True, hide_index=True)
        if tg_token and tg_chat and st.button("Send to Telegram"):
            msg = "Scan: " + ", ".join(st.session_state["chosen"]) + "\n" + \
                  "\n".join(f"{r.Symbol} {r.Close} ({r['Chg %']}%)"
                            for _, r in res.iterrows())
            st.success("Sent") if telegram(tg_token, tg_chat, msg) else st.error("Failed")
    else:
        st.info("No stocks match right now.")
