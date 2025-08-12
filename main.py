import os
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ===== Handelszeit-Filter =====
def is_us_rth_utc(dt: datetime) -> bool:
    wd = dt.weekday()  # 0=Mo … 6=So
    if wd >= 5:
        return False
    mins = dt.hour * 60 + dt.minute
    return (8 * 60 <= mins < 24 * 60) or (0 <= mins < 2 * 60)  # 08:00–23:59 + 00:00–01:59 UTC

# ===== Config =====
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
QTY = int(os.getenv("POSITION_QTY", 10))
BREAKOUT_LEN = int(os.getenv("BREAKOUT_LEN", 20))
MIN_PCT_MOVE = float(os.getenv("MIN_PCT_MOVE", "5"))
MIN_RSI_MOM = float(os.getenv("MIN_RSI_MOM", "50"))
KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL")
TP_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL")

# ===== Funktionen =====
def download_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval=TIMEFRAME, progress=False)
        if df.empty:
            print(f"[WARN] {ticker}: keine Daten")
            return None
        df["rsi"] = ta_rsi(df["Close"], 14)
        return df
    except Exception as e:
        print(f"[ERR] {ticker} Download: {e}")
        return None

def ta_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def breakout_signal(df):
    if df is None or len(df) < BREAKOUT_LEN:
        return False, 0, 0
    price = float(df["Close"].iloc[-1])
    rsi = float(df["rsi"].iloc[-1])
    prev_high = float(df["High"].iloc[-BREAKOUT_LEN:].max())
    pct_move = ((price - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100
    breakout = price > prev_high
    momentum = rsi >= MIN_RSI_MOM
    return breakout and momentum, price, rsi

def send_to_traderspost(ticker, side, price, rsi):
    payload = {
        "symbol": ticker,
        "action": side,
        "price": round(price, 2),
        "rsi": round(rsi, 2)
    }
    try:
        resp = requests.post(TP_WEBHOOK_URL, json=payload, timeout=10)
        ok = resp.status_code == 200
        print(f"[CF] {ticker} {side} -> status={resp.status_code} ok={ok} resp='{resp.text[:50]}'")
    except Exception as e:
        print(f"[ERR] send_to_traderspost: {e}")

# ===== Main Run =====
def run():
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    if not is_us_extended_utc(now_utc):
        print(f"[SKIP] Outside US extended session (now UTC {now_utc.isoformat()})")
        return

    tickers = [
        "OMI", "MSOS", "LAR", "AIYY", "WOW", "YOLO",
        "VTEX", "UA", "ETHD", "AMC", "MX", "BLND", "LAC"
    ]
    print(f"[{now_utc}] Scan {len(tickers)} Symbole…")
    print(f"Tickers: {tickers}")

    for t in tickers:
        print(f"[{t}] downloading…")
        df = download_data(t)
        if df is None:
            continue
        signal, price, rsi = breakout_signal(df)
        print(f"[{t}] price={price:.2f} rsi={rsi:.2f} breakout={signal}")
        if signal:
            send_to_traderspost(t, "buy", price, rsi)
        time.sleep(0.2)  # API freundlich

    print("[DONE] cycle complete")

if __name__ == "__main__":
    print(">>> Starting main.py …")
    run()
