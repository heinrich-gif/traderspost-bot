import os
import time
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import requests

# ==== Parameter aus Umgebungsvariablen ====
KI_MODE = os.getenv("KI_MODE", "webhook")
KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL", "")
TP_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL", "")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
POSITION_QTY = int(os.getenv("POSITION_QTY", "10"))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "60"))
BREAKOUT_LEN = int(os.getenv("BREAKOUT_LEN", "20"))
MIN_PCT_MOVE = float(os.getenv("MIN_PCT_MOVE", "5.0"))
MIN_RSI_MOM = float(os.getenv("MIN_RSI_MOM", "50.0"))

# ==== Handelszeiten-Filter ====
def is_us_extended_utc(dt: datetime) -> bool:
    """
    Gibt True zurück, wenn wir uns im US Pre-Market, Regular oder After-Hours befinden.
    Zeiten in UTC:
        Pre-Market: 08:00 – 13:30 UTC  (10:00 – 15:30 DE)
        Regular:    13:30 – 20:00 UTC  (15:30 – 22:00 DE)
        After-Hours:20:00 – 00:00 UTC  (22:00 – 02:00 DE)
    """
    wd = dt.weekday()  # 0=Mo … 6=So
    if wd >= 5:
        return False
    mins = dt.hour * 60 + dt.minute
    return (8 * 60 <= mins < 24 * 60) or (0 <= mins < 2 * 60)

# ==== Daten-Download ====
def download_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="5d", interval=TIMEFRAME, progress=False)
        if df.empty:
            print(f"[WARN] {ticker}: Keine Daten")
            return pd.DataFrame()
        df["rsi"] = compute_rsi(df["Close"], 14)
        return df
    except Exception as e:
        print(f"[ERR] {ticker}: Download-Fehler {e}")
        return pd.DataFrame()

# ==== RSI-Berechnung ====
def compute_rsi(series, period: int):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ==== Strategie ====
def check_trade(df: pd.DataFrame, ticker: str):
    price = float(df["Close"].iloc[-1])
    rsi_val = float(df["rsi"].iloc[-1])
    pct_move = ((price - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100
    prev_high = float(df["High"].tail(BREAKOUT_LEN).max())
    breakout = price > prev_high
    momentum = rsi_val > MIN_RSI_MOM

    print(f"[{ticker}] price={price:.2f} rsi={rsi_val:.2f} pct={pct_move:.2f}% prevHigh{BREAKOUT_LEN}={prev_high} breakout={breakout} mom={momentum}")

    buy = breakout and momentum and abs(pct_move) >= MIN_PCT_MOVE
    if buy:
        send_order(ticker, "buy", price)
    return buy

# ==== Order senden ====
def send_order(ticker: str, side: str, price: float):
    if TP_WEBHOOK_URL:
        payload = {"symbol": ticker, "side": side, "qty": POSITION_QTY, "price": price}
        try:
            r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=5)
            print(f"[TP] {ticker} {side} -> status={r.status_code} resp='{r.text.strip()}'")
        except Exception as e:
            print(f"[ERR] Order-Webhook fehlgeschlagen: {e}")

# ==== Hauptlauf ====
def run():
    now_utc = datetime.now(timezone.utc)
    if not is_us_extended_utc(now_utc):
        print(f"[SKIP] Outside US extended session (now UTC {now_utc.isoformat()})")
        return

    print(f"[{now_utc.isoformat()}] Scan startet…")

    tickers = ["OMI", "MSOS", "LAR", "AIYY", "WOW", "YOLO", "VTEX", "UA", "ETHD", "AMC", "MX", "BLND", "LAC"]

    for t in tickers:
        df = download_data(t)
        if not df.empty:
            check_trade(df, t)
        time.sleep(0.15)  # sanftes Rate-Limit für yfinance

    print("[DONE] cycle complete")

if __name__ == "__main__":
    print(">>> Starting main.py …")
    run()
