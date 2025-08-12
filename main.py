import os
import time
import json
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import requests

# --- Config ---
BREAKOUT_LEN = int(os.getenv("BREAKOUT_LEN", "20"))
MIN_PCT_MOVE = float(os.getenv("MIN_PCT_MOVE", "5.0"))
MIN_RSI_MOM = float(os.getenv("MIN_RSI_MOM", "50.0"))
KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL", "").strip()
TP_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL", "").strip()
KI_SCORER_URL = os.getenv("KI_SCORER_URL", "").strip()  # Optional
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
POSITION_QTY = int(os.getenv("POSITION_QTY", "10"))

TICKERS_FILE = Path("tickers.txt")
WEB_LOG_FILE = Path("signals.json")


# --- Helper ---
def now_utc_iso():
    return datetime.utcnow().isoformat()


def last_n_closes(df, n):
    closes = df["Close"].tail(n)
    return [round(float(x), 6) for x in closes.tolist()]


def rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, min_periods=length).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/length, min_periods=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def download_df(ticker, period="60d", interval="5m"):
    return yf.download(ticker, period=period, interval=interval, progress=False)


def evaluate_ticker(df, ticker):
    price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    prev_high = float(df["High"].tail(BREAKOUT_LEN).max())
    pct_move = (price - prev_close) / prev_close * 100
    rsi_now = float(rsi(df["Close"]).iloc[-1])

    breakout = price > prev_high
    momentum = rsi_now >= MIN_RSI_MOM
    buy_signal = breakout and momentum
    sell_signal = rsi_now > 70

    score = None
    if KI_SCORER_URL:
        try:
            resp = requests.post(KI_SCORER_URL, json={"ticker": ticker, "price": price, "rsi": rsi_now}, timeout=5)
            if resp.ok:
                score = resp.json().get("score")
        except Exception as e:
            print(f"[WARN] Score-API failed for {ticker}: {e}")

    return {
        "ticker": ticker,
        "price": round(price, 4),
        "rsi": round(rsi_now, 2),
        "pct": round(pct_move, 2),
        "breakout": breakout,
        "momentum": momentum,
        "buy": buy_signal,
        "sell": sell_signal,
        "score": score,
        "spark": last_n_closes(df, 50),
        "time": now_utc_iso()
    }


def send_signal(sig):
    payload = {
        "symbol": sig["ticker"],
        "side": "buy" if sig["buy"] else "sell" if sig["sell"] else "hold",
        "qty": POSITION_QTY,
        "price": sig["price"]
    }

    if KI_WEBHOOK_URL:
        try:
            r = requests.post(KI_WEBHOOK_URL, json=payload, timeout=5)
            print(f"[KI] {sig['ticker']} -> {r.status_code} {r.text}")
        except Exception as e:
            print(f"[KI] send failed: {e}")

    if TP_WEBHOOK_URL:
        try:
            r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=5)
            print(f"[TP] {sig['ticker']} -> {r.status_code} {r.text}")
        except Exception as e:
            print(f"[TP] send failed: {e}")


# --- Main ---
def run():
    if not TICKERS_FILE.exists():
        print("[ERR] tickers.txt fehlt!")
        return

    tickers = [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip()]
    print(f"[{now_utc_iso()}] Scan {len(tickers)} Symbole…")

    results = []
    for t in tickers:
        try:
            print(f"[{t}] downloading…")
            df = download_df(t, period="60d", interval=TIMEFRAME)
            if df.empty:
                print(f"[WARN] {t} hat keine Daten")
                continue
            sig = evaluate_ticker(df, t)
            results.append(sig)
            print(f"[{t}] price={sig['price']} rsi={sig['rsi']} pct={sig['pct']} buy={sig['buy']} sell={sig['sell']}")
            if sig["buy"] or sig["sell"]:
                send_signal(sig)
        except Exception as e:
            print(f"[ERR] {t} failed: {e}")

    try:
        WEB_LOG_FILE.write_text(json.dumps(results, indent=2))
        print(f"[WEB] {len(results)} Signale in {WEB_LOG_FILE} gespeichert")
    except Exception as e:
        print(f"[ERR] write signals.json failed: {e}")


if __name__ == "__main__":
    print(">>> Starting main.py …")
    run()
