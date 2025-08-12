import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from finvizfinance.screener.overview import Overview
import requests

# ==== CONFIG ====
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL", "")
TP_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL", "")
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.02"))  # 2%

BREAKOUT_LEN = 20
DELTA_TOL = 0.3  # % Toleranz für Breakout
RSI_BUY_MAX = 65.0
PCT_MOVE_MIN = 0.5

STATE_FILE = "state/last.json"
SIGNALS_FILE = "docs/signals.json"
EXITS_FILE = "docs/exits.json"

# ==== HELPER ====
def now_utc():
    return datetime.now(timezone.utc).isoformat()

def last_n_closes(df, n=50):
    return [round(float(x), 6) for x in df["Close"].tail(n).tolist()]

def scan_finviz():
    ov = Overview()
    filters = [
        "sh_price_u5",     # price <= 5 USD
        "sh_relvol_o2",    # rel vol >= 2
        "sh_avgvol_o500"   # avg vol >= 500k
    ]
    print(f"[{now_utc()}] [SCAN] Finviz… price<=5.0 relvol>=2.0 avgvol>=500000")
    ov.set_filter(filter_list=filters)  # FIX: ohne named arg
    df = ov.screener_view()
    tickers = df['Ticker'].tolist() if 'Ticker' in df.columns else []
    print(f"[{now_utc()}] [SCAN] Finviz-Kandidaten: {len(tickers)} -> {tickers}")
    return tickers

def analyze_with_ki(signal: dict) -> dict:
    if not KI_WEBHOOK_URL:
        return {}
    payload = {
        "symbol": signal["symbol"],
        "price": signal["price"],
        "recommendation": signal["recommendation"],
        "rsi": signal["rsi"],
        "pct_move": signal["pct_move"],
        "breakout": signal["breakout"],
        "momentum": signal["momentum"],
        "equity": ACCOUNT_EQUITY,
        "risk_pct": RISK_PCT,
    }
    try:
        r = requests.post(KI_WEBHOOK_URL, json=payload, timeout=10)
        if r.ok:
            return r.json()
        else:
            print(f"[KI] Error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[KI] Exception: {e}")
    return {}

def evaluate_ticker(df, t):
    price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    prev_high = float(df["High"].tail(BREAKOUT_LEN).max())

    pct_move = round((price - prev_close) / prev_close * 100, 2)
    rsi = calc_rsi(df["Close"])
    breakout = price >= prev_high * (1 - DELTA_TOL/100)
    momentum = pct_move >= PCT_MOVE_MIN and rsi <= RSI_BUY_MAX

    buy_raw = breakout and momentum
    rec = "BUY" if buy_raw else "HOLD"

    return {
        "symbol": t,
        "price": round(price, 4),
        "pct_move": pct_move,
        "rsi": round(rsi, 2),
        "breakout": breakout,
        "momentum": momentum,
        "recommendation": rec,
        "spark": last_n_closes(df, 50)
    }

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ==== MAIN ====
def run():
    tickers = scan_finviz()
    signals = []
    exits = []

    for t in tickers:
        print(f"[{t}] downloading…")
        df = yf.download(t, period="5d", interval=TIMEFRAME, progress=False)
        if df.empty:
            continue

        signal = evaluate_ticker(df, t)
        ki_data = analyze_with_ki(signal)
        if ki_data:
            signal.update(ki_data)

        print(f"[{t}] price={signal['price']:.4f} rsi={signal['rsi']} Δ%={signal['pct_move']} breakout={signal['breakout']} mom={signal['momentum']} -> {signal['recommendation']}")

        if signal["recommendation"] == "BUY":
            signals.append(signal)
            if TP_WEBHOOK_URL:
                try:
                    requests.post(TP_WEBHOOK_URL, json=signal, timeout=10)
                except Exception as e:
                    print(f"[TP] Error sending: {e}")
        elif signal["recommendation"] == "SELL":
            exits.append(signal)

    save_json(SIGNALS_FILE, signals)
    save_json(EXITS_FILE, exits)
    print(f"[DONE] {len(signals)} BUY, {len(exits)} SELL")

if __name__ == "__main__":
    run()
