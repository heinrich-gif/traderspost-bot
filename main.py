import os, json, time
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- Parameter (kannst du auch per GitHub-Env überschreiben) ---
TP_WEBHOOK = os.getenv("TP_WEBHOOK_URL")               # NUR falls KI_MODE=webhook nicht forwardet (Failover)
KI_MODE = os.getenv("KI_MODE", "webhook").lower()      # wir nutzen "webhook"
KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL", "")       # Cloudflare Worker URL
POSITION_QTY = int(os.getenv("POSITION_QTY", "10"))
TIMEFRAME = os.getenv("TIMEFRAME", "5m")               # 5m empfohlen, da Actions alle 5 Min
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))
RSI_BUY_CROSS = float(os.getenv("RSI_BUY_CROSS", "30"))
RSI_SELL_CROSS = float(os.getenv("RSI_SELL_CROSS", "70"))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "60"))    # mind. 60 Min zwischen gleichen Signalen
STATE_PATH = "state/last_signals.json"

os.makedirs("state", exist_ok=True)

def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r") as f:
            st = json.load(f)
    except Exception:
        st = {}
    # alte Einträge (älter als 3 Tage) entsorgen
    cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=3)
    return {k:v for k,v in st.items() if datetime.fromisoformat(v) >= cutoff}

def save_state(st):
    with open(STATE_PATH, "w") as f:
        json.dump(st, f)

def read_tickers(path="tickers.txt"):
    with open(path) as f:
        return [x.strip() for x in f if x.strip() and not x.startswith("#")]

def download_hist(t, period="60d", interval="5m"):
    return yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True)

def compute_inds(df):
    df = df.copy()
    df["rsi"] = RSIIndicator(close=df["Close"], window=RSI_LEN).rsi()
    df["ema_fast"] = EMAIndicator(close=df["Close"], window=EMA_FAST).ema_indicator()
    df["ema_slow"] = EMAIndicator(close=df["Close"], window=EMA_SLOW).ema_indicator()
    return df.dropna()

def crossover_up(s, level):
    return len(s) >= 2 and s.iloc[-2] <= level and s.iloc[-1] > level

def crossover_down(s, level):
    return len(s) >= 2 and s.iloc[-2] >= level and s.iloc[-1] < level

def trend_ok(df):
    return (df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]) and (df["Close"].iloc[-1] > df["ema_slow"].iloc[-1])

def cooldown_ok(state, symbol, action, now_utc):
    key = f"{symbol}:{action}"
    if key not in state:
        return True
    last = datetime.fromisoformat(state[key])
    return (now_utc - last) >= timedelta(minutes=COOLDOWN_MIN)

def mark_sent(state, symbol, action, now_utc):
    state[f"{symbol}:{action}"] = now_utc.isoformat()

def send_to_cloudflare(symbol, action, qty, price=None):
    body = {"symbol": symbol, "action": action, "quantity": int(qty)}
    if price is not None:
        body["price"] = float(price)
    r = requests.post(KI_WEBHOOK_URL, json=body, timeout=15)
    # Der Worker forwardet nur bei gutem Score (200 mit TP-Antwort oder 2xx).
    # Wenn er "filtered" antwortet, ignorieren wir’s.
    ok = 200 <= r.status_code < 300 and "filtered" not in r.text.lower()
    if not ok:
        print(f"[CF] filtered/err {r.status_code}: {r.text[:120]}")
    return ok

def run():
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    tickers = read_tickers()
    state = load_state()
    print(f"[{now_utc.isoformat()}] Scan {len(tickers)} Symbole…")

    for t in tickers:
        try:
            df = download_hist(t, period="30d", interval=TIMEFRAME)
            if df is None or df.empty:
                continue
            df = compute_inds(df)
            if df.empty:
                continue

            price = float(df["Close"].iloc[-1])
            buy = trend_ok(df) and crossover_up(df["rsi"], RSI_BUY_CROSS)
            sell = crossover_down(df["rsi"], RSI_SELL_CROSS)

            if buy and cooldown_ok(state, t, "buy", now_utc):
                if send_to_cloudflare(t, "buy", POSITION_QTY, price):
                    mark_sent(state, t, "buy", now_utc)
                    print(f"[OK] BUY {t} @ {price:.2f}")

            if sell and cooldown_ok(state, t, "sell", now_utc):
                if send_to_cloudflare(t, "sell", POSITION_QTY, price):
                    mark_sent(state, t, "sell", now_utc)
                    print(f"[OK] SELL {t} @ {price:.2f}")

            time.sleep(0.15)  # sanfte Drossel

        except Exception as e:
            print(f"[WARN] {t}: {e}")

    save_state(state)

if __name__ == "__main__":
    run()
