import os, json, time, traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# ========= Parameter (können via GitHub Actions gesetzt werden) =========
TP_WEBHOOK        = os.getenv("TP_WEBHOOK_URL")                # nur Failover, normal leitet der KI-Worker weiter
KI_MODE           = os.getenv("KI_MODE", "webhook").lower()    # "webhook" (Cloudflare), "openai", "none"
KI_WEBHOOK_URL    = os.getenv("KI_WEBHOOK_URL", "")            # Cloudflare Worker URL
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

POSITION_QTY      = int(os.getenv("POSITION_QTY", "10"))
TIMEFRAME         = os.getenv("TIMEFRAME", "5m")               # 1m,5m,15m,1h,1d
RSI_LEN           = int(os.getenv("RSI_LEN", "14"))
EMA_FAST          = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW          = int(os.getenv("EMA_SLOW", "200"))
RSI_BUY_CROSS     = float(os.getenv("RSI_BUY_CROSS", "30"))
RSI_SELL_CROSS    = float(os.getenv("RSI_SELL_CROSS", "70"))
COOLDOWN_MIN      = int(os.getenv("COOLDOWN_MIN", "60"))
STATE_PATH        = "state/last_signals.json"

os.makedirs("state", exist_ok=True)

print("[BOOT] starting…")
print(f"[CFG] KI_MODE={KI_MODE} TIMEFRAME={TIMEFRAME} QTY={POSITION_QTY} RSI={RSI_LEN} EMA={EMA_FAST}/{EMA_SLOW}")
print(f"[CFG] KI_WEBHOOK_URL set? {'yes' if bool(KI_WEBHOOK_URL) else 'no'}")

# ========= Hilfsfunktionen =========

def load_state():
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r") as f:
            st = json.load(f)
    except Exception:
        st = {}
    cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=3)
    cleaned = {}
    for k, v in st.items():
        try:
            if datetime.fromisoformat(v) >= cutoff:
                cleaned[k] = v
        except Exception:
            pass
    return cleaned

def save_state(st):
    with open(STATE_PATH, "w") as f:
        json.dump(st, f)

def read_tickers(path="tickers.txt"):
    try:
        with open(path) as f:
            lst = [x.strip() for x in f if x.strip() and not x.startswith("#")]
            return lst
    except FileNotFoundError:
        print("[ERR] tickers.txt nicht gefunden!")
        return []

def download_hist(t, period="30d", interval="5m"):
    # flache Spalten & stabil in Actions
    df = yf.download(
        t, period=period, interval=interval,
        progress=False, auto_adjust=True, group_by="column", threads=False
    )
    return df

def to_series_close(df):
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    if isinstance(df, pd.DataFrame):
        if 'Close' in df.columns:
            s = df['Close']
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        else:
            s = df.iloc[:, -1]
    else:
        s = df

    if hasattr(s, "to_numpy") and getattr(s, "ndim", 1) > 1:
        s = pd.Series(s.to_numpy().ravel(), index=df.index, name='Close')

    try:
        s = s.astype("float64")
    except Exception:
        s = pd.to_numeric(s, errors="coerce")

    return s

def compute_inds(df):
    df = df.copy()
    close = to_series_close(df)
    if close.empty:
        return pd.DataFrame()
    df['Close'] = close
    df["rsi"] = RSIIndicator(close=close, window=RSI_LEN).rsi()
    df["ema_fast"] = EMAIndicator(close=close, window=EMA_FAST).ema_indicator()
    df["ema_slow"] = EMAIndicator(close=close, window=EMA_SLOW).ema_indicator()
    return df.dropna()

def crossover_up(series, level):
    return len(series) >= 2 and series.iloc[-2] <= level and series.iloc[-1] > level

def crossover_down(series, level):
    return len(series) >= 2 and series.iloc[-2] >= level and series.iloc[-1] < level

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
    if not KI_WEBHOOK_URL:
        print("[CFG] KI_WEBHOOK_URL fehlt - überspringe.")
        return False
    body = {"symbol": symbol, "action": action, "quantity": int(qty)}
    if price is not None:
        body["price"] = float(price)
    r = requests.post(KI_WEBHOOK_URL, json=body, timeout=20)
    ok = 200 <= r.status_code < 300 and "filtered" not in r.text.lower()
    print(f"[CF] {symbol} {action} -> status={r.status_code} ok={ok} resp='{r.text[:120]}'")
    return ok

def send_traderspost_direct(symbol, action, qty):
    if not TP_WEBHOOK:
        print("[TP] TP_WEBHOOK_URL fehlt (Failover nicht möglich).")
        return False
    body = {"symbol": symbol, "action": action, "quantity": int(qty)}
    r = requests.post(TP_WEBHOOK, json=body, timeout=20)
    ok = 200 <= r.status_code < 300
    print(f"[TP] {symbol} {action} -> status={r.status_code} ok={ok} resp='{r.text[:120]}'")
    return ok

def ki_pass(symbol, action, price):
    mode = KI_MODE
    if mode == "none":
        return True
    if mode == "webhook":
        return True
    if mode == "openai":
        try:
            import requests as rq
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            prompt = (
                f"Bewerte kurz: Symbol={symbol}, Aktion={action}, Preis={price}. "
                f"Strategie: RSI-Cross {RSI_BUY_CROSS}/{RSI_SELL_CROSS} + EMA({EMA_FAST}>{EMA_SLOW}). "
                f"Antworte NUR 'JA' oder 'NEIN'."
            )
            body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0}
            resp = rq.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=20)
            txt = resp.json()["choices"][0]["message"]["content"].strip().lower()
            print(f"[KI] OpenAI antwortet: {txt}")
            return "ja" in txt
        except Exception as e:
            print(f"[KI] OpenAI-Fehler: {e}")
            return False
    return False

# ========= Hauptlauf =========

def run():
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    tickers = read_tickers()
    print(f"[{now_utc.isoformat()}] Scan {len(tickers)} Symbole…")
    print("Tickers:", tickers)

    state = load_state()

    for t in tickers:
        print(f"[{t}] downloading…", flush=True)
        try:
            df = download_hist(t, period="30d", interval=TIMEFRAME)
            if df is None or df.empty:
                print(f"[{t}] empty df, skip")
                time.sleep(0.15)
                continue

            print(f"[{t}] rows={len(df)}")
            df = compute_inds(df)
            if df.empty or not all(c in df.columns for c in ["Close", "rsi", "ema_fast", "ema_slow"]):
                print(f"[{t}] indicators missing/empty, skip")
                time.sleep(0.15)
                continue

            price = float(df["Close"].iloc[-1])
            rsi_val = float(df["rsi"].iloc[-1])
            ema_fast_val = float(df["ema_fast"].iloc[-1])
            ema_slow_val = float(df["ema_slow"].iloc[-1])

            buy = trend_ok(df) and crossover_up(df["rsi"], RSI_BUY_CROSS)
            sell = crossover_down(df["rsi"], RSI_SELL_CROSS)

            print(f"[{t}] price={price:.2f} rsi={rsi_val:.2f} emaF/S={ema_fast_val:.2f}/{ema_slow_val:.2f} buy={buy} sell={sell}")

            # BUY
            if buy and cooldown_ok(state, t, "buy", now_utc):
                if ki_pass(t, "buy", price):
                    ok = True
                    if KI_MODE == "webhook":
                        ok = send_to_cloudflare(t, "buy", POSITION_QTY, price)
                    elif KI_MODE in ("none", "openai"):
                        ok = send_traderspost_direct(t, "buy", POSITION_QTY)
                    if ok:
                        mark_sent(state, t, "buy", now_utc)
                        print(f"[OK] BUY {t} @ {price:.2f}")

            # SELL (optional)
            if sell and cooldown_ok(state, t, "sell", now_utc):
                if ki_pass(t, "sell", price):
                    ok = True
                    if KI_MODE == "webhook":
                        ok = send_to_cloudflare(t, "sell", POSITION_QTY, price)
                    elif KI_MODE in ("none", "openai"):
                        ok = send_traderspost_direct(t, "sell", POSITION_QTY)
                    if ok:
                        mark_sent(state, t, "sell", now_utc)
                        print(f"[OK] SELL {t} @ {price:.2f}")

        except Exception as e:
            print(f"[WARN] {t}: {e}")
            traceback.print_exc()

        # sanfte Drossel außerhalb des try/except
        time.sleep(0.15)

    save_state(state)
    print("[DONE] cycle complete")

if __name__ == "__main__":
    run()
