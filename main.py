import os, json, time, traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator  # optional behalten, falls du später Trendfilter willst

# ========= Konfiguration (per GitHub Actions Vars/Secrets überschreibbar) =========
KI_MODE           = os.getenv("KI_MODE", "webhook").lower()    # "webhook" (Cloudflare), "openai", "none"
KI_WEBHOOK_URL    = os.getenv("KI_WEBHOOK_URL", "")            # Cloudflare Worker URL (Pflicht bei webhook)
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")            # nur falls KI_MODE=openai
TP_WEBHOOK        = os.getenv("TP_WEBHOOK_URL")                # Failover (normal nicht genutzt)

TIMEFRAME         = os.getenv("TIMEFRAME", "5m")               # 1m, 5m, 15m, 1h, 1d
POSITION_QTY      = int(os.getenv("POSITION_QTY", "10"))
COOLDOWN_MIN      = int(os.getenv("COOLDOWN_MIN", "60"))       # mind. X Min zwischen gleichen Signalen/Symbol

# Breakout/Momentum-Parameter (aggressiv)
BREAKOUT_LEN      = int(os.getenv("BREAKOUT_LEN", "20"))       # Donchian High über letzte N Kerzen
MIN_PCT_MOVE      = float(os.getenv("MIN_PCT_MOVE", "5"))      # min. +% seit Tages-Open
MIN_RSI_MOM       = float(os.getenv("MIN_RSI_MOM", "50"))      # RSI-Mindestmomentum

# RSI/EMA werden noch berechnet (hilfreich fürs Logging oder spätere Filter)
RSI_LEN           = int(os.getenv("RSI_LEN", "14"))
EMA_FAST          = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW          = int(os.getenv("EMA_SLOW", "200"))

STATE_PATH        = "state/last_signals.json"

os.makedirs("state", exist_ok=True)

print("[BOOT] starting…")
print(f"[CFG] KI_MODE={KI_MODE} TIMEFRAME={TIMEFRAME} QTY={POSITION_QTY} COOLDOWN_MIN={COOLDOWN_MIN}")
print(f"[CFG] BREAKOUT_LEN={BREAKOUT_LEN} MIN_PCT_MOVE={MIN_PCT_MOVE}% MIN_RSI_MOM={MIN_RSI_MOM}")
print(f"[CFG] KI_WEBHOOK_URL set? {'yes' if bool(KI_WEBHOOK_URL) else 'no'}")

# ========= State / Ticker =========

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

# ========= yfinance robust =========

def download_hist(t, period="30d", interval="5m"):
    # flache Spalten; threads=False stabilisiert Runs in Actions
    df = yf.download(
        t, period=period, interval=interval,
        progress=False, auto_adjust=True, group_by="column", threads=False
    )
    return df

def to_series(df, col_name: str):
    """Gibt garantiert eine 1D-Series (float) für die gewünschte Spalte zurück."""
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if isinstance(df, pd.DataFrame):
        if col_name in df.columns:
            s = df[col_name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        else:
            # Fallback: letzte Spalte
            s = df.iloc[:, -1]
    else:
        s = df
    if hasattr(s, "to_numpy") and getattr(s, "ndim", 1) > 1:
        s = pd.Series(s.to_numpy().ravel(), index=df.index, name=col_name)
    try:
        s = s.astype("float64")
    except Exception:
        s = pd.to_numeric(s, errors="coerce")
    return s

def ensure_series(x):
    """Macht aus 1-spaltigem DataFrame eine Series."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x

def last_scalar(x):
    """Letzter Wert als float."""
    s = ensure_series(x)
    val = s.iloc[-1]
    try:
        return float(val)
    except Exception:
        return float(pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0])

# ========= Indikatoren =========

def compute_inds(df):
    """Berechnet RSI und optionale EMAs. Liefert DataFrame mit Close/Open/High/Low + rsi/ema_*."""
    df = df.copy()
    close = to_series(df, "Close")
    if close.empty:
        return pd.DataFrame()
    open_ = to_series(df, "Open")
    high  = to_series(df, "High")
    low   = to_series(df, "Low")

    df["Close"] = close
    df["Open"]  = open_
    df["High"]  = high
    df["Low"]   = low

    df["rsi"]       = RSIIndicator(close=close, window=RSI_LEN).rsi()
    df["ema_fast"]  = EMAIndicator(close=close, window=EMA_FAST).ema_indicator()
    df["ema_slow"]  = EMAIndicator(close=close, window=EMA_SLOW).ema_indicator()
    return df.dropna()

def rolling_high(series, n):
    """Höchstes Hoch der letzten n Kerzen VOR der aktuellen Kerze (klassischer Ausbruch)."""
    s = ensure_series(series)
    if len(s) < n + 1:
        return None
    return float(s.iloc[-(n+1):-1].max())

# ========= KI / TradersPost =========

def send_to_cloudflare(symbol, action, qty, price=None):
    if not KI_WEBHOOK_URL:
        print("[CFG] KI_WEBHOOK_URL fehlt - überspringe.")
        return False
    body = {"symbol": symbol, "action": action, "quantity": int(qty)}
    if price is not None:
        body["price"] = float(price)
    r = requests.post(KI_WEBHOOK_URL, json=body, timeout=20)

    # Erwartet JSON vom Worker (siehe empfohlene Worker-Version)
    score = None
    result = None
    try:
        j = r.json()
        score = j.get("score")
        result = j.get("result")
    except Exception:
        pass

    # 204 = gefiltert; oder "result=filtered"
    if r.status_code == 204 or (isinstance(result, str) and result.lower() == "filtered"):
        print(f"[CF] {symbol} {action} -> FILTERED score={score}")
        return False

    ok = 200 <= r.status_code < 300 and "filtered" not in r.text.lower()
    print(f"[CF] {symbol} {action} -> status={r.status_code} ok={ok} score={score} resp='{str(r.text)[:200]}'")
    return ok

def send_traderspost_direct(symbol, action, qty):
    """Failover: normal nicht genutzt (Worker forwarded an TP)."""
    if not TP_WEBHOOK:
        print("[TP] TP_WEBHOOK_URL fehlt (Failover nicht möglich).")
        return False
    body = {"symbol": symbol, "action": action, "quantity": int(qty)}
    r = requests.post(TP_WEBHOOK, json=body, timeout=20)
    ok = 200 <= r.status_code < 300
    print(f"[TP] {symbol} {action} -> status={r.status_code} ok={ok} resp='{r.text[:200]}'")
    return ok

def ki_pass(symbol, action, price):
    """Nur relevant, wenn KI_MODE != 'webhook'."""
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
                f"Strategie: Breakout über {BREAKOUT_LEN}-Kerzen Hoch + Intraday Move>={MIN_PCT_MOVE}%. "
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

# ========= Cooldown =========

def cooldown_ok(state, symbol, action, now_utc):
    key = f"{symbol}:{action}"
    if key not in state:
        return True
    last = datetime.fromisoformat(state[key])
    return (now_utc - last) >= timedelta(minutes=COOLDOWN_MIN)

def mark_sent(state, symbol, action, now_utc):
    state[f"{symbol}:{action}"] = now_utc.isoformat()

# ========= Hauptlauf =========

def run():
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    tickers = read_tickers()
    print(f"[{now_utc.isoformat()}] Scan {len(tickers)} Symbole…")
    print("Tickers:", tickers)

    if not tickers:
        print("[WARN] Keine Ticker gefunden – beende Run.")
        return

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
            needed = ["Open", "High", "Low", "Close", "rsi"]
            if df.empty or not all(c in df.columns for c in needed):
                print(f"[{t}] indicators missing/empty, skip")
                time.sleep(0.15)
                continue

            # sichere Series & Skalare
            open_s = ensure_series(df["Open"])
            high_s = ensure_series(df["High"])
            close_s= ensure_series(df["Close"])
            rsi_s  = ensure_series(df["rsi"])

            price       = last_scalar(close_s)
            rsi_val     = last_scalar(rsi_s)

            # Intraday %-Move (gegen aktuelle Kerze Open)
            try:
                day_open = float(open_s.iloc[-1])
                pct_move = (price - day_open) / day_open * 100.0 if day_open > 0 else 0.0
            except Exception:
                pct_move = 0.0

            prev_break_high = rolling_high(high_s, BREAKOUT_LEN)
            cond_breakout   = (prev_break_high is not None) and (price > prev_break_high)
            cond_momentum   = (pct_move >= MIN_PCT_MOVE) and (rsi_val >= MIN_RSI_MOM)

            # Aggressiv: kein Trendfilter; NUR BUYs, Exits via TradersPost SL/TP/Trailing
            buy  = cond_breakout and cond_momentum

            print(f"[{t}] price={price:.2f} rsi={rsi_val:.2f} pct={pct_move:.2f}% "
                  f"prevHigh{BREAKOUT_LEN}={prev_break_high} breakout={cond_breakout} mom={cond_momentum} buy={buy}")

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

        except Exception as e:
            print(f"[WARN] {t}: {e}")
            traceback.print_exc()

        # sanfte Drossel außerhalb try/except
        time.sleep(0.15)

    save_state(state)
    print("[DONE] cycle complete")

if __name__ == "__main__":
    run()
