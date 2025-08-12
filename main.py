#!/usr/bin/env python3
import os
import time
import math
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ========= Konfiguration =========
PRICE_MAX = float(os.getenv("PENNY_PRICE_MAX", 5))
RELVOL_MIN = float(os.getenv("PENNY_REL_VOL_MIN", 2))
AVG_VOL_MIN = int(os.getenv("PENNY_AVG_VOL_MIN", 500_000))
RANGE_PCT_MIN = float(os.getenv("PENNY_RANGE_PCT_MIN", 5))
MAX_TICKERS = int(os.getenv("PENNY_MAX_TICKERS", 30))
KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL", "")
TP_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL", "")
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", 1000))
RISK_PCT = float(os.getenv("RISK_PCT", 0.25))

BREAKOUT_LEN = 20
TOL_PCT = 0.3

STATE_FILE = "state/signals.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# ========= Hilfsfunktionen =========
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

def log(msg):
    print(f"[{now_utc()}] {msg}")

def scan_finviz(price_max=PRICE_MAX, relvol_min=RELVOL_MIN, avgvol_min=AVG_VOL_MIN, max_pages=10, pause=0.4):
    f_price = f"sh_price_u{int(price_max)}"
    f_relvo = f"sh_relvol_o{int(relvol_min)}"
    f_avgvo = f"sh_avgvol_o{int(avgvol_min // 1000)}"

    tickers = []
    base = "https://finviz.com/screener.ashx"
    for page in range(max_pages):
        r_start = page * 20 + 1
        params = {
            "v": 111,
            "f": ",".join([f_price, f_relvo, f_avgvo]),
            "r": r_start
        }
        try:
            resp = requests.get(base, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            if not tables:
                break
            df = tables[0]
            if "Ticker" not in df.columns:
                df.columns = df.iloc[0]
                df = df[1:]
            if "Ticker" not in df.columns:
                break
            page_tickers = [str(t) for t in df["Ticker"].dropna().astype(str).tolist()]
            if not page_tickers:
                break
            tickers.extend(page_tickers)
            if len(page_tickers) < 20:
                break
        except Exception as e:
            log(f"[SCAN] Finviz-Fehler (Seite {page+1}): {e}")
            break
        time.sleep(pause)

    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    log(f"[SCAN] Finviz-Kandidaten: {len(uniq)} -> {uniq[:20]} …")
    return uniq

def last_n_closes(df, n):
    s = df["Close"].tail(n)
    return [round(float(x), 6) for x in s.tolist()]

def analyze_with_ki(signal: dict) -> dict:
    if not KI_WEBHOOK_URL:
        return {}
    try:
        resp = requests.post(KI_WEBHOOK_URL, json=signal, timeout=15)
        if resp.ok:
            return resp.json()
        else:
            log(f"[KI] Fehler {resp.status_code}: {resp.text}")
    except Exception as e:
        log(f"[KI] Exception: {e}")
    return {}

def evaluate_ticker(df, ticker):
    price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    pct_move = round((price - prev_close) / prev_close * 100, 2)
    prev_high = float(df["High"].tail(BREAKOUT_LEN).max())
    tol = prev_high * TOL_PCT / 100
    breakout = price >= prev_high - tol
    momentum = pct_move >= 0.5
    rsi = calc_rsi(df["Close"], 14)

    buy_raw = breakout and momentum and rsi <= 65
    rec = "BUY" if buy_raw else "HOLD"

    signal = {
        "timestamp": now_utc(),
        "symbol": ticker,
        "recommendation": rec,
        "price": price,
        "pct_move": pct_move,
        "rsi": round(rsi, 2),
        "breakout": breakout,
        "momentum": momentum,
        "prev_high": round(prev_high, 4),
        "spark": last_n_closes(df, 50)
    }

    if rec == "BUY":
        ki_data = analyze_with_ki(signal)
        signal.update(ki_data)

    return signal

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def run():
    tickers = scan_finviz()
    final_tickers = []
    for t in tickers:
        try:
            df = yf.download(t, period="5d", interval="1d", progress=False)
            if df.empty:
                continue
            rng = (df["High"].max() - df["Low"].min()) / df["Low"].min() * 100
            if rng >= RANGE_PCT_MIN:
                final_tickers.append(t)
        except Exception as e:
            log(f"[{t}] yfinance error: {e}")

    log(f"[SCAN] final: {final_tickers[:MAX_TICKERS]}")

    recs = []
    for t in final_tickers[:MAX_TICKERS]:
        try:
            df = yf.download(t, period="6mo", interval="1d", progress=False)
            if df.empty:
                continue
            recs.append(evaluate_ticker(df, t))
        except Exception as e:
            log(f"[{t}] Eval error: {e}")

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(recs, f, indent=2)
    log(f"[WRITE] {STATE_FILE} -> {len(recs)} Zeilen")

if __name__ == "__main__":
    run()
