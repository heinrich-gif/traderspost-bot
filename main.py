#!/usr/bin/env python3
import os
import re
import json
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ===== CONFIG =====
PRICE_MAX     = float(os.getenv("PENNY_PRICE_MAX", 5.0))
RELVOL_MIN    = float(os.getenv("PENNY_REL_VOL_MIN", 2.0))
AVGVOL_MIN    = int(os.getenv("PENNY_AVG_VOL_MIN", 500_000))
RANGE_PCT_MIN = float(os.getenv("PENNY_RANGE_PCT_MIN", 5.0))
TIMEFRAME     = os.getenv("TIMEFRAME", "5m")
ACCOUNT_EQUITY= float(os.getenv("ACCOUNT_EQUITY", 1000))
RISK_PCT      = float(os.getenv("RISK_PCT", 0.25))
KI_WEBHOOK_URL= os.getenv("KI_WEBHOOK_URL", "").strip()
TP_WEBHOOK_URL= os.getenv("TP_WEBHOOK_URL", "").strip()

BREAKOUT_LEN  = 20
TOLERANCE_PCT = 0.3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finviz.com/screener.ashx?v=111"
}

# ===== FUNKTIONEN =====
def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}")

def scan_finviz(price_max=PRICE_MAX, relvol_min=RELVOL_MIN, avgvol_min=AVGVOL_MIN, max_pages=10, pause=0.5):
    f_price = f"sh_price_u{int(price_max)}"
    f_relvo = f"sh_relvol_o{int(relvol_min)}"
    f_avgvo = f"sh_avgvol_o{int(avgvol_min // 1000)}"

    base = "https://finviz.com/screener.ashx"
    tickers = []

    for page in range(max_pages):
        r_start = page * 20 + 1
        params = {"v": 111, "f": ",".join([f_price, f_relvo, f_avgvo]), "r": r_start}
        try:
            resp = requests.get(base, params=params, headers=HEADERS, timeout=15)
            html = resp.text
            log(f"[SCAN] page={page+1} url={resp.url} len={len(html)}")

            # 1) Primär: Pandas HTML
            page_tickers = []
            try:
                tables = pd.read_html(pd.io.common.StringIO(html))
                for df in tables:
                    if "Ticker" in df.columns:
                        page_tickers = [str(t) for t in df["Ticker"].dropna().astype(str).tolist()]
                        break
            except ValueError:
                pass

            # 2) Fallback: Regex
            if not page_tickers:
                m = re.findall(r"/quote\.ashx\?t=([A-Z0-9\.\-]{1,10})", html, flags=re.I)
                page_tickers = [t.upper() for t in m if 1 <= len(t) <= 8 and t.isalnum()]
                log(f"[SCAN] fallback tickers found: {len(page_tickers)}")

            tickers.extend(page_tickers)
            if not page_tickers or len(page_tickers) < 20:
                break

        except Exception as e:
            log(f"[SCAN] Finviz-Fehler (Seite {page+1}): {e}")
            break

        time.sleep(pause)

    uniq = []
    seen = set()
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    log(f"[SCAN] Finviz-Kandidaten: {len(uniq)} -> {uniq[:20]} …")
    return uniq

def last_n_closes(df, n=50):
    s = df["Close"].tail(n)
    return [round(float(x), 6) for x in s.tolist()]

def analyze_with_ki(signal: dict) -> dict:
    if not KI_WEBHOOK_URL:
        return {}
    try:
        resp = requests.post(KI_WEBHOOK_URL, json=signal, timeout=20)
        if resp.ok:
            return resp.json()
        else:
            log(f"[KI] Fehler {resp.status_code}: {resp.text}")
    except Exception as e:
        log(f"[KI] Exception: {e}")
    return {}

def send_to_traderspost(order: dict):
    if not TP_WEBHOOK_URL:
        return
    try:
        resp = requests.post(TP_WEBHOOK_URL, json=order, timeout=15)
        log(f"[TP] {order['symbol']} -> {order.get('qty', '?')} {resp.text}")
    except Exception as e:
        log(f"[TP] Exception: {e}")

def evaluate_ticker(df, symbol):
    price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    prev_high = float(df["High"].tail(BREAKOUT_LEN).max())
    pct_move = round((price - prev_close) / prev_close * 100, 2)
    rsi = calc_rsi(df["Close"])

    breakout = price >= prev_high * (1 - TOLERANCE_PCT / 100)
    momentum = price > df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1]

    buy_raw = (breakout and momentum and rsi <= 65 and pct_move >= 0.5)
    rec = "BUY" if buy_raw else "SELL" if rsi >= 80 else "HOLD"

    signal = {
        "symbol": symbol,
        "price": price,
        "rsi": round(rsi, 2),
        "pct_move": pct_move,
        "breakout": breakout,
        "momentum": momentum,
        "recommendation": rec,
        "spark": last_n_closes(df)
    }

    if rec == "BUY":
        ki_data = analyze_with_ki(signal)
        signal.update(ki_data)
        if ki_data:
            order = {
                "symbol": symbol,
                "action": "buy",
                "qty": ki_data.get("qty", 10),
                "stop_loss_percent": ki_data.get("stop_loss", 0.09),
                "take_profit_percent": ki_data.get("take_profit", 0.25),
                "trailing_stop_percent": ki_data.get("trailing", 0.06)
            }
            send_to_traderspost(order)

    return signal

def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def run():
    tickers = scan_finviz()
    final_signals = []
    for t in tickers:
        try:
            log(f"[{t}] downloading…")
            df = yf.download(t, period="5d", interval="5m", progress=False)
            if df.empty:
                continue
            df["EMA50"] = df["Close"].ewm(span=50).mean()
            df["EMA200"] = df["Close"].ewm(span=200).mean()
            sig = evaluate_ticker(df, t)
            final_signals.append(sig)
            log(f"[{t}] {sig}")
        except Exception as e:
            log(f"[ERR] {t}: {e}")

    # WRITE
    os.makedirs("state", exist_ok=True)
    with open("state/signals.json", "w") as f:
        json.dump(final_signals, f, indent=2)
    log(f"[WRITE] state/signals.json -> {len(final_signals)} Zeilen")

if __name__ == "__main__":
    log(">>> Starting main.py …")
    run()
