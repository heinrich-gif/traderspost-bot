import os
import re
import json
import time
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ==================== CONFIG ====================
PRICE_MAX = 5.0
RELVOL_MIN = 2.0
AVGVOL_MIN = 500_000

TICKERS_FILE = Path("tickers.txt")
STATE_FILE = Path("state/signals.json")

# ==================== LOGGING ====================
def log(msg):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)

# ==================== Ticker-Quelle ====================
def get_tickers_from_file():
    if TICKERS_FILE.exists():
        xs = [l.strip().upper() for l in TICKERS_FILE.read_text().splitlines()]
        xs = [x for x in xs if x and not x.startswith("#")]
        if xs:
            log(f"[TICKERS] using tickers.txt -> {xs}")
            return xs
    return []

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finviz.com/screener.ashx?v=111"
}

def scan_finviz(price_max=5.0, relvol_min=2.0, avgvol_min=500_000, max_pages=10, pause=0.6):
    f_price = f"sh_price_u{int(price_max)}"
    f_relvo = f"sh_relvol_o{int(relvol_min)}"
    f_avgvo = f"sh_avgvol_o{int(avgvol_min // 1000)}"

    base = "https://finviz.com/screener.ashx"
    seen, tickers = set(), []
    for page in range(max_pages):
        r_start = page * 20 + 1
        params = {"v": 111, "f": ",".join([f_price, f_relvo, f_avgvo]), "r": r_start}
        try:
            resp = requests.get(base, params=params, headers=HEADERS, timeout=15)
            html = resp.text
            log(f"[SCAN] page={page+1} url={resp.url} len={len(html)}")
            page_tickers = re.findall(
                r'class="screener-link-primary"[^>]*>\s*([A-Z0-9.\-]{1,8})\s*</a>',
                html, flags=re.I
            )
            page_tickers = [t.upper() for t in page_tickers if t.isalnum() and 1 <= len(t) <= 8]
            page_tickers = [t for t in page_tickers if not any(ch in t for ch in "%()")]
            added = 0
            for t in page_tickers:
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
                    added += 1
            log(f"[SCAN] page {page+1}: +{added} tickers, total={len(tickers)}")
            if len(page_tickers) < 20:
                break
        except Exception as e:
            log(f"[SCAN] Finviz-Fehler (Seite {page+1}): {e}")
            break
        time.sleep(pause)

    log(f"[SCAN] Finviz-Kandidaten: {len(tickers)} -> {tickers[:20]} …")
    return tickers

# ==================== Analyse ====================
def evaluate_ticker(df, symbol):
    try:
        price = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        pct_move = round((price - prev_close) / prev_close * 100, 2)
        rsi = calc_rsi(df["Close"], 14)
        prev_high = float(df["High"].tail(20).max())
        tol = 0.003
        breakout = price >= prev_high * (1 - tol)
        momentum = rsi > 50 and pct_move > 0
        buy_raw = breakout and momentum and rsi < 70
        rec = "BUY" if buy_raw else "HOLD"
        return {
            "symbol": symbol,
            "price": price,
            "pct_move": pct_move,
            "rsi": round(rsi, 2),
            "breakout": breakout,
            "momentum": momentum,
            "recommendation": rec
        }
    except Exception as e:
        log(f"[EVAL-ERR] {symbol}: {e}")
        return None

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)).iloc[-1]

# ==================== Hauptlauf ====================
def run():
    tickers = get_tickers_from_file()
    if not tickers:
        tickers = scan_finviz(price_max=PRICE_MAX, relvol_min=RELVOL_MIN, avgvol_min=AVGVOL_MIN)

    final_signals = []
    for t in tickers:
        try:
            log(f"[{t}] downloading…")
            df = yf.download(t, period="5d", interval="5m", progress=False)
            if df.empty:
                continue
            sig = evaluate_ticker(df, t)
            if sig:
                final_signals.append(sig)
                log(f"[{t}] {sig}")
        except Exception as e:
            log(f"[ERR] {t}: {e}")

    os.makedirs("state", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(final_signals, f, indent=2)
    log(f"[WRITE] {STATE_FILE} -> {len(final_signals)} Zeilen")

if __name__ == "__main__":
    log(">>> Starting main.py …")
    run()
