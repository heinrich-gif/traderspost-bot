#!/usr/bin/env python3
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
import yfinance as yf
import pandas as pd

# ===================== Konfiguration =====================
PRICE_MAX     = float(os.getenv("PENNY_PRICE_MAX", 5.0))
RELVOL_MIN    = float(os.getenv("PENNY_REL_VOL_MIN", 2.0))
AVGVOL_MIN    = int(os.getenv("PENNY_AVG_VOL_MIN", 500_000))
TIMEFRAME     = os.getenv("TIMEFRAME", "5m")   # 1m/5m/15m …
BREAKOUT_LEN  = int(os.getenv("BREAKOUT_LEN", "20"))
TOLERANCE_PCT = float(os.getenv("NEAR_BREAKOUT_PCT", "0.3"))  # 0.3%
RSI_BUY_MAX   = float(os.getenv("BUY_RSI_MAX", "70"))
DELTA_MIN     = float(os.getenv("BUY_DELTA_MIN", "0.0"))      # Mindest-Δ% für Momentum

TICKERS_FILE  = Path("tickers.txt")
STATE_FILE    = Path("state/signals.json")

# ===================== Utils / Logging =====================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str) -> None:
    print(f"[{now_utc_iso()}] {msg}", flush=True)

def to_float_scalar(x) -> float:
    """
    Wandelt ein beliebiges Pandas/NumPy-Scalar in float um.
    """
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float("nan")

def series_last_float(s: pd.Series, idx: int = -1) -> float:
    """
    Sichere letzte(n) Wert(e) aus einer Series als float.
    """
    v = s.iloc[idx]
    return to_float_scalar(v)

# ===================== Tickerquelle =====================
def get_tickers_from_file() -> list[str]:
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

def scan_finviz(price_max=PRICE_MAX, relvol_min=RELVOL_MIN, avgvol_min=AVGVOL_MIN,
                max_pages=10, pause=0.6) -> list[str]:
    """
    Robuster Finviz-Scanner (Regex auf screener-link-primary).
    Nur Fallback, wenn tickers.txt leer ist.
    """
    f_price = f"sh_price_u{int(price_max)}"
    f_relvo = f"sh_relvol_o{int(relvol_min)}"         # Finviz: Ganzzahl-Schwellen
    f_avgvo = f"sh_avgvol_o{int(avgvol_min // 1000)}" # 500k -> 500

    base = "https://finviz.com/screener.ashx"
    seen, tickers = set(), []

    for page in range(max_pages):
        r_start = page * 20 + 1
        params = {"v": 111, "f": ",".join([f_price, f_relvo, f_avgvo]), "r": r_start}
        try:
            resp = requests.get(base, params=params, headers=HEADERS, timeout=15)
            html = resp.text
            log(f"[SCAN] page={page+1} url={resp.url} len={len(html)}")

            # Nur Primär-Links der Ergebnistabelle:
            # <a class="screener-link-primary" ...>TICKER</a>
            page_tickers = re.findall(
                r'class="screener-link-primary"[^>]*>\s*([A-Z0-9.\-]{1,8})\s*</a>',
                html, flags=re.I
            )
            # filtern: echte Tickers (keine Prozent/Leerwörter)
            page_tickers = [t.upper() for t in page_tickers if 1 <= len(t) <= 8 and t.replace('.', '').replace('-', '').isalnum()]
            page_tickers = [t for t in page_tickers if not any(ch in t for ch in "%()")]

            added = 0
            for t in page_tickers:
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
                    added += 1
            log(f"[SCAN] page {page+1}: +{added} tickers, total={len(tickers)}")

            # Ende, wenn weniger als 20 Einträge auf Seite
            if len(page_tickers) < 20:
                break

        except Exception as e:
            log(f"[SCAN] Finviz-Fehler (Seite {page+1}): {e}")
            break

        time.sleep(pause)

    log(f"[SCAN] Finviz-Kandidaten: {len(tickers)} -> {tickers[:20]} …")
    return tickers

# ===================== Technische Indikatoren =====================
def rsi(series: pd.Series, period: int = 14) -> float:
    s = pd.to_numeric(series, errors="coerce")
    d = s.diff()
    gain = d.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-d.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    return to_float_scalar(rsi_series.iloc[-1])

# ===================== Evaluierung pro Ticker =====================
def evaluate_ticker(df: pd.DataFrame, symbol: str) -> dict | None:
    if df is None or df.empty:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce")
    high  = pd.to_numeric(df["High"], errors="coerce")

    price = series_last_float(close, -1)
    prev_close = series_last_float(close, -2)

    # Schutz, falls NaN
    if not (price == price) or not (prev_close == prev_close) or prev_close == 0.0:
        log(f"[{symbol}] skip: bad price/prev_close (price={price}, prev_close={prev_close})")
        return None

    pct_move = ((price - prev_close) / prev_close) * 100.0

    # Vorheriges Hoch (ohne letzte Kerze) + Toleranz
    tail = high.tail(BREAKOUT_LEN + 1)
    if len(tail) >= 2:
        prev_high = to_float_scalar(tail.iloc[:-1].max())
    else:
        prev_high = to_float_scalar(high.max())

    tol_abs = prev_high * (TOLERANCE_PCT / 100.0)
    breakout = bool(price >= (prev_high - tol_abs))

    rsi_now = rsi(close, 14)
    momentum = bool(pct_move >= DELTA_MIN)

    # BUY-Bedingung (aggressiv, aber begrenzt über RSI)
    buy_raw = bool(breakout and momentum and (rsi_now <= RSI_BUY_MAX))
    recommendation = "BUY" if buy_raw else "HOLD"

    log(
        f"[{symbol}] price={price:.4f} rsi={rsi_now:.2f} Δ%={pct_move:.2f} "
        f"prevHigh{BREAKOUT_LEN}={prev_high:.4f} tol={TOLERANCE_PCT:.3f}% "
        f"breakout={breakout} momentum={momentum} -> {recommendation}"
    )

    return {
        "timestamp": now_utc_iso(),
        "symbol": symbol,
        "price": round(price, 4),
        "pct_move": round(pct_move, 2),
        "rsi": round(rsi_now, 2),
        "breakout": breakout,
        "momentum": momentum,
        "prev_high_n": BREAKOUT_LEN,
        "prev_high_value": round(prev_high, 4),
        "near_breakout_pct": TOLERANCE_PCT,
        "recommendation": recommendation
    }

# ===================== Main-Run =====================
def run():
    log(">>> Starting main.py …")

    # 1) tickers.txt bevorzugen
    tickers = get_tickers_from_file()

    # 2) Fallback: Finviz-Scan
    if not tickers:
        tickers = scan_finviz(PRICE_MAX, RELVOL_MIN, AVGVOL_MIN)

    final_signals = []

    for t in tickers:
        try:
            log(f"[{t}] downloading…")
            df = yf.download(
                t, period="5d", interval=TIMEFRAME,
                progress=False, auto_adjust=True, prepost=True,
                threads=False
            )
            if df is None or df.empty:
                log(f"[{t}] empty dataframe")
                continue

            sig = evaluate_ticker(df, t)
            if sig:
                final_signals.append(sig)

        except Exception as e:
            log(f"[ERR] {t}: {e}")

        time.sleep(0.12)  # sanfte Rate

    # Schreiben
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(final_signals, indent=2))
    log(f"[WRITE] {STATE_FILE} -> {len(final_signals)} Zeilen")

if __name__ == "__main__":
    run()
