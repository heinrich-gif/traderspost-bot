#!/usr/bin/env python3
import os, json, time, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# -------------------- Config / ENV --------------------
# Scanner / Filter
PRICE_MAX         = float(os.getenv("PENNY_PRICE_MAX", "5"))           # $-Kappung für Finviz
RELVOL_MIN        = float(os.getenv("PENNY_REL_VOL_MIN", "2"))          # min Relative Volume
AVGVOL_MIN        = int(os.getenv("PENNY_AVG_VOL_MIN", "500000"))       # min Avg Vol
RANGE_PCT_MIN     = float(os.getenv("PENNY_RANGE_PCT_MIN", "5"))        # min Tages-Range in %
PENNY_MAX_TICKERS = int(os.getenv("PENNY_MAX_TICKERS", "30"))

# Evaluation
BREAKOUT_LEN      = int(os.getenv("BREAKOUT_LEN", "20"))                # n-Perioden-High
NEAR_BREAKOUT_PCT = float(os.getenv("NEAR_BREAKOUT_PCT", "0.3"))        # Toleranz um n-High (in %)
DELTA_MIN         = float(os.getenv("BUY_DELTA_MIN", "0.2"))            # min Intraday Bar-Delta (%)
RSI_LEN           = int(os.getenv("RSI_LEN", "14"))
RSI_BUY_MAX       = float(os.getenv("RSI_BUY_MAX", "70"))
DAY_DELTA_MIN     = float(os.getenv("DAY_DELTA_MIN", "10"))             # min Tagesänderung (%); 0 = aus

# KI / Orders / Risk
KI_WEBHOOK_URL    = os.getenv("KI_WEBHOOK_URL", "").strip()
ALWAYS_KI         = os.getenv("ALWAYS_KI", "false").lower() in ("1","true","yes")
TP_WEBHOOK_URL    = os.getenv("TP_WEBHOOK_URL", "").strip()
ACCOUNT_EQUITY    = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_PCT          = float(os.getenv("RISK_PCT", "1.0"))
DEFAULT_QTY       = int(os.getenv("POSITION_QTY", "10"))

# Output
DOCS_DIR          = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE        = DOCS_DIR / "signals.json"
EXITS_FILE        = DOCS_DIR / "exits.json"

# yfinance Defaults
YF_INTRA_PERIOD   = os.getenv("PENNY_YF_PERIOD", "5d")
YF_INTRA_INTERVAL = os.getenv("PENNY_YF_INTERVAL", "5m")

# -------------------- Utils --------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str):
    print(f"[{now_utc_iso()}] {msg}", flush=True)

def series_last_float(ser: pd.Series, idx: int=-1) -> Optional[float]:
    try:
        return float(ser.iloc[idx])
    except Exception:
        return None

def rsi(close: pd.Series, period: int) -> float:
    # klassisches RSI (Wilder)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    return float(out.iloc[-1])

def get_prev_close_daily(ticker: str) -> Optional[float]:
    try:
        df = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True, prepost=False, threads=False)
        if df is None or len(df) < 2:
            return None
        return float(df["Close"].iloc[-2])
    except Exception:
        return None

def spark_from_series(close: pd.Series, n: int=50) -> List[float]:
    s = close.tail(n).astype(float).tolist()
    return [round(x, 6) for x in s]

# -------------------- Finviz Scan (HTML) --------------------
def scan_finviz(price_max=PRICE_MAX, relvol_min=RELVOL_MIN, avgvol_min=AVGVOL_MIN, limit=PENNY_MAX_TICKERS) -> List[str]:
    """
    Holt Ticker aus Finviz per read_html + HTML-Filter.
    Wir verwenden die Filter via URL:
      sh_price_u{max}, sh_relvol_o{min}, sh_avgvol_o{min}
    """
    base = "https://finviz.com/screener.ashx"
    params = f"v=111&f=sh_price_u{int(price_max)}%2Csh_relvol_o{relvol_min}%2Csh_avgvol_o{int(avgvol_min)}&r=1"
    url = f"{base}?{params}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        log(f"[SCAN] page=1 url={url} len={len(resp.text)}")
        # Tabellen ziehen
        tables = pd.read_html(resp.text)
        # Ticker steht i. d. R. in der großen Ergebnistabelle in Spalte "Ticker"
        tickers: List[str] = []
        for t in tables:
            cols = [c for c in t.columns]
            if any("Ticker" == str(c) for c in cols):
                tickers += [str(x).strip().upper() for x in t["Ticker"].tolist() if isinstance(x, str)]
        # dedup + begrenzen
        out = []
        for tk in tickers:
            if tk not in out and tk.isalnum():
                out.append(tk)
            if len(out) >= limit:
                break
        log(f"[SCAN] Finviz-Kandidaten: {len(out)} -> {out[:18]} …")
        return out
    except Exception as e:
        log(f"[SCAN-ERR] Finviz: {e}")
        return []

def get_tickers_from_file(path="tickers.txt") -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    raw = [ln.strip().upper() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    return [t for t in raw if t and not t.startswith("#")]

# -------------------- KI / TP / Exits --------------------
def analyze_with_ki(signal: Dict[str, Any]) -> Dict[str, Any]:
    if not KI_WEBHOOK_URL:
        return {}
    payload = {
        "symbol": signal["symbol"],
        "price": signal["price"],
        "recommendation": signal["recommendation"],
        "rsi": signal["rsi"],
        "pct_move": signal["pct_move"],
        "day_change": signal.get("day_change"),
        "breakout": signal["breakout"],
        "momentum": signal["momentum"],
        "equity": ACCOUNT_EQUITY,
        "risk_pct": RISK_PCT,
    }
    try:
        r = requests.post(KI_WEBHOOK_URL, json=payload, timeout=15)
        if r.ok:
            data = r.json()
            # Wir akzeptieren sl/tp/trailing als % oder price; qty optional
            out = {}
            for k in ("sl_percent","tp_percent","trailing_percent","sl_price","tp_price","qty_sized","ki_score"):
                if k in data:
                    out[k] = data[k]
            log(f"[KI] {signal['symbol']} -> {data}")
            return out
        else:
            log(f"[KI-ERR] {signal['symbol']} -> {r.status_code} {r.text[:240]}")
            return {}
    except Exception as e:
        log(f"[KI-EXC] {signal['symbol']}: {e}")
        return {}

def size_qty(price: float) -> int:
    # sehr einfache risikobasierte Positionsgröße
    risk_abs = ACCOUNT_EQUITY * (RISK_PCT/100.0)
    # als Proxy nehmen wir 10 % des Kurses als "Stop-Abstand" – konservativ
    stop_proxy = max(price * 0.10, 0.01)
    qty = int(max(1, risk_abs / stop_proxy))
    return qty

def tp_send_buy(sig: Dict[str, Any]):
    if not TP_WEBHOOK_URL:
        return
    qty = int(sig.get("qty_sized") or sig.get("qty") or DEFAULT_QTY)
    payload: Dict[str, Any] = {
        "ticker": sig["symbol"],
        "action": "buy",
        "quantity": qty,
        "order_type": "market",
        "time_in_force": "day",
        "meta": {"source": "gh-actions-bot"}
    }
    # Nur Felder setzen, die existieren (sonst überschreibt TP mit Defaults)
    if sig.get("tp_percent") is not None:       payload["take_profit_percent"] = sig["tp_percent"]
    if sig.get("sl_percent") is not None:       payload["stop_loss_percent"] = sig["sl_percent"]
    if sig.get("trailing_percent") is not None: payload["trailing_stop_percent"] = sig["trailing_percent"]
    if sig.get("tp_price") is not None:         payload["take_profit_price"] = sig["tp_price"]
    if sig.get("sl_price") is not None:         payload["stop_loss_price"] = sig["sl_price"]

    log(f"[TP BUY][payload] {payload}")
    try:
        r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=20)
        log(f"[TP BUY] {sig['symbol']} qty={qty} -> {r.status_code} {r.text[:240]}")
    except Exception as e:
        log(f"[TP BUY] send failed: {e}")

def log_exit(symbol: str, price: float, reason: str):
    try:
        if EXITS_FILE.exists():
            exits = json.loads(EXITS_FILE.read_text(encoding="utf-8"))
            if not isinstance(exits, list): exits = []
        else:
            exits = []
    except Exception:
        exits = []
    exits.append({
        "timestamp": now_utc_iso(),
        "symbol": symbol,
        "price": round(price, 4),
        "reason": reason
    })
    EXITS_FILE.write_text(json.dumps(exits, indent=2), encoding="utf-8")
    log(f"[EXIT] {symbol} @ {price} -> {reason} (exits.json)")

# -------------------- Evaluation --------------------
def evaluate_ticker(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        log(f"[{symbol}] downloading…")
        df = yf.download(symbol, period=YF_INTRA_PERIOD, interval=YF_INTRA_INTERVAL,
                         progress=False, auto_adjust=True, prepost=True, threads=False)
        if df is None or len(df) < (RSI_LEN + BREAKOUT_LEN + 2):
            log(f"[{symbol}] not enough data ({len(df) if df is not None else 0})")
            return None

        close = df["Close"].astype(float)
        high  = df["High"].astype(float)

        price       = series_last_float(close, -1)
        prev_close  = series_last_float(close, -2)
        if price is None or prev_close is None:
            return None

        # Intraday Momentum (letzte Bar vs. vorherige)
        pct_move = ((price - prev_close) / max(1e-9, prev_close)) * 100.0

        # Tages-Change (vs. Vortages-Close)
        prev_day_close = get_prev_close_daily(symbol)
        day_change = None
        if prev_day_close and prev_day_close != 0:
            day_change = ((price - prev_day_close) / prev_day_close) * 100.0

        # Breakout-Check
        prev_high = float(high.tail(BREAKOUT_LEN).max())
        tol_abs   = prev_high * (NEAR_BREAKOUT_PCT / 100.0)
        breakout  = bool(price >= (prev_high - tol_abs))

        # RSI
        rsi_now = rsi(close, RSI_LEN)

        # Momentum
        momentum = bool(pct_move >= DELTA_MIN)

        # Day-Filter
        day_ok = True
        if day_change is not None and DAY_DELTA_MIN > 0:
            day_ok = (day_change >= DAY_DELTA_MIN)

        buy_raw = bool(breakout and momentum and (rsi_now <= RSI_BUY_MAX) and day_ok)
        recommendation = "BUY" if buy_raw else "HOLD"

        log(
            f"[{symbol}] price={price:.4f} rsi={rsi_now:.2f} Δ%={pct_move:.2f} "
            f"DayΔ%={(day_change if day_change is not None else float('nan')):.2f} "
            f"prevHigh{BREAKOUT_LEN}={prev_high:.4f} tol={NEAR_BREAKOUT_PCT:.3f}% "
            f"breakout={breakout} momentum={momentum} day_ok={day_ok} -> {recommendation}"
        )

        rec = {
            "timestamp": now_utc_iso(),
            "symbol": symbol,
            "price": round(price, 4),
            "pct_move": round(pct_move, 2),
            "day_change": round(day_change, 2) if day_change is not None else None,
            "rsi": round(rsi_now, 2),
            "breakout": breakout,
            "momentum": momentum,
            "prev_high_n": BREAKOUT_LEN,
            "prev_high_value": round(prev_high, 4),
            "near_breakout_pct": NEAR_BREAKOUT_PCT,
            "recommendation": recommendation,
            "spark": spark_from_series(close, 50),
        }

        # KI anreichern (bei BUY oder wenn ALWAYS_KI=true)
        run_ki = (recommendation == "BUY") or ALWAYS_KI
        if run_ki:
            enriched = analyze_with_ki(rec)
            if enriched:
                rec.update(enriched)

        # TradersPost nur bei BUY
        if recommendation == "BUY":
            if rec.get("qty_sized") is None:
                rec["qty_sized"] = size_qty(price)
            tp_send_buy(rec)

        return rec

    except Exception as e:
        log(f"[ERR] {symbol}: {e}")
        return None

# -------------------- Main Loop --------------------
def run():
    print(">>> Starting main.py …", flush=True)

    # Ticker-Liste bauen: tickers.txt + Finviz (merge, dedup)
    tickers: List[str] = []
    watch = get_tickers_from_file()
    if watch:
        tickers.extend([t for t in watch if t not in tickers])
    scan  = scan_finviz()
    for t in scan:
        if t not in tickers:
            tickers.append(t)

    if not tickers:
        log("[TICKERS] none found")
        STATE_FILE.write_text("[]", encoding="utf-8")
        return

    log(f"[TICKERS] merged total={len(tickers)} -> {tickers[:20]}")

    results: List[Dict[str, Any]] = []
    for t in tickers:
        rec = evaluate_ticker(t)
        if rec:
            results.append(rec)

    # Schreiben fürs Dashboard
    STATE_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"[WRITE] {STATE_FILE} -> {len(results)} Zeilen")

if __name__ == "__main__":
    run()
