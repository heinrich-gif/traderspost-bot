#!/usr/bin/env python3
import os, json, math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# -------------------- Config / ENV --------------------
PRICE_MAX         = float(os.getenv("PENNY_PRICE_MAX", "5"))
RELVOL_MIN        = float(os.getenv("PENNY_REL_VOL_MIN", "2"))
AVGVOL_MIN        = int(os.getenv("PENNY_AVG_VOL_MIN", "500000"))
RANGE_PCT_MIN     = float(os.getenv("PENNY_RANGE_PCT_MIN", "5"))
PENNY_MAX_TICKERS = int(os.getenv("PENNY_MAX_TICKERS", "30"))

BREAKOUT_LEN      = int(os.getenv("BREAKOUT_LEN", "20"))
NEAR_BREAKOUT_PCT = float(os.getenv("NEAR_BREAKOUT_PCT", "0.3"))
DELTA_MIN         = float(os.getenv("BUY_DELTA_MIN", "0.2"))
RSI_LEN           = int(os.getenv("RSI_LEN", "14"))
RSI_BUY_MAX       = float(os.getenv("RSI_BUY_MAX", "70"))
DAY_DELTA_MIN     = float(os.getenv("DAY_DELTA_MIN", "10"))  # 0 = aus

KI_WEBHOOK_URL    = os.getenv("KI_WEBHOOK_URL", "").strip()
ALWAYS_KI         = os.getenv("ALWAYS_KI", "false").lower() in ("1","true","yes")
TP_WEBHOOK_URL    = os.getenv("TP_WEBHOOK_URL", "").strip()
ACCOUNT_EQUITY    = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_PCT          = float(os.getenv("RISK_PCT", "1.0"))
DEFAULT_QTY       = int(os.getenv("POSITION_QTY", "10"))

DOCS_DIR          = Path("docs"); DOCS_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE        = DOCS_DIR / "signals.json"
EXITS_FILE        = DOCS_DIR / "exits.json"

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
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    return float(out.iloc[-1])

def get_prev_close_daily(ticker: str) -> Optional[float]:
    try:
        df = yf.download(ticker, period="2d", interval="1d", progress=False,
                         auto_adjust=True, prepost=False, threads=False)
        if df is None or len(df) < 2: return None
        return float(df["Close"].iloc[-2])
    except Exception:
        return None

def to_series(obj) -> pd.Series:
    """Sicherstellen, dass wir wirklich eine Series bekommen (nicht DataFrame/MultiIndex)."""
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        # nimm die erste Spalte (typischerweise die einzige)
        return obj.iloc[:, 0].squeeze()
    # Fallback: in Series casten
    return pd.Series(obj)

def spark_from_series(close_like, n: int=50) -> List[float]:
    s = to_series(close_like)
    s = s.tail(n).astype(float)
    # Series hat .tolist(); DataFrame nicht – aber wir sind jetzt sicher Series
    return [round(x, 6) for x in s.tolist()]

# -------------------- Finviz Scan --------------------
def scan_finviz(price_max=PRICE_MAX, relvol_min=RELVOL_MIN, avgvol_min=AVGVOL_MIN, limit=PENNY_MAX_TICKERS) -> List[str]:
    base = "https://finviz.com/screener.ashx"
    params = f"v=111&f=sh_price_u{int(price_max)}%2Csh_relvol_o{relvol_min}%2Csh_avgvol_o{int(avgvol_min)}&r=1"
    url = f"{base}?{params}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        log(f"[SCAN] page=1 url={url} len={len(resp.text)}")
        tables = pd.read_html(resp.text)
        tickers: List[str] = []
        for t in tables:
            if any(str(c) == "Ticker" for c in t.columns):
                tickers += [str(x).strip().upper() for x in t["Ticker"].tolist() if isinstance(x, str)]
        out = []
        for tk in tickers:
            if tk not in out and tk.isalnum():
                out.append(tk)
            if len(out) >= limit: break
        log(f"[SCAN] Finviz-Kandidaten: {len(out)} -> {out[:18]} …")
        return out
    except Exception as e:
        log(f"[SCAN-ERR] Finviz: {e}")
        return []

def get_tickers_from_file(path="tickers.txt") -> List[str]:
    p = Path(path)
    if not p.exists(): return []
    raw = [ln.strip().upper() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
    return [t for t in raw if t and not t.startswith("#")]

# -------------------- KI / TP / Exits --------------------
def analyze_with_ki(signal: Dict[str, Any]) -> Dict[str, Any]:
    if not KI_WEBHOOK_URL: return {}
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
            out = {}
            for k in ("sl_percent","tp_percent","trailing_percent","sl_price","tp_price","qty_sized","ki_score"):
                if k in data: out[k] = data[k]
            log(f"[KI] {signal['symbol']} -> {data}")
            return out
        log(f"[KI-ERR] {signal['symbol']} -> {r.status_code} {r.text[:240]}")
        return {}
    except Exception as e:
        log(f"[KI-EXC] {signal['symbol']}: {e}")
        return {}

def size_qty(price: float) -> int:
    risk_abs = ACCOUNT_EQUITY * (RISK_PCT/100.0)
    stop_proxy = max(price * 0.10, 0.01)
    return int(max(1, risk_abs / stop_proxy))

def tp_send_buy(sig: Dict[str, Any]):
    if not TP_WEBHOOK_URL: return
    qty = int(sig.get("qty_sized") or sig.get("qty") or DEFAULT_QTY)
    payload: Dict[str, Any] = {
        "ticker": sig["symbol"],
        "action": "buy",
        "quantity": qty,
        "order_type": "market",
        "time_in_force": "day",
        "meta": {"source": "gh-actions-bot"}
    }
    if sig.get("tp_percent") is not None:       payload["take_profit_percent"] = sig["tp_percent"]
    if sig.get("sl_percent") is not None:       payload["stop_loss_percent"]  = sig["sl_percent"]
    if sig.get("trailing_percent") is not None: payload["trailing_stop_percent"] = sig["trailing_percent"]
    if sig.get("tp_price") is not None:         payload["take_profit_price"]  = sig["tp_price"]
    if sig.get("sl_price") is not None:         payload["stop_loss_price"]    = sig["sl_price"]

    log(f"[TP BUY][payload] {payload}")
    try:
        r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=20)
        log(f"[TP BUY] {sig['symbol']} qty={qty} -> {r.status_code} {r.text[:240]}")
    except Exception as e:
        log(f"[TP BUY] send failed: {e}")

def log_exit(symbol: str, price: float, reason: str):
    try:
        exits = json.loads(EXITS_FILE.read_text(encoding="utf-8")) if EXITS_FILE.exists() else []
        if not isinstance(exits, list): exits = []
    except Exception:
        exits = []
    exits.append({"timestamp": now_utc_iso(), "symbol": symbol, "price": round(price, 4), "reason": reason})
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

        # Robust gegen DataFrame/MultiIndex
        close = to_series(df["Close"]).astype(float)
        high  = to_series(df["High"]).astype(float)

        price       = series_last_float(close, -1)
        prev_close  = series_last_float(close, -2)
        if price is None or prev_close is None:
            return None

        pct_move = ((price - prev_close) / max(1e-9, prev_close)) * 100.0

        prev_day_close = get_prev_close_daily(symbol)
        day_change = None
        if prev_day_close and prev_day_close != 0:
            day_change = ((price - prev_day_close) / prev_day_close) * 100.0

        prev_high = float(high.tail(BREAKOUT_LEN).max())
        tol_abs   = prev_high * (NEAR_BREAKOUT_PCT / 100.0)
        breakout  = bool(price >= (prev_high - tol_abs))

        rsi_now = rsi(close, RSI_LEN)
        momentum = bool(pct_move >= DELTA_MIN)

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

        run_ki = (recommendation == "BUY") or ALWAYS_KI
        if run_ki:
            enriched = analyze_with_ki(rec)
            if enriched: rec.update(enriched)

        if recommendation == "BUY":
            if rec.get("qty_sized") is None:
                rec["qty_sized"] = size_qty(price)
            tp_send_buy(rec)

        return rec

    except Exception as e:
        log(f"[ERR] {symbol}: {e}")
        return None

# -------------------- Main --------------------
def run():
    print(">>> Starting main.py …", flush=True)

    tickers: List[str] = []
    watch = get_tickers_from_file()
    if watch: tickers.extend([t for t in watch if t not in tickers])
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
        if rec: results.append(rec)

    STATE_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"[WRITE] {STATE_FILE} -> {len(results)} Zeilen")

if __name__ == "__main__":
    run()
