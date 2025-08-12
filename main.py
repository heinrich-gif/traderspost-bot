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

# KI + Risk
KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL", "").strip()
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_PCT       = float(os.getenv("RISK_PCT", "1.0"))  # in %

# TradersPost Webhook (optional)
TP_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL", "").strip()
DEFAULT_QTY    = int(os.getenv("POSITION_QTY", "10"))

TICKERS_FILE  = Path("tickers.txt")
STATE_FILE    = Path("docs/signals.json")

# ===================== Utils / Logging =====================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str) -> None:
    print(f"[{now_utc_iso()}] {msg}", flush=True)

def to_float_scalar(x) -> float:
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float("nan")

def ensure_series(col) -> pd.Series:
    """
    Macht aus df['Close'] o.ä. sicher eine 1D-Series (falls DataFrame geliefert wurde).
    """
    s = col
    if isinstance(s, pd.DataFrame):
        # nimm die erste Spalte
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    return s

def series_last_float(s: pd.Series, idx: int = -1) -> float:
    v = s.iloc[idx]
    return to_float_scalar(v)

def last_n_closes(df: pd.DataFrame, n: int = 50):
    s = ensure_series(df["Close"]).tail(n)
    return [round(float(x), 6) for x in s.tolist()]

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
            page_tickers = [t.upper() for t in page_tickers if 1 <= len(t) <= 8 and t.replace('.', '').replace('-', '').isalnum()]
            page_tickers = [t for t in page_tickers if not any(ch in t for ch in "%()")]
            added = 0
            for t in page_tickers:
                if t not in seen:
                    seen.add(t); tickers.append(t); added += 1
            log(f"[SCAN] page {page+1}: +{added} tickers, total={len(tickers)}")
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
    s = ensure_series(series)
    if len(s) < period + 1:
        return float("nan")
    d = s.diff()
    gain = d.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-d.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    return to_float_scalar(rsi_series.iloc[-1])

# ===================== KI-Analyse (SL/TP/Trailing/Qty) =====================
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
        "spark": signal.get("spark", []),
    }

    for attempt in range(2):
        try:
            r = requests.post(KI_WEBHOOK_URL, json=payload, timeout=20)
            if not r.ok:
                log(f"[KI] HTTP {r.status_code}: {r.text[:300]}")
                return {}
            data = r.json()
            res = data.get("result") if isinstance(data, dict) and "result" in data else data
            if not isinstance(res, dict):
                log("[KI] invalid response shape")
                return {}

            out = {}
            # Schema A
            if "stopLossPct" in res or "takeProfitPct" in res or "trailingStopPct" in res:
                if res.get("stopLossPct") is not None:     out["sl_percent"] = float(res["stopLossPct"])
                if res.get("takeProfitPct") is not None:   out["tp_percent"] = float(res["takeProfitPct"])
                if res.get("trailingStopPct") is not None: out["trailing_percent"] = float(res["trailingStopPct"])
                if res.get("qty") is not None:             out["qty_sized"] = int(res["qty"])

            # Schema B (+ Extras)
            for k in ("sl_percent","tp_percent","trailing_percent","sl_price","tp_price","qty_sized",
                      "equity_used","risk_pct_used","ki_score","ki_min","ki_pass","rationale","analyzed_at"):
                if k in res and res[k] is not None:
                    out[k] = res[k]

            # Guardrails & Ableitung
            price = float(signal["price"])
            slp = out.get("sl_percent"); tpp = out.get("tp_percent"); trp = out.get("trailing_percent")

            if isinstance(slp, (int,float)) and slp < 0: out["sl_percent"] = None
            if isinstance(tpp, (int,float)) and tpp < 0: out["tp_percent"] = None
            if isinstance(trp, (int,float)) and trp < 0: out["trailing_percent"] = None

            if "sl_price" not in out and isinstance(out.get("sl_percent"), (int,float)):
                out["sl_price"] = round(price * (1 - out["sl_percent"]/100), 4)
            if "tp_price" not in out and isinstance(out.get("tp_percent"), (int,float)):
                out["tp_price"] = round(price * (1 + out["tp_percent"]/100), 4)

            if isinstance(slp, (int,float)) and isinstance(tpp, (int,float)) and tpp <= slp * 1.2:
                out["tp_percent"] = round(slp * 1.6, 2)
                out["tp_price"]   = round(price * (1 + out["tp_percent"]/100), 4)

            return out

        except requests.RequestException as e:
            log(f"[KI] attempt {attempt+1} error: {e}")
            time.sleep(0.6)

    return {}

# ===================== TradersPost Versand =====================
def tp_send_buy(sig: dict):
    if not TP_WEBHOOK_URL:
        return

    qty = int(sig.get("qty_sized") or sig.get("qty") or DEFAULT_QTY)

    payload = {
        "ticker": sig["symbol"],
        "action": "buy",
        "quantity": qty,
        "order_type": "market",
        "time_in_force": "day",
        "take_profit_percent": sig.get("tp_percent"),
        "stop_loss_percent": sig.get("sl_percent"),
        "trailing_stop_percent": sig.get("trailing_percent"),
        "take_profit_price": sig.get("tp_price"),
        "stop_loss_price": sig.get("sl_price"),
        "meta": {"source": "gh-actions-bot"}
    }

    try:
        r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=15)
        log(f"[TP BUY] {sig['symbol']} qty={qty} -> {r.status_code} {r.text[:240]}")
    except Exception as e:
        log(f"[TP BUY] send failed: {e}")

# ===================== Evaluierung pro Ticker =====================
def evaluate_ticker(df: pd.DataFrame, symbol: str) -> dict | None:
    if df is None or df.empty:
        return None

    close = ensure_series(df["Close"])
    high  = ensure_series(df["High"])

    if len(close) < 3 or len(high) < 3:
        log(f"[{symbol}] skip: not enough rows (len={len(close)})")
        return None

    price = series_last_float(close, -1)
    prev_close = series_last_float(close, -2)

    if not (price == price) or not (prev_close == prev_close) or prev_close == 0.0:
        log(f"[{symbol}] skip: bad price/prev_close (price={price}, prev_close={prev_close})")
        return None

    pct_move = ((price - prev_close) / prev_close) * 100.0

    # Vorheriges Hoch (ohne letzte Kerze)
    tail = high.tail(BREAKOUT_LEN + 1)
    if len(tail) >= 2:
        prev_high = to_float_scalar(tail.iloc[:-1].max())
    else:
        prev_high = to_float_scalar(high.max())

    tol_abs = prev_high * (TOLERANCE_PCT / 100.0)
    breakout = bool(price >= (prev_high - tol_abs))

    rsi_now = rsi(close, 14)
    momentum = bool(pct_move >= DELTA_MIN)

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
            if not sig:
                continue

            # Sparkline für Dashboard & KI
            sig["spark"] = last_n_closes(df, 50)

            # KI + TP bei BUY
            if sig["recommendation"] == "BUY":
                enriched = analyze_with_ki(sig)
                if enriched:
                    sig.update(enriched)
                if not sig.get("qty_sized"):
                    price = float(sig["price"])
                    risk_abs = ACCOUNT_EQUITY * (RISK_PCT / 100.0)
                    qty = int(max(1, risk_abs / max(price * 0.1, 0.01)))
                    sig["qty_sized"] = qty
                tp_send_buy(sig)

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
