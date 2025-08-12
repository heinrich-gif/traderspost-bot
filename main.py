import os
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests

# ========= Konfiguration (ENV) =========
TIMEFRAME          = os.getenv("TIMEFRAME", "5m")
POSITION_QTY       = int(os.getenv("POSITION_QTY", "10"))

# Setup-Regeln
BREAKOUT_LEN       = int(os.getenv("BREAKOUT_LEN", "20"))
NEAR_BREAKOUT_PCT  = float(os.getenv("NEAR_BREAKOUT_PCT", "0.3"))  # % Toleranz für Near-Breakout

# Aggressive BUY-Schwellen
BUY_RSI_MAX        = float(os.getenv("BUY_RSI_MAX", "65"))
BUY_DELTA_MIN      = float(os.getenv("BUY_DELTA_MIN", "0.5"))

# Info/Badge (keine Kaufentscheidung)
MIN_PCT_MOVE       = float(os.getenv("MIN_PCT_MOVE", "5.0"))
MIN_RSI_MOM        = float(os.getenv("MIN_RSI_MOM", "50.0"))

# KI-Analyse (SL/TP/Trailing/Qty)
KI_WEBHOOK_URL     = os.getenv("KI_WEBHOOK_URL", "").strip()
ACCOUNT_EQUITY     = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_PCT           = float(os.getenv("RISK_PCT", "1"))

# TradersPost (optional)
TP_WEBHOOK_URL     = os.getenv("TP_WEBHOOK_URL", "").strip()

# Bot-Exit-Engine (Backup-TP/SL/Trailing)
BOT_MANAGE_EXITS   = os.getenv("BOT_MANAGE_EXITS", "1") == "1"
EXIT_CHECK_LOOKBACK= int(os.getenv("EXIT_CHECK_LOOKBACK", "3"))
STATE_DIR          = Path("state")
STATE_FILE         = STATE_DIR / "positions.json"

# Dateien
OUTPUT_PATH        = Path(os.getenv("OUTPUT_PATH", "docs/signals.json"))
TICKERS_FILE       = Path("tickers.txt")


# ========= Zeit / Session =========
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def is_us_extended_utc(dt: datetime) -> bool:
    # Mo–Fr 08:00–24:00 UTC & 00:00–02:00 UTC
    wd = dt.weekday()
    if wd >= 5:
        return False
    mins = dt.hour * 60 + dt.minute
    return (8 * 60 <= mins < 24 * 60) or (0 <= mins < 2 * 60)


# ========= Pandas/YF Hilfen =========
def to_1d_series(obj) -> pd.Series:
    s = obj
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    return s

def close_series(df: pd.DataFrame) -> pd.Series:
    return to_1d_series(df["Close"])

def high_series(df: pd.DataFrame) -> pd.Series:
    return to_1d_series(df["High"])

def last_n_closes(df: pd.DataFrame, n: int = 50):
    s = close_series(df).tail(n)
    return [round(float(v), 6) for v in s.tolist()]

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    s = to_1d_series(series)
    d = s.diff()
    gain  = d.clip(lower=0).ewm(alpha=1/length, min_periods=length).mean()
    loss  = (-d.clip(upper=0)).ewm(alpha=1/length, min_periods=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def download_df(ticker: str, period="60d", interval="5m") -> pd.DataFrame:
    if interval == "1m":
        period = "7d"
    return yf.download(
        ticker, period=period, interval=interval,
        progress=False, auto_adjust=True, prepost=True,
        group_by="column", threads=False
    )


# ========= KI-Analyse (SL/TP/Trailing/Qty) =========
def analyze_with_ki(signal: dict) -> dict:
    """
    Ruft den Cloudflare-Worker (KI_WEBHOOK_URL) auf.
    Unterstützt zwei Schemata:
      A) { stopLossPct, takeProfitPct, trailingStopPct, qty, sl_price?, tp_price? }
      B) { sl_percent, tp_percent, trailing_percent, qty_sized, sl_price?, tp_price? }
    Gibt einheitlich gemappte Felder zurück.
    """
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
        "spark": signal.get("spark", []),  # Worker kann Volatilität daraus schätzen
    }

    for attempt in range(2):
        try:
            r = requests.post(KI_WEBHOOK_URL, json=payload, timeout=20)
            if not r.ok:
                print(f"[KI] HTTP {r.status_code}: {r.text[:240]}")
                return {}
            data = r.json()
            # neuer Worker liefert plain-Objekt oder { ok, result }
            res = data.get("result") if isinstance(data, dict) and "result" in data else data
            if not isinstance(res, dict):
                print("[KI] invalid response shape")
                return {}

            # Mapping auf interne Namen
            out = {}
            # Schema A
            if "stopLossPct" in res or "takeProfitPct" in res or "trailingStopPct" in res:
                if res.get("stopLossPct") is not None:
                    out["sl_percent"] = float(res["stopLossPct"])
                if res.get("takeProfitPct") is not None:
                    out["tp_percent"] = float(res["takeProfitPct"])
                if res.get("trailingStopPct") is not None:
                    out["trailing_percent"] = float(res["trailingStopPct"])
                if res.get("qty") is not None:
                    out["qty_sized"] = int(res["qty"])
            # Schema B (oder ergänzend)
            for k in ("sl_percent","tp_percent","trailing_percent","sl_price","tp_price","qty_sized",
                      "equity_used","risk_pct_used","ki_score","ki_min","ki_pass","rationale","analyzed_at"):
                if k in res and res[k] is not None:
                    out[k] = res[k]

            # Guardrails
            price = float(signal["price"])
            slp = out.get("sl_percent")
            tpp = out.get("tp_percent")
            trp = out.get("trailing_percent")

            if isinstance(slp, (int,float)) and slp < 0: out["sl_percent"] = None
            if isinstance(tpp, (int,float)) and tpp < 0: out["tp_percent"] = None
            if isinstance(trp, (int,float)) and trp < 0: out["trailing_percent"] = None

            # Preise ableiten, falls nur Prozent geliefert wurden
            if "sl_price" not in out and isinstance(out.get("sl_percent"), (int,float)):
                out["sl_price"] = round(price * (1 - out["sl_percent"]/100), 4)
            if "tp_price" not in out and isinstance(out.get("tp_percent"), (int,float)):
                out["tp_price"] = round(price * (1 + out["tp_percent"]/100), 4)

            # Mindest-RR
            if isinstance(slp, (int,float)) and isinstance(tpp, (int,float)) and tpp <= slp * 1.2:
                out["tp_percent"] = round(slp * 1.6, 2)
                out["tp_price"]   = round(price * (1 + out["tp_percent"]/100), 4)

            return out

        except requests.RequestException as e:
            print(f"[KI] attempt {attempt+1} error: {e}")
            time.sleep(0.6)

    return {}


# ========= Exit-State =========
def load_positions() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception as e:
        print(f"[STATE] load failed: {e}")
    return {"positions": {}}

def save_positions(state: dict):
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        print(f"[STATE] save failed: {e}")


# ========= Strategie (Entry) =========
def evaluate_ticker(df: pd.DataFrame, ticker: str) -> dict:
    # Stale-Check
    try:
        last_ts = df.index[-1]
        if hasattr(last_ts, "to_pydatetime"):
            last_ts = last_ts.to_pydatetime()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60.0
        if age_min > 8:
            print(f"[WARN] {ticker} stale data ({age_min:.1f} min alt)")
    except Exception:
        pass

    c = close_series(df); h = high_series(df)
    if c.empty or h.empty or len(c) < 3:
        raise ValueError("Not enough data")

    price = float(c.iloc[-1])
    prev_close = float(c.iloc[-2])
    pct_move = (price - prev_close) / prev_close * 100.0 if prev_close else 0.0

    # vorheriges Hoch (ohne letzte Kerze) + Toleranz
    window = max(1, BREAKOUT_LEN)
    if len(h) > window:
        prev_high_raw = float(h.iloc[-(window+1):-1].max())
    else:
        prev_high_raw = float(h.iloc[:-1].max()) if len(h) > 1 else float(h.max())
    tolerance = prev_high_raw * (NEAR_BREAKOUT_PCT / 100.0)
    near_breakout = (price >= prev_high_raw - tolerance)

    rsi_now = float(rsi(c).iloc[-1])
    momentum = (rsi_now >= MIN_RSI_MOM)

    # Aggressive BUY-Logik
    cond_rsi_delta = (rsi_now <= BUY_RSI_MAX) and (pct_move >= BUY_DELTA_MIN)
    cond_near_bo   = near_breakout
    buy_raw = cond_rsi_delta or cond_near_bo
    recommendation = "BUY" if buy_raw else "HOLD"

    print(
        f"[{ticker}] price={price:.4f} rsi={rsi_now:.2f} Δ%={pct_move:.2f} "
        f"prevHigh{window}={prev_high_raw:.4f} tol={NEAR_BREAKOUT_PCT:.3f}% "
        f"nearBO={near_breakout} | cond_rsi_delta={cond_rsi_delta} cond_near_bo={cond_near_bo} "
        f"-> {recommendation}"
    )

    return {
        "symbol": ticker,
        "price": round(price, 4),
        "rsi": round(rsi_now, 2),
        "pct_move": round(pct_move, 2),
        "prev_high_n": window,
        "prev_high_value": round(prev_high_raw, 4),
        "near_breakout_pct": NEAR_BREAKOUT_PCT,
        "breakout": bool(near_breakout),
        "momentum": bool(momentum),
        "recommendation": recommendation,
        "qty": POSITION_QTY if recommendation == "BUY" else 0,
        "timestamp": now_utc_iso(),
        "spark": last_n_closes(df, 50),
        # werden bei BUY gefüllt:
        "sl_percent": None, "tp_percent": None, "trailing_percent": None,
        "sl_price": None, "tp_price": None, "qty_sized": None,
    }


# ========= TradersPost Versand =========
def tp_send_buy(sig: dict):
    if not TP_WEBHOOK_URL:
        return
    qty = int(sig.get("qty_sized") or sig.get("qty") or POSITION_QTY)
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
        r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=12)
        print(f"[TP BUY] {sig['symbol']} -> {r.status_code} {r.text[:260]}")
    except Exception as e:
        print(f"[TP BUY] send failed: {e}")

def tp_send_sell(symbol: str, qty: int | None = None):
    if not TP_WEBHOOK_URL:
        return
    payload = {
        "ticker": symbol,
        "action": "sell",
        "order_type": "market",
        "time_in_force": "day",
    }
    if qty and qty > 0:
        payload["quantity"] = int(qty)
    try:
        r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=12)
        print(f"[TP SELL] {symbol} -> {r.status_code} {r.text[:260]}")
    except Exception as e:
        print(f"[TP SELL] send failed: {e}")


# ========= Exit-Engine (Backup-TP/SL/Trailing) =========
def update_and_check_exits(state: dict, df_map: dict):
    if not BOT_MANAGE_EXITS:
        return
    positions = state.get("positions", {})
    to_close = []

    for sym, pos in list(positions.items()):
        df = df_map.get(sym)
        if df is None or df.empty:
            continue
        last_price = float(close_series(df).tail(max(2, EXIT_CHECK_LOOKBACK)).iloc[-1])

        entry = float(pos.get("entry", last_price))
        tp = pos.get("tp")
        sl = pos.get("sl")
        trail_pct = pos.get("trail_pct")
        hi_water = float(pos.get("hi_water", entry))

        if last_price > hi_water:
            hi_water = last_price
        trail_stop = hi_water * (1 - float(trail_pct)/100) if trail_pct else None

        trigger = None
        if tp and last_price >= tp:
            trigger = "TP"
        elif sl and last_price <= sl:
            trigger = "SL"
        elif trail_stop and last_price <= trail_stop:
            trigger = "TRAIL"

        pos["hi_water"] = hi_water
        positions[sym] = pos

        if trigger:
            print(f"[EXIT] {sym} trigger={trigger} last={last_price:.4f} tp={tp or '-'} sl={sl or '-'} trail={trail_stop or '-'}")
            to_close.append(sym)

    for sym in to_close:
        qty = positions[sym].get("qty")
        tp_send_sell(sym, qty)
        positions.pop(sym, None)

    state["positions"] = positions


# ========= JSON fürs Dashboard =========
def write_signals(signals: list, path: Path = OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": now_utc_iso(),
        "timeframe": TIMEFRAME,
        "breakout_len": BREAKOUT_LEN,
        "min_pct_move": MIN_PCT_MOVE,
        "min_rsi_mom": MIN_RSI_MOM,
        "buy_logic": {
            "buy_rsi_max": BUY_RSI_MAX,
            "buy_delta_min": BUY_DELTA_MIN,
            "near_breakout_pct": NEAR_BREAKOUT_PCT
        },
        "signals": signals,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[WRITE] {path} -> {len(signals)} Zeilen")


# ========= Main =========
def run():
    now = datetime.now(timezone.utc)
    print(f"[CFG] TF={TIMEFRAME} BUY_RSI_MAX={BUY_RSI_MAX} BUY_DELTA_MIN={BUY_DELTA_MIN} "
          f"NEAR_BO%={NEAR_BREAKOUT_PCT} KI_URL={'set' if KI_WEBHOOK_URL else 'unset'} "
          f"BOT_EXITS={'on' if BOT_MANAGE_EXITS else 'off'}")

    if not is_us_extended_utc(now):
        print(f"[SKIP] Outside US extended session (UTC {now.isoformat()})")
        write_signals([])
        return

    if not TICKERS_FILE.exists():
        print("[ERR] tickers.txt fehlt!")
        write_signals([])
        return
    tickers = [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip() and not t.startswith("#")]

    state = load_positions()
    print(f"[{now_utc_iso()}] Scan {len(tickers)} Symbole…")

    signals, df_map = [], {}
    for t in tickers:
        try:
            print(f"[{t}] downloading…")
            df = download_df(t, period="60d", interval=TIMEFRAME)
            if df is None or df.empty:
                print(f"[WARN] {t} empty df"); continue

            df_map[t] = df
            sig = evaluate_ticker(df, t)
            decided = sig["recommendation"]

            if sig["recommendation"] == "BUY":
                enriched = analyze_with_ki(sig)
                if enriched:
                    sig.update(enriched)
                if not sig.get("qty_sized"):
                    sig["qty_sized"] = sig.get("qty", POSITION_QTY)

                # BUY zu TradersPost
                tp_send_buy(sig)

                # Position für Bot-Exits merken
                if BOT_MANAGE_EXITS:
                    entry = float(sig["price"])
                    sl = float(sig.get("sl_price") or 0) or None
                    tp = float(sig.get("tp_price") or 0) or None
                    trail_pct = float(sig.get("trailing_percent") or 0) or None
                    state["positions"][t] = {
                        "qty": int(sig.get("qty_sized") or sig.get("qty") or POSITION_QTY),
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "trail_pct": trail_pct,
                        "hi_water": entry
                    }
                    print(f"[STATE] add {t}: entry={entry} sl={sl} tp={tp} trail%={trail_pct}")

            if sig["recommendation"] != decided:
                print(f"[WARN] {t}: recommendation changed {decided} -> {sig['recommendation']} AFTER enrichment")

            signals.append(sig)

        except Exception as e:
            print(f"[ERR] {t} failed: {e}")

        time.sleep(0.12)

    if BOT_MANAGE_EXITS and state.get("positions"):
        update_and_check_exits(state, df_map)

    save_positions(state)
    write_signals(signals)


if __name__ == "__main__":
    print(">>> Starting main.py …")
    run()
