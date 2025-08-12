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
NEAR_BREAKOUT_PCT  = float(os.getenv("NEAR_BREAKOUT_PCT", "0.3"))  # % Toleranz fürs Near-Breakout

# Aggressive BUY-Schwellen
BUY_RSI_MAX        = float(os.getenv("BUY_RSI_MAX", "65"))
BUY_DELTA_MIN      = float(os.getenv("BUY_DELTA_MIN", "0.5"))

# Info/Badge (keine Kaufentscheidung)
MIN_PCT_MOVE       = float(os.getenv("MIN_PCT_MOVE", "5.0"))
MIN_RSI_MOM        = float(os.getenv("MIN_RSI_MOM", "50.0"))

# KI-Scoring (Badge/Filter)
KI_SCORER_URL      = os.getenv("KI_SCORER_URL", "").strip()
KI_SCORE_MIN       = int(os.getenv("KI_SCORE_MIN", "65"))

# KI-Analyse (SL/TP/Trailing/Qty)
KI_WEBHOOK_URL     = os.getenv("KI_WEBHOOK_URL", "").strip()
ACCOUNT_EQUITY     = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_PCT           = float(os.getenv("RISK_PCT", "1"))

# TradersPost (optional)
TP_WEBHOOK_URL     = os.getenv("TP_WEBHOOK_URL", "").strip()

# Bot-Exit-Engine (Backup-TP/SL/Trailing)
BOT_MANAGE_EXITS   = os.getenv("BOT_MANAGE_EXITS", "1") == "1"   # 1=an, 0=aus
EXIT_CHECK_LOOKBACK= int(os.getenv("EXIT_CHECK_LOOKBACK", "3"))  # wie viele Kerzen zur Prüfung
STATE_DIR          = Path("state")
STATE_FILE         = STATE_DIR / "positions.json"

# Dateien
OUTPUT_PATH        = Path(os.getenv("OUTPUT_PATH", "docs/signals.json"))
TICKERS_FILE       = Path("tickers.txt")


# ========= Zeit / Session =========
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def is_us_extended_utc(dt: datetime) -> bool:
    """
    US Extended Hours (Mo–Fr):
      Pre   08:00–13:30 UTC
      RTH   13:30–20:00 UTC
      After 20:00–24:00 UTC + 00:00–02:00 UTC
    => Effektiv: 08:00–23:59 + 00:00–01:59 UTC
    """
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
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/length, min_periods=length).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/length, min_periods=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def download_df(ticker: str, period="60d", interval="5m") -> pd.DataFrame:
    """
    Stabiler Download; auto_adjust=True gegen Splits/Dividenden.
    prepost=True: Extended Hours.
    """
    if interval == "1m":
        period = "7d"
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        prepost=True,
        group_by="column",
        threads=False,
    )
    return df


# ========= KI-Scoring =========
def score_with_ki(symbol: str, price: float, rsi_now: float, pct_move: float, breakout: bool, momentum: bool):
    ki_score, ki_pass = None, True
    if not KI_SCORER_URL:
        return ki_score, ki_pass
    try:
        resp = requests.post(
            KI_SCORER_URL,
            json={
                "symbol": symbol,
                "price": round(price, 4),
                "rsi": round(rsi_now, 2),
                "pctChange": round(pct_move, 2),
                "breakout": bool(breakout),
                "momentum": bool(momentum),
            },
            timeout=12
        )
        if resp.ok:
            j = resp.json()
            if isinstance(j, dict) and "score" in j:
                ki_score = int(j["score"])
    except Exception as e:
        print(f"[KI] scorer error {symbol}: {e}")
    return ki_score, ki_pass


# ========= KI-Analyse (SL/TP/Trailing/Qty) =========
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
    }
    try:
        r = requests.post(KI_WEBHOOK_URL, json=payload, timeout=15)
        if not r.ok:
            print(f"[KI-ANALYSE] HTTP {r.status_code}: {r.text[:160]}")
            return {}
        data = r.json()
        res = data.get("result")
        if isinstance(res, list) and res:
            return res[0]
        if isinstance(res, dict):
            return res
        if isinstance(data, dict) and "sl_price" in data:
            return data
        return {}
    except Exception as e:
        print(f"[KI-ANALYSE] error: {e}")
        return {}


# ========= Exit-State (persistente Positionsverwaltung) =========
def load_positions() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception as e:
        print(f"[STATE] load failed: {e}")
    return {"positions": {}}  # { symbol: {qty, entry, sl, tp, trail_pct, hi_water} }

def save_positions(state: dict):
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        print(f"[STATE] save failed: {e}")


# ========= Strategie (Entry) =========
def evaluate_ticker(df: pd.DataFrame, ticker: str) -> dict:
    # ---- Stale Check ----
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

    c = close_series(df)
    h = high_series(df)
    if c.empty or h.empty or len(c) < 3:
        raise ValueError("Not enough data")

    price = float(c.iloc[-1])
    prev_close = float(c.iloc[-2])
    pct_move = (price - prev_close) / prev_close * 100.0 if prev_close else 0.0

    # PrevHigh OHNE aktuelle Kerze + Near-Breakout-Toleranz
    window = max(1, BREAKOUT_LEN)
    if len(h) > window:
        prev_high_raw = float(h.iloc[-(window+1):-1].max())
    else:
        prev_high_raw = float(h.iloc[:-1].max()) if len(h) > 1 else float(h.max())
    tolerance = prev_high_raw * (NEAR_BREAKOUT_PCT / 100.0)
    prev_high = prev_high_raw
    near_breakout = (price >= prev_high_raw - tolerance)

    rsi_now = float(rsi(c).iloc[-1])
    momentum = rsi_now >= MIN_RSI_MOM  # Info/Badge

    # Aggressive BUY-Logik
    cond_rsi_delta = (rsi_now <= BUY_RSI_MAX) and (pct_move >= BUY_DELTA_MIN)
    cond_near_bo   = near_breakout
    buy_raw = cond_rsi_delta or cond_near_bo

    # KI-Score (Badge immer; BUY blockt nur wenn Score < MIN)
    ki_score, _ = score_with_ki(ticker, price, rsi_now, pct_move, near_breakout, momentum)
    ki_ok = (KI_SCORER_URL == "") or (ki_score is None) or (ki_score >= KI_SCORE_MIN)

    recommendation = "BUY" if (buy_raw and ki_ok) else "HOLD"

    print(
        f"[{ticker}] price={price:.4f} rsi={rsi_now:.2f} Δ%={pct_move:.2f} "
        f"prevHigh{window}={prev_high:.4f} tol={NEAR_BREAKOUT_PCT:.3f}% "
        f"nearBO={near_breakout} | cond_rsi_delta={cond_rsi_delta} cond_near_bo={cond_near_bo} "
        f"raw={buy_raw} ki={ki_score} ki_ok={ki_ok} -> {recommendation}"
    )

    rec = {
        "symbol": ticker,
        "price": round(price, 4),
        "rsi": round(rsi_now, 2),
        "pct_move": round(pct_move, 2),
        "prev_high_n": window,
        "prev_high_value": round(prev_high, 4),
        "near_breakout_pct": NEAR_BREAKOUT_PCT,
        "breakout": bool(near_breakout),
        "momentum": bool(momentum),
        "recommendation": recommendation,
        "qty": POSITION_QTY if recommendation == "BUY" else 0,
        "ki_score": ki_score,
        "ki_pass": (ki_score is None) or (ki_score >= KI_SCORE_MIN),
        "timestamp": now_utc_iso(),
        "spark": last_n_closes(df, 50),
        # Platzhalter für KI-Analyse (werden bei BUY gefüllt)
        "sl_percent": None,
        "tp_percent": None,
        "trailing_percent": None,
        "sl_price": None,
        "tp_price": None,
        "qty_sized": None,
    }
    return rec


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
        # prefer percent; TP ignoriert unbekannte Felder
        "take_profit_percent": sig.get("tp_percent"),
        "stop_loss_percent": sig.get("sl_percent"),
        "trailing_stop_percent": sig.get("trailing_percent"),
        # alternativ: fixe Preise
        "take_profit_price": sig.get("tp_price"),
        "stop_loss_price": sig.get("sl_price"),
        "meta": {
            "source": "gh-actions-bot",
            "ki_score": sig.get("ki_score"),
            "ki_pass": sig.get("ki_pass")
        }
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
        "action": "sell",       # falls TP 'close' verlangt, einfach hier anpassen
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
    """
    state: { positions: { symbol: {qty, entry, sl, tp, trail_pct, hi_water} } }
    df_map: { symbol: df } – aktuelle Daten je Symbol
    Prüft für jedes gehaltene Symbol:
      - trailing: hi_water = max(hi_water, last_price)
                  trail_stop = hi_water * (1 - trail_pct/100)
      - exit, wenn last_price >= tp  -> SELL
              oder last_price <= min(sl, trail_stop)
    """
    if not BOT_MANAGE_EXITS:
        return

    positions = state.get("positions", {})
    to_close = []

    for sym, pos in positions.items():
        df = df_map.get(sym)
        if df is None or df.empty:
            continue

        closes = close_series(df).tail(max(2, EXIT_CHECK_LOOKBACK))
        last_price = float(closes.iloc[-1])

        entry = float(pos.get("entry", last_price))
        tp = float(pos.get("tp", 0) or 0)
        sl = float(pos.get("sl", 0) or 0)
        trail_pct = float(pos.get("trail_pct", 0) or 0)
        hi_water = float(pos.get("hi_water", entry))

        # hi-water aktualisieren
        if last_price > hi_water:
            hi_water = last_price

        trail_stop = hi_water * (1 - trail_pct/100) if trail_pct > 0 else None

        trigger = None
        if tp and last_price >= tp:
            trigger = "TP"
        elif sl and last_price <= sl:
            trigger = "SL"
        elif trail_stop and last_price <= trail_stop:
            trigger = "TRAIL"

        # speichern
        pos["hi_water"] = hi_water
        positions[sym] = pos

        if trigger:
            print(f"[EXIT] {sym} trigger={trigger} last={last_price:.4f} "
                  f"tp={tp or '-'} sl={sl or '-'} trail={trail_stop or '-'}")
            to_close.append(sym)

    # SELL für getriggerte Positionen
    for sym in to_close:
        qty = positions[sym].get("qty")
        tp_send_sell(sym, qty)
        # aus State entfernen
        positions.pop(sym, None)

    state["positions"] = positions


# ========= JSON-Ausgabe fürs Dashboard =========
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
        "ki": {"scorer_url_set": bool(KI_SCORER_URL), "min_score": KI_SCORE_MIN},
        "signals": signals,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[WRITE] {path} -> {len(signals)} Zeilen")


# ========= Main =========
def run():
    now = datetime.now(timezone.utc)
    print(f"[CFG] TF={TIMEFRAME} BUY_RSI_MAX={BUY_RSI_MAX} BUY_DELTA_MIN={BUY_DELTA_MIN} "
          f"NEAR_BO%={NEAR_BREAKOUT_PCT} KI_MIN={KI_SCORE_MIN} "
          f"SCORER={'on' if KI_SCORER_URL else 'off'} ANALYSE={'on' if KI_WEBHOOK_URL else 'off'} "
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

    # State laden
    state = load_positions()

    print(f"[{now_utc_iso()}] Scan {len(tickers)} Symbole…")

    signals = []
    df_map = {}
    for t in tickers:
        try:
            print(f"[{t}] downloading…")
            df = download_df(t, period="60d", interval=TIMEFRAME)
            if df is None or df.empty:
                print(f"[WARN] {t} empty df")
                continue

            df_map[t] = df
            sig = evaluate_ticker(df, t)
            decided = sig["recommendation"]

            # BUY → KI-Analyse (SL/TP/Trailing/Qty) + TP-Order + State-Add
            if sig["recommendation"] == "BUY":
                enriched = analyze_with_ki(sig)
                if enriched:
                    for k in ("sl_percent","tp_percent","trailing_percent",
                              "sl_price","tp_price","qty_sized",
                              "equity_used","risk_pct_used",
                              "ki_score","ki_min","ki_pass","rationale"):
                        if k in enriched:
                            sig[k] = enriched[k]
                if not sig.get("qty_sized"):
                    sig["qty_sized"] = sig.get("qty", POSITION_QTY)

                # BUY an TradersPost
                tp_send_buy(sig)

                # ---- Position in State aufnehmen (für Bot-Exits) ----
                if BOT_MANAGE_EXITS:
                    entry = float(sig["price"])
                    # Priorität: feste Preise > Prozent
                    sl = float(sig.get("sl_price") or 0)
                    tp = float(sig.get("tp_price") or 0)

                    if not sl and sig.get("sl_percent"):
                        sl = round(entry * (1 - float(sig["sl_percent"])/100), 4)
                    if not tp and sig.get("tp_percent"):
                        tp = round(entry * (1 + float(sig["tp_percent"])/100), 4)

                    state["positions"][t] = {
                        "qty": int(sig.get("qty_sized") or sig.get("qty") or POSITION_QTY),
                        "entry": entry,
                        "sl": sl or None,
                        "tp": tp or None,
                        "trail_pct": float(sig.get("trailing_percent") or 0) or None,
                        "hi_water": entry
                    }
                    print(f"[STATE] add {t}: entry={entry} sl={state['positions'][t]['sl']} tp={state['positions'][t]['tp']} trail%={state['positions'][t]['trail_pct']}")

            if sig["recommendation"] != decided:
                print(f"[WARN] {t}: recommendation changed from {decided} -> {sig['recommendation']} AFTER enrichment")

            signals.append(sig)

        except Exception as e:
            print(f"[ERR] {t} failed: {e}")

        time.sleep(0.12)  # freundlich zu yfinance

    # ---- Bot-Exit-Engine prüfen (auf Basis der gerade geladenen Daten) ----
    if BOT_MANAGE_EXITS and state.get("positions"):
        update_and_check_exits(state, df_map)

    # State speichern
    save_positions(state)

    # Dashboard schreiben
    write_signals(signals)


if __name__ == "__main__":
    print(">>> Starting main.py …")
    run()
