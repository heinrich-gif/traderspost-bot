import os
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests

# ========= Konfiguration (ENV) =========
TIMEFRAME        = os.getenv("TIMEFRAME", "5m")
POSITION_QTY     = int(os.getenv("POSITION_QTY", "10"))          # Fallback, falls KI keine qty liefert

# Setup-Regeln
BREAKOUT_LEN     = int(os.getenv("BREAKOUT_LEN", "20"))
MIN_PCT_MOVE     = float(os.getenv("MIN_PCT_MOVE", "5.0"))
MIN_RSI_MOM      = float(os.getenv("MIN_RSI_MOM", "50.0"))

# KI-Scoring (Badge/Filter)
KI_SCORER_URL    = os.getenv("KI_SCORER_URL", "").strip()        # z. B. Cloudflare Scorer
KI_SCORE_MIN     = int(os.getenv("KI_SCORE_MIN", "65"))          # BUY nur wenn Score >= MIN (sofern buy_raw True)

# KI-Analyse (SL/TP/Trailing/Qty)
KI_WEBHOOK_URL   = os.getenv("KI_WEBHOOK_URL", "").strip()       # ← dein neuer KI-Analyse-Worker
ACCOUNT_EQUITY   = float(os.getenv("ACCOUNT_EQUITY", "10000"))   # für Positions-sizing im Worker
RISK_PCT         = float(os.getenv("RISK_PCT", "1"))             # Risiko pro Trade in %

# TradersPost (optional)
TP_WEBHOOK_URL   = os.getenv("TP_WEBHOOK_URL", "").strip()

# Dateien
OUTPUT_PATH      = Path(os.getenv("OUTPUT_PATH", "docs/signals.json"))
TICKERS_FILE     = Path("tickers.txt")


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
    wd = dt.weekday()  # 0=Mo … 6=So
    if wd >= 5:
        return False
    mins = dt.hour * 60 + dt.minute
    return (8 * 60 <= mins < 24 * 60) or (0 <= mins < 2 * 60)


# ========= Pandas/YF Hilfen =========
def to_1d_series(obj) -> pd.Series:
    """ Erzwingt 1D Serie aus Series oder 1-spaltigem DataFrame. """
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
    Stabiler Download ohne MultiIndex; auto_adjust=True gegen Splits/Dividenden.
    Passt period automatisch für 1m an.
    """
    if interval == "1m":
        period = "7d"
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=False,
    )
    return df


# ========= KI-Scoring (Badge / BUY-Filter) =========
def score_with_ki(symbol: str, price: float, rsi_now: float, pct_move: float, breakout: bool, momentum: bool):
    """
    Fragt optional den KI_SCORER_URL ab und gibt (score:int|None, pass:bool) zurück.
    pass=True, wenn (kein Score nötig) oder Score >= KI_SCORE_MIN, sonst False.
    """
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

    # Score beeinflusst nur BUY, wenn das Roh-Setup grundsätzlich passt (entschieden im Aufrufer)
    return ki_score, ki_pass  # ki_pass wird im Aufrufer gesetzt, wenn buy_raw True und score vorhanden


# ========= KI-Analyse (SL/TP/Trailing/Qty) =========
def analyze_with_ki(signal: dict) -> dict:
    """
    Ruft den KI_WEBHOOK_URL Worker auf.
    Erwartet Response-Format:
      { analyzed_at: "...", result: { ...enriched signal... } }
    oder bei Array: { result: [ ... ] }
    Gibt das angereicherte Einzel-Resultat zurück; bei Fehler -> {}.
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
        # Falls Worker direkt das angereicherte Dict zurückgibt
        if isinstance(data, dict) and "sl_price" in data:
            return data
        return {}
    except Exception as e:
        print(f"[KI-ANALYSE] error: {e}")
        return {}


# ========= Strategie =========
def evaluate_ticker(df: pd.DataFrame, ticker: str) -> dict:
    c = close_series(df)
    h = high_series(df)
    if c.empty or h.empty or len(c) < 3:
        raise ValueError("Not enough data")

    price = float(c.iloc[-1])
    prev_close = float(c.iloc[-2])
    pct_move = (price - prev_close) / prev_close * 100.0 if prev_close else 0.0
    prev_high = float(h.tail(BREAKOUT_LEN).max())
    rsi_now = float(rsi(c).iloc[-1])

    breakout = price > prev_high
    momentum = rsi_now >= MIN_RSI_MOM
    buy_raw  = breakout and momentum and (pct_move >= MIN_PCT_MOVE)

    # KI-Score immer berechnen (für Anzeige), aber BUY nur filtern, wenn buy_raw True
    ki_score, _ = score_with_ki(ticker, price, rsi_now, pct_move, breakout, momentum)
    ki_pass = True
    if buy_raw and (ki_score is not None):
        ki_pass = ki_score >= KI_SCORE_MIN

    recommendation = "BUY" if (buy_raw and ki_pass) else "HOLD"

    rec = {
        "symbol": ticker,
        "price": round(price, 4),
        "rsi": round(rsi_now, 2),
        "pct_move": round(pct_move, 2),
        "prev_high_n": BREAKOUT_LEN,
        "breakout": bool(breakout),
        "momentum": bool(momentum),
        "recommendation": recommendation,
        "qty": POSITION_QTY if recommendation == "BUY" else 0,
        "ki_score": ki_score,
        "ki_pass": ki_pass,
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

    print(
        f"[{ticker}] price={price:.2f} rsi={rsi_now:.2f} Δ%={pct_move:.2f} "
        f"prevHigh{BREAKOUT_LEN}={prev_high:.4f} breakout={breakout} mom={momentum} "
        f"ki={ki_score} pass={ki_pass} -> {recommendation}"
    )
    return rec


# ========= TradersPost Versand =========
def send_to_traderspost(sig: dict):
    if not TP_WEBHOOK_URL:
        return
    # Nutze angereicherte Werte, wenn vorhanden
    qty = int(sig.get("qty_sized") or sig.get("qty") or POSITION_QTY)
    payload = {
        "symbol": sig["symbol"],
        "action": "buy",
        "quantity": qty,
        "price": sig["price"],
        # Zusatz-Infos – falls der TP-Webhook sie versteht, werden sie genutzt; sonst ignoriert.
        "tp_percent": sig.get("tp_percent"),
        "sl_percent": sig.get("sl_percent"),
        "trailing_percent": sig.get("trailing_percent"),
        "tp_price": sig.get("tp_price"),
        "sl_price": sig.get("sl_price"),
        "meta": {
            "ki_score": sig.get("ki_score"),
            "ki_pass": sig.get("ki_pass"),
        }
    }
    try:
        r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=12)
        print(f"[TP] {sig['symbol']} -> {r.status_code} {r.text[:160]}")
    except Exception as e:
        print(f"[TP] send failed: {e}")


# ========= JSON-Ausgabe fürs Dashboard =========
def write_signals(signals: list, path: Path = OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": now_utc_iso(),
        "timeframe": TIMEFRAME,
        "breakout_len": BREAKOUT_LEN,
        "min_pct_move": MIN_PCT_MOVE,
        "min_rsi_mom": MIN_RSI_MOM,
        "ki": {"scorer_url_set": bool(KI_SCORER_URL), "min_score": KI_SCORE_MIN},
        "signals": signals,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[WRITE] {path} -> {len(signals)} Zeilen")


# ========= Main =========
def run():
    now = datetime.now(timezone.utc)
    if not is_us_extended_utc(now):
        print(f"[SKIP] Outside US extended session (UTC {now.isoformat()})")
        write_signals([])
        return

    if not TICKERS_FILE.exists():
        print("[ERR] tickers.txt fehlt!")
        write_signals([])
        return
    tickers = [t.strip() for t in TICKERS_FILE.read_text().splitlines() if t.strip() and not t.startswith("#")]

    print(f"[{now_utc_iso()}] Scan {len(tickers)} Symbole…")

    signals = []
    for t in tickers:
        try:
            print(f"[{t}] downloading…")
            df = download_df(t, period="60d", interval=TIMEFRAME)
            if df is None or df.empty:
                print(f"[WARN] {t} empty df")
                continue

            sig = evaluate_ticker(df, t)

            # BUY → KI-Analyse (SL/TP/Trailing/Qty) + optional TP-Order
            if sig["recommendation"] == "BUY":
                enriched = analyze_with_ki(sig)
                if enriched:
                    # Mache die relevanten Felder in sig sichtbar
                    for k in ("sl_percent","tp_percent","trailing_percent","sl_price","tp_price","qty_sized","equity_used","risk_pct_used","ki_score","ki_min","ki_pass","rationale"):
                        if k in enriched:
                            sig[k] = enriched[k]
                # Falls keine qty_sized geliefert → Fallback qty
                if not sig.get("qty_sized"):
                    sig["qty_sized"] = sig.get("qty", POSITION_QTY)

                # Optional: Order an TradersPost schicken
                send_to_traderspost(sig)

            signals.append(sig)

        except Exception as e:
            print(f"[ERR] {t} failed: {e}")

        time.sleep(0.15)  # freundlich zu yfinance

    write_signals(signals)


if __name__ == "__main__":
    print(">>> Starting main.py …")
    run()
