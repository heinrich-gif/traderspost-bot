import os, time, json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests

# ===== Config =====
BREAKOUT_LEN = int(os.getenv("BREAKOUT_LEN", "20"))
MIN_PCT_MOVE = float(os.getenv("MIN_PCT_MOVE", "5.0"))
MIN_RSI_MOM  = float(os.getenv("MIN_RSI_MOM", "50.0"))
TIMEFRAME    = os.getenv("TIMEFRAME", "5m")
POSITION_QTY = int(os.getenv("POSITION_QTY", "10"))

KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL", "").strip()
TP_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL", "").strip()
KI_SCORER_URL  = os.getenv("KI_SCORER_URL", "").strip()  # optional

TICKERS_FILE = Path("tickers.txt")
OUTPUT_PATH  = Path(os.getenv("OUTPUT_PATH", "docs/signals.json"))

# ===== Helpers =====
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def is_us_extended_utc(dt: datetime) -> bool:
    wd = dt.weekday()  # 0=Mo…6=So
    if wd >= 5: return False
    mins = dt.hour*60 + dt.minute
    # 08:00–23:59 UTC + 00:00–01:59 UTC
    return (8*60 <= mins < 24*60) or (0 <= mins < 2*60)

def to_1d_series(obj) -> pd.Series:
    """
    Erzwingt aus Series/1-spaltigem DataFrame eine 1D-Nummernserie.
    """
    s = obj
    if isinstance(s, pd.DataFrame):
        # nimm erste Spalte
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
    gain = delta.clip(lower=0).ewm(alpha=1/length, min_periods=length).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/length, min_periods=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def download_df(ticker: str, period="60d", interval="5m") -> pd.DataFrame:
    # group_by="column" + threads=False verhindert MultiIndex-Salat
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

# ===== Scoring / Strategy =====
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
    buy_signal = breakout and momentum and (pct_move >= MIN_PCT_MOVE)
    sell_signal = False  # Exits machst du via TP/SL/Trailing in TradersPost

    # Optionaler KI-Score
    ki_score = None
    if KI_SCORER_URL and buy_signal:
        try:
            resp = requests.post(
                KI_SCORER_URL,
                json={"symbol": ticker, "price": price, "rsi": rsi_now, "pctChange": round(pct_move,2)},
                timeout=10,
            )
            if resp.ok:
                j = resp.json()
                if isinstance(j, dict) and "score" in j:
                    ki_score = int(j["score"])
        except Exception as e:
            print(f"[KI] scorer error {ticker}: {e}")

    rec = {
        "symbol": ticker,
        "price": round(price, 4),
        "rsi": round(rsi_now, 2),
        "pct_move": round(pct_move, 2),
        "prev_high_n": BREAKOUT_LEN,
        "breakout": bool(breakout),
        "momentum": bool(momentum),
        "recommendation": "BUY" if buy_signal else "HOLD",
        "qty": POSITION_QTY if buy_signal else 0,
        "ki_score": ki_score,
        "timestamp": now_utc_iso(),
        "spark": last_n_closes(df, 50),
    }
    print(f"[{ticker}] price={price:.2f} rsi={rsi_now:.2f} Δ%={pct_move:.2f} "
          f"prevHigh{BREAKOUT_LEN}={prev_high:.4f} breakout={breakout} mom={momentum} -> {rec['recommendation']}")
    return rec

def send_signal_to_endpoints(sig: dict):
    side = "buy" if sig["recommendation"] == "BUY" else "hold"
    payload = {"symbol": sig["symbol"], "action": side, "quantity": sig["qty"], "price": sig["price"]}

    if KI_WEBHOOK_URL:
        try:
            r = requests.post(KI_WEBHOOK_URL, json=payload, timeout=10)
            print(f"[CF] {sig['symbol']} -> {r.status_code} {r.text[:120]}")
        except Exception as e:
            print(f"[CF] send failed: {e}")

    if TP_WEBHOOK_URL and side == "buy":
        try:
            r = requests.post(TP_WEBHOOK_URL, json=payload, timeout=10)
            print(f"[TP] {sig['symbol']} -> {r.status_code} {r.text[:120]}")
        except Exception as e:
            print(f"[TP] send failed: {e}")

# ===== Output =====
def write_signals(signals: list, path: Path = OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": now_utc_iso(),
        "timeframe": TIMEFRAME,
        "breakout_len": BREAKOUT_LEN,
        "min_pct_move": MIN_PCT_MOVE,
        "min_rsi_mom": MIN_RSI_MOM,
        "ki": {"scorer_url_set": bool(KI_SCORER_URL)},
        "signals": signals,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[WRITE] {path} -> {len(signals)} Zeilen")

# ===== Main =====
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
            signals.append(sig)
            if sig["recommendation"] == "BUY":
                send_signal_to_endpoints(sig)
        except Exception as e:
            print(f"[ERR] {t} failed: {e}")
        time.sleep(0.15)

    write_signals(signals)

if __name__ == "__main__":
    print(">>> Starting main.py …")
    run()
