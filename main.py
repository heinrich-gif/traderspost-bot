import os, time, json
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import requests

# ==== Env / Parameter ====
TIMEFRAME      = os.getenv("TIMEFRAME", "5m")
POSITION_QTY   = int(os.getenv("POSITION_QTY", "10"))
BREAKOUT_LEN   = int(os.getenv("BREAKOUT_LEN", "20"))
MIN_PCT_MOVE   = float(os.getenv("MIN_PCT_MOVE", "5.0"))
MIN_RSI_MOM    = float(os.getenv("MIN_RSI_MOM", "50.0"))

# Risk-Empfehlungen
TP_PCT         = float(os.getenv("TP_PCT", "25"))  # %
SL_PCT         = float(os.getenv("SL_PCT", "9"))   # %
TRAIL_PCT      = float(os.getenv("TRAIL_PCT", "6"))# %

OUTPUT_PATH    = os.getenv("OUTPUT_PATH", "docs/signals.json")

# Optional: KI-Score (separater Scoring-Endpoint)
# Erwartet JSON-Response mit {"score": 0..100, "result": "..."} – wenn None/Fehler: kein Score
KI_SCORER_URL  = os.getenv("KI_SCORER_URL", "").strip()
KI_SCORE_MIN   = int(os.getenv("KI_SCORE_MIN", "60"))

# ==== Session-Filter: US extended (Pre/Regular/After) ====
def is_us_extended_utc(dt: datetime) -> bool:
    wd = dt.weekday()  # 0=Mo … 6=So
    if wd >= 5: return False
    mins = dt.hour*60 + dt.minute
    # 08:00–23:59 UTC + 00:00–01:59 UTC
    return (8*60 <= mins < 24*60) or (0 <= mins < 2*60)

# ==== Indikatoren ====
def compute_rsi(series, period: int):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==== Daten holen ====
def download_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="5d", interval=TIMEFRAME, progress=False, auto_adjust=True, group_by="column", threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df["rsi"] = compute_rsi(df["Close"], 14)
        return df.dropna()
    except Exception as e:
        print(f"[WARN] {ticker} download: {e}")
        return pd.DataFrame()

def last_n_closes(df: pd.DataFrame, n: int = 50):
    s = df["Close"].tail(n)
    return [round(float(x), 6) for x in s.to_list()]

# ==== Strategie (Breakout + Momentum) ====
def evaluate_ticker(df: pd.DataFrame, ticker: str):
    price = float(df["Close"].iloc[-1])
    rsi   = float(df["rsi"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    pct_move = (price - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0
    prev_high = float(df["High"].tail(BREAKOUT_LEN).max())
    breakout  = price > prev_high
    momentum  = rsi >= MIN_RSI_MOM
    buy = breakout and momentum and (pct_move >= MIN_PCT_MOVE)

    # Optionaler KI-Score
    ki_score = None
    if KI_SCORER_URL and buy:
        try:
            r = requests.post(KI_SCORER_URL, json={"symbol": ticker, "action": "buy", "price": price}, timeout=10)
            if r.status_code >= 200 and r.status_code < 300:
                j = r.json()
                if isinstance(j, dict) and "score" in j:
                    ki_score = int(j["score"])
        except Exception as e:
            print(f"[KI] score error {ticker}: {e}")

    rec = {
        "symbol": ticker,
        "price": round(price, 4),
        "rsi": round(rsi, 2),
        "pct_move": round(pct_move, 2),
        "prev_high_n": BREAKOUT_LEN,
        "breakout": bool(breakout),
        "momentum": bool(momentum),
        "recommendation": "BUY" if buy else "HOLD",
        "qty": POSITION_QTY if buy else 0,
        # Empfehlungen
        "tp_percent": TP_PCT if buy else None,
        "sl_percent": SL_PCT if buy else None,
        "trailing_percent": TRAIL_PCT if buy else None,
        "tp_price": round(price * (1 + TP_PCT/100), 4) if buy else None,
        "sl_price": round(price * (1 - SL_PCT/100), 4) if buy else None,
        "ki_score": ki_score,
        "ki_pass": (ki_score is None) or (ki_score >= KI_SCORE_MIN),
        "spark": last_n_closes(df, 50),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    print(f"[{ticker}] price={price:.2f} rsi={rsi:.2f} Δ%={pct_move:.2f} breakout={breakout} mom={momentum} "
          f"-> {rec['recommendation']} ki={ki_score}")
    return rec

# ==== JSON schreiben ====
def write_signals(signals, path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": TIMEFRAME,
        "breakout_len": BREAKOUT_LEN,
        "min_pct_move": MIN_PCT_MOVE,
        "min_rsi_mom": MIN_RSI_MOM,
        "risk_profile": {"tp_percent": TP_PCT, "sl_percent": SL_PCT, "trailing_percent": TRAIL_PCT},
        "ki": {"min_score": KI_SCORE_MIN, "scorer_url_set": bool(KI_SCORER_URL)},
        "signals": signals
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[WRITE] {path} -> {len(signals)} Zeilen")

# ==== Main ====
def run():
    now_utc = datetime.now(timezone.utc)
    if not is_us_extended_utc(now_utc):
        print(f"[SKIP] Outside US extended session (UTC {now_utc.isoformat()})")
        write_signals([])
        return

    # Ticker aus tickers.txt (vom Finviz-Scanner) – sonst Fallback
    try:
        with open("tickers.txt") as f:
            tickers = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    except FileNotFoundError:
        tickers = ["OMI","MSOS","LAR","AIYY","WOW","YOLO","VTEX","UA","ETHD","AMC","MX","BLND","LAC"]

    print(f"[{now_utc.isoformat()}] Scan {len(tickers)} Symbole…")
    recs = []
    for t in tickers:
        df = download_data(t)
        if df.empty:
            continue
        recs.append(evaluate_ticker(df, t))
        time.sleep(0.1)

    write_signals(recs)

if __name__ == "__main__":
    print(">>> Starting main.py …")
    run()
