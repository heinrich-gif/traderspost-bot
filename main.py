import os
import json
import requests
import yfinance as yf
from datetime import datetime
from finvizfinance.screener.overview import Overview

# ========= Einstellungen =========
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "1000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.25"))  # 25% Standard
KI_WEBHOOK_URL = os.getenv("KI_WEBHOOK_URL", "").strip()
TP_API_KEY = os.getenv("TRADERSPOST_API_KEY", "").strip()
TP_STRATEGY_ID = os.getenv("TRADERSPOST_STRATEGY_ID", "").strip()
SIGNALS_FILE = "docs/signals.json"
EXITS_FILE = "docs/exits.json"

# ========= Parameter =========
PRICE_MAX = 5.0
RELVOL_MIN = 2.0
AVGVOL_MIN = 500_000
RANGE_PCT_MIN = 5.0
BREAKOUT_TOL = 0.003  # 0.3%

# ========= Hilfsfunktionen =========
def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] {msg}")

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))
    return []

# ========= KI-Analyse (SL/TP/Trailing) =========
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
        res = requests.post(KI_WEBHOOK_URL, json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()
        log(f"[KI] Fehler {res.status_code}: {res.text}")
    except Exception as e:
        log(f"[KI] Exception: {e}")
    return {}

# ========= TraderPost Order =========
def send_traderspost_order(symbol, qty, side, sl=None, tp=None, trail=None):
    if not TP_API_KEY or not TP_STRATEGY_ID:
        return
    payload = {
        "symbol": symbol,
        "quantity": qty,
        "side": side,
        "type": "market",
        "time_in_force": "gtc",
    }
    if sl: payload["stop_loss_percent"] = sl
    if tp: payload["take_profit_percent"] = tp
    if trail: payload["trailing_stop_percent"] = trail

    headers = {
        "Authorization": f"Bearer {TP_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post(
        f"https://api.traderspost.io/api/v2/strategies/{TP_STRATEGY_ID}/orders",
        headers=headers,
        json=payload
    )
    log(f"[TP] {symbol} -> {side.upper()} {qty} | SL={sl}% TP={tp}% Trail={trail}% | {r.status_code} {r.text}")

# ========= Scanner =========
def scan_finviz():
    log(f"[SCAN] Finviz… price<={PRICE_MAX} relvol>={RELVOL_MIN} avgvol>={AVGVOL_MIN}")
    ov = Overview()
    filters = [
        f"sh_price_u{PRICE_MAX}",
        f"sh_relvol_o{RELVOL_MIN}",
        f"sh_avgvol_o{AVGVOL_MIN}"
    ]
    ov.set_filter(filters=filters)
    df = ov.screener_view()
    tickers = df["Ticker"].tolist()
    log(f"[SCAN] Finviz-Kandidaten: {len(tickers)} -> {tickers[:20]} …")
    return tickers

# ========= Evaluierung =========
def evaluate_ticker(ticker):
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if df.empty:
            return None
        price = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        prev_high = float(df["High"].tail(20).max())
        pct_move = round(((price - prev_close) / prev_close) * 100, 2)
        rsi_now = calc_rsi(df["Close"])

        breakout = price >= prev_high * (1 - BREAKOUT_TOL)
        momentum = rsi_now >= 60
        buy_raw = breakout and momentum and pct_move >= 0.5 and rsi_now <= 70

        signal = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": ticker,
            "price": round(price, 4),
            "pct_move": pct_move,
            "rsi": rsi_now,
            "breakout": breakout,
            "momentum": momentum,
            "recommendation": "BUY" if buy_raw else "HOLD",
        }

        if buy_raw:
            ki = analyze_with_ki(signal)
            if ki:
                signal.update(ki)
                qty = ki.get("quantity", int((ACCOUNT_EQUITY * RISK_PCT) / price))
                send_traderspost_order(ticker, qty, "buy", ki.get("stop_loss"), ki.get("take_profit"), ki.get("trailing_stop"))
        return signal
    except Exception as e:
        log(f"[ERR] {ticker} {e}")
        return None

# ========= RSI =========
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return round(100 - (100 / (1 + rs)).iloc[-1], 2)

# ========= Main =========
if __name__ == "__main__":
    tickers = scan_finviz()
    final_signals = []
    exits = load_json(EXITS_FILE)

    for t in tickers:
        sig = evaluate_ticker(t)
        if sig:
            final_signals.append(sig)

            # Exit-Logik
            if sig["recommendation"] == "SELL":
                exits.append({
                    "timestamp": sig["timestamp"],
                    "symbol": sig["symbol"],
                    "price": sig["price"],
                    "reason": sig.get("exit_reason", "Signal SELL"),
                })

    save_json(SIGNALS_FILE, final_signals)
    save_json(EXITS_FILE, exits)

    log(f"[WRITE] {SIGNALS_FILE} -> {len(final_signals)} Zeilen")
    log(f"[WRITE] {EXITS_FILE} -> {len(exits)} Zeilen")
