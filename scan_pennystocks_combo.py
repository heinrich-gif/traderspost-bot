#!/usr/bin/env python3
import os, re, sys, math, time, json
from typing import Iterable, List, Set, Tuple
import requests
from bs4 import BeautifulSoup
import yfinance as yf

# ---------- ENV / Defaults ----------
PRICE_MIN        = float(os.getenv("PENNY_PRICE_MIN", "0.2"))
PRICE_MAX        = float(os.getenv("PENNY_PRICE_MAX", "6"))
AVG_VOL_MIN      = int(os.getenv("PENNY_AVG_VOL_MIN", "100000"))
REL_VOL_MIN      = float(os.getenv("PENNY_REL_VOL_MIN", "1.0"))
MAX_TICKERS      = int(os.getenv("PENNY_MAX_TICKERS", "40"))
YF_PERIOD        = os.getenv("PENNY_YF_PERIOD", "5d")
YF_INTERVAL      = os.getenv("PENNY_YF_INTERVAL", "1d")
SOURCES          = os.getenv("PENNY_SOURCES", "finviz,marketbeat").lower().split(",")
MARKETBEAT_URL   = os.getenv("MARKETBEAT_URL", "https://www.marketbeat.com/market-data/low-priced-stocks/")
# Optional: zusätzliche manuelle Liste (CSV)
EXTRA_TICKERS    = os.getenv("EXTRA_TICKERS", "")

OUTFILE = "tickers.txt"

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PennyScanner/1.0"}

TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")  # einfach & robust

def log(msg): print(msg, flush=True)

# ---------- Scraper ----------
def fetch_finviz() -> List[str]:
    """
    Holt Ticker aus Finviz-Screener (Top-Gainer / Preis / Volumen / RelVol).
    Hinweis: Finviz-Filter sind grob; wir validieren später mit Yahoo Finance.
    """
    # baue Filter-URL dynamisch
    # sh_price_u{PRICE_MAX}, sh_avgvol_o{AVG_VOL_MIN}, sh_relvol_o{REL_VOL_MIN}
    f = [
        f"sh_price_u{int(PRICE_MAX)}",
        f"sh_avgvol_o{int(AVG_VOL_MIN)}",
        f"sh_relvol_o{REL_VOL_MIN}",
    ]
    url = f"https://finviz.com/screener.ashx?v=111&f={','.join(f)}&o=-change"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
    except Exception as e:
        log(f"[FINVIZ] fetch error: {e}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    tickers = []
    # Finviz Tabelle: Links in erster Spalte mit class 'screener-link-primary'
    for a in soup.select("a.screener-link-primary"):
        t = a.get_text(strip=True).upper()
        if TICKER_RE.match(t):
            tickers.append(t)
    tickers = list(dict.fromkeys(tickers))
    log(f"[FINVIZ] {len(tickers)} tickers")
    return tickers

def fetch_marketbeat(url: str) -> List[str]:
    """
    Holt Ticker aus einer MarketBeat-ähnlichen Seite („Top Penny Stocks …“).
    Fällt auf generische <a>-Ticker zurück, wenn Struktur abweicht.
    Du kannst die URL über MARKETBEAT_URL setzen.
    """
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
    except Exception as e:
        log(f"[MB] fetch error: {e}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    # 1) Versuche Tabellen mit dem Wort "Penny"
    tables = soup.find_all("table")
    tickers = []
    for tbl in tables:
        header_txt = (tbl.find_previous("h2") or tbl.find_previous("h3") or tbl.find_previous("h1"))
        if header_txt and "penny" not in header_txt.get_text(" ", strip=True).lower():
            continue
        for a in tbl.select("a"):
            t = a.get_text(strip=True).upper()
            if TICKER_RE.match(t):
                tickers.append(t)
    # 2) Fallback: Alle <a>-Texte, die wie ein Ticker aussehen
    if not tickers:
        for a in soup.select("a"):
            t = a.get_text(strip=True).upper()
            if TICKER_RE.match(t):
                tickers.append(t)
    tickers = list(dict.fromkeys(tickers))
    log(f"[MB] {len(tickers)} tickers ({url})")
    return tickers

def parse_extra(csv: str) -> List[str]:
    vals = [x.strip().upper() for x in csv.split(",") if x.strip()]
    return [t for t in vals if TICKER_RE.match(t)]

# ---------- Validation via Yahoo Finance ----------
def yf_validate(tickers: List[str]) -> List[Tuple[str, float, float, float]]:
    """
    Lädt Price/Volume und berechnet:
      - last close price
      - avg volume (über Zeitraum)
      - relative volume (letzter Tag / Durchschnitt, grob)
    Rückgabe: Liste (ticker, price, avg_vol, rel_vol)
    """
    out = []
    for t in tickers:
        try:
            df = yf.download(t, period=YF_PERIOD, interval=YF_INTERVAL, progress=False, auto_adjust=True, prepost=True, group_by="column", threads=False)
            if df is None or df.empty:
                continue
            close = float(df["Close"].iloc[-1])
            vols = df["Volume"].dropna()
            if vols.empty:
                continue
            avg_vol = float(vols.mean())
            last_vol = float(vols.iloc[-1])
            rel_vol = (last_vol / avg_vol) if avg_vol > 0 else 1.0
            out.append((t, close, avg_vol, rel_vol))
        except Exception as e:
            log(f"[YF] {t} error: {e}")
        time.sleep(0.05)
    return out

# ---------- Main ----------
def main():
    all_ticks: List[str] = []

    if "finviz" in SOURCES:
        all_ticks += fetch_finviz()
    if "marketbeat" in SOURCES:
        all_ticks += fetch_marketbeat(MARKETBEAT_URL)

    all_ticks += parse_extra(EXTRA_TICKERS)

    # Dedupe & sanitize
    ticks = []
    seen: Set[str] = set()
    for t in all_ticks:
        t = t.strip().upper()
        if not TICKER_RE.match(t): 
            continue
        if t in seen: 
            continue
        seen.add(t)
        ticks.append(t)

    if not ticks:
        log("[SCAN] keine Roh-Ticker gefunden.")
        open(OUTFILE, "w").close()
        return

    log(f"[SCAN] Roh-Ticker: {len(ticks)} → YF-Validierung…")
    vals = yf_validate(ticks)

    # Filter anwenden
    filtered = []
    for t, price, avg_vol, rel_vol in vals:
        if not (PRICE_MIN <= price <= PRICE_MAX):
            continue
        if avg_vol < AVG_VOL_MIN:
            continue
        if rel_vol < REL_VOL_MIN:
            continue
        filtered.append((t, price, avg_vol, rel_vol))

    # Sortierung: erst nach rel_vol, dann nach %PriceChange schätzen wir nicht → daher rel_vol/avg_vol
    filtered.sort(key=lambda x: (x[3], x[2]), reverse=True)

    final = [t for t, _, _, _ in filtered[:MAX_TICKERS]]
    log(f"[SCAN] final: {len(final)} → {final}")

    with open(OUTFILE, "w") as f:
        f.write("\n".join(final))

    # Optionales Debug-JSON
    debug = [{"ticker": t, "price": p, "avg_vol": int(av), "rel_vol": round(rv, 2)} for t, p, av, rv in filtered]
    with open("scan_debug.json", "w") as f:
        json.dump({"source_count": len(ticks), "kept": len(final), "items": debug}, f, indent=2)
    log("[DONE] tickers.txt + scan_debug.json geschrieben.")

if __name__ == "__main__":
    main()
