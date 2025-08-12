# scan_pennystocks_finviz.py
import os
import time
import math
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

# -------- Parameter (per GitHub Vars leicht änderbar) --------
PRICE_MAX = float(os.getenv("PENNY_PRICE_MAX", "5"))          # <= 5$
REL_VOL_MIN = float(os.getenv("PENNY_REL_VOL_MIN", "2"))      # Relative Volume >= 2 (Finviz-Filter)
AVG_VOL_MIN = int(os.getenv("PENNY_AVG_VOL_MIN", "500000"))   # Avg Volume >= 500k (Finviz-Filter)
RANGE_PCT_MIN = float(os.getenv("PENNY_RANGE_PCT_MIN", "5"))  # Zusätzliche Validierung per yfinance
MAX_TICKERS = int(os.getenv("PENNY_MAX_TICKERS", "30"))       # Begrenzung für tickers.txt
YF_PERIOD = os.getenv("PENNY_YF_PERIOD", "5d")                # 5d/10d
YF_INTERVAL = os.getenv("PENNY_YF_INTERVAL", "1d")            # 1d

# Exch-Filter: NASDAQ/NYSE/AMEX (kein OTC)
FINVIZ_BASE = (
    "https://finviz.com/screener.ashx?"
    "v=111&ft=4"  # Tabelle mit vielen Spalten, fund/tech gemischt
    "&f=exch_amex,exch_nasd,exch_nyse"
    ",sh_price_u{price_max}"
    ",sh_relvol_o{rel_vol_min}"
    ",sh_avgvol_o{avg_vol_min}"
    "&o=-relativevolume"  # sortiere absteigend nach RelVol
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://finviz.com/",
}

def finviz_list():
    url = FINVIZ_BASE.format(
        price_max=int(PRICE_MAX),
        rel_vol_min=int(REL_VOL_MIN),
        avg_vol_min=int(AVG_VOL_MIN/1000)*1000  # runde etwas
    )
    tickers = []
    page = 0
    while True:
        start = page * 20 + 1  # Finviz paginiert 20/Seite: r=1,21,41,...
        paged = url + f"&r={start}"
        resp = requests.get(paged, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            if page == 0:
                raise RuntimeError(f"Finviz HTTP {resp.status_code}")
            break
        soup = BeautifulSoup(resp.text, "lxml")
        # Alle Ticker-Links im Table erkennen:
        anchors = soup.select("a.screener-link-primary")
        # Falls Finviz Layout wechselt, fallback:
        if not anchors:
            anchors = soup.select("td a[href*='quote.ashx?t=']")
        page_tickers = []
        for a in anchors:
            sym = a.get_text(strip=True)
            if sym and sym.isupper() and len(sym) <= 6 and sym.isalpha():
                page_tickers.append(sym)
        # Duplikate weg:
        page_tickers = [t for t in page_tickers if t not in tickers]
        if not page_tickers:
            break
        tickers.extend(page_tickers)
        # Max. 5 Seiten (100 Ticker) abklappern:
        page += 1
        if page >= 5:
            break
        time.sleep(0.8)  # freundlich drosseln
    return tickers

def validate_with_yf(cands):
    """
    Filtert mit yfinance nach tatsächlicher Intraday/Day-Range und Liquidität.
    RANGE_PCT_MIN: (High-Low)/Close * 100 der letzten Kerze
    """
    final = []
    for t in cands:
        try:
            df = yf.download(t, period=YF_PERIOD, interval=YF_INTERVAL, progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            vol = int(row.get("Volume", 0))
            if close <= 0:
                continue
            day_range_pct = (high - low) / close * 100.0
            # harte Regeln: Penny, Liquid, Volatil
            if close <= PRICE_MAX and vol >= AVG_VOL_MIN and day_range_pct >= RANGE_PCT_MIN:
                final.append((t, close, vol, round(day_range_pct, 2)))
        except Exception as e:
            print(f"[WARN] yfinance {t}: {e}")
        time.sleep(0.15)
    # sortiere volatilste zuerst
    final.sort(key=lambda x: x[3], reverse=True)
    return [x[0] for x in final[:MAX_TICKERS]]

def main():
    print(f"[SCAN] Finviz… price<=${PRICE_MAX} relvol>={REL_VOL_MIN} avgvol>={AVG_VOL_MIN}")
    raw = finviz_list()
    print(f"[SCAN] Finviz-Kandidaten: {len(raw)} -> {raw[:20]}{' …' if len(raw)>20 else ''}")

    if not raw:
        # Safety: falls Finviz nichts liefert, fallback auf kleine Watchlist
        fallback = ["SGBX","PHUN","COSM","CEI","BBIG","BNGO"]
        print("[SCAN] Finviz leer – fallback:", fallback)
        raw = fallback

    print(f"[SCAN] yfinance-Validierung (RANGE_PCT_MIN={RANGE_PCT_MIN}%) …")
    final_tickers = validate_with_yf(raw)
    print(f"[SCAN] final: {final_tickers}")

    with open("tickers.txt", "w") as f:
        for t in final_tickers:
            f.write(t + "\n")

    if not final_tickers:
        # leeres File vermeiden: schreibe wenigstens 1 Dummy (kein Trade, aber Bot läuft)
        with open("tickers.txt", "w") as f:
            f.write("SGBX\n")
        print("[SCAN] Achtung: kein Kandidat nach Validierung – SGBX als Platzhalter gesetzt.")

if __name__ == "__main__":
    main()
