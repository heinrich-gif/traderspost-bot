#!/usr/bin/env python3
import os, re, time, json
import requests
from bs4 import BeautifulSoup
import yfinance as yf

# ---- Parameter (ENV) ----
PRICE_MIN        = float(os.getenv("PENNY_PRICE_MIN", "0.2"))
PRICE_MAX        = float(os.getenv("PENNY_PRICE_MAX", "6"))          # 6, damit 5.xx mit reinkommt
AVG_VOL_MIN      = int(os.getenv("PENNY_AVG_VOL_MIN", "150000"))      # etwas lockerer
REL_VOL_MIN      = float(os.getenv("PENNY_REL_VOL_MIN", "1.2"))       # >1.0 = heute mehr als Schnitt
MAX_TICKERS      = int(os.getenv("PENNY_MAX_TICKERS", "40"))
YF_PERIOD        = os.getenv("PENNY_YF_PERIOD", "5d")
YF_INTERVAL      = os.getenv("PENNY_YF_INTERVAL", "1d")

# Nur FINVIZ
FINVIZ_FILTERS   = os.getenv("FINVIZ_FILTERS", "")  # optional zusätzl. finviz f= Filter (CSV)
EXTRA_TICKERS    = os.getenv("EXTRA_TICKERS", "")   # CSV, um manuell zu ergänzen

OUTFILE          = "tickers.txt"
DEBUG_JSON       = "scan_debug.json"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) FinvizPennyScanner/1.0"}

TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

def log(msg): print(msg, flush=True)

def finviz_tickers():
    # Basisfilter: Preis <= PRICE_MAX, AvgVol >= AVG_VOL_MIN, RelVol >= REL_VOL_MIN
    f = [
        f"sh_price_u{int(PRICE_MAX)}",
        f"sh_avgvol_o{int(AVG_VOL_MIN)}",
        f"sh_relvol_o{REL_VOL_MIN}",
    ]
    if FINVIZ_FILTERS.strip():
        f += [x.strip() for x in FINVIZ_FILTERS.split(",") if x.strip()]
    url = f"https://finviz.com/screener.ashx?v=111&f={','.join(f)}&o=-change"
    log(f"[FINVIZ] URL: {url}")
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
    except Exception as e:
        log(f"[FINVIZ] fetch error: {e}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    tickers = []
    for a in soup.select("a.screener-link-primary"):
        t = a.get_text(strip=True).upper()
        if TICKER_RE.match(t):
            tickers.append(t)
    # Dedupe, Reihenfolge bewahren
    return list(dict.fromkeys(tickers))

def yf_validate(tickers):
    out = []
    for t in tickers:
        try:
            df = yf.download(
                t, period=YF_PERIOD, interval=YF_INTERVAL,
                progress=False, auto_adjust=True, prepost=True,
                group_by="column", threads=False
            )
            if df is None or df.empty:
                continue
            price = float(df["Close"].iloc[-1])
            vols = df["Volume"].dropna()
            if vols.empty:
                continue
            avg_vol = float(vols.mean())
            last_vol = float(vols.iloc[-1])
            rel_vol = (last_vol / avg_vol) if avg_vol > 0 else 1.0

            # endgültige Filter (untere Preisgrenze hier anwenden)
            if not (PRICE_MIN <= price <= PRICE_MAX):
                continue
            if avg_vol < AVG_VOL_MIN:
                continue
            if rel_vol < REL_VOL_MIN:
                continue

            out.append((t, price, avg_vol, rel_vol))
        except Exception as e:
            log(f"[YF] {t} error: {e}")
        time.sleep(0.05)
    # Sortiere nach RelVol, dann AvgVol
    out.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return out

def parse_extra(csv_):
    return [x.strip().upper() for x in csv_.split(",") if TICKER_RE.match(x.strip().upper())]

def main():
    base = finviz_tickers()
    log(f"[FINVIZ] Roh: {len(base)}")
    # Optional manuell ergänzen (z. B. PRPH,XFOR,ZENA,RMBL,ANY)
    base = list(dict.fromkeys(base + parse_extra(EXTRA_TICKERS)))

    if not base:
        open(OUTFILE, "w").close()
        log("[DONE] keine Ticker gefunden.")
        return

    validated = yf_validate(base)
    final = [t for t, _, _, _ in validated[:MAX_TICKERS]]

    with open(OUTFILE, "w") as f:
        f.write("\n".join(final))
    with open(DEBUG_JSON, "w") as f:
        json.dump(
            [{"ticker": t, "price": p, "avg_vol": int(av), "rel_vol": round(rv,2)} for t, p, av, rv in validated],
            f, indent=2
        )
    log(f"[DONE] tickers.txt: {len(final)} Ticker → {final}")

if __name__ == "__main__":
    main()
