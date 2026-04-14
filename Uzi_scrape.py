
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, re, sys

#data taken from kworb.net

URL = "https://kworb.net/spotify/artist/4O15NlyKLIASxsJ0PrXPfz_songs.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def clean_num(s):
    """'1,048,576' -> 1048576"""
    try:
        return int(re.sub(r"[^\d]", "", str(s)))
    except:
        return None

def scrape():
    print(f"Fetching {URL} ...")
    resp = requests.get(URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    print(f"Status: {resp.status_code} | Content length: {len(resp.text):,} chars")

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Debug: show all tables found ──────────────────────────
    tables = soup.find_all("table")
    print(f"\nTables found on page: {len(tables)}")
    for i, t in enumerate(tables):
        ths = [th.get_text(strip=True) for th in t.find_all("th")]
        nrows = len(t.find_all("tr"))
        print(f"  Table {i}: {nrows} rows | headers: {ths}")

    if not tables:
        print("\n[ERROR] No tables found. Saving raw HTML for inspection...")
        with open("debug_page.html", "w") as f:
            f.write(resp.text)
        print("Saved debug_page.html — open in a browser to see what kworb returned.")
        sys.exit(1)

    # ── Pick the biggest table (the song list) ─────────────────
    table = max(tables, key=lambda t: len(t.find_all("tr")))
    raw_headers = [th.get_text(strip=True) for th in table.find_all("th")]
    print(f"\nUsing table with headers: {raw_headers}")

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)

    if not rows:
        print("[ERROR] Table has no data rows.")
        sys.exit(1)

    # Pad / trim every row to match header count
    ncols = len(raw_headers)
    rows = [r[:ncols] + [""] * max(0, ncols - len(r)) for r in rows]

    df = pd.DataFrame(rows, columns=raw_headers)
    print(f"\nRaw shape: {df.shape}")
    print(df.head(5).to_string())

    # ── Standardise column names ───────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {}
    for col in df.columns:
        if col in ("song", "title", "track", "name"):
            col_map[col] = "Song"
        elif "stream" in col or col == "total":
            col_map[col] = "Streams"
        elif col in ("daily", "day"):
            col_map[col] = "Daily"
        elif col in ("pk", "peak", "peak_position"):
            col_map[col] = "Peak"
        elif col in ("+/-", "change", "d"):
            col_map[col] = "Change"
    df = df.rename(columns=col_map)
    print(f"\nRenamed columns: {list(df.columns)}")

    # ── Clean numerics ─────────────────────────────────────────
    for col in ["Streams", "Daily", "Peak"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_num)

    # Drop empty song name rows
    name_col = "Song" if "Song" in df.columns else df.columns[0]
    df = df[df[name_col].notna() & (df[name_col].str.strip() != "")]

    print(f"\nFinal shape: {df.shape}")
    if "Streams" in df.columns:
        print(f"Streams range: {df['Streams'].min():,} → {df['Streams'].max():,}")

    df.to_csv("uzi_songs.csv", index=False)
    print(f"\n✓ Saved {len(df)} songs to uzi_songs.csv")
    return df


if __name__ == "__main__":
    df = scrape()
    print("\n── Top 10 by streams ──")
    if "Streams" in df.columns and "Song" in df.columns:
        print(df.nlargest(10, "Streams")[["Song", "Streams", "Daily", "Peak"]].to_string(index=False))