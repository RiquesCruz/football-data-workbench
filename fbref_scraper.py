"""
FBref Defensive Stats Scraper — v2.0 (Selenium)
Hendrick / football-data-workbench

Scrapes 4 tables × 5 ligues FBref 2025-26 → CSV compatible FBREF_DEF_FEATURES
Output columns: tackles, interceptions, blocks, clearances, pressures,
                progressive_passes, progressive_carries, progressive_passes_received,
                take_ons_won, miscontrols, fouls, cards_yellow, cards_red, aerials_won

Usage:
    pip install selenium webdriver-manager pandas beautifulsoup4 lxml
    python fbref_scraper.py
    python fbref_scraper.py --season 2024-2025
    python fbref_scraper.py --leagues eng esp
"""

import argparse
import time
import io
import logging
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Comment
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# ─── Configuration ────────────────────────────────────────────────────────────

LEAGUES = {
    "eng": {"name": "Premier League", "fbref_id": "9",  "slug": "Premier-League"},
    "esp": {"name": "La Liga",        "fbref_id": "12", "slug": "La-Liga"},
    "ger": {"name": "Bundesliga",     "fbref_id": "20", "slug": "Bundesliga"},
    "ita": {"name": "Serie A",        "fbref_id": "11", "slug": "Serie-A"},
    "fra": {"name": "Ligue 1",        "fbref_id": "13", "slug": "Ligue-1"},
}

TABLES = {
    # stat_type (URL path) → target feature names
    "defense":    ["tackles", "interceptions", "blocks", "clearances"],
    "possession": ["progressive_carries", "take_ons_won", "miscontrols", "progressive_passes_received"],
    "misc":       ["fouls", "cards_yellow", "cards_red"],
    "passing":    ["progressive_passes"],
    # Note: aerials_won and pressures not available in FBref 2025-26 player stat pages
}

# Exact flattened column name → feature name
COLUMN_MAP = {
    # defense
    "tackles_tklw":       "tackles",
    "int":                "interceptions",
    "blocks_blocks":      "blocks",
    "clr":                "clearances",
    # possession
    "carries_1/3":        "progressive_carries",   # carries into final third ≈ progressive carries
    "take-ons_succ":      "take_ons_won",
    "carries_mis":        "miscontrols",
    "rec":                "progressive_passes_received",
    # misc
    "performance_fls":    "fouls",
    "performance_crdy":   "cards_yellow",
    "performance_crdr":   "cards_red",
    "performance_won":    "aerials_won",
    # passing
    "1/3":                "progressive_passes",    # passes into final third ≈ progressive passes
    # standard stats
    "expected_aerialswon":  "aerials_won",
    "aerialswon":           "aerials_won",
    "performance_aerialswon": "aerials_won",
}

RATE_LIMIT = 6
DEFAULT_SEASON = "2025-2026"
OUTPUT_FILE = "fbref_def_features.csv"

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Driver Selenium ──────────────────────────────────────────────────────────

_driver = None

def get_driver(headless: bool = False):
    global _driver
    if _driver is None:
        opts = uc.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        _driver = uc.Chrome(options=opts, headless=headless, use_subprocess=True)
        log.info("✓ Driver Chrome (undetected) initialisé")
    return _driver


def quit_driver():
    global _driver
    if _driver:
        _driver.quit()
        _driver = None


# ─── Fetch ────────────────────────────────────────────────────────────────────

def fetch_page(url: str) -> Optional[BeautifulSoup]:
    clean_url = url.split("#")[0]
    log.info(f"  GET {clean_url}")
    try:
        driver = get_driver()
        driver.get(clean_url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(6)  # laisse JS finir + Cloudflare challenge
        return BeautifulSoup(driver.page_source, "lxml")
    except Exception as e:
        log.error(f"  Échec fetch : {e}")
        return None


# ─── Parsing ──────────────────────────────────────────────────────────────────

def find_table(soup: BeautifulSoup, table_id: str) -> Optional[pd.DataFrame]:
    tag = soup.find("table", {"id": table_id})
    if tag is None:
        for c in soup.find_all(string=lambda t: isinstance(t, Comment) and table_id in t):
            sub = BeautifulSoup(c, "lxml")
            tag = sub.find("table", {"id": table_id})
            if tag:
                break
    if tag is None:
        return None
    try:
        return pd.read_html(io.StringIO(str(tag)), header=[0, 1])[0]
    except Exception as e:
        log.warning(f"  Erreur parsing {table_id}: {e}")
        return None


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c).strip().lower() for c in col if "unnamed" not in str(c).lower()).strip("_")
            for col in df.columns
        ]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def clean_player_df(df: pd.DataFrame) -> pd.DataFrame:
    if "player" in df.columns:
        df = df[df["player"].notna() & (df["player"] != "Player")]
    if "squad" in df.columns:
        df = df[df["squad"].notna() & (df["squad"] != "Squad")]
    return df.reset_index(drop=True)


def extract_target_columns(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col in COLUMN_MAP and COLUMN_MAP[col]:
            rename[col] = COLUMN_MAP[col]
        else:
            for key, val in COLUMN_MAP.items():
                if col.endswith(f"_{key}") and val:
                    rename[col] = val
                    break
    df = df.rename(columns=rename)
    keep = ["player", "squad"]
    for t in targets:
        if t in df.columns:
            keep.append(t)
        else:
            log.warning(f"    Colonne absente : '{t}'")
    return df[[c for c in keep if c in df.columns]]


def numeric_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            col = df[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            df[c] = pd.to_numeric(col, errors="coerce").fillna(0)
    return df


# ─── Scraper principal ────────────────────────────────────────────────────────

def build_url(league_id: str, slug: str, stat_type: str, season: str) -> str:
    return (
        f"https://fbref.com/en/comps/{league_id}/{season}/{stat_type}/"
        f"{season}-{slug}-Stats"
    )


def scrape_league_table(league_key: str, stat_type: str, season: str) -> Optional[pd.DataFrame]:
    league = LEAGUES[league_key]
    url = build_url(league["fbref_id"], league["slug"], stat_type, season)
    soup = fetch_page(url)
    if soup is None:
        return None
    table_id = f"stats_{stat_type}"
    df = find_table(soup, table_id)
    if df is None:
        log.warning(f"  Table '{table_id}' introuvable — {league['name']}")
        return None
    df = flatten_columns(df)
    df = clean_player_df(df)
    df = extract_target_columns(df, TABLES[stat_type])
    df = numeric_columns(df, TABLES[stat_type])
    df["league"] = league["name"]
    df["season"] = season
    log.info(f"  ✓ {league['name']} / {stat_type} → {len(df)} joueurs")
    return df


def scrape_all(season: str, league_keys: list) -> pd.DataFrame:
    table_dfs = {t: [] for t in TABLES}
    total = len(league_keys) * len(TABLES)
    count = 0

    for lkey in league_keys:
        for stat_type in TABLES:
            count += 1
            log.info(f"[{count}/{total}] {LEAGUES[lkey]['name']} — {stat_type}")
            df = scrape_league_table(lkey, stat_type, season)
            if df is not None:
                table_dfs[stat_type].append(df)
            if count < total:
                time.sleep(RATE_LIMIT)

    merged_tables = []
    for stat_type, dfs in table_dfs.items():
        if dfs:
            merged_tables.append(pd.concat(dfs, ignore_index=True))

    if not merged_tables:
        log.error("Aucune donnée collectée.")
        return pd.DataFrame()

    base = merged_tables[0]
    for other in merged_tables[1:]:
        dup = [c for c in other.columns if c in base.columns and c not in ["player", "squad", "league", "season"]]
        other = other.drop(columns=dup, errors="ignore")
        base = pd.merge(base, other, on=["player", "squad", "league", "season"], how="outer")

    for col in {c for cols in TABLES.values() for c in cols}:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    return base


def validate_output(df: pd.DataFrame) -> None:
    required = list({c for cols in TABLES.values() for c in cols})
    present = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]
    log.info("─" * 60)
    log.info(f"Joueurs total : {len(df)}")
    log.info(f"Colonnes présentes ({len(present)}/14) : {present}")
    if missing:
        log.warning(f"Colonnes manquantes : {missing}")
    if "league" in df.columns:
        for league, n in df["league"].value_counts().items():
            log.info(f"  {league}: {n} joueurs")
    log.info("─" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="FBref scraper (Selenium)")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUES.keys()), choices=list(LEAGUES.keys()))
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--delay", type=float, default=RATE_LIMIT)
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless (may be blocked by Cloudflare)")
    return parser.parse_args()


def main():
    args = parse_args()
    global RATE_LIMIT, _driver
    RATE_LIMIT = args.delay

    # Pre-init driver with correct headless setting
    get_driver(headless=args.headless)

    log.info("=" * 60)
    log.info(f"FBref Scraper v2 (Selenium) — saison {args.season}")
    log.info(f"Ligues : {args.leagues} | Délai : {RATE_LIMIT}s")
    log.info("=" * 60)

    try:
        df = scrape_all(season=args.season, league_keys=args.leagues)
    finally:
        quit_driver()

    if df.empty:
        log.error("Aucune donnée — CSV non généré.")
        return

    validate_output(df)

    stat_cols_ordered = [
        "tackles", "interceptions", "blocks", "clearances", "pressures",
        "progressive_passes", "progressive_carries", "progressive_passes_received",
        "take_ons_won", "miscontrols", "fouls", "cards_yellow", "cards_red", "aerials_won",
    ]
    meta_cols = ["player", "squad", "league", "season"]
    final_cols = meta_cols + [c for c in stat_cols_ordered if c in df.columns]
    df[final_cols].to_csv(args.output, index=False, encoding="utf-8")
    log.info(f"✓ CSV sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
