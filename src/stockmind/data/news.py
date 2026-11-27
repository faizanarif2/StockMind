"""News for the Sentiment Agent (Agent 4): company announcements and media
headlines.

Sources. Each one allows automated access in its robots.txt, needs no API
key, and is a recognised source of Pakistani business news:

    PSX company announcements   Official disclosures the company files with
      dps.psx.com.pk            the exchange: results, dividends, gas
      /announcements            discoveries, contracts, buy-backs, board
                                changes. Timestamped to the minute and
                                published the same day, with years of history.
    Publisher RSS feeds         What the business press reports: Business
      config.NEWS_FEEDS         Recorder, Dawn, Express Tribune and
                                ProPakistani. A feed only shows its latest
                                ~30 headlines, so this history builds up from
                                the first run onwards (see history_store.py).
    Press archive               Express Tribune's sitemap: every story of the
      config.NEWS_PRESS_ARCHIVE last year at once (about 2.5 MB compressed,
                                once a month), so sector and market news
                                exist from the first run. The headline is read
                                from the web address, so it is lower-case and
                                without punctuation, and the day is estimated
                                (see estimate_publication_days). When the same
                                story is also in an RSS feed, the RSS version
                                is kept.

Sources tested and rejected (2026-11-25):
    Google News search feeds   robots.txt forbids automated access.
    Business Recorder search   robots.txt forbids it (/search).
    ProPakistani older pages   one week of news is ~20 feed pages, so a year
                               would take ~1,000 requests.
    GDELT news API             12-17 s per request, rate-limited, and almost
                               no coverage of PSX companies.
    Text of announcement PDFs  most are scanned images with no text, and
                               downloads took up to 90 s.

What is kept: the headline, the publisher's short summary (RSS only), the
source, a link and the time, not full articles. Headlines are what FinBERT
(Agent 4) is designed for (its benchmark, Financial PhraseBank, is
sentence-level). This also keeps storage small and respects copyright.

Every item is tagged so Agent 4 can weigh it:
    kind         "announcement" (PSX) or "article" (media)
    relevance    "company"  names the company (every announcement)
                 "sector"   about the company's industry, e.g. urea prices
                            for a fertilizer maker (config.NEWS_SECTOR_KEYWORDS)
                 "market"   about the market as a whole, e.g. KSE-100 reports
    category     announcement type: results, dividend, operations, ...
    informative  False when the headline itself carries no news: procedural
                 notices (board meetings, report transmissions, dividend
                 payment notices) and a bare "Material Information". There is
                 nothing in those for FinBERT to read.

Fetching and parsing are separate so the parsers can be tested offline.
"""

import hashlib
import html
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import etree
from psxdata import BaseScraper
from psxdata.constants import BASE_URL
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stockmind import config


NEWS_COLUMNS = [
    "article_id", "published_at", "kind", "relevance", "category",
    "informative", "title", "summary", "source", "url",
]

ANNOUNCEMENTS_PAGE_SIZE = 100   # the most PSX returns per request
MAX_ANNOUNCEMENT_PAGES = 30     # safety stop when paging through history
REQUEST_TIMEOUT = 30            # seconds


def empty_news() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)


def _article_id(prefix: str, *parts) -> str:
    text = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha1(text.encode()).hexdigest()[:16]}"


def _words(text: str) -> str:
    """Upper-case text with punctuation turned into single spaces, padded so
    a term can be matched at a word start: " BUY BACK " for "Buy-back"."""
    return " " + re.sub(r"[^A-Z0-9&]+", " ", text.upper()).strip() + " "


# ---------------------------------------------------------------------------
# PSX company announcements
# ---------------------------------------------------------------------------

class AnnouncementScraper(BaseScraper):
    """PSX's announcement search (psxdata's BaseScraper handles the request
    token, retries and rate limiting)."""

    def search(
        self,
        symbol: str,
        date_from: date | None = None,
        date_to: date | None = None,
        offset: int = 0,
    ) -> str:
        return self._request("POST", f"{BASE_URL}/announcements", data={
            "type": "C", "symbol": symbol, "query": "",
            "count": ANNOUNCEMENTS_PAGE_SIZE, "offset": offset,
            "date_from": date_from.isoformat() if date_from else "",
            "date_to": date_to.isoformat() if date_to else "",
            "page": "annc",
        }).text


_scraper: AnnouncementScraper | None = None


def _get_scraper() -> AnnouncementScraper:
    global _scraper
    if _scraper is None:
        _scraper = AnnouncementScraper()
    return _scraper


def fetch_announcements(symbol: str) -> pd.DataFrame:
    """The latest page of a company's announcements (up to 100)."""
    return parse_announcements(_get_scraper().search(symbol))


def fetch_announcement_history(symbol: str, years: int) -> pd.DataFrame:
    """
    Every announcement of the last `years` years, page by page, including
    those filed under the company's earlier symbols
    (config.NEWS_PREDECESSOR_SYMBOLS). Those keep the old symbol in `source`,
    e.g. "PSX (as DAWH)", so Agent 4 can tell them apart.

    Raises ValueError if PSX returns nothing for any of the symbols: a listed
    company always has announcements over several years, and an empty answer
    (PSX does send these occasionally) must not be cached as if it were the
    real history.
    """
    end = date.today()
    start = end - timedelta(days=round(365.25 * years))
    frames = []
    for filed_as in (symbol, *config.NEWS_PREDECESSOR_SYMBOLS.get(symbol, ())):
        history = _symbol_history(filed_as, start, end)
        if history.empty:
            raise ValueError(f"PSX returned no announcements for {filed_as}")
        if filed_as != symbol:
            history["source"] = f"PSX (as {filed_as})"
        frames.append(history)
    # During a restructuring both companies often file the same notice;
    # keep one copy (the current symbol's, which comes first).
    combined = pd.concat(frames, ignore_index=True)
    same_notice = pd.DataFrame({"day": combined["published_at"].dt.date,
                                "title": combined["title"].map(_words)})
    return combine_news([combined[~same_notice.duplicated()]])


def _symbol_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    """All announcements filed under one symbol between two dates."""
    pages, offset = [], 0
    for _ in range(MAX_ANNOUNCEMENT_PAGES):
        page_html = _get_scraper().search(symbol, start, end, offset)
        page = parse_announcements(page_html)
        pages.append(page)
        offset += ANNOUNCEMENTS_PAGE_SIZE
        if page.empty or offset >= parse_total(page_html):
            break
    return combine_news(pages)


def parse_total(page_html: str) -> int:
    """How many announcements match the search in total (0 if unknown)."""
    tag = BeautifulSoup(page_html, "lxml").find(attrs={"data-total": True})
    try:
        return int(tag["data-total"]) if tag else 0
    except ValueError:
        return 0


def parse_announcements(page_html: str) -> pd.DataFrame:
    """
    One row per announcement, in the NEWS_COLUMNS layout.

    The article id is PSX's own document number when there is one
    ("psx-289733"), so the same announcement is never stored twice.
    """
    records = []
    for row in BeautifulSoup(page_html, "lxml").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        day, clock = (cells[i].get_text(" ", strip=True) for i in (0, 1))
        title = re.sub(r"\s+", " ", cells[4].get_text(" ", strip=True))
        published = pd.to_datetime(f"{day} {clock}", format="%b %d, %Y %I:%M %p",
                                   errors="coerce")
        if not title or pd.isna(published):
            continue

        pdf = row.find("a", href=re.compile(r"\.pdf$", re.I))
        image = row.find(attrs={"data-images": True})
        document = (re.search(r"(\d+)", pdf["href"]) if pdf else
                    re.search(r"(\d+)", image["data-images"]) if image else None)
        category, informative = classify_announcement(title)
        records.append({
            "article_id": (f"psx-{document.group(1)}" if document
                           else _article_id("psx", published, title)),
            "published_at": published,
            "kind": "announcement",
            "relevance": "company",
            "category": category,
            "informative": informative,
            "title": title,
            "summary": None,
            "source": "PSX",
            "url": f"{BASE_URL}{pdf['href']}" if pdf else None,
        })
    return pd.DataFrame(records, columns=NEWS_COLUMNS)


# (category, headline carries news?, terms matched at a word start).
# The first matching rule wins, so procedural notices are caught before
# the subjects they mention (e.g. "Credit of ... Dividend").
ANNOUNCEMENT_RULES: tuple[tuple[str, bool, tuple[str, ...]], ...] = (
    ("board_meeting", False, ("BOARD MEETING", "CLOSED PERIOD")),
    ("general_meeting", False, ("GENERAL MEETING", "AGM", "EOGM", "BRIEFING",
                                "PROXY", "POSTAL BALLOT", "ELECTRONIC VOTING",
                                "E VOTING")),
    # "Unusual movement in the price of the shares": PSX asks the company to
    # confirm it has no undisclosed news; the headline says nothing about
    # the company ("UNUSAL" is a misspelling PSX itself has published).
    ("notice", False, ("NEWSPAPER", "ADVERTISEMENT", "WITHHOLDING",
                       "UNUSUAL MOVEMENT", "UNUSAL MOVEMENT")),
    ("report", False, ("TRANSMISSION OF", "ANNUAL REPORT", "QUARTERLY REPORT",
                       "SHARIAH", "AUDITOR")),
    ("insider_disclosure", False, ("DISCLOSURE OF INTEREST", "TRANSACTION IN SHARES",
                                   "SHAREHOLDING")),
    ("dividend_payment", False, ("CREDIT OF", "UNCLAIMED", "BOOK CLOSURE")),
    ("financial_results", False, ("FINANCIAL RESULT",)),
    ("dividend", True, ("DIVIDEND", "BONUS", "RIGHT SHARE", "RIGHT ISSUE")),
    ("operations", True, ("DISCOVERY", "PRODUCTION", "COMMENCEMENT", "AGREEMENT",
                          "CONTRACT", "ACQUISITION", "MERGER", "AMALGAMATION",
                          "BUY BACK", "BUYBACK", "EXPANSION", "PROJECT", "PLANT",
                          "INVESTMENT", "TARIFF", "LICENCE", "LICENSE")),
    ("governance", True, ("APPOINT", "RESIGN", "ELECTION OF DIRECTOR",
                          "CHIEF EXECUTIVE", "CEO", "CHAIRMAN")),
    ("material_information", True, ("MATERIAL INFORMATION", "PRICE SENSITIVE",
                                     "ADDITIONAL INFORMATION")),
)

# Words that say nothing about what happened in a material-information title.
_GENERIC_WORDS = {"DISCLOSURE", "OF", "MATERIAL", "INFORMATION", "ADDITIONAL",
                  "PRICE", "SENSITIVE", "PSX", "THE", "LIMITED", "LTD", "CO",
                  "COMPANY", "AND", "&", "TO", "REGARDING", "RE", "SUBJECT"}


def classify_announcement(title: str) -> tuple[str, bool]:
    """(category, informative) for an announcement title."""
    words = _words(title)
    for category, informative, terms in ANNOUNCEMENT_RULES:
        if any(f" {term}" in words for term in terms):
            if category == "material_information":
                # "Disclosure of Material Information of Meezan Bank" names
                # only the company; the news itself is in a scanned PDF.
                remaining = [w for w in words.split()
                             if w not in _GENERIC_WORDS and not w.isdigit()]
                informative = len(remaining) > 3
            return category, informative
    return "other", True


# ---------------------------------------------------------------------------
# Publisher RSS feeds
# ---------------------------------------------------------------------------


def reclassify_announcements(items: pd.DataFrame) -> pd.DataFrame:
    """
    Classify stored announcements again with today's rules, so a change to
    ANNOUNCEMENT_RULES also applies to announcements kept from earlier runs.
    """
    rows = items["kind"] == "announcement"
    if not rows.any():
        return items
    items = items.copy()
    classified = items.loc[rows, "title"].map(classify_announcement)
    items.loc[rows, "category"] = classified.map(lambda result: result[0])
    items.loc[rows, "informative"] = classified.map(lambda result: result[1])
    items["informative"] = items["informative"].astype(bool)
    return items

def _session() -> requests.Session:
    """HTTP session with a browser User-Agent and automatic retries."""
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    )
    retries = Retry(total=2, backoff_factor=1,
                    status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_feeds(feeds: dict[str, tuple[str, str]] | None = None) -> dict:
    """
    Download every feed in parallel.

    Returns {"items": DataFrame of all headlines, "failed": [feed names]}.
    One feed failing does not stop the others; all of them failing raises,
    so the cache can fall back to the previous download.
    """
    feeds = feeds or config.NEWS_FEEDS

    def download(name: str) -> pd.DataFrame | None:
        source, url = feeds[name]
        try:
            response = _session().get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException:
            return None
        return parse_feed(response.content, source)

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
        results = dict(zip(feeds, pool.map(download, feeds)))

    failed = [name for name, frame in results.items() if frame is None]
    if len(failed) == len(feeds):
        raise requests.ConnectionError("none of the news feeds could be downloaded")
    return {"items": combine_news([f for f in results.values() if f is not None]),
            "failed": failed}


def _plain_text(markup: str, limit: int) -> str | None:
    """Strip HTML and cut to `limit` characters at a word boundary."""
    text = BeautifulSoup(html.unescape(markup or ""), "lxml").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text


def _pakistan_time(value: str | None) -> pd.Timestamp:
    """An RSS date as a Pakistan-time timestamp without a time zone, like
    every other date in StockMind."""
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        return pd.NaT
    return stamp.tz_convert(config.MARKET_TIMEZONE).tz_localize(None)


def parse_feed(content: bytes | str, source: str) -> pd.DataFrame:
    """One row per RSS item, in the NEWS_COLUMNS layout (relevance unset)."""
    records = []
    for item in BeautifulSoup(content, "xml").find_all("item"):
        title = html.unescape(item.title.get_text(" ", strip=True)) if item.title else ""
        link = item.link.get_text(strip=True) if item.link else ""
        published = _pakistan_time(item.pubDate.get_text(strip=True)
                                   if item.pubDate else None)
        if not title or pd.isna(published):
            continue
        records.append({
            "article_id": _article_id("rss", link or title),
            "published_at": published,
            "kind": "article",
            "relevance": None,
            "category": None,
            "informative": True,
            "title": re.sub(r"\s+", " ", title),
            "summary": _plain_text(item.description.get_text() if item.description
                                   else "", config.NEWS_SUMMARY_CHARS),
            "source": source,
            "url": link or None,
        })
    return pd.DataFrame(records, columns=NEWS_COLUMNS)


# ---------------------------------------------------------------------------
# Press archive (a publisher's sitemap)
# ---------------------------------------------------------------------------

# Article ids of archive headlines start with this, which also marks their
# titles as rebuilt from a web address (lower case, no punctuation).
ARCHIVE_PREFIX = "archive"
MAX_ARCHIVE_PAGES = 10          # safety stop: 10 pages are about 4 years
ARCHIVE_TIMEOUT = 90            # seconds; a page is ~800 KB compressed
_STORY = re.compile(r"/story/(\d+)/([^/?#]+)")


def is_archive(items: pd.DataFrame) -> pd.Series:
    """True for the rows that come from the press archive."""
    return items["article_id"].astype(str).str.startswith(ARCHIVE_PREFIX + "-")


def headline_from_url(url: str) -> str | None:
    """
    The headline in a story's web address:
    ".../story/2585222/cement-sales-grow-97-to-258m-tonnes"
        -> "Cement sales grow 97 to 258m tonnes"
    None when the address has no story id or no words.
    """
    match = _STORY.search(url or "")
    if not match or match.group(2).isdigit():
        return None
    words = " ".join(match.group(2).replace("-", " ").split())
    return words[:1].upper() + words[1:] if words else None


def parse_sitemap(content: bytes | str, source: str) -> pd.DataFrame:
    """
    One row per story in a sitemap, in the NEWS_COLUMNS layout (relevance
    unset). `published_at` is the day the story was last modified; see
    estimate_publication_days.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"{source} sitemap is not valid XML ({exc})") from exc

    records = []
    for entry in root.iterfind("{*}url"):
        url = (entry.findtext("{*}loc") or "").strip()
        title = headline_from_url(url)
        modified = pd.to_datetime(entry.findtext("{*}lastmod"), errors="coerce")
        if not title or pd.isna(modified):
            continue
        if modified.tzinfo is not None:
            modified = modified.tz_convert(config.MARKET_TIMEZONE).tz_localize(None)
        records.append({
            "article_id": _article_id(ARCHIVE_PREFIX, url),
            "published_at": modified.normalize(),
            "kind": "article",
            "relevance": None,
            "category": None,
            "informative": True,
            "title": title,
            "summary": None,
            "source": source,
            "url": url,
        })
    return pd.DataFrame(records, columns=NEWS_COLUMNS)


def estimate_publication_days(items: pd.DataFrame) -> pd.DataFrame:
    """
    Replace each story's last-modified day with the day it was published.

    A sitemap gives the day a story was last MODIFIED: an old story edited
    today looks like today's news (2.7% of Express Tribune's stories). Story
    ids are numbered in publication order, so a story was published no later
    than any story with a higher id was last modified. The earliest of
    those days is used; checked against the RSS feeds' exact times, it is
    the correct day or the next one, never earlier.

    The time is set to the end of that day: a backtest on that day then
    cannot see a story that may have been published after the market
    closed. Stories published today count from tomorrow; the RSS feeds,
    with exact times, cover today.
    """
    if items.empty:
        return items
    items = items.copy()
    story = items["url"].str.extract(_STORY, expand=True)[0].astype("int64")
    newest_first = story.sort_values(ascending=False).index
    published = items.loc[newest_first, "published_at"].cummin()
    items.loc[newest_first, "published_at"] = (published + pd.Timedelta(days=1)
                                               - pd.Timedelta(seconds=1))
    return items


def fetch_press_archive(days: int | None, max_pages: int = MAX_ARCHIVE_PAGES) -> pd.DataFrame:
    """
    Every story of the last `days` days from config.NEWS_PRESS_ARCHIVE,
    page by page (newest first) until the pages reach that far back.
    With `days=None`, every page up to `max_pages` is read.
    """
    source, address = config.NEWS_PRESS_ARCHIVE
    today = pd.Timestamp.now(tz=config.MARKET_TIMEZONE).tz_localize(None).normalize()
    oldest = today - pd.Timedelta(days=days) if days else None
    session = _session()
    pages = []
    for page in range(1, max_pages + 1):
        response = session.get(address.format(page=page), timeout=ARCHIVE_TIMEOUT)
        if response.status_code == 404 and pages:
            break                                   # past the last page
        response.raise_for_status()
        items = parse_sitemap(response.content, source)
        if items.empty:
            break
        pages.append(items)
        if oldest is not None and items["published_at"].min() < oldest:
            break
    if not pages:
        raise ValueError(f"{source} sitemap has no stories")

    items = estimate_publication_days(pd.concat(pages, ignore_index=True))
    if oldest is not None:
        items = items[items["published_at"] >= oldest]
    return combine_news([items])


def fetch_press_latest() -> pd.DataFrame:
    """The press archive's newest page (~5 months): keeps it up to date."""
    return fetch_press_archive(None, max_pages=1)


# ---------------------------------------------------------------------------
# Matching headlines to a company
# ---------------------------------------------------------------------------

def company_names(ticker: str, name: str | None) -> list[str]:
    """
    The names the press uses for a company, derived from its PSX name:

        "Fauji Fertilizer Company Limited" -> Fauji Fertilizer Company,
                                              Fauji Fertilizer
        "Oil & Gas Development Company Limited"
                -> Oil & Gas Development Company,
                   Oil and Gas Development Company  (+ OGDCL from config)
        "Systems Limited" -> Systems Limited  (one word alone is too vague)

    plus config.NEWS_EXTRA_NAMES for well-known short names.
    """
    names = set(config.NEWS_EXTRA_NAMES.get(ticker, ()))
    if name:
        full = re.sub(r"^\s*the\s+", "", name, flags=re.I).strip()
        base = re.sub(r"[\s,.]*\b(limited|ltd)\.?\s*$", "", full, flags=re.I).strip()
        if len(base.split()) < 2:
            base = full
        candidates = {base}
        short = re.sub(r"\s+(company|corporation|co\.?)$", "", base, flags=re.I)
        # Dropping "Company" is safe for short names ("Hub Power") but not
        # for descriptive ones ("Oil & Gas Development" is a generic phrase).
        if short != base and 2 <= len(short.split()) <= 3 and "&" not in short:
            candidates.add(short)
        for candidate in list(candidates):
            candidates.add(re.sub(r"\s*&\s*", " and ", candidate))
            # Web addresses drop "&": "oil-gas-development-company".
            candidates.add(re.sub(r"\s*&\s*", " ", candidate))
        names |= candidates
    return sorted(names, key=len, reverse=True)


def _pattern(terms, ignore_case: bool) -> re.Pattern | None:
    terms = [t for t in terms if t]
    if not terms:
        return None
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.I if ignore_case else 0)


def match_articles(
    items: pd.DataFrame, ticker: str, name: str | None, sector: str | None
) -> pd.DataFrame:
    """
    Keep the headlines that concern this company and tag their relevance:

        company  the headline or summary names the company: one of its
                 names (any case) or its ticker (upper case only, so the
                 ticker LUCK does not match the word "luck")
        sector   the headline uses one of the sector's keywords
        market   the headline is about the market as a whole

    Archive headlines (rebuilt from web addresses, so lower case and from
    every section of the paper) follow two extra rules: the ticker matches
    in any case unless it is an ordinary word (config.NEWS_WORD_TICKERS),
    and a sector keyword needs a business term with it
    (config.NEWS_BUSINESS_TERMS).

    Everything else is dropped.
    """
    if items is None or items.empty:
        return empty_news()

    by_name = _pattern(company_names(ticker, name), ignore_case=True)
    by_ticker = _pattern([ticker], ignore_case=False)
    by_sector = _pattern(config.NEWS_SECTOR_KEYWORDS.get(sector or "", ()), ignore_case=True)
    by_market = _pattern(config.NEWS_MARKET_KEYWORDS, ignore_case=True)

    def found(text: pd.Series, pattern: re.Pattern | None) -> pd.Series:
        if pattern is None:
            return pd.Series(False, index=text.index)
        return text.str.contains(pattern, na=False)

    title = items["title"].fillna("").astype(str)
    text = title + " " + items["summary"].fillna("").astype(str)
    archive = is_archive(items)

    company = found(text, by_name) | found(text, by_ticker)
    if ticker.upper() not in config.NEWS_WORD_TICKERS:
        company |= archive & found(text, _pattern([ticker], ignore_case=True))
    sector_news = found(title, by_sector)
    sector_news &= ~archive | found(title, _pattern(config.NEWS_BUSINESS_TERMS, True))
    market = found(title, by_market)

    relevance = np.select([company, sector_news, market],
                          ["company", "sector", "market"], default="")
    tagged = items.assign(relevance=relevance)
    return tagged[tagged["relevance"] != ""].reset_index(drop=True)


def combine_news(frames) -> pd.DataFrame:
    """
    Join news tables: one row per article id, newest first.

    A story both in an RSS feed and in the press archive (same web address)
    is kept once, as the RSS item: its real headline and exact time.
    """
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return empty_news()
    news = pd.concat(frames, ignore_index=True).reindex(columns=NEWS_COLUMNS)
    news["published_at"] = pd.to_datetime(news["published_at"])
    news["informative"] = news["informative"].astype(bool)
    news = news.drop_duplicates("article_id", keep="first")

    archive = is_archive(news)
    if archive.any():
        covered = set(news.loc[~archive, "url"].dropna())
        news = news[~(archive & news["url"].isin(covered))]
    return (news.sort_values("published_at", ascending=False, kind="stable")
                .reset_index(drop=True))
