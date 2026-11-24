"""Tests for news collection: parsing, classification and matching (offline)."""

import pandas as pd
import pytest
import requests

from stockmind.data import news
from stockmind.data.news import (
    classify_announcement,
    combine_news,
    company_names,
    match_articles,
    parse_announcements,
    parse_feed,
    parse_total,
)

# conftest.py replaces the fetchers with "offline" stubs; keep the real ones.
REAL_FETCH_FEEDS = news.fetch_feeds
REAL_FETCH_HISTORY = news.fetch_announcement_history


ANNOUNCEMENTS_HTML = """
<a class="form__button next" data-offset="0" data-total="281">Next</a>
<table>
  <tr><th>DATE</th><th>TIME</th><th>SYMBOL</th><th>NAME</th><th>TITLE</th><th></th></tr>
  <tr><td>Nov 11, 2026</td><td>8:00 AM</td><td>OGDC</td><td>Oil &amp; Gas Development</td>
      <td>GAS/CONDENSATE DISCOVERY AT BITRISIM NORTHWEST-1 EXPLORATORY WELL</td>
      <td><a data-images="288951-1.gif">View</a><a href="/download/document/288951.pdf">PDF</a></td></tr>
  <tr><td>Nov 6, 2026</td><td>4:21 PM</td><td>OGDC</td><td>Oil &amp; Gas Development</td>
      <td>Dissemination of Video Recording of Corporate Briefing Session</td>
      <td><a data-images="288835-1.gif">View</a></td></tr>
  <tr><td>not a date</td><td></td><td>OGDC</td><td></td><td>Broken row</td><td></td></tr>
</table>
"""

FEED_XML = """<?xml version="1.0"?>
<rss><channel>
  <item>
    <title>OGDCL profit surges 53pc &amp; declares dividend</title>
    <link>https://www.brecorder.com/news/1</link>
    <description>&lt;p&gt;Oil and Gas Development Company Limited reported a
      &lt;b&gt;higher&lt;/b&gt; profit.&lt;/p&gt;</description>
    <pubDate>Tue, 24 Nov 2026 13:00:00 +0000</pubDate>
  </item>
  <item>
    <title>No date, so it is skipped</title>
    <link>https://www.brecorder.com/news/2</link>
  </item>
</channel></rss>"""


def articles(*rows):
    """Feed items (relevance not yet set) from (title, summary) pairs."""
    return pd.DataFrame([{
        "article_id": f"rss-{i}", "published_at": pd.Timestamp("2026-11-24 10:00"),
        "kind": "article", "relevance": None, "category": None, "informative": True,
        "title": title, "summary": summary, "source": "Dawn", "url": None,
    } for i, (title, summary) in enumerate(rows)], columns=news.NEWS_COLUMNS)


# --- PSX announcements -----------------------------------------------------

def test_announcements_are_parsed_with_document_ids():
    table = parse_announcements(ANNOUNCEMENTS_HTML)

    assert list(table.columns) == news.NEWS_COLUMNS
    assert len(table) == 2  # header and broken rows skipped
    first = table.iloc[0]
    assert first["article_id"] == "psx-288951"
    assert first["published_at"] == pd.Timestamp("2026-11-11 08:00")
    assert first["url"] == "https://dps.psx.com.pk/download/document/288951.pdf"
    assert (first["kind"], first["relevance"], first["source"]) == ("announcement", "company", "PSX")
    # No PDF: the id comes from the scanned image instead.
    assert table.iloc[1]["article_id"] == "psx-288835"
    assert pd.isna(table.iloc[1]["url"])


def test_total_is_read_from_the_pager():
    assert parse_total(ANNOUNCEMENTS_HTML) == 281
    assert parse_total("<table></table>") == 0


@pytest.mark.parametrize("title, category, informative", [
    ("GAS/CONDENSATE DISCOVERY AT BITRISIM NORTHWEST-1 WELL", "operations", True),
    ("Completion of purchase period for buy-back of shares", "operations", True),
    ("Credit of First Interim Cash Dividend (D-100)", "dividend_payment", False),
    ("NOTICE OF BOARD MEETING", "board_meeting", False),
    ("Financial Results for the Quarter Ended September 30, 2026", "financial_results", False),
    ("Transmission of Quarterly Report for the Period Ended June 30, 2026", "report", False),
    ("Resolutions passed at the 29th Annual General Meeting", "general_meeting", False),
    ("Appointment of Chief Executive Officer", "governance", True),
    ("Disclosure of Material Information of Meezan Bank", "material_information", False),
    ("HUBCO - PSX - 8379 Disclosure of Material Information 16-11-", "material_information", False),
    ("Material Information - New gas field brought into production", "operations", True),
    ("Material Information: Revision in tariff approved by regulator", "operations", True),
    ("Transfer of BESOS shares back to Government of Pakistan", "other", True),
    # Procedural notices seen on PSX that carry no news in the headline.
    ("Publication of Postal Ballot and Provision of Electronic Voting", "general_meeting", False),
    ("SYS | Systems Limited Investor briefing", "general_meeting", False),
    ("SYS | Systems Limited Auditors Certificate", "report", False),
    ("Newspaper Advertisement regarding Withholding of Tax on Dividend", "notice", False),
    ("Additional Information of The Hub Power Company Limited", "material_information", False),
])
def test_announcement_classification(title, category, informative):
    assert classify_announcement(title) == (category, informative)


def test_history_pages_until_total(monkeypatch):
    def page(start_id, count, total):
        rows = "".join(
            f"<tr><td>Nov 1, 2026</td><td>9:00 AM</td><td>X</td><td>X</td>"
            f"<td>Item {i}</td><td><a href='/download/document/{i}.pdf'>PDF</a></td></tr>"
            for i in range(start_id, start_id + count))
        return f"<a data-total='{total}'></a><table>{rows}</table>"

    class FakeScraper:
        offsets = []

        def search(self, symbol, date_from=None, date_to=None, offset=0):
            self.offsets.append(offset)
            return page(offset, min(100, 250 - offset), 250)

    monkeypatch.setattr(news, "_get_scraper", lambda: FakeScraper())
    history = REAL_FETCH_HISTORY("OGDC", years=5)

    assert FakeScraper.offsets == [0, 100, 200]
    assert len(history) == 250


def test_history_includes_announcements_under_earlier_symbols(monkeypatch):
    def row(doc, day, title):
        return (f"<tr><td>{day}</td><td>9:00 AM</td><td>X</td><td>X</td><td>{title}</td>"
                f"<td><a href='/download/document/{doc}.pdf'>PDF</a></td></tr>")

    pages = {
        "NEWCO": row(3, "Jan 2, 2026", "Buy-back of shares")
                 + row(2, "Dec 1, 2025", "Scheme of arrangement approved"),
        "OLDCO": row(1, "Dec 1, 2025", "SCHEME OF ARRANGEMENT APPROVED")  # same notice
                 + row(0, "Jun 1, 2024", "Gas discovery"),
    }

    class FakeScraper:
        def search(self, symbol, date_from=None, date_to=None, offset=0):
            return f"<a data-total='2'></a><table>{pages[symbol]}</table>"

    monkeypatch.setattr(news, "_get_scraper", lambda: FakeScraper())
    monkeypatch.setattr(news.config, "NEWS_PREDECESSOR_SYMBOLS", {"NEWCO": ("OLDCO",)})
    history = REAL_FETCH_HISTORY("NEWCO", years=5)

    assert history["article_id"].tolist() == ["psx-3", "psx-2", "psx-0"]
    assert history["source"].tolist() == ["PSX", "PSX", "PSX (as OLDCO)"]

    # An empty answer for the old symbol is a failed fetch, not "no history".
    pages["OLDCO"] = ""
    with pytest.raises(ValueError):
        REAL_FETCH_HISTORY("NEWCO", years=5)


def test_empty_history_is_an_error_not_a_result(monkeypatch):
    class EmptyScraper:
        def search(self, *args, **kwargs):
            return "<table></table>"

    monkeypatch.setattr(news, "_get_scraper", lambda: EmptyScraper())
    with pytest.raises(ValueError):
        REAL_FETCH_HISTORY("OGDC", years=5)


# --- RSS feeds -------------------------------------------------------------

def test_feed_items_are_parsed_into_plain_text_in_pakistan_time():
    items = parse_feed(FEED_XML, "Business Recorder")

    assert len(items) == 1
    item = items.iloc[0]
    assert item["title"] == "OGDCL profit surges 53pc & declares dividend"
    assert item["summary"] == ("Oil and Gas Development Company Limited reported a "
                               "higher profit.")
    assert item["published_at"] == pd.Timestamp("2026-11-24 18:00")  # UTC+5
    assert item["source"] == "Business Recorder"
    assert item["article_id"].startswith("rss-")


def test_long_summaries_are_shortened(monkeypatch):
    monkeypatch.setattr(news.config, "NEWS_SUMMARY_CHARS", 20)
    item = parse_feed(FEED_XML, "Business Recorder").iloc[0]
    assert len(item["summary"]) <= 23 and item["summary"].endswith("...")


def test_one_failing_feed_does_not_stop_the_others(monkeypatch):
    class Response:
        content = FEED_XML.encode()

        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, timeout):
            if "broken" in url:
                raise requests.ConnectionError("down")
            return Response()

    monkeypatch.setattr(news, "_session", Session)
    feeds = {"good": ("Dawn", "https://example.com/good"),
             "bad": ("Dawn", "https://example.com/broken")}
    result = REAL_FETCH_FEEDS(feeds)

    assert len(result["items"]) == 1
    assert result["failed"] == ["bad"]

    with pytest.raises(requests.ConnectionError):
        REAL_FETCH_FEEDS({"bad": ("Dawn", "https://example.com/broken")})


# --- matching headlines to companies --------------------------------------

@pytest.mark.parametrize("ticker, name, expected", [
    ("FFC", "Fauji Fertilizer Company Limited",
     {"Fauji Fertilizer Company", "Fauji Fertilizer"}),
    ("HUBC", "The Hub Power Company Limited",
     {"Hub Power Company", "Hub Power", "HUBCO"}),
    ("OGDC", "Oil & Gas Development Company Limited",
     {"Oil & Gas Development Company", "Oil and Gas Development Company", "OGDCL"}),
    ("SYS", "Systems Limited", {"Systems Limited"}),
    ("MEBL", None, set()),
])
def test_company_names(ticker, name, expected):
    assert set(company_names(ticker, name)) == expected


def test_headlines_are_tagged_by_relevance():
    items = articles(
        ("Fauji Fertilizer posts record profit", ""),            # company (name)
        ("FFC announces interim dividend", ""),                  # company (ticker)
        ("Gas supply plan", "Fauji Fertilizer gets more gas"),   # company (summary)
        ("Urea offtake rises 20pc in October", ""),              # sector
        ("KSE-100 gains 1,000 points", ""),                      # market
        ("Fauji Foods appoints new CEO", ""),                    # another company
        ("Petrol price unchanged", ""),                          # unrelated
    )
    tagged = match_articles(items, "FFC", "Fauji Fertilizer Company Limited", "FERTILIZER")

    assert tagged["relevance"].tolist() == ["company", "company", "company", "sector", "market"]


def test_foreign_markets_are_not_market_news():
    items = articles(("Asian stocks rally as AI trade revives", ""),
                     ("US stocks surge after last week's losses", ""),
                     ("PSX: buying at bourse, KSE-100 gains over 1,000 points", ""))
    tagged = match_articles(items, "FFC", "Fauji Fertilizer Company Limited", "FERTILIZER")
    assert tagged["title"].tolist() == ["PSX: buying at bourse, KSE-100 gains over 1,000 points"]


def test_tickers_only_match_in_upper_case():
    items = articles(("Investors try their luck with small caps", ""),
                     ("LUCK expands cement capacity", ""))
    tagged = match_articles(items, "LUCK", "Lucky Cement Limited", None)
    assert tagged["title"].tolist() == ["LUCK expands cement capacity"]


def test_sector_terms_are_specific():
    items = articles(("World Bank approves loan for Pakistan", ""),
                     ("Banks' deposits grow 12pc", ""))
    tagged = match_articles(items, "MEBL", "Meezan Bank Limited", "COMMERCIAL BANKS")
    assert tagged["title"].tolist() == ["Banks' deposits grow 12pc"]


def test_no_items_gives_empty_news():
    assert match_articles(pd.DataFrame(), "FFC", "Fauji Fertilizer", None).empty


def test_combine_removes_duplicates_and_sorts_newest_first():
    older = articles(("Old", "")).assign(published_at=pd.Timestamp("2026-01-01"))
    newer = articles(("New", ""), ("Newer", "")).assign(
        article_id=["rss-a", "rss-b"],
        published_at=[pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-01")])
    combined = combine_news([newer, older, newer])

    assert combined["title"].tolist() == ["Newer", "New", "Old"]
    assert combine_news([]).columns.tolist() == news.NEWS_COLUMNS
