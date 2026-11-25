# StockMind

StockMind is a PSX-focused stock analysis and decision-support system developed as a Final Year Project.

## Which Companies Are Analysed

StockMind analyses companies from the **KMI-30** index. A selection stage (Stage 0) runs before Agent 1: it reads the current KMI-30 constituents live from PSX, groups them into categories, and selects the most promising companies in each category using the criteria described below. Agent 1 then collects data for the selected companies. Because the index is read live, PSX's rebalancing needs no code changes.

**Stage 0 is the default way to run.** `python scripts/collect_data.py` collects the Stage 0 selection; `python scripts/collect_data.py OGDC PRL` collects named KMI-30 companies instead. The selection itself is kept in the response cache until the next trading session has ended, so agents run at different times analyse the same companies. If PSX is unavailable, the last selection is reused with a warning. If there is no previous selection, the script stops with a clear error; it never silently switches to a fixed list.

`SUPPORTED_TICKERS` in `src/stockmind/config.py` (FFC, ENGROH, MEBL, OGDC, HUBC, the five largest by index weight on 2026-10-30) remains only as the default of `DataCollectionAgent` when it is used directly. Run `python scripts/kmi30_weights.py` to see the current index weights.

## Current Development Scope

### Stage 0 - Stock Selection (implemented)

`src/stockmind/agents/stock_selection.py` decides **which** companies the pipeline analyses. Whether to buy them is still decided by the analysis and decision agents, so selecting "the best of a weak category" is fine: they can still say HOLD or SELL.

```text
KMI-30 (live) -> categories -> eligibility -> rank within category -> top picks -> Agent 1 -> Agents 2, 3, ...
```

**Categories.** Each company's sector is read live from PSX and mapped to a broad, GICS-style category (Energy, Materials, Financials, ...) by `SECTOR_CATEGORIES` in `config.py`. The table covers all 37 PSX sectors, not individual companies, so a company that joins the index is classified automatically. A sector PSX adds in future becomes its own category, with a warning, until it is added to the table.

**Eligibility.** A company is only ranked if it has a full year of prices with at least 200 trading days, has traded in the last 10 days (not suspended), and trades at least PKR 20 million a day on average over the last 3 months.

**Ranking.** Within its category, each measure becomes a percentile rank (best company = 1), and the selection score is the weighted sum (0-100):

| Measure | Definition | Weight |
|---|---|---|
| 12-month return | price change, adjusted for bonus issues | 0.25 |
| Risk-adjusted return | 12-month return / annualised volatility | 0.20 |
| Maximum drawdown | worst fall from a previous peak (smaller is better) | 0.10 |
| Earnings yield | 1 / P/E (no P/E ranks last) | 0.20 |
| Dividend yield | PSX trailing dividend yield | 0.10 |
| Liquidity | average daily traded value | 0.15 |

Ranks are used instead of raw values so that measures in different units can be combined, one extreme value cannot dominate, and companies are only compared with their own category (a bank's low P/E is compared with other banks). Only market data is used: profitability and balance-sheet strength are Agent 3's job, so the selection does not pre-judge it.

**Picks.** One company per category, plus one more for every further 5 eligible companies (`CATEGORY_SIZE_PER_PICK`), so Energy (9 companies) gets 2. Every selection comes with a reason, for example: *"FCCL ranks 1 of 6 eligible companies in Materials (selection score 78/100). 12-month return +12.9%, ..."*.

**Speed.** One price history and one payouts request per constituent, shared with Agent 1's cache: about 30 s cold, under a second afterwards, and Agent 1 then reuses the cached prices.

```python
from stockmind.agents.data_collection import DataCollectionAgent
from stockmind.agents.stock_selection import StockSelectionAgent

selection = StockSelectionAgent().current()  # the session's selection
selection.selected      # e.g. ["GHNI", "TREET", "PRL", "ATRL", "MEBL", ...]
selection.by_category   # {"Energy": ["PRL", "ATRL"], ...}
selection.reasons       # ticker -> plain-English reason
selection.ranking       # every constituent: measures, ranks, score, exclusions
selection.warnings      # e.g. unknown sectors, categories with no eligible company

agent = DataCollectionAgent(universe=selection.universe)
data = agent.collect_many(selection.selected)
```

Files: `agents/stock_selection.py`, `data/universe.py` (index constituents and categories), `analysis/price_adjustments.py` (bonus adjustment, shared with Agent 2), `scripts/select_stocks.py`.

Limitations: past performance does not predict future performance, so this is a screen, not a forecast. A very low P/E can come from one-off profits (e.g. refiners) and still ranks well on earnings yield. Some companies outside the original five have gaps on stockanalysis.com (e.g. PAEL has no dividend page); Agent 1 reports these as warnings.

### Agent 1 - Data Collection Agent (implemented)

`src/stockmind/agents/data_collection.py` fetches everything later agents need for one company, live from the sources:

| `CompanyData` field | Contents | History | Source |
|---|---|---|---|
| `prices` | Daily open, high, low, close, volume, validated | ~10 years | PSX (via `psxdata`) |
| `price_issues` | Rows rejected by validation, with reasons | — | computed |
| `profile` | Name, sector, fiscal year end, shares outstanding, face value, free float, market cap, price, P/E (TTM), 52-week range | current | PSX Data Portal |
| `financials` | Sales / income, profit after tax, EPS (as reported to PSX, may be standalone) | 4 years + 4 quarters | PSX Data Portal |
| `ratios` | Gross and net margin, EPS growth, PEG | 4 years | PSX Data Portal |
| `payouts` | Official cash dividend / bonus / right announcements, book closure | ~2 years | PSX Data Portal |
| `income_statement` | Consolidated revenue, net income, EPS, margins, EBITDA | 5 years + TTM | stockanalysis.com |
| `balance_sheet` | Consolidated assets, liabilities, shareholders' equity, debt, book value per share | 5 years | stockanalysis.com |
| `dividend_history` | Cash dividend per share with ex, record and pay dates | ~5-7 years | stockanalysis.com |
| `peers` | Sector name, the largest other companies in the same PSX sector, their market cap / price / P/E / dividend yield, and their income statements and balance sheets | 5 years | PSX listing + stockanalysis.com |
| `news` | For the Sentiment Agent: the company's official announcements and business-news headlines about the company, its sector or the market, tagged for relevance (see below) | 5 years of announcements; a year of press headlines from the first run, then everything kept | PSX announcements + Business Recorder, Dawn, Express Tribune, ProPakistani + Express Tribune archive |

Amounts are in PKR thousands, per-share values in PKR. For ratios such as ROE, use `income_statement` and `balance_sheet` together: both are consolidated, so they are consistent with each other.

**Sector peers (industry averages).** To judge whether a ratio is good, the Fundamental Agent compares it with the company's industry. Agent 1 finds the `PEERS_PER_COMPANY` (default 4) largest other companies in the same PSX sector by market cap, for example MEBL → UBL, MCB, HBL, NBP. Peer statements use the same layout as the company's own, with an extra `ticker` column.

**News for the Sentiment Agent (Agent 4).** `CompanyData.news` is one table, newest first, built by `src/stockmind/data/news.py` from two kinds of source:

| Source | What it gives | History | Refreshed |
|---|---|---|---|
| PSX company announcements (`dps.psx.com.pk/announcements`) | Official disclosures the company files itself, e.g. results, dividends, "Gas/condensate discovery at Bitrisim Northwest-1", buy-backs, board changes. Timestamped to the minute, published the same day. | 5 years (`NEWS_HISTORY_YEARS`), loaded the first time a company is collected | hourly |
| Business-news RSS feeds (`NEWS_FEEDS`): Business Recorder, Dawn, Express Tribune, ProPakistani | What the press reports. One download serves every company. | Feeds show only their latest ~30 headlines, so headlines are kept from the first run onwards | every 2 hours |
| Press archive (`NEWS_PRESS_ARCHIVE`): Express Tribune's sitemap | Every story the paper published in the last year (`NEWS_PRESS_HISTORY_DAYS`), so sector and market news exist from the first run instead of building up from the feeds. About 2.5 MB compressed | 1 year, loaded on the first run | newest page twice a day, the whole year monthly |

Every source allows automated access in its robots.txt and needs no API key. Sources tested and rejected: Google News search feeds (robots.txt forbids automated access), Business Recorder's search (also forbidden), ProPakistani's older feed pages (one week is ~20 pages, so a year would take ~1,000 requests), the GDELT news API (12-17 s per request, rate-limited, almost no PSX coverage), and the text inside announcement PDFs (mostly scanned images with no text, up to 90 s to download).

Columns (`news.NEWS_COLUMNS`):

| Column | Meaning |
|---|---|
| `article_id` | Stable id (`psx-<PSX document number>`, `rss-<hash of the link>` or `archive-<hash of the link>`), so nothing is stored or analysed twice |
| `published_at` | Pakistan time, like every other date in StockMind |
| `kind` | `announcement` (PSX) or `article` (press) |
| `relevance` | `company` (names the company: every announcement, and headlines using one of its names or its ticker), `sector` (an industry term from `NEWS_SECTOR_KEYWORDS`, e.g. urea for fertilizer makers), `market` (the PSX as a whole, e.g. KSE-100 reports) |
| `category` | Announcements only: `operations`, `dividend`, `governance`, `financial_results`, `board_meeting`, ... |
| `informative` | `False` when the headline itself carries no news: board-meeting notices, report transmissions, dividend payment notices, a bare "Material Information". Over 50% of PSX announcements are of this kind |
| `title`, `summary`, `source`, `url` | The headline, the publisher's summary (at most 400 characters, press only), the source name, and a link to the article or PSX document |

**Why only headlines and short summaries?** Agent 1 does not store or analyse full articles. Headlines are what FinBERT was built and benchmarked for (Financial PhraseBank is made of single sentences), they keep the data small, and storing whole articles would copy copyrighted text. Sentiment scoring belongs to Agent 4: FinBERT is a large model and would slow down every agent that needs Agent 1. Agent 4 should score each `article_id` once and cache the score.

**Companies that changed symbol.** PSX files each announcement under the symbol the company had at the time, so the history also includes the company's earlier symbols (`NEWS_PREDECESSOR_SYMBOLS`), marked in `source`, e.g. `PSX (as DAWH)`. ENGROH is Dawood Hercules (DAWH) renamed in January 2025, when Engro Corporation (ENGRO) became its subsidiary and was delisted, so its history includes both. A notice that both companies filed on the same day is kept once.

**Matching headlines to a company.** Names are derived from PSX's official name, for example "The Hub Power Company Limited" gives "Hub Power Company" and "Hub Power". Short names the press uses are added in `NEWS_EXTRA_NAMES` (OGDCL, HUBCO, ...), and the ticker counts only in upper case, so LUCK does not match "luck". A headline about another company with a similar name, such as Fauji **Foods** for Fauji Fertilizer, is not matched. Market news needs PSX-specific terms, because generic words like "stocks" also match "Asian stocks rally". Stored headlines are matched again, and stored announcements classified again, with the current rules on every run, so changing the keywords applies to older items too.

**The press archive.** A sitemap lists each story's web address and the day it was last modified, not its headline or publication time. Agent 1 therefore:
- reads the headline from the address: `.../story/2585222/cement-sales-grow-97-to-258m-tonnes` gives "Cement sales grow 97 to 258m tonnes" (lower case, and "9.7" becomes "97");
- estimates the publication day: an old story edited today would otherwise look like today's news (2.7% of stories). Story numbers follow publication order, so a story was published no later than any later story was last modified. Checked against the RSS feeds' exact times, the estimate is the correct day or the next, never earlier. The time is set to the end of that day, so a backtest never sees a story before it could have been read;
- applies two extra matching rules, because the archive covers every section of the paper in lower case. The ticker matches in any case ("Ogdc plans partnerships in Libya") unless it is an ordinary word (`NEWS_WORD_TICKERS`: LUCK, ...). A sector keyword needs a business term with it (`NEWS_BUSINESS_TERMS`), so "Bethell looks to cement place" (cricket) is not cement news;
- keeps a story found both in a feed and in the archive once, as the feed version with its real headline and time.

The archive is mostly general news: of its ~57,000 stories a year, about 500 are PSX market reports, and sector stories number in the tens (e.g. ~30 on fertiliser). Stories naming a company are rare (OGDC: about 25).

**Timing: why news is a slower signal than prices.** Announcements reach PSX the same day, often during trading. The press usually reports the following morning, after the price has already moved. Agent 4 should therefore treat news sentiment as a trend over days and weeks rather than a same-day trigger. How to use the table:

- Use `informative` rows.
- Weight by relevance, for example company > sector > market.
- Weight recent items more heavily, as the proposal plans.
- Use the long announcement history for a company's sentiment over time.

Company-specific press headlines are rare: on 2026-11-24, 2 of 184 feed headlines named one of the original five companies. Official announcements are therefore the main per-company source, and sector and market headlines add context.

**How it stays fast.** The agent fetches on demand:

1. All sources, including the peers, are requested in parallel, so a cold request takes about as long as the slowest source (usually 7-10 s; the very first request of the day also builds the market listing, ~20 s).
2. Responses pass through an automatic cache in the user's home folder (`~/.stockmind/cache`, built on `diskcache`). A repeat request takes about 0.15 s (under 0.1 s within the same run). This is the same pattern `psxdata`, `yfinance`, and Hugging Face use. The cache is never edited by hand and is safe to delete.
3. If a source is down, the last fetched response is used and a warning is attached instead of failing. If nothing is cached either, the permanent history store (below) is used.

**Market-aware refreshing.** Market data (prices, the company page, the market listing) is refreshed according to PSX's trading activity, not just a fixed age (`src/stockmind/data/market_clock.py`):

| When | What happens |
|---|---|
| PSX is trading | refreshed every 15 minutes (`SESSION_REFRESH`), so a crash shows up within 15 minutes |
| A session has just ended | fetched once more to get the final closing prices |
| Nights, weekends | nothing is fetched (no request at all) |
| Holidays, unscheduled closures | nothing is fetched after one check of PSX's market state |

All times are Pakistan time. A broad weekly schedule (`TRADING_WINDOWS`, wider than the official hours so pre-open, post-close and Ramadan hours fit) says when the market *may* trade. Inside it, PSX's own live market **State** (Data Portal homepage, regular market) decides, so moon-sighted Eid dates and other closures need no holiday calendar. If the State cannot be read, the market is assumed to be trading, so data is refreshed rather than going stale. Statements, payouts and dividends keep their fixed refresh ages in `CACHE_MAX_AGE`. The age limits there still apply to market data as an upper limit (1 day).

**Permanent history store.** Every source shows only a rolling window: PSX shows 10 years of prices (the oldest day drops off daily) and about 2 years of payouts, and stockanalysis.com shows 5 fiscal years. The cache only keeps the latest download, so older records used to be lost. They are now merged into a permanent store in `~/.stockmind/data` (`src/stockmind/data/history_store.py`, one Parquet file per company and dataset):

| Dataset | One record per | Updated by |
|---|---|---|
| `prices` | date | adding new days; only finished sessions are stored, never a partial day |
| `payouts` | announcement date + details | adding new announcements |
| `dividend_history` | ex-date | adding new dividends |
| `income_statement`, `balance_sheet` | fiscal year + period end + metric | adding new periods; the TTM column is replaced |
| PSX `financials`, `ratios` | period + metric | adding new periods |
| `announcements`, `articles` (news) | article id | adding new items; headlines stay after they leave a feed |

The data handed to the other agents has exactly the same columns as before; over time it simply reaches further back. Company profiles, the market listing and the KMI-30 list describe "now", so they stay in the cache only. When the PSX company page shows a reporting quarter not seen before, the statements are fetched again at once rather than at their weekly refresh.

The store protects its data:

- **Merge, never replace.** Records are only added or corrected. An empty or failed download changes nothing.
- **No duplicates.** One record per key, enforced on every write and read.
- **Corrections are logged.** If a source changes a record we already have, the new value is used and the old and new values are appended to `~/.stockmind/data/corrections.csv`.
- **Suspicious updates are refused.** A download that would change more than 5% of the overlapping records (at least 5 of them), for example after a unit change or a broken page, is not written. A warning asks for review, and the download itself is still passed on, as before.
- **Safe writes.** Each new version is written to a temporary file, read back and checked, then swapped in atomically. A crash never leaves a half-written file, and the previous 3 versions are kept in `_backups/`.
- **Corruption recovery.** An unreadable file is moved aside (never deleted) and the newest readable backup is restored automatically, with a warning.
- **Locking.** Writes to one file are serialised across threads and processes.

**Backing up the history store.** Unlike the cache, the store *cannot* be rebuilt by downloading again, because the sources no longer show its oldest records. Back it up regularly, preferably to another drive or a cloud-synced folder:

```powershell
python scripts/backup_history.py backup                  # zip to ~/.stockmind/backups
python scripts/backup_history.py backup --to D:\Backups  # zip to another folder
python scripts/backup_history.py check                   # read every file, report problems
python scripts/backup_history.py restore <file.zip>      # restore a backup
```

A restore never deletes anything: the current store is first renamed to `data.before-restore-<time>`. Set the `STOCKMIND_DATA_DIR` environment variable to keep the store somewhere else.

**Tested against a live session.** On 2026-11-23, PSX's data was sampled every minute for a full trading day (501 samples):

- **The normal price download is live during a session.** From the open it includes today's unfinished prices, so a fall shows up within 15 minutes (`SESSION_REFRESH`). Agent 1 warns when the latest row is unfinished, and that row is only stored in the history once the session has ended.
- **The market clock reads PSX correctly.** PSX's State moved Ready → Pre-Open → Open (09:35) → Post-Close → Closed (15:50). Prices kept changing until Closed, and the final prices matched the official daily history for every company checked.
- **A faster alternative was tested and dropped.** PSX's Market Watch page lists every company's current prices in one request, but it was never more up to date than the normal download (it only updates every 5 minutes). It would also have needed extra safeguards: it still shows the previous day's prices before the open, and early in the session its total volume ran up to 2.8% ahead of PSX's own figure. It was therefore removed.

**Scaling to more companies.** Nothing in the code assumes five companies. To support more, add their symbols to `SUPPORTED_TICKERS` in `config.py`. Peers are found automatically from the PSX sector listing, and every source is fetched and cached per symbol, so a company that is both supported and a peer is fetched only once. `DataCollectionAgent().collect_many()` collects all supported companies in one call.

Technical, fundamental, sentiment, risk, and decision-making agents will be developed in later stages.

## Technology Stack

- Python
- psxdata (PSX prices) and the PSX Data Portal
- stockanalysis.com (financial statements, dividend history)
- pandas, BeautifulSoup, requests
- diskcache (response caching)
- PyTorch and Hugging Face Transformers (FinBERT sentiment model)
- pytest
- Git and GitHub

## Project Structure

```text
StockMind/
├── src/
│   └── stockmind/
│       ├── config.py              # supported tickers, cache settings
│       ├── agents/
│       │   └── data_collection.py # Agent 1
│       └── data/
│           ├── tickers.py         # ticker validation (e.g. HUBCXD -> HUBC)
│           ├── fetcher.py         # price history (psxdata)
│           ├── company.py         # profile, financials, ratios, payouts (PSX portal)
│           ├── stockanalysis.py   # income statement, balance sheet, dividends
│           ├── market.py          # PSX-wide listing, sector-peer selection
│           ├── cleaning.py        # price validation and anomaly flags
│           └── cache.py           # automatic response cache
├── scripts/
│   ├── collect_data.py            # run Agent 1 and print a summary
│   └── kmi30_weights.py           # current KMI-30 ranking
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

The project structure will expand as additional components are implemented.

## Setup Instructions

### 1. Clone the Repository

```powershell
git clone <repository-url>
cd <repository-folder>
```

Replace `<repository-url>` and `<repository-folder>` with the actual values for the team's GitHub repository.

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` opens the Microsoft Store on Windows, use `py` instead (e.g. `py -m venv .venv`).

### 5. Run the Agent and Tests

```powershell
python scripts/select_stocks.py           # Stage 0: select from the KMI-30
python scripts/backup_history.py backup   # back up the permanent history store

python scripts/collect_data.py          # the Stage 0 selection
python scripts/collect_data.py OGDC     # one company
python scripts/collect_data.py --force  # ignore the cache, fetch live
python -m pytest
```

Using the agent from Python:

```python
from stockmind.agents.data_collection import DataCollectionAgent

data = DataCollectionAgent().collect("OGDC")
data.prices            # DataFrame: date, open, high, low, close, volume, ...
data.profile           # dict: name, sector, equity, market
data.income_statement  # DataFrame: fiscal_year, period_end, metric, value, unit
data.balance_sheet     # same layout, e.g. metric == "shareholders_equity"
data.dividend_history  # DataFrame: ex_date, cash_per_share, record_date, pay_date
data.peers.tickers     # e.g. ["MARI", "PPL", "POL"] for OGDC
data.peers.market      # peers' market cap, price, P/E, dividend yield
data.peers.balance_sheet  # peers' statements, with a "ticker" column
data.warnings          # e.g. "OGDC:prices: source unavailable ... using data fetched at ..."
```

## Agent 4 - Sentiment Analysis Agent (implemented)

`src/stockmind/agents/sentiment_analysis.py` reads the news Agent 1 collected (PSX announcements and business-news headlines about the company, its sector and the market), judges each headline with **FinBERT**, and returns a signal for the Decision Agent. Its result has the same shape and the same signal and confidence rules as Agents 2 and 3 (`analysis/scoring.py`), so Agent 5 can combine all three directly:

```text
OGDC: BUY (29% confidence, sentiment score 67/100, positive)
  - Company news (recent): 2 items in the last 60 days (0 positive, 0 negative, 2 neutral); recency-weighted sentiment +0.06 (neutral).
  - Company news (past year): 59 items in the last 365 days (18 positive, 1 negative, 40 neutral); recency-weighted sentiment +0.25 (positive). Most positive: "Ogdc turns to us firm to boost oil output" (2026-09-02).
  - Sector news: no news in the last 30 days.
  - Market news: 27 items in the last 14 days (14 positive, 10 negative, 3 neutral); recency-weighted sentiment +0.14 (positive).
  WARNING: OGDC: no OIL & GAS EXPLORATION COMPANIES sector headlines in the last 30 days.
```

**Warnings name the cause** of a missing signal, so it can be fixed when it can be: procedural notices only ("only 14 procedural notices, e.g. board meetings"), a sector with no keywords in `NEWS_SECTOR_KEYWORDS`, or headlines stored for less time than the signal's window. When nothing is known about the company itself, the summary says the result reflects only sector or market news.

```powershell
python scripts/analyse_sentiment.py              # the Stage 0 selection
python scripts/analyse_sentiment.py OGDC         # one company
python scripts/analyse_sentiment.py OGDC --headlines 10   # show more headlines
```

```python
from stockmind.agents.data_collection import DataCollectionAgent
from stockmind.agents.sentiment_analysis import SentimentAnalysisAgent

result = SentimentAnalysisAgent().analyse(DataCollectionAgent().collect("OGDC"))
result.signal, result.confidence, result.score   # same fields as Agents 2 and 3
result.sentiment          # "positive", "neutral" or "negative"
result.signals            # one row per signal: items, value, score, evidence, vote
result.articles           # every analysed item with P(positive/negative/neutral)
result.history            # monthly sentiment for company, sector and market news
```

**FinBERT.** FinBERT (`ProsusAI/finbert`, Araci 2019) is BERT trained further on financial text and fine-tuned on Financial PhraseBank. General sentiment models misread financial language ("costs fell" is good news), so the proposal uses FinBERT. It is used as published, with no training. For each headline it gives P(positive), P(negative) and P(neutral); the headline's sentiment is P(positive) - P(negative), from -1 to +1. It reads headlines, not summaries: headlines are what it was built for, and a summary often mentions other companies.

**How the signal is made** (settings in `config.SENTIMENT_*`):

| Signal | News used | Window | Half-life | Weight |
|---|---|---|---|---|
| `company_recent` | The company's announcements and headlines naming it | 60 days | 14 days | 0.45 |
| `company_long_term` | The same, over the past year | 365 days | 120 days | 0.15 |
| `sector` | Industry headlines (e.g. urea prices for a fertilizer maker) | 30 days | 7 days | 0.20 |
| `market` | PSX market headlines (e.g. KSE-100 reports) | 14 days | 3 days | 0.20 |

- **Only informative items** are used. A "Notice of Board Meeting" gives FinBERT nothing to read.
- **Recent news counts more**, as the proposal plans: an item's weight halves every half-life.
- **A signal's score** is its recency-weighted average sentiment / 0.5, capped at +/-1 (most financial headlines are neutral, so averages rarely come near 1).
- **Little news means low confidence, not a confident HOLD.** A signal needs about 4 recent items to count fully; one fresh headline counts a quarter. With no news in its window a signal is skipped.
- **Signal, confidence and the 0-100 sentiment score** follow the same rules as Agents 2 and 3: score >= 0.2 is BUY, <= -0.2 is SELL, and `confidence = 100 x coverage x (agreement + strength) / 2`.

**Speed.** Each headline is analysed once and its score is remembered in `~/.stockmind/cache/sentiment` (keyed by model and text). When every headline has been seen before, the model is not even loaded and the analysis takes about 0.1 s per company (measured 2026-11-25); Agent 1's collection takes about 1.5 s per company with a cached press archive. When there are new headlines, loading PyTorch and FinBERT takes 20-40 s once per run, then about 100 headlines are scored per second on a normal CPU. The first run downloads FinBERT (~440 MB) into the Hugging Face cache (`~/.cache/huggingface`), outside the project, and the model revision is pinned (`SENTIMENT_MODEL_REVISION`) so scores never change silently. PyTorch is installed in its CPU-only version (`requirements.txt` points pip to PyTorch's CPU package index), which is much smaller and needs no graphics card.

**Timing and backtesting.** PSX announcements appear the same day; the press usually reports the next morning, after the price has moved. Read the sentiment as a trend over days and weeks rather than a same-day trigger. `analyse(data, as_of=...)` uses only news published by that date, so a backtest (Agent 7) can ask what the sentiment was at any earlier point.

**Preparing the text.** Pakistani newspapers write "53pc" for "53 percent", which FinBERT does not know: "OGDC profit surges 53pc and declares dividend" scored -0.25, and +0.61 with "percent". "pc" after a number is therefore replaced before scoring.

**Limitations.** Company-specific press headlines are rare, so the company signals rest mostly on PSX announcements, whose titles are short and often neutral. Many companies file only procedural titles for months (e.g. HUBC: "Financial Result for the First Quarter...", "Credit of Interim Cash Dividend"), whose content is in the PDF, so their company signals are often empty and their confidence low. FinBERT also reads plain statements of fact as neutral ("GAS/CONDENSATE DISCOVERY AT BITRISIM NORTHWEST-1..." scored +0.06), because Financial PhraseBank's annotators labelled facts neutral unless they were clearly good or bad. Low confidence is the intended result in these cases: the Decision Agent weighs each agent by its confidence. The press archive fills the market signal from the first run, but sector headlines are few (often none in a 30-day window), and a company with no informative news of its own gets a result from market news alone, at low confidence (e.g. HUBC: BUY, 15%). FinBERT reads one headline at a time and does not know which company a market report is mainly about.

## Data Quality Principles

Price history is validated for issues such as:

- Duplicate trading dates.
- Missing or invalid values.
- Impossible relationships between opening, high, low, and closing prices.
- Unusual price movements requiring investigation.
- Anomalies already identified by the source library.

Unusual values will not automatically be treated as errors because genuine market movements and corporate actions can also produce large price changes. Invalid rows are returned separately in `price_issues` with their reasons rather than silently dropped.

## Development Status

- Done: Agent 1 (prices, profile, financial statements, ratios, dividends) with on-demand fetching, caching, and automated tests.
- Next: Technical Analysis Agent.
- Later: news collection, Fundamental, Sentiment, Decision, Risk, and Learning agents, and the web application.

## Known Data Limitations

- During a trading session the latest price row is today's unfinished session. Agent 1 adds a warning, and the row is not stored in the history until the session has ended.
- History depth is set by the sources: about 10 years of prices (PSX always returns its full window), 5 years of statements, and about 5-7 years of dividends.
- PSX `financials` can be standalone figures (e.g. ENGROH), while stockanalysis.com statements are consolidated. Do not mix them in one ratio.
- ENGROH has no payout announcements on PSX since it was renamed from Dawood Hercules (DAWH). `dividend_history` still shows DAWH-era dividends (2019-2024), because it is the same listed company.
- `dividend_history` can include upcoming ex-dates that have been announced but not yet reached.
- Blank or "-" cells at the source are left missing rather than set to zero.
- Peers follow PSX's own sector classification. ENGROH is a holding company, but PSX places it under "Inv. Banks / Inv. Cos. / Securities Cos.", so its peers (e.g. PSX, DLL, IMS) are a weak comparison. Small sectors can have fewer than 4 peers (OGDC has 3).
- Peer averages can be distorted by one outlier (e.g. a very high P/E when earnings are tiny). Use the median for industry comparisons.
- stockanalysis.com is a third-party website. If it changes its page layout, only `stockanalysis.py` needs updating.
