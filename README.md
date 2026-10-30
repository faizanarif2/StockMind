# StockMind

StockMind is a PSX-focused stock analysis and decision-support system developed as a Final Year Project.

## Supported Companies

StockMind is specialised for the five largest KMI-30 constituents by index weight (checked 2026-10-30):

| Ticker | Company | PSX sector |
|---|---|---|
| FFC | Fauji Fertilizer Company Limited | Fertilizer |
| ENGROH | Engro Holdings Limited | Inv. Banks / Inv. Cos. / Securities Cos. |
| MEBL | Meezan Bank Limited | Commercial Banks |
| OGDC | Oil & Gas Development Company Limited | Oil & Gas Exploration Companies |
| HUBC | The Hub Power Company Limited | Power Generation & Distribution |

The list lives in `src/stockmind/config.py`. Run `python scripts/kmi30_weights.py` to check the current ranking after PSX rebalances the index.

## Current Development Scope

### Agent 1 - Data Collection Agent (implemented, news pending)

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

Amounts are in PKR thousands, per-share values in PKR. For ratios such as ROE, use `income_statement` and `balance_sheet` together: both are consolidated, so they are consistent with each other.

**Sector peers (industry averages).** To judge whether a ratio is good, the Fundamental Agent compares it with the company's industry. Agent 1 finds the `PEERS_PER_COMPANY` (default 4) largest other companies in the same PSX sector by market cap, for example MEBL → UBL, MCB, HBL, NBP. Peer statements use the same layout as the company's own, with an extra `ticker` column.

News collection for the Sentiment Agent will be added later.

**How it stays fast.** Nothing is saved in the project. The agent fetches on demand:

1. All sources, including the peers, are requested in parallel, so a cold request takes about as long as the slowest source (usually 7-10 s; the very first request of the day also builds the market listing, ~20 s).
2. Responses pass through an automatic cache in the user's home folder (`~/.stockmind/cache`, built on `diskcache`). Repeat requests within the max age set in `config.py` take under 0.1 s. This is the same pattern `psxdata`, `yfinance`, and Hugging Face use. The cache is never edited by hand and is safe to delete.
3. If a source is down, the last fetched response is used and a warning is attached instead of failing.

**Scaling to more companies.** Nothing in the code assumes five companies. To support more, add their symbols to `SUPPORTED_TICKERS` in `config.py`. Peers are found automatically from the PSX sector listing, and every source is fetched and cached per symbol, so a company that is both supported and a peer is fetched only once. `DataCollectionAgent().collect_many()` collects all supported companies in one call.

Technical, fundamental, sentiment, risk, and decision-making agents will be developed in later stages.

## Technology Stack

- Python
- psxdata (PSX prices) and the PSX Data Portal
- stockanalysis.com (financial statements, dividend history)
- pandas, BeautifulSoup, requests
- diskcache (response caching)
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
python scripts/collect_data.py          # all supported companies
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

- History depth is set by the sources: about 10 years of prices (PSX always returns its full window), 5 years of statements, and about 5-7 years of dividends.
- PSX `financials` can be standalone figures (e.g. ENGROH), while stockanalysis.com statements are consolidated. Do not mix them in one ratio.
- ENGROH has no payout announcements on PSX since it was renamed from Dawood Hercules (DAWH). `dividend_history` still shows DAWH-era dividends (2019-2024), because it is the same listed company.
- `dividend_history` can include upcoming ex-dates that have been announced but not yet reached.
- Blank or "-" cells at the source are left missing rather than set to zero.
- Peers follow PSX's own sector classification. ENGROH is a holding company, but PSX places it under "Inv. Banks / Inv. Cos. / Securities Cos.", so its peers (e.g. PSX, DLL, IMS) are a weak comparison. Small sectors can have fewer than 4 peers (OGDC has 3).
- Peer averages can be distorted by one outlier (e.g. a very high P/E when earnings are tiny). Use the median for industry comparisons.
- stockanalysis.com is a third-party website. If it changes its page layout, only `stockanalysis.py` needs updating.
