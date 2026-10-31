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

### Agent 3 - Fundamental Analysis Agent (implemented)

`src/stockmind/agents/fundamental_analysis.py` judges how financially healthy a company is, compares it with its sector, and returns a signal for the Decision Agent:

```text
OGDC: BUY (50% confidence, health 62/100)
  - P/E ratio is 5.44x, below the sector median of 5.66x (similar); strong on its own.
  - Return on equity is 16.0%, below the sector median of 28.8% (worse); average on its own.
  ...
```

It only calculates from Agent 1's `CompanyData`. It fetches nothing, so it runs in milliseconds.

**Ratios.** These are calculated in `analysis/ratios.py` from the consolidated statements, for the latest period: TTM if the company has reported quarters since its year end, otherwise the latest fiscal year. The balance sheet is taken at the same date.

| Ratio | Formula | Weight |
|---|---|---|
| P/E | latest close / EPS (a company with no profit scores -1) | 0.20 |
| Return on equity | net income to common / common equity | 0.20 |
| EPS growth | EPS vs the previous 12 months | 0.15 |
| Revenue growth | revenue vs the previous 12 months | 0.15 |
| Net profit margin | net income / revenue | 0.10 |
| Debt-to-equity | total debt / equity (not used for banks) | 0.10 |
| Price-to-book | latest close / book value per share | 0.05 |
| Dividend yield | cash dividends with an ex-date in the last 12 months / price | 0.05 |
| EPS, 3-year revenue growth | shown for context, not scored | — |

**Industry comparison.** The same ratios are calculated for every sector peer from Agent 1. The industry benchmark is the peers' median, not their mean, so one outlier cannot distort it. Peers whose statements are more than ~18 months older than the company's are left out (e.g. KEL for HUBC). With fewer than 2 usable peers, only the absolute benchmarks are used and a warning is attached.

**Signal and confidence.** These are defined in the module docstring so results are reproducible:

1. Each ratio scores -1 (weak) to +1 (strong): the average of an absolute score from fixed benchmarks and an industry score from its gap to the sector median.
2. Overall score = weighted average. A score of +0.2 or more gives **BUY**, -0.2 or less gives **SELL**, anything in between gives **HOLD**. Each ratio also votes with the same rule.
3. `confidence = 100 × coverage × (agreement + strength) / 2`. Coverage is the share of the weight whose ratio could be calculated, and agreement is the share that votes with the final signal. Strength measures how clear the signal is: for BUY or SELL it is how far the score is from 0, and for HOLD how close it is to 0.
4. Health score = 50 × (score + 1), from 0 to 100.

Weights, benchmarks, and thresholds are all in `config.py` (`FUNDAMENTAL_METRICS` and the settings below it).

```python
from stockmind.agents.data_collection import DataCollectionAgent
from stockmind.agents.fundamental_analysis import FundamentalAnalysisAgent

result = FundamentalAnalysisAgent().analyse(DataCollectionAgent().collect("OGDC"))
result.signal        # "BUY", "HOLD" or "SELL"
result.confidence    # 0-100
result.health_score  # 0-100
result.summary       # plain-English overview
result.explanation   # one sentence per ratio
result.metrics       # DataFrame: value, peer_median, scores, weight, vote, comment
result.peer_ratios   # the same ratios for each peer
result.warnings      # e.g. peers left out, ratios that could not be calculated
```

Technical, sentiment, risk, and decision-making agents will be developed in later stages.

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
│       │   ├── data_collection.py # Agent 1
│       │   └── fundamental_analysis.py # Agent 3
│       ├── analysis/
│       │   └── ratios.py          # P/E, ROE, growth, margins from statements
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
│   ├── analyse_fundamentals.py    # run Agents 1 + 3 and print the signal
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
python scripts/analyse_fundamentals.py       # fundamental signal, all companies
python scripts/analyse_fundamentals.py MEBL  # one company
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
- Done: Agent 3 (fundamental ratios, sector comparison, Buy/Hold/Sell signal with confidence and explanation) with automated tests.
- In progress: Technical Analysis Agent (Agent 2).
- Next: Decision Agent v1 (combines Agents 2 and 3) and the basic web application.
- Later: news collection, Sentiment, Risk, and Learning agents.

## Known Data Limitations

- History depth is set by the sources: about 10 years of prices (PSX always returns its full window), 5 years of statements, and about 5-7 years of dividends.
- PSX `financials` can be standalone figures (e.g. ENGROH), while stockanalysis.com statements are consolidated. Do not mix them in one ratio.
- ENGROH has no payout announcements on PSX since it was renamed from Dawood Hercules (DAWH). `dividend_history` still shows DAWH-era dividends (2019-2024), because it is the same listed company.
- `dividend_history` can include upcoming ex-dates that have been announced but not yet reached.
- Blank or "-" cells at the source are left missing rather than set to zero.
- Peers follow PSX's own sector classification. ENGROH is a holding company, but PSX places it under "Inv. Banks / Inv. Cos. / Securities Cos.", so its peers (e.g. PIAHCLA, DLL, PSX, IMS) are a weak comparison, and its fundamental signal should be read with that in mind. Small sectors can have fewer than 4 peers (OGDC has 3).
- stockanalysis.com's TTM `dividend_per_share` can be incomplete (e.g. MEBL shows 7.00 while 29.50 was paid in the last 12 months). Agent 3 therefore takes the company's dividend yield from `dividend_history` and peers' yields from PSX.
- Year-on-year growth can be extreme when the previous year was unusually small. Some power producers' revenue grew over 200%, which makes HUBC's revenue growth look weak next to the sector median. The weights keep any single ratio from deciding the signal.
- Peer averages can be distorted by one outlier (e.g. a very high P/E when earnings are tiny). Use the median for industry comparisons.
- stockanalysis.com is a third-party website. If it changes its page layout, only `stockanalysis.py` needs updating.
