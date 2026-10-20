# StockMind

StockMind is a PSX-focused stock analysis and decision-support system developed as a Final Year Project.

## Current Development Scope

The initial development phase focuses on collecting, storing, and validating historical stock market data from the Pakistan Stock Exchange (PSX).

The current scope includes:

- Fetching historical daily stock prices using the `psxdata` Python library.
- Storing downloaded data locally.
- Updating previously downloaded data without unnecessarily reprocessing the entire history.
- Validating historical price records.
- Detecting and flagging anomalous or invalid records.
- Producing cleaned datasets for future analysis while preserving the original data.

Fundamental analysis, sentiment analysis, technical indicators, risk management, and decision-making agents will be developed in later stages.

## Technology Stack

- Python
- psxdata
- pandas
- Git and GitHub

## Project Structure

```text
StockMind/
├── src/
│   └── stockmind/
│       ├── __init__.py
│       └── data/
│           └── __init__.py
├── scripts/
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── quarantine/
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

## Data Storage

Downloaded stock data and generated datasets are stored locally.

- `data/raw/` stores original fetched data.
- `data/processed/` stores cleaned data for analysis.
- `data/quarantine/` stores records flagged for investigation.

Raw data should be preserved so that cleaning decisions remain traceable. Suspicious records should be flagged and investigated rather than silently deleted.

Generated datasets and local virtual environments are excluded from Git.

## Data Quality Principles

The historical-price pipeline will validate records for issues such as:

- Duplicate trading dates.
- Missing or invalid values.
- Impossible relationships between opening, high, low, and closing prices.
- Unusual price movements requiring investigation.
- Anomalies already identified by the source library.

Unusual values will not automatically be treated as errors because genuine market movements and corporate actions can also produce large price changes.

## Development Status

The historical stock-price pipeline is under development.

Planned components include data fetching, local storage, incremental updates, anomaly detection, data cleaning, and automated tests.
