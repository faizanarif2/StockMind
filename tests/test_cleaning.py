import pandas as pd

from stockmind.data.cleaning import clean_historical_data


def test_invalid_ohlc_record_is_quarantined():
    df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "volume": 100,
                "is_anomaly": False,
            },
            {
                "date": "2026-01-02",
                "open": 10,
                "high": 9,
                "low": 8,
                "close": 10,
                "volume": 100,
                "is_anomaly": False,
            },
        ]
    )

    clean, quarantine = clean_historical_data(df)

    assert len(clean) == 1
    assert len(quarantine) == 1
    assert "high_below_ohlc_value" in quarantine.iloc[0]["anomaly_flags"]


def test_large_price_move_is_flagged_not_removed():
    df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "volume": 100,
                "is_anomaly": False,
            },
            {
                "date": "2026-01-02",
                "open": 10,
                "high": 16,
                "low": 9,
                "close": 15,
                "volume": 100,
                "is_anomaly": False,
            },
        ]
    )

    clean, quarantine = clean_historical_data(df)

    assert len(clean) == 2
    assert quarantine.empty
    assert "daily_move_over_30%" in clean.iloc[1]["anomaly_flags"]


def test_duplicate_date_is_quarantined():
    df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "volume": 100,
                "is_anomaly": False,
            },
            {
                "date": "2026-01-01",
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11,
                "volume": 120,
                "is_anomaly": False,
            },
        ]
    )

    clean, quarantine = clean_historical_data(df)

    assert len(clean) == 1
    assert len(quarantine) == 1
    assert "duplicate_date" in quarantine.iloc[0]["anomaly_flags"]