"""Validation and anomaly handling for historical stock prices."""

import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close"]
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def clean_historical_data(
    df: pd.DataFrame,
    large_move_threshold: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate OHLCV data.

    Returns:
        clean_df: Valid rows, including review flags where appropriate.
        quarantine_df: Invalid or duplicate rows with reasons.

    Large price movements are flagged for review, not automatically removed.
    """
    if large_move_threshold <= 0:
        raise ValueError("large_move_threshold must be greater than zero.")

    if df.empty:
        empty = df.copy()
        return empty, empty.copy()

    data = df.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Preserve the source library's anomaly flag.
    if "is_anomaly" in data.columns:
        data = data.rename(columns={"is_anomaly": "source_is_anomaly"})
    else:
        data["source_is_anomaly"] = False

    # Convert the source anomaly flag safely.
    data["source_is_anomaly"] = (
        data["source_is_anomaly"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # Parse dates and numeric values.
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for column in PRICE_COLUMNS + ["volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Record every reason rather than stopping at the first problem.
    reasons = [[] for _ in range(len(data))]
    hard_invalid = [False] * len(data)

    def flag(mask: pd.Series, reason: str, invalid: bool = True) -> None:
        """Add a reason to each row selected by a boolean mask."""
        for position, selected in enumerate(mask.fillna(False).tolist()):
            if selected:
                reasons[position].append(reason)
                if invalid:
                    hard_invalid[position] = True

    flag(data["date"].isna(), "invalid_date")

    for column in PRICE_COLUMNS + ["volume"]:
        flag(data[column].isna(), f"missing_or_invalid_{column}")

    # Price values must be positive.
    for column in PRICE_COLUMNS:
        flag(data[column].le(0), f"non_positive_{column}")

    # Trading volume cannot be negative.
    flag(data["volume"].lt(0), "negative_volume")

    # OHLC consistency checks.
    flag(data["high"].lt(data[["open", "low", "close"]].max(axis=1)),
         "high_below_ohlc_value")
    flag(data["low"].gt(data[["open", "high", "close"]].min(axis=1)),
         "low_above_ohlc_value")

    # Source flags are preserved and surfaced for review.
    flag(
        data["source_is_anomaly"],
        "flagged_by_psxdata",
        invalid=False,
    )

    # Work chronologically when checking daily movements.
    data["_original_order"] = range(len(data))
    data = data.sort_values(
        ["date", "_original_order"],
        na_position="last",
    ).reset_index(drop=True)

    # Reorder the reason lists to match the sorted data.
    original_reasons = reasons
    original_invalid = hard_invalid
    order = data["_original_order"].astype(int).tolist()
    reasons = [original_reasons[i] for i in order]
    hard_invalid = [original_invalid[i] for i in order]

    # Duplicate dates: keep the last occurrence, quarantine earlier ones.
    valid_dates = data["date"].notna()
    duplicate_earlier = (
        data.loc[valid_dates, "date"].duplicated(keep="last")
    )
    duplicate_mask = pd.Series(False, index=data.index)
    duplicate_mask.loc[valid_dates] = duplicate_earlier.to_numpy()

    flag(duplicate_mask, "duplicate_date")

    # Large price movements need investigation but are not automatically invalid.
    daily_return = data["close"].pct_change(fill_method=None)
    large_move = daily_return.abs().gt(large_move_threshold)

    flag(
        large_move,
        f"daily_move_over_{large_move_threshold:.0%}",
        invalid=False,
    )

    data["anomaly_flags"] = [
        ";".join(row_reasons) for row_reasons in reasons
    ]
    data["is_valid"] = [not invalid for invalid in hard_invalid]

    # Invalid records go to quarantine. Review-only flags remain in clean data.
    clean_df = data.loc[data["is_valid"]].copy()
    quarantine_df = data.loc[~data["is_valid"]].copy()

    # Remove internal helper column and sort chronologically.
    for frame in (clean_df, quarantine_df):
        frame.drop(columns=["_original_order"], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    clean_df = clean_df.sort_values("date").reset_index(drop=True)
    quarantine_df = quarantine_df.sort_values(
        "date", na_position="last"
    ).reset_index(drop=True)

    return clean_df, quarantine_df