from pathlib import Path

import pandas as pd
import polars as pl

try:
    from .paths import DATA_DIR
except ImportError:
    from paths import DATA_DIR


CSV_PATH = DATA_DIR / "filtereddata.csv"
PARQUET_PATH = DATA_DIR / "filtereddata.parquet"

ROAD_FEATURE_COLUMNS = [
    "Crossing",
    "Give_Way",
    "Junction",
    "Stop",
    "Traffic_Signal",
]

DASHBOARD_COLUMNS = [
    "Severity",
    "Weather_Group",
    "Start_Time",
    "State",
    "Start_Lat",
    "Start_Lng",
    "City",
    "Street",
    *ROAD_FEATURE_COLUMNS,
]

OPTIONAL_DASHBOARD_COLUMNS = [
    "Weather_Condition",
    "Temperature(F)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Pressure(in)",
    "Humidity(%)",
]

REQUIRED_DASHBOARD_COLUMNS = {"Weather_Group", "Start_Time", "Severity", "State"}
MISSING_DATA_MESSAGE = (
    "Dataset not found. Run `python scripts/download_data.py` to download "
    "data/filtereddata.parquet."
)


def _prepare_polars_dataframe(dataframe: pl.DataFrame) -> pl.DataFrame:
    available_columns = [
        column
        for column in DASHBOARD_COLUMNS + OPTIONAL_DASHBOARD_COLUMNS
        if column in dataframe.columns
    ]
    data = dataframe.select(available_columns)

    if "Start_Time" in data.columns:
        data = data.with_columns(
            pl.col("Start_Time")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, strict=False)
            .alias("Start_Time")
        )
        data = data.with_columns(
            pl.col("Start_Time").dt.truncate("1mo").alias("Month_Start")
        )

    if "Severity" in data.columns:
        data = data.with_columns(pl.col("Severity").cast(pl.Int64, strict=False))

    for feature in ROAD_FEATURE_COLUMNS:
        if feature in data.columns:
            data = data.with_columns(
                pl.col(feature).cast(pl.Boolean, strict=False).fill_null(False)
            )

    if "Weather_Group" in data.columns:
        data = data.with_columns(
            pl.when(pl.col("Weather_Group").is_null() | (pl.col("Weather_Group") == ""))
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("Weather_Group"))
            .alias("Weather_Group")
        )

    return data


def load_accident_data_polars(prefer_parquet: bool = True) -> pl.DataFrame:
    """Load accident data for the dashboard using Polars."""
    if prefer_parquet and PARQUET_PATH.exists():
        try:
            parquet_df = pl.read_parquet(PARQUET_PATH)
            if REQUIRED_DASHBOARD_COLUMNS.issubset(set(parquet_df.columns)):
                return _prepare_polars_dataframe(parquet_df)
        except Exception:
            pass

    if CSV_PATH.exists():
        csv_df = pl.read_csv(CSV_PATH, try_parse_dates=True)
        return _prepare_polars_dataframe(csv_df)

    raise FileNotFoundError(MISSING_DATA_MESSAGE)


def load_accident_data(prefer_parquet: bool = True) -> pd.DataFrame:
    """Load the full dataset as Pandas for legacy prototype apps."""
    if prefer_parquet and PARQUET_PATH.exists():
        try:
            parquet_df = pd.read_parquet(PARQUET_PATH)
            if REQUIRED_DASHBOARD_COLUMNS.issubset(parquet_df.columns):
                return parquet_df
        except (ImportError, ValueError, OSError):
            pass

    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)

    raise FileNotFoundError(MISSING_DATA_MESSAGE)


def add_time_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add reusable datetime-derived columns for Pandas-based prototype apps."""
    data = dataframe.copy()
    data["Start_Time"] = pd.to_datetime(data["Start_Time"], errors="coerce", utc=True)
    data["Year"] = data["Start_Time"].dt.year
    data["Month"] = data["Start_Time"].dt.month
    return data
