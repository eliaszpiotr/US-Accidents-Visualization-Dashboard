import math

import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import orjson
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from flask_caching import Cache

try:
    from .data_utils import load_accident_data_polars
    from .paths import ASSETS_DIR, GENERATED_DIR
except ImportError:
    from data_utils import load_accident_data_polars
    from paths import ASSETS_DIR, GENERATED_DIR


severity_levels = [1, 2, 3, 4]
features = ["Crossing", "Give_Way", "Junction", "Stop", "Traffic_Signal"]
SEVERITY_COLORS = {
    1: "#009E73",
    2: "#0072B2",
    3: "#E69F00",
    4: "#D55E00",
}
SEVERITY_COLOR_SCALES = {
    1: [[0.0, "#E6F5EF"], [1.0, "#009E73"]],
    2: [[0.0, "#E6F0F8"], [1.0, "#0072B2"]],
    3: [[0.0, "#FFF4D6"], [1.0, "#E69F00"]],
    4: [[0.0, "#FDE7DD"], [1.0, "#D55E00"]],
}
DEFAULT_FILTER_STATE = {
    "state": None,
    "selected_states": None,
    "selected_point_ids": None,
    "feature": None,
    "severity": None,
    "weather_group": None,
    "month": None,
    "month_range": None,
    "condition_ranges": None,
}
DEFAULT_MAP_CENTER = [39.8283, -98.5795]
DEFAULT_MAP_ZOOM = 3
FILTERED_MAP_ZOOM = 6
# Approximate rendered size of the accident map, used to shape the viewport box on
# the minimap whenever plotly does not hand us real corner coordinates.
MAP_VIEWPORT_PIXELS = (900, 520)
MAX_MAP_POINTS = 10000
MAX_PARALLEL_LINES = 5000
CONDITION_DIMENSIONS = [
    ("Temperature(F)", "Temperature (°F)"),
    ("Humidity(%)", "Humidity (%)"),
    ("Visibility(mi)", "Visibility (mi)"),
    ("Wind_Speed(mph)", "Wind speed (mph)"),
    ("Pressure(in)", "Pressure (in)"),
]
PARALLEL_DIMENSIONS = [("Severity", "Severity"), *CONDITION_DIMENSIONS]
CHOROPLETH_IDS = [f"{feature}-{severity}-choropleth" for feature in features for severity in severity_levels]

PAGE_STYLE = {
    "width": "100%",
    "margin": "0 auto",
    "backgroundColor": "#f4f7fb",
    "minHeight": "100vh",
}
HEADER_STYLE = {
    "width": "100%",
    "minHeight": "68px",
    "background": "linear-gradient(120deg, #0f172a 0%, #1e3a5f 100%)",
    "color": "white",
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "21px",
    "fontWeight": "700",
    "letterSpacing": "0.2px",
    "display": "flex",
    "alignItems": "center",
    "padding": "0 24px",
    "boxSizing": "border-box",
    "boxShadow": "0 4px 16px rgba(15, 23, 42, 0.16)",
}
CONTROL_BAR_STYLE = {
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
    "padding": "14px 16px",
    "margin": "18px 20px",
    "gap": "16px",
    "backgroundColor": "#ffffff",
    "border": "1px solid #dce4ef",
    "borderRadius": "12px",
    "boxShadow": "0 4px 14px rgba(15, 23, 42, 0.05)",
}
GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(min(100%, 560px), 1fr))",
    "gap": "20px",
    "padding": "0 20px 20px 20px",
    "alignItems": "stretch",
}
CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #dce4ef",
    "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.07)",
    "borderRadius": "14px",
    "padding": "18px",
    "boxSizing": "border-box",
    "height": "100%",
    "display": "flex",
    "flexDirection": "column",
}
CARD_TITLE_STYLE = {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "17px",
    "fontWeight": "bold",
    "color": "#172033",
    "marginBottom": "6px",
}
CARD_DESCRIPTION_STYLE = {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "13px",
    "color": "#64748b",
    "marginBottom": "12px",
}
BUTTON_STYLE = {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontWeight": "600",
    "padding": "10px 16px",
    "border": "none",
    "borderRadius": "8px",
    "backgroundColor": "#1d4ed8",
    "color": "white",
    "fontSize": "14px",
    "cursor": "pointer",
}
FOOTER_STYLE = {
    "width": "100%",
    "height": "36px",
    "backgroundColor": "#0f172a",
    "color": "white",
    "fontFamily": "Arial",
    "fontSize": "12px",
    "fontWeight": "600",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "marginTop": "24px",
}
ROAD_GRID_MATRIX_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "110px repeat(4, minmax(0, 1fr))",
    "gridAutoRows": "142px",
    "gap": "10px 12px",
    "alignItems": "stretch",
}
ROAD_GRID_HEADER_STYLE = {
    "fontFamily": "Arial",
    "fontSize": "13px",
    "fontWeight": "bold",
    "textAlign": "center",
    "color": "#111111",
    "alignSelf": "end",
    "paddingBottom": "4px",
}
ROAD_GRID_ROW_LABEL_STYLE = {
    "fontFamily": "Arial",
    "fontSize": "13px",
    "fontWeight": "bold",
    "color": "#111111",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "flex-start",
    "paddingLeft": "4px",
}
ROAD_GRID_SCALE_ROW_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "110px repeat(4, minmax(0, 1fr))",
    "gap": "10px 12px",
    "alignItems": "start",
    "marginTop": "10px",
}

state_full_names = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}
CONTIGUOUS_STATE_CODES = [
    state for state in sorted({*state_full_names.keys(), "DC"}) if state not in {"AK", "HI"}
]
MINIMAP_EXTENT = {"west": -128.0, "east": -65.0, "south": 23.0, "north": 51.0}
MINIMAP_MIN_BOX_FRACTION = 0.09
MINIMAP_VIEWPORT_MAX_COVERAGE = 0.85
MINIMAP_RECTANGLE_EDGE_POINTS = 24
ACCIDENTS_PL = load_accident_data_polars()
if "Accident_ID" not in ACCIDENTS_PL.columns:
    ACCIDENTS_PL = ACCIDENTS_PL.with_row_count("Accident_ID")

app = Dash(__name__, assets_folder=str(ASSETS_DIR))
CACHE_DIR = GENERATED_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "FileSystemCache",
        "CACHE_DIR": str(CACHE_DIR),
        "CACHE_DEFAULT_TIMEOUT": 300,
    },
)


def normalize_month_value(value):
    if value in (None, "", "None"):
        return None
    value_str = str(value)
    if len(value_str) >= 7:
        return value_str[:7]
    return value_str[:7]


def normalize_filter_state(filter_state):
    normalized = DEFAULT_FILTER_STATE.copy()
    if filter_state:
        for key in DEFAULT_FILTER_STATE:
            normalized[key] = filter_state.get(key)

    for key, value in normalized.items():
        if value in ("", "None", "All"):
            normalized[key] = None

    if normalized["severity"] is not None:
        try:
            normalized["severity"] = int(normalized["severity"])
        except (TypeError, ValueError):
            normalized["severity"] = None

    if normalized.get("selected_states") is not None:
        selected_states = normalized["selected_states"]
        if not isinstance(selected_states, list):
            normalized["selected_states"] = None
        else:
            cleaned_states = sorted(
                {
                    str(state)
                    for state in selected_states
                    if state not in (None, "", "None")
                }
            )
            normalized["selected_states"] = cleaned_states or None

    if normalized.get("selected_point_ids") is not None:
        selected_point_ids = normalized["selected_point_ids"]
        if not isinstance(selected_point_ids, list):
            normalized["selected_point_ids"] = None
        else:
            cleaned_ids = []
            for point_id in selected_point_ids:
                try:
                    cleaned_ids.append(int(point_id))
                except (TypeError, ValueError):
                    continue
            normalized["selected_point_ids"] = sorted(set(cleaned_ids)) or None

    normalized["month"] = normalize_month_value(normalized["month"])
    if normalized.get("month_range") is not None:
        month_range = normalized["month_range"]
        if not isinstance(month_range, list) or len(month_range) != 2:
            normalized["month_range"] = None
        else:
            normalized["month_range"] = [
                normalize_month_value(month_range[0]),
                normalize_month_value(month_range[1]),
            ]

    condition_ranges = normalized.get("condition_ranges")
    if not isinstance(condition_ranges, dict):
        normalized["condition_ranges"] = None
    else:
        cleaned_ranges = {}
        valid_columns = {column for column, _ in PARALLEL_DIMENSIONS}
        for column, value in condition_ranges.items():
            if column not in valid_columns or not isinstance(value, list):
                continue
            ranges = value if value and isinstance(value[0], list) else [value]
            cleaned = []
            for range_pair in ranges:
                if not isinstance(range_pair, list) or len(range_pair) != 2:
                    continue
                try:
                    lower, upper = sorted([float(range_pair[0]), float(range_pair[1])])
                except (TypeError, ValueError):
                    continue
                cleaned.append([lower, upper])
            if cleaned:
                cleaned_ranges[column] = cleaned[0] if len(cleaned) == 1 else cleaned
        normalized["condition_ranges"] = cleaned_ranges or None
    return normalized


def make_filter_cache_key(filter_state, scope="default", ignore_keys=None):
    normalized = normalize_filter_state(filter_state)
    for key in ignore_keys or []:
        normalized[key] = None
    payload = {"scope": scope, **normalized}
    return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def parse_filter_cache_key(filter_key):
    payload = orjson.loads(filter_key)
    return normalize_filter_state({key: payload.get(key) for key in DEFAULT_FILTER_STATE})


def get_color(severity):
    return SEVERITY_COLORS.get(int(severity), "gray")


def create_empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#444444"),
                xanchor="center",
                yanchor="middle",
            )
        ],
    )
    return fig


def create_empty_map_figure(message):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#444444"),
                xanchor="center",
                yanchor="middle",
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def apply_filters(dataframe, filter_state, ignore_keys=None):
    filters = normalize_filter_state(filter_state)
    ignored = set(ignore_keys or [])
    filtered = dataframe

    if (
        filters.get("selected_point_ids") is not None
        and "selected_point_ids" not in ignored
        and "Accident_ID" in filtered.columns
    ):
        filtered = filtered.filter(pl.col("Accident_ID").is_in(filters["selected_point_ids"]))
    elif (
        filters.get("selected_states") is not None
        and "selected_states" not in ignored
        and "State" in filtered.columns
    ):
        filtered = filtered.filter(pl.col("State").is_in(filters["selected_states"]))
    elif filters["state"] is not None and "state" not in ignored and "State" in filtered.columns:
        filtered = filtered.filter(pl.col("State") == filters["state"])

    if filters["feature"] is not None and "feature" not in ignored and filters["feature"] in filtered.columns:
        filtered = filtered.filter(pl.col(filters["feature"]).cast(pl.Boolean, strict=False).fill_null(False))

    if filters["severity"] is not None and "severity" not in ignored and "Severity" in filtered.columns:
        filtered = filtered.filter(pl.col("Severity") == filters["severity"])

    if filters["weather_group"] is not None and "weather_group" not in ignored and "Weather_Group" in filtered.columns:
        filtered = filtered.filter(
            pl.col("Weather_Group").fill_null("Unknown") == filters["weather_group"]
        )

    if (
        filters.get("month_range") is not None
        and "month_range" not in ignored
        and "Month_Start" in filtered.columns
    ):
        start_month, end_month = filters["month_range"]
        filtered = filtered.filter(
            (pl.col("Month_Start").dt.strftime("%Y-%m") >= start_month)
            & (pl.col("Month_Start").dt.strftime("%Y-%m") <= end_month)
        )
    elif filters["month"] is not None and "month" not in ignored and "Month_Start" in filtered.columns:
        filtered = filtered.filter(
            pl.col("Month_Start").dt.strftime("%Y-%m") == filters["month"]
        )

    if filters.get("condition_ranges") and "condition_ranges" not in ignored:
        for column, selected_ranges in filters["condition_ranges"].items():
            if column not in filtered.columns:
                continue
            ranges = selected_ranges if isinstance(selected_ranges[0], list) else [selected_ranges]
            range_filter = None
            for lower, upper in ranges:
                current_range = pl.col(column).cast(pl.Float64, strict=False).is_between(lower, upper)
                range_filter = current_range if range_filter is None else (range_filter | current_range)
            if range_filter is not None:
                filtered = filtered.filter(range_filter)

    return filtered


@cache.memoize()
def get_weather_severity_agg_cached(filter_key):
    filter_state = parse_filter_cache_key(filter_key)
    filtered = apply_filters(ACCIDENTS_PL, filter_state)
    if filtered.is_empty():
        return pd.DataFrame(columns=["Weather_Group", "Severity", "Accident_Count", "Total_Count", "Percentage"])
    if "Weather_Group" not in filtered.columns:
        return pd.DataFrame(columns=["Weather_Group", "Severity", "Accident_Count", "Total_Count", "Percentage"])

    grouped = (
        filtered.filter(pl.col("Severity").is_not_null())
        .with_columns(
            pl.col("Weather_Group").fill_null("Unknown").alias("Weather_Group"),
            pl.col("Severity").cast(pl.Int64, strict=False).alias("Severity"),
        )
        .group_by(["Weather_Group", "Severity"])
        .len()
        .rename({"len": "Accident_Count"})
    )
    if grouped.is_empty():
        return pd.DataFrame(columns=["Weather_Group", "Severity", "Accident_Count", "Total_Count", "Percentage"])

    totals = grouped.group_by("Weather_Group").agg(
        pl.col("Accident_Count").sum().alias("Total_Count")
    )
    result = (
        grouped.join(totals, on="Weather_Group", how="left")
        .with_columns(
            (pl.col("Accident_Count") / pl.col("Total_Count") * 100).alias("Percentage")
        )
        .sort(["Total_Count", "Weather_Group", "Severity"], descending=[True, False, False])
    )
    return result.to_pandas()


def get_weather_severity_agg(filter_state):
    return get_weather_severity_agg_cached(make_filter_cache_key(filter_state, scope="weather"))


@cache.memoize()
def get_monthly_severity_agg_cached(filter_key):
    filter_state = parse_filter_cache_key(filter_key)
    filtered = apply_filters(ACCIDENTS_PL, filter_state)
    if filtered.is_empty() or "Month_Start" not in filtered.columns:
        return pd.DataFrame(columns=["Month_Start", "Severity", "Accident_Count"])

    grouped = (
        filtered.filter(pl.col("Start_Time").is_not_null() & pl.col("Severity").is_not_null())
        .with_columns(pl.col("Severity").cast(pl.Int64, strict=False).alias("Severity"))
        .group_by(["Month_Start", "Severity"])
        .len()
        .rename({"len": "Accident_Count"})
        .sort(["Month_Start", "Severity"])
    )
    return grouped.to_pandas()


def get_monthly_severity_agg(filter_state):
    return get_monthly_severity_agg_cached(make_filter_cache_key(filter_state, scope="monthly"))


@cache.memoize()
def get_road_feature_state_agg_cached(filter_key):
    filter_state = parse_filter_cache_key(filter_key)
    base = apply_filters(ACCIDENTS_PL, filter_state, ignore_keys={"state", "feature"})
    results = []

    if base.is_empty():
        return pd.DataFrame(columns=["Feature", "Severity", "State", "Accident_Count"])

    for feature in features:
        if feature not in base.columns:
            continue
        feature_filtered = base.filter(pl.col(feature).cast(pl.Boolean, strict=False).fill_null(False))
        if feature_filtered.is_empty():
            continue
        counts = (
            feature_filtered.group_by(["State", "Severity"])
            .len()
            .rename({"len": "Accident_Count"})
            .with_columns(pl.lit(feature).alias("Feature"))
        )
        results.append(counts)

    if not results:
        return pd.DataFrame(columns=["Feature", "Severity", "State", "Accident_Count"])

    aggregated = pl.concat(results)
    return aggregated.to_pandas()


def get_road_feature_state_agg(filter_state):
    return get_road_feature_state_agg_cached(
        make_filter_cache_key(filter_state, scope="road-grid", ignore_keys={"state", "feature"})
    )


@cache.memoize()
def get_map_points_cached(filter_key):
    filter_state = parse_filter_cache_key(filter_key)
    filtered = apply_filters(ACCIDENTS_PL, filter_state)

    if filtered.is_empty():
        return {
            "records": [],
            "center": DEFAULT_MAP_CENTER,
            "zoom": DEFAULT_MAP_ZOOM,
            "sampled": False,
            "displayed": 0,
            "total": 0,
        }

    if "Start_Lat" not in filtered.columns or "Start_Lng" not in filtered.columns:
        return {
            "records": [],
            "center": DEFAULT_MAP_CENTER,
            "zoom": DEFAULT_MAP_ZOOM,
            "sampled": False,
            "displayed": 0,
            "total": 0,
        }

    filtered = filtered.drop_nulls(["Start_Lat", "Start_Lng"])
    total_matches = filtered.height
    if total_matches == 0:
        return {
            "records": [],
            "center": DEFAULT_MAP_CENTER,
            "zoom": DEFAULT_MAP_ZOOM,
            "sampled": False,
            "displayed": 0,
            "total": 0,
        }

    point_cap = MAX_MAP_POINTS
    sampled = total_matches > point_cap
    if sampled:
        filtered = filtered.sample(n=point_cap, seed=42, shuffle=True)

    center = DEFAULT_MAP_CENTER
    zoom = DEFAULT_MAP_ZOOM
    if filter_state.get("state") is not None:
        center = [
            float(filtered["Start_Lat"].mean()),
            float(filtered["Start_Lng"].mean()),
        ]
        zoom = FILTERED_MAP_ZOOM

    selected_columns = [
        column
        for column in [
            "Accident_ID",
            "Start_Lat",
            "Start_Lng",
            "Severity",
            "State",
            "City",
            "Street",
            "Weather_Group",
            "Start_Time",
            "Month_Start",
        ]
        if column in filtered.columns
    ]
    sampled_points = filtered.select(selected_columns).to_dicts()

    return {
        "records": sampled_points,
        "center": center,
        "zoom": zoom,
        "sampled": sampled,
        "displayed": len(sampled_points),
        "total": total_matches,
    }


def get_map_points(filter_state):
    return get_map_points_cached(make_filter_cache_key(filter_state, scope="map"))


@cache.memoize()
def get_condition_lines_cached(filter_key):
    filter_state = parse_filter_cache_key(filter_key)
    filtered = apply_filters(
        ACCIDENTS_PL,
        filter_state,
        ignore_keys={"condition_ranges"},
    )
    available_columns = [
        column for column, _ in CONDITION_DIMENSIONS if column in filtered.columns
    ]
    if filtered.is_empty() or not available_columns or "Severity" not in filtered.columns:
        return pd.DataFrame(columns=["Severity", *available_columns])

    condition_data = (
        filtered.select(["Severity", *available_columns])
        .with_columns(
            pl.col("Severity").cast(pl.Float64, strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in available_columns
            ],
        )
        .drop_nulls(["Severity", *available_columns])
    )
    if condition_data.height > MAX_PARALLEL_LINES:
        condition_data = condition_data.sample(
            n=MAX_PARALLEL_LINES,
            seed=42,
            shuffle=True,
        )
    return condition_data.to_pandas()


def get_condition_lines(filter_state):
    return get_condition_lines_cached(
        make_filter_cache_key(
            filter_state,
            scope="conditions",
            ignore_keys={"condition_ranges"},
        )
    )


def create_accident_scatter_map(
    df,
    filter_state=None,
    map_revision=0,
    map_height=520,
):
    if df.empty:
        return create_empty_map_figure("No accident points match the current filters")
    required_columns = {"Start_Lat", "Start_Lng", "Severity"}
    if not required_columns.issubset(df.columns):
        return create_empty_map_figure("Map coordinates or severity data are missing")

    map_df = df.copy()
    map_df["Severity"] = pd.to_numeric(map_df["Severity"], errors="coerce")
    map_df = map_df.dropna(subset=["Start_Lat", "Start_Lng", "Severity"])
    if map_df.empty:
        return create_empty_map_figure("No valid accident coordinates are available")

    map_df["Severity"] = map_df["Severity"].astype(int)
    map_df["Severity_Label"] = map_df["Severity"].astype(str)
    if "Accident_ID" in map_df.columns:
        map_df["Accident_ID"] = pd.to_numeric(map_df["Accident_ID"], errors="coerce").fillna(-1).astype(int)
    else:
        map_df["Accident_ID"] = range(len(map_df))
    map_df["State"] = map_df.get("State", pd.Series(index=map_df.index, dtype=object)).fillna("Unknown")
    map_df["City"] = map_df.get("City", pd.Series(index=map_df.index, dtype=object)).fillna("Unknown")
    map_df["Street"] = map_df.get("Street", pd.Series(index=map_df.index, dtype=object)).fillna("Unknown")
    map_df["Weather_Group"] = map_df.get(
        "Weather_Group", pd.Series(index=map_df.index, dtype=object)
    ).fillna("Unknown")
    if "Month_Start" in map_df.columns:
        month_series = pd.to_datetime(map_df["Month_Start"], errors="coerce")
        map_df["Month_Key"] = month_series.dt.strftime("%Y-%m").fillna("Unknown")
    else:
        map_df["Month_Key"] = "Unknown"
    start_time_source = (
        map_df["Start_Time"]
        if "Start_Time" in map_df.columns
        else pd.Series(index=map_df.index, dtype=object)
    )
    map_df["Start_Time_Label"] = pd.to_datetime(
        start_time_source,
        errors="coerce",
    ).dt.strftime("%Y-%m-%d %H:%M").fillna("Unknown")

    normalized_filters = normalize_filter_state(filter_state)
    center = {"lat": DEFAULT_MAP_CENTER[0], "lon": DEFAULT_MAP_CENTER[1]}
    zoom = DEFAULT_MAP_ZOOM
    if normalized_filters.get("state") is not None:
        center = {
            "lat": float(map_df["Start_Lat"].mean()),
            "lon": float(map_df["Start_Lng"].mean()),
        }
        zoom = FILTERED_MAP_ZOOM

    fig = px.scatter_map(
        map_df,
        lat="Start_Lat",
        lon="Start_Lng",
        color="Severity_Label",
        category_orders={"Severity_Label": [str(level) for level in severity_levels]},
        color_discrete_map={str(level): color for level, color in SEVERITY_COLORS.items()},
        custom_data=["Accident_ID", "State", "Severity", "Weather_Group", "Month_Key", "City", "Street"],
        zoom=zoom,
        center=center,
        height=map_height,
    )
    fig.update_traces(
        # Density is read off overlap, so the alpha has to leave headroom: coverage
        # grows as 1-(1-a)^N, and at a=0.7 three stacked points already look the same
        # as thirty. At 0.35 the ramp keeps separating well past a dozen. The smaller
        # radius stops dense metro areas from turning into one solid blob.
        marker=dict(size=6, opacity=0.35),
        selected=dict(marker=dict(opacity=0.9, size=9)),
        unselected=dict(marker=dict(opacity=0.08)),
        hovertemplate=(
            "State: %{customdata[1]}<br>"
            "City: %{customdata[5]}<br>"
            "Street: %{customdata[6]}<br>"
            "Severity: %{customdata[2]}<br>"
            "Weather: %{customdata[3]}<br>"
            "Month: %{customdata[4]}<br>"
            "Start Time: %{text}<extra></extra>"
        ),
        text=map_df["Start_Time_Label"],
    )
    fig.update_layout(
        map=dict(style="carto-positron", center=center, zoom=zoom),
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="Severity",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        clickmode="event+select",
        # Pan is the default gesture: dragging the map should move it, which is what
        # people expect from a map. Lasso and box select live in the modebar, and
        # keeping the map out of select mode also lets plotly's own double-click reset
        # work, since it only reaches maplibre when dragPan is enabled.
        dragmode="pan",
        uirevision=f"accident-map-{map_revision}",
    )
    return fig


def get_map_center_from_relayout(relayout_data):
    center = {"lat": DEFAULT_MAP_CENTER[0], "lon": DEFAULT_MAP_CENTER[1]}
    if not isinstance(relayout_data, dict):
        return center

    map_center = relayout_data.get("map.center")
    if isinstance(map_center, dict):
        try:
            return {
                "lat": float(map_center.get("lat", center["lat"])),
                "lon": float(map_center.get("lon", center["lon"])),
            }
        except (TypeError, ValueError):
            return center

    try:
        return {
            "lat": float(relayout_data.get("map.center.lat", center["lat"])),
            "lon": float(relayout_data.get("map.center.lon", center["lon"])),
        }
    except (TypeError, ValueError):
        return center


def create_map_view_bounds(center, zoom):
    """Approximate the map viewport in degrees, in Web Mercator terms.

    Degrees per pixel is 360 / (512 * 2**zoom) horizontally; vertically the same
    figure is scaled by cos(latitude). Leaving that factor out (and using 180
    instead of 360) made the box roughly twice as flat as the real viewport.
    """
    zoom = max(float(zoom), 0)
    world_pixels = 512 * 2**zoom
    width_pixels, height_pixels = MAP_VIEWPORT_PIXELS
    degrees_per_pixel = 360.0 / world_pixels
    latitude_scale = math.cos(math.radians(max(-85.0, min(85.0, float(center["lat"])))))
    longitude_half_span = min(180.0, degrees_per_pixel * width_pixels) / 2
    latitude_half_span = min(170.0, degrees_per_pixel * height_pixels * latitude_scale) / 2
    return {
        "west": max(-180.0, center["lon"] - longitude_half_span),
        "east": min(180.0, center["lon"] + longitude_half_span),
        "south": max(-85.0, center["lat"] - latitude_half_span),
        "north": min(85.0, center["lat"] + latitude_half_span),
    }


def create_default_map_view_state():
    center = {"lat": DEFAULT_MAP_CENTER[0], "lon": DEFAULT_MAP_CENTER[1]}
    return {
        "center": center,
        "zoom": DEFAULT_MAP_ZOOM,
        "bounds": create_map_view_bounds(center, DEFAULT_MAP_ZOOM),
    }


def update_map_view_state(relayout_data, current_view=None):
    current_view = current_view or create_default_map_view_state()
    previous_center = dict(
        current_view.get("center") or create_default_map_view_state()["center"]
    )
    previous_zoom = float(current_view.get("zoom", DEFAULT_MAP_ZOOM))
    previous_bounds = current_view.get("bounds") or create_map_view_bounds(
        previous_center, previous_zoom
    )
    center = dict(previous_center)
    zoom = previous_zoom
    if isinstance(relayout_data, dict):
        updated_center = get_map_center_from_relayout(relayout_data)
        if "map.center" in relayout_data or any(
            key in relayout_data for key in ("map.center.lat", "map.center.lon")
        ):
            center = updated_center
        try:
            zoom = float(relayout_data.get("map.zoom", zoom))
        except (TypeError, ValueError):
            pass

        derived = relayout_data.get("map._derived") or relayout_data.get("mapbox._derived")
        coordinates = derived.get("coordinates") if isinstance(derived, dict) else None
        if isinstance(coordinates, list) and len(coordinates) >= 4:
            try:
                longitudes = [float(point[0]) for point in coordinates]
                latitudes = [float(point[1]) for point in coordinates]
                return {
                    "center": center,
                    "zoom": zoom,
                    "bounds": {
                        "west": min(longitudes),
                        "east": max(longitudes),
                        "south": min(latitudes),
                        "north": max(latitudes),
                    },
                }
            except (TypeError, ValueError, IndexError):
                pass

    zoom_scale = 2 ** (previous_zoom - zoom)
    longitude_span = (previous_bounds["east"] - previous_bounds["west"]) * zoom_scale
    latitude_span = (previous_bounds["north"] - previous_bounds["south"]) * zoom_scale
    return {
        "center": center,
        "zoom": zoom,
        "bounds": {
            "west": center["lon"] - longitude_span / 2,
            "east": center["lon"] + longitude_span / 2,
            "south": center["lat"] - latitude_span / 2,
            "north": center["lat"] + latitude_span / 2,
        },
    }


def clip_bounds_to_minimap(view_bounds):
    clipped = {
        "west": max(MINIMAP_EXTENT["west"], view_bounds["west"]),
        "east": min(MINIMAP_EXTENT["east"], view_bounds["east"]),
        "south": max(MINIMAP_EXTENT["south"], view_bounds["south"]),
        "north": min(MINIMAP_EXTENT["north"], view_bounds["north"]),
    }
    if clipped["west"] >= clipped["east"] or clipped["south"] >= clipped["north"]:
        return None
    return clipped


def expand_bounds_to_minimum(view_bounds):
    """Keep a deeply zoomed viewport box large enough to stay visible on the minimap."""
    extent_width = MINIMAP_EXTENT["east"] - MINIMAP_EXTENT["west"]
    extent_height = MINIMAP_EXTENT["north"] - MINIMAP_EXTENT["south"]
    min_width = extent_width * MINIMAP_MIN_BOX_FRACTION
    min_height = extent_height * MINIMAP_MIN_BOX_FRACTION

    west, east = view_bounds["west"], view_bounds["east"]
    south, north = view_bounds["south"], view_bounds["north"]
    if east - west < min_width:
        center_lon = (west + east) / 2
        west, east = center_lon - min_width / 2, center_lon + min_width / 2
    if north - south < min_height:
        center_lat = (south + north) / 2
        south, north = center_lat - min_height / 2, center_lat + min_height / 2

    # Push the enlarged box back inside the minimap frame instead of letting it spill out.
    if west < MINIMAP_EXTENT["west"]:
        west, east = MINIMAP_EXTENT["west"], MINIMAP_EXTENT["west"] + (east - west)
    if east > MINIMAP_EXTENT["east"]:
        west, east = MINIMAP_EXTENT["east"] - (east - west), MINIMAP_EXTENT["east"]
    if south < MINIMAP_EXTENT["south"]:
        south, north = MINIMAP_EXTENT["south"], MINIMAP_EXTENT["south"] + (north - south)
    if north > MINIMAP_EXTENT["north"]:
        south, north = MINIMAP_EXTENT["north"] - (north - south), MINIMAP_EXTENT["north"]

    return {
        "west": max(MINIMAP_EXTENT["west"], west),
        "east": min(MINIMAP_EXTENT["east"], east),
        "south": max(MINIMAP_EXTENT["south"], south),
        "north": min(MINIMAP_EXTENT["north"], north),
    }


def create_minimap_rectangle_outline(bounds, edge_points=MINIMAP_RECTANGLE_EDGE_POINTS):
    """Build a closed viewport ring that the geo projection renders as a real rectangle.

    Two things matter here:
    * Winding. d3-geo reads a ring counter-clockwise in lon/lat as the *complement* of
      the area, so the old corner order made the fill cover the whole minimap with the
      box punched out of it. The ring must run clockwise: W->N->E->S.
    * Density. Edges are resampled along great circles, which bows the horizontal sides
      once the box gets wide. Sampling each edge keeps every side on a constant lat/lon.
    """
    west, east = bounds["west"], bounds["east"]
    south, north = bounds["south"], bounds["north"]
    steps = range(edge_points + 1)
    lons_west_to_east = [west + (east - west) * step / edge_points for step in steps]
    lats_south_to_north = [south + (north - south) * step / edge_points for step in steps]

    lon = (
        [west] * len(lats_south_to_north)
        + lons_west_to_east
        + [east] * len(lats_south_to_north)
        + lons_west_to_east[::-1]
    )
    lat = (
        lats_south_to_north
        + [north] * len(lons_west_to_east)
        + lats_south_to_north[::-1]
        + [south] * len(lons_west_to_east)
    )
    return lat, lon


def should_show_minimap_viewport(view_bounds):
    clipped = clip_bounds_to_minimap(view_bounds) if view_bounds else None
    if not clipped:
        return False
    width_ratio = (clipped["east"] - clipped["west"]) / (
        MINIMAP_EXTENT["east"] - MINIMAP_EXTENT["west"]
    )
    height_ratio = (clipped["north"] - clipped["south"]) / (
        MINIMAP_EXTENT["north"] - MINIMAP_EXTENT["south"]
    )
    return (
        width_ratio < MINIMAP_VIEWPORT_MAX_COVERAGE
        and height_ratio < MINIMAP_VIEWPORT_MAX_COVERAGE
    )


def create_map_minimap(
    filter_state=None,
    center=None,
    view_bounds=None,
    show_viewport=False,
):
    filters = normalize_filter_state(filter_state)
    selected_states = set(filters.get("selected_states") or [])
    if filters.get("state"):
        selected_states.add(filters["state"])
    center = center or {"lat": DEFAULT_MAP_CENTER[0], "lon": DEFAULT_MAP_CENTER[1]}
    view_bounds = view_bounds or create_map_view_bounds(center, DEFAULT_MAP_ZOOM)
    view_bounds = clip_bounds_to_minimap(view_bounds)
    if view_bounds is None:
        show_viewport = False

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=CONTIGUOUS_STATE_CODES,
            z=[1 if state in selected_states else 0 for state in CONTIGUOUS_STATE_CODES],
            locationmode="USA-states",
            colorscale=[
                [0, "#e2e8f0"],
                [0.49, "#e2e8f0"],
                [0.5, "#93c5fd"],
                [1, "#3b82f6"],
            ],
            zmin=0,
            zmax=1,
            showscale=False,
            marker_line_color="#ffffff",
            marker_line_width=0.6,
            hoverinfo="skip",
        )
    )
    if show_viewport:
        box_lat, box_lon = create_minimap_rectangle_outline(
            expand_bounds_to_minimum(view_bounds)
        )
        # The viewport box is an annotation, not data, so it must not compete with the
        # selected-state fill. Selection stays a filled area in blue; the box is an
        # outline in near-black ink over a white ring, which separates by channel and
        # by lightness instead of by hue - it stays readable on top of a selected state.
        fig.add_trace(
            go.Scattergeo(
                lat=box_lat,
                lon=box_lon,
                mode="lines",
                fill="toself",
                fillcolor="rgba(255, 255, 255, 0.42)",
                line=dict(width=3.4, color="rgba(255, 255, 255, 0.9)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lat=box_lat,
                lon=box_lon,
                mode="lines",
                line=dict(width=1.4, color="#0f172a"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        geo=dict(
            scope="north america",
            projection_type="equirectangular",
            center=dict(lat=37.0, lon=-96.5),
            lonaxis=dict(range=[MINIMAP_EXTENT["west"], MINIMAP_EXTENT["east"]]),
            lataxis=dict(range=[MINIMAP_EXTENT["south"], MINIMAP_EXTENT["north"]]),
            bgcolor="rgba(0,0,0,0)",
            showland=False,
            showlakes=False,
            showcoastlines=False,
            showcountries=False,
            showframe=False,
        ),
        height=68,
        margin=dict(l=1, r=1, t=1, b=1),
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision="map-minimap",
    )
    return fig


def create_conditions_parallel_coordinates(dataframe, filter_state=None):
    available_dimensions = [
        (column, label)
        for column, label in PARALLEL_DIMENSIONS
        if column in dataframe.columns
    ]
    if dataframe.empty or not available_dimensions or "Severity" not in dataframe.columns:
        return create_empty_figure("No condition data available for selected filters")

    chart_data = dataframe.copy()
    numeric_columns = list(dict.fromkeys(column for column, _ in available_dimensions))
    for column in numeric_columns:
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna(subset=numeric_columns)
    if chart_data.empty:
        return create_empty_figure("No complete condition records match the filters")

    active_ranges = normalize_filter_state(filter_state).get("condition_ranges") or {}
    dimensions = []
    for column, label in available_dimensions:
        values = chart_data[column]
        dimension = dict(
            range=[float(values.min()), float(values.max())],
            label=label,
            values=values,
        )
        if column == "Severity":
            dimension.update(
                range=[1, 4],
                tickvals=severity_levels,
                ticktext=[str(level) for level in severity_levels],
            )
        if column in active_ranges:
            dimension["constraintrange"] = active_ranges[column]
        dimensions.append(dimension)

    severity_colorscale = [
        [0.0, SEVERITY_COLORS[1]],
        [1 / 6, SEVERITY_COLORS[1]],
        [1 / 6, SEVERITY_COLORS[2]],
        [0.5, SEVERITY_COLORS[2]],
        [0.5, SEVERITY_COLORS[3]],
        [5 / 6, SEVERITY_COLORS[3]],
        [5 / 6, SEVERITY_COLORS[4]],
        [1.0, SEVERITY_COLORS[4]],
    ]
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=chart_data["Severity"],
                colorscale=severity_colorscale,
                cmin=1,
                cmax=4,
                showscale=False,
            ),
            dimensions=dimensions,
            labelfont=dict(size=12, color="#222222"),
            tickfont=dict(size=10, color="#555555"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=55, r=35, t=45, b=45),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        uirevision="conditions-parallel-coordinates",
    )
    return fig


def create_weather_severity_stacked_bar(agg_df):
    if agg_df.empty:
        return create_empty_figure("No weather data available for selected filters")
    if "Weather_Group" not in agg_df.columns:
        return create_empty_figure("Weather_Group column is missing")

    chart_data = agg_df.copy()
    chart_data["Severity"] = chart_data["Severity"].astype(int)
    chart_data["Severity_Label"] = chart_data["Severity"].astype(str)
    weather_order = (
        chart_data[["Weather_Group", "Total_Count"]]
        .drop_duplicates()
        .sort_values("Total_Count", ascending=False)["Weather_Group"]
        .tolist()
    )

    fig = px.bar(
        chart_data,
        x="Weather_Group",
        y="Percentage",
        color="Severity_Label",
        category_orders={
            "Weather_Group": weather_order,
            "Severity_Label": [str(level) for level in severity_levels],
        },
        color_discrete_map={str(level): color for level, color in SEVERITY_COLORS.items()},
        custom_data=["Weather_Group", "Severity", "Accident_Count", "Percentage"],
    )
    fig.update_traces(
        hovertemplate=(
            "Weather_Group: %{customdata[0]}<br>"
            "Severity: %{customdata[1]}<br>"
            "Accident count: %{customdata[2]}<br>"
            "Percentage: %{customdata[3]:.2f}%<extra></extra>"
        )
    )
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        clickmode="event+select",
        title=None,
        xaxis_title="Weather condition group",
        # Without a uirevision plotly discards UI state on every redraw, so switching a
        # severity off in the legend was undone as soon as anything rebuilt the figure.
        uirevision="conditions-stacked-bars",
        yaxis_title="Percentage of accidents",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend_title_text="Severity",
        height=520,
        margin=dict(l=50, r=30, t=30, b=85),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def create_monthly_severity_line_chart(agg_df, smoothing_window=1, month_revision="all"):
    if agg_df.empty:
        return create_empty_figure("No time data available for selected filters")
    if "Month_Start" not in agg_df.columns:
        return create_empty_figure("Start_Time column is missing")

    chart_data = agg_df.copy()
    chart_data["Month_Start"] = pd.to_datetime(chart_data["Month_Start"], errors="coerce")
    chart_data = chart_data.dropna(subset=["Month_Start", "Severity"])
    if chart_data.empty:
        return create_empty_figure("No valid Start_Time values available")

    chart_data["Severity"] = chart_data["Severity"].astype(int)
    chart_data["Severity_Label"] = chart_data["Severity"].astype(str)
    chart_data["Month_Key"] = chart_data["Month_Start"].dt.strftime("%Y-%m")
    smoothing_window = max(int(smoothing_window or 1), 1)
    chart_data = chart_data.sort_values(["Severity", "Month_Start"])
    chart_data["Trend_Value"] = chart_data.groupby("Severity")["Accident_Count"].transform(
        lambda values: values.rolling(smoothing_window, min_periods=1).mean()
    )

    fig = px.line(
        chart_data.sort_values("Month_Start"),
        x="Month_Start",
        y="Trend_Value",
        color="Severity_Label",
        markers=smoothing_window == 1,
        category_orders={"Severity_Label": [str(level) for level in severity_levels]},
        color_discrete_map={str(level): color for level, color in SEVERITY_COLORS.items()},
        custom_data=["Month_Key", "Severity", "Accident_Count", "Trend_Value"],
    )
    fig.update_traces(
        hovertemplate=(
            "Month: %{customdata[0]}<br>"
            "Severity: %{customdata[1]}<br>"
            "Monthly count: %{customdata[2]:,.0f}<br>"
            "Displayed trend: %{customdata[3]:,.1f}<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        clickmode="event+select",
        dragmode="zoom",
        title=None,
        xaxis_title="Month",
        yaxis_title=(
            "Number of accidents"
            if smoothing_window == 1
            else f"{smoothing_window}-month moving average"
        ),
        legend_title_text="Severity",
        height=570,
        margin=dict(l=50, r=30, t=30, b=70),
        # Same reason as the stacked bars. Keyed to the month filter so that clearing
        # the filter still snaps the range slider back to the full period.
        uirevision=f"monthly-trend-{month_revision}",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        annotations=[
            dict(
                text="Note: 2023 is incomplete in this dataset.",
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                font=dict(size=12, color="#555555"),
            )
        ],
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.11, bgcolor="#f1f5f9")
    )
    return fig


def create_choropleth_map(feature, severity_level, aggregated_df, filter_state, cmax):
    cell_df = aggregated_df[
        (aggregated_df["Feature"] == feature) & (aggregated_df["Severity"] == severity_level)
    ][["State", "Accident_Count"]].copy()
    cell_df["Feature"] = feature
    cell_df["Severity"] = severity_level
    cmax = max(cmax, 1)
    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=cell_df["State"] if not cell_df.empty else [],
            z=cell_df["Accident_Count"] if not cell_df.empty else [],
            locationmode="USA-states",
            colorscale=SEVERITY_COLOR_SCALES[severity_level],
            zmin=0,
            zmax=cmax,
            marker_line_color="rgba(0,0,0,0.35)",
            marker_line_width=0.5,
            showscale=False,
            customdata=(
                cell_df[["State", "Feature", "Severity", "Accident_Count"]].to_numpy()
                if not cell_df.empty
                else []
            ),
            hovertemplate=(
                "State: %{customdata[0]}<br>"
                "Feature: %{customdata[1]}<br>"
                "Severity: %{customdata[2]}<br>"
                "Accident count: %{customdata[3]}<extra></extra>"
            ),
        )
    )

    selected_state = filter_state.get("state")
    selected_feature = filter_state.get("feature")
    if not cell_df.empty:
        highlight_active = selected_state is not None and selected_feature == feature
        line_widths = [
            3 if highlight_active and state == selected_state else 0.5
            for state in cell_df["State"]
        ]
        line_colors = [
            "black" if highlight_active and state == selected_state else "rgba(0,0,0,0.35)"
            for state in cell_df["State"]
        ]
        fig.update_traces(
            marker_line_width=line_widths,
            marker_line_color=line_colors,
        )

    fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            bgcolor="#ffffff",
            lakecolor="#ffffff",
            landcolor="#f9f9f9",
            showlakes=False,
            showland=True,
            subunitcolor="#ffffff",
        ),
        margin={"r": 2, "t": 2, "l": 2, "b": 2},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def create_road_grid_scale_legend(severity_level, cmax):
    start_color = SEVERITY_COLOR_SCALES[severity_level][0][1]
    end_color = SEVERITY_COLOR_SCALES[severity_level][1][1]
    return html.Div(
        [
            html.Div(
                style={
                    "height": "10px",
                    "width": "100%",
                    "background": f"linear-gradient(90deg, {start_color} 0%, {end_color} 100%)",
                }
            ),
            html.Div(
                [
                    html.Span("0"),
                    html.Span(f"{int(max(cmax, 1))}"),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "fontFamily": "Arial",
                    "fontSize": "11px",
                    "color": "#555555",
                    "marginTop": "4px",
                },
            ),
        ]
    )


def build_choropleth_bundle(filter_state):
    normalized_filters = normalize_filter_state(filter_state)
    aggregated_df = get_road_feature_state_agg(normalized_filters)
    severity_cmax = {}
    for severity in severity_levels:
        severity_slice = aggregated_df[aggregated_df["Severity"] == severity]
        severity_cmax[severity] = int(severity_slice["Accident_Count"].max()) if not severity_slice.empty else 1
    figures = []
    for feature in features:
        for severity in severity_levels:
            figures.append(
                create_choropleth_map(
                    feature,
                    severity,
                    aggregated_df,
                    normalized_filters,
                    cmax=severity_cmax[severity],
                )
            )
    return tuple(figures), severity_cmax


def build_choropleth_figures(filter_state):
    figures, _ = build_choropleth_bundle(filter_state)
    return figures


def create_active_filter_summary(filter_state):
    filters = normalize_filter_state(filter_state)
    parts = []
    if filters.get("selected_point_ids") is not None:
        parts.append(f"Selected points = {len(filters['selected_point_ids'])} accidents")
    elif filters.get("selected_states") is not None:
        states_label = ", ".join(filters["selected_states"][:4])
        if len(filters["selected_states"]) > 4:
            states_label += ", ..."
        parts.append(f"Selected states = {states_label}")
    elif filters["state"] is not None:
        parts.append(f"State = {filters['state']}")
    if filters["feature"] is not None:
        parts.append(f"Feature = {filters['feature']}")
    if filters["severity"] is not None:
        parts.append(f"Severity = {filters['severity']}")
    if filters["weather_group"] is not None:
        parts.append(f"Weather = {filters['weather_group']}")
    if filters["month_range"] is not None:
        parts.append(
            f"Month range = {filters['month_range'][0]} to {filters['month_range'][1]}"
        )
    elif filters["month"] is not None:
        parts.append(f"Month = {filters['month']}")
    if filters.get("condition_ranges"):
        parts.append(
            f"Parallel-coordinate ranges = {len(filters['condition_ranges'])} active axes"
        )

    if not parts:
        return "Active filters: None"
    return "Active filters: " + " | ".join(parts)


def toggle_pair(current_value_a, current_value_b, clicked_value_a, clicked_value_b):
    if current_value_a == clicked_value_a and current_value_b == clicked_value_b:
        return None, None
    return clicked_value_a, clicked_value_b


def parse_choropleth_id(component_id):
    base = component_id.replace("-choropleth", "")
    feature_name, severity_str = base.rsplit("-", 1)
    return feature_name, int(severity_str)


def update_condition_ranges_from_restyle(restyle_data, current_ranges):
    if not isinstance(restyle_data, list) or not restyle_data:
        return current_ranges
    changes = restyle_data[0]
    if not isinstance(changes, dict):
        return current_ranges

    updated_ranges = dict(current_ranges or {})
    dimension_columns = [column for column, _ in PARALLEL_DIMENSIONS]
    for property_name, raw_value in changes.items():
        prefix = "dimensions["
        suffix = "].constraintrange"
        if not property_name.startswith(prefix) or not property_name.endswith(suffix):
            continue
        try:
            dimension_index = int(property_name[len(prefix) : -len(suffix)])
            column = dimension_columns[dimension_index]
        except (ValueError, IndexError):
            continue

        value = raw_value
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if value is None or value == []:
            updated_ranges.pop(column, None)
        else:
            updated_ranges[column] = value

    return normalize_filter_state(
        {"condition_ranges": updated_ranges}
    )["condition_ranges"]


def build_initial_state():
    initial_filters = normalize_filter_state(DEFAULT_FILTER_STATE)
    initial_map_data = get_map_points(initial_filters)
    initial_map_fig = create_accident_scatter_map(
        pd.DataFrame(initial_map_data["records"]),
        initial_filters,
    )
    conditions_fig = create_conditions_parallel_coordinates(
        get_condition_lines(initial_filters),
        initial_filters,
    )
    monthly_fig = create_monthly_severity_line_chart(get_monthly_severity_agg(initial_filters))
    choropleth_figures, severity_cmax = build_choropleth_bundle(initial_filters)
    return initial_map_fig, conditions_fig, monthly_fig, choropleth_figures, severity_cmax


(
    initial_map_fig,
    initial_conditions_fig,
    initial_monthly_fig,
    initial_choropleth_figures,
    initial_severity_cmax,
) = build_initial_state()

app.layout = html.Div(
    [
        dcc.Store(id="filter-state", data=DEFAULT_FILTER_STATE),
        dcc.Store(id="map-viewport", data=create_default_map_view_state()),
        dcc.Store(id="conditions-render-key", data=None),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("US Accidents Explorer", style={"lineHeight": "1.2"}),
                        html.Div(
                            "Interactive patterns across location, conditions, roads and time · 2016–2023",
                            style={
                                "fontSize": "12px",
                                "fontWeight": "400",
                                "color": "#cbd5e1",
                                "marginTop": "4px",
                            },
                        ),
                    ]
                )
            ],
            style=HEADER_STYLE,
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            id="active-filter-summary",
                            children=create_active_filter_summary(DEFAULT_FILTER_STATE),
                            style={
                                "font-family": "Arial",
                                "font-size": "14px",
                                "color": "#222222",
                            },
                        ),
                        html.Button(
                            "Reset visualizations",
                            id="reset-button",
                            style=BUTTON_STYLE,
                        ),
                    ],
                    style=CONTROL_BAR_STYLE,
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Accident Map", style=CARD_TITLE_STYLE),
                                        dcc.Checklist(
                                            id="minimap-toggle",
                                            options=[
                                                {"label": " Overview", "value": "show"}
                                            ],
                                            value=["show"],
                                            className="minimap-toggle",
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-between",
                                        "alignItems": "center",
                                        "gap": "12px",
                                    },
                                ),
                                html.Div(
                                    "Drag to pan; pick lasso or box select in the toolbar to filter by an area. The outlined minimap box shows your current view; double-click to reset.",
                                    style=CARD_DESCRIPTION_STYLE,
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="accident-map",
                                            figure=initial_map_fig,
                                            config={
                                                "displayModeBar": "hover",
                                                "responsive": True,
                                                "doubleClick": "reset",
                                                "scrollZoom": True,
                                            },
                                            style={"width": "100%", "height": "520px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Div("OVERVIEW", className="map-minimap-title"),
                                                dcc.Graph(
                                                    id="map-minimap",
                                                    figure=create_map_minimap(DEFAULT_FILTER_STATE),
                                                    config={
                                                        "displayModeBar": False,
                                                        "responsive": True,
                                                        "staticPlot": True,
                                                    },
                                                    className="map-minimap-graph",
                                                ),
                                            ],
                                            id="minimap-container",
                                            className="map-minimap",
                                        ),
                                    ],
                                    className="map-stage",
                                ),
                                html.Div(
                                    "Map displays up to 10,000 sampled points for performance.",
                                    style={
                                        "fontFamily": "Arial",
                                        "fontSize": "12px",
                                        "color": "#555555",
                                        "paddingTop": "8px",
                                    },
                                ),
                            ],
                            id="map-card",
                            style={**CARD_STYLE, "minHeight": "620px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Conditions and Accident Severity",
                                    style=CARD_TITLE_STYLE,
                                ),
                                html.Div(
                                    id="conditions-description",
                                    children="Drag along any axis to filter every visualization by one or more condition ranges.",
                                    style=CARD_DESCRIPTION_STYLE,
                                ),
                                dcc.RadioItems(
                                    id="conditions-view-mode",
                                    options=[
                                        {"label": "Parallel coordinates", "value": "parallel"},
                                        {"label": "Stacked bars", "value": "bars"},
                                    ],
                                    value="parallel",
                                    inline=True,
                                    className="segmented-control",
                                ),
                                dcc.Graph(
                                    id="conditions-visualization",
                                    figure=initial_conditions_fig,
                                    config={"displayModeBar": "hover", "responsive": True},
                                    style={"width": "100%", "height": "520px"},
                                ),
                            ],
                            style={**CARD_STYLE, "minHeight": "620px"},
                        ),
                    ],
                    style=GRID_STYLE,
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Road Features and Severity by State",
                                    style=CARD_TITLE_STYLE,
                                ),
                                html.Div(
                                    "State-level accident counts by road feature and severity.",
                                    style=CARD_DESCRIPTION_STYLE,
                                ),
                                html.Div(
                                    [
                                        html.Div("", style=ROAD_GRID_HEADER_STYLE),
                                        *[
                                            html.Div(f"Severity {severity}", style=ROAD_GRID_HEADER_STYLE)
                                            for severity in severity_levels
                                        ],
                                        *[
                                            element
                                            for row_index, feature in enumerate(features)
                                            for element in [
                                                html.Div(feature, style=ROAD_GRID_ROW_LABEL_STYLE),
                                                *[
                                                    dcc.Graph(
                                                        id=f"{feature}-{severity}-choropleth",
                                                        figure=initial_choropleth_figures[
                                                            row_index * len(severity_levels) + (severity - 1)
                                                        ],
                                                        config={"displayModeBar": False, "responsive": True},
                                                        style={"height": "142px", "width": "100%"},
                                                    )
                                                    for severity in severity_levels
                                                ],
                                            ]
                                        ],
                                    ],
                                    style=ROAD_GRID_MATRIX_STYLE,
                                ),
                                html.Div(
                                    [
                                        html.Div("", style={"height": "1px"}),
                                        *[
                                            html.Div(
                                                id=f"road-grid-scale-{severity}",
                                                children=create_road_grid_scale_legend(
                                                    severity,
                                                    initial_severity_cmax[severity],
                                                ),
                                            )
                                            for severity in severity_levels
                                        ],
                                    ],
                                    style=ROAD_GRID_SCALE_ROW_STYLE,
                                ),
                            ],
                            style={**CARD_STYLE, "minHeight": "720px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Monthly Accident Trend by Severity",
                                    style=CARD_TITLE_STYLE,
                                ),
                                html.Div(
                                    "Monthly accident counts by severity. Note: 2023 is incomplete.",
                                    style=CARD_DESCRIPTION_STYLE,
                                ),
                                dcc.Graph(
                                    id="monthly-severity-line-chart",
                                    figure=initial_monthly_fig,
                                    config={"displayModeBar": "hover", "responsive": True},
                                    style={"width": "100%", "height": "570px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "Trend smoothing",
                                            style={
                                                "fontSize": "12px",
                                                "fontWeight": "700",
                                                "color": "#334155",
                                                "marginBottom": "7px",
                                            },
                                        ),
                                        dcc.RadioItems(
                                            id="trend-window",
                                            options=[
                                                {"label": "Monthly", "value": 1},
                                                {"label": "3 months", "value": 3},
                                                {"label": "6 months", "value": 6},
                                                {"label": "12 months", "value": 12},
                                            ],
                                            value=1,
                                            inline=True,
                                            className="segmented-control trend-window-control",
                                        ),
                                        html.Div(
                                            id="trend-help",
                                            children="Monthly values show every short-term change. Drag the overview below the chart to focus on a period.",
                                            style={
                                                "fontSize": "12px",
                                                "color": "#64748b",
                                                "lineHeight": "1.45",
                                            },
                                        ),
                                    ],
                                    className="trend-controls",
                                ),
                            ],
                            style={**CARD_STYLE, "minHeight": "790px"},
                        ),
                    ],
                    style=GRID_STYLE,
                ),
            ],
            style={
                "marginTop": "0",
                "paddingBottom": "20px",
            },
        ),
        html.Div("Aarhus University 2024", style=FOOTER_STYLE),
    ],
    style=PAGE_STYLE,
)


@app.callback(
    Output("filter-state", "data"),
    [Input(component_id, "clickData") for component_id in CHOROPLETH_IDS]
    + [
        Input("accident-map", "clickData"),
        Input("accident-map", "selectedData"),
        Input("conditions-visualization", "restyleData"),
        Input("conditions-visualization", "clickData"),
        Input("monthly-severity-line-chart", "clickData"),
        Input("monthly-severity-line-chart", "relayoutData"),
        Input("reset-button", "n_clicks"),
    ],
    State("filter-state", "data"),
    prevent_initial_call=True,
)
def update_filter_state(*args):
    current_filter_state = normalize_filter_state(args[-1])
    input_values = args[:-1]
    triggered_id = ctx.triggered_id
    triggered_prop = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # assets/graph_click_toggle.js clears clickData back to null after a click so that
    # clicking the same mark again reads as a change and can toggle its filter off.
    # That null must never write to the store: this callback would otherwise race the
    # in-flight run for the real click and could hand back the pre-click state,
    # silently undoing the selection the user just made.
    if triggered_prop.endswith(("clickData", "selectedData")):
        triggered_value = ctx.triggered[0].get("value")
        if not triggered_value:
            return no_update

    if triggered_id == "reset-button":
        return DEFAULT_FILTER_STATE.copy()

    updated_state = current_filter_state.copy()

    if triggered_id == "accident-map" and triggered_prop.endswith("clickData"):
        click_data = input_values[len(CHOROPLETH_IDS)]
        if not click_data or "points" not in click_data:
            return current_filter_state
        point = click_data["points"][0]
        clicked_state = None
        if point.get("customdata"):
            clicked_state = point["customdata"][1]
        if clicked_state is None:
            return current_filter_state
        updated_state["state"] = (
            None if current_filter_state.get("state") == clicked_state else clicked_state
        )
        updated_state["selected_states"] = None
        updated_state["selected_point_ids"] = None
        return normalize_filter_state(updated_state)

    if triggered_id == "accident-map" and triggered_prop.endswith("selectedData"):
        selected_data = input_values[len(CHOROPLETH_IDS) + 1]
        if not selected_data or "points" not in selected_data:
            return current_filter_state
        selected_ids = []
        selected_states = {
            point["customdata"][1]
            for point in selected_data["points"]
            if point.get("customdata") and len(point["customdata"]) > 1 and point["customdata"][1]
        }
        for point in selected_data["points"]:
            if not point.get("customdata"):
                continue
            point_id = point["customdata"][0]
            try:
                selected_ids.append(int(point_id))
            except (TypeError, ValueError):
                continue

        if selected_ids:
            updated_state["selected_point_ids"] = sorted(set(selected_ids))
            updated_state["selected_states"] = sorted(selected_states) or None
            updated_state["state"] = next(iter(selected_states)) if len(selected_states) == 1 else None
            return normalize_filter_state(updated_state)
        if len(selected_states) == 1:
            updated_state["state"] = next(iter(selected_states))
            updated_state["selected_states"] = None
            updated_state["selected_point_ids"] = None
            return normalize_filter_state(updated_state)
        if len(selected_states) > 1:
            updated_state["selected_states"] = sorted(selected_states)
            updated_state["selected_point_ids"] = None
            updated_state["state"] = None
            return normalize_filter_state(updated_state)
        return current_filter_state

    if triggered_id == "conditions-visualization" and triggered_prop.endswith("restyleData"):
        restyle_data = input_values[len(CHOROPLETH_IDS) + 2]
        updated_state["condition_ranges"] = update_condition_ranges_from_restyle(
            restyle_data,
            current_filter_state.get("condition_ranges"),
        )
        return normalize_filter_state(updated_state)

    if triggered_id == "conditions-visualization" and triggered_prop.endswith("clickData"):
        click_data = input_values[len(CHOROPLETH_IDS) + 3]
        if not click_data or "points" not in click_data:
            return current_filter_state
        point = click_data["points"][0]
        if not point.get("customdata") or len(point["customdata"]) < 2:
            return current_filter_state
        clicked_weather = point["customdata"][0]
        try:
            clicked_severity = int(point["customdata"][1])
        except (TypeError, ValueError):
            return current_filter_state
        updated_state["weather_group"], updated_state["severity"] = toggle_pair(
            current_filter_state.get("weather_group"),
            current_filter_state.get("severity"),
            clicked_weather,
            clicked_severity,
        )
        return normalize_filter_state(updated_state)

    if triggered_prop.endswith("relayoutData") and triggered_id == "monthly-severity-line-chart":
        relayout_data = input_values[len(CHOROPLETH_IDS) + 5]
        if not relayout_data:
            return current_filter_state
        if relayout_data.get("xaxis.autorange"):
            updated_state["month_range"] = None
            updated_state["month"] = None
            return normalize_filter_state(updated_state)

        start_value = relayout_data.get("xaxis.range[0]") or (
            relayout_data.get("xaxis.range", [None, None])[0]
            if isinstance(relayout_data.get("xaxis.range"), list)
            else None
        )
        end_value = relayout_data.get("xaxis.range[1]") or (
            relayout_data.get("xaxis.range", [None, None])[1]
            if isinstance(relayout_data.get("xaxis.range"), list)
            else None
        )
        start_month = normalize_month_value(start_value)
        end_month = normalize_month_value(end_value)
        if start_month and end_month:
            if start_month > end_month:
                start_month, end_month = end_month, start_month
            updated_state["month_range"] = [start_month, end_month]
            updated_state["month"] = None
            updated_state["severity"] = None
            return normalize_filter_state(updated_state)
        return current_filter_state

    if triggered_id == "monthly-severity-line-chart":
        click_data = input_values[len(CHOROPLETH_IDS) + 4]
        if not click_data or "points" not in click_data:
            return current_filter_state
        point = click_data["points"][0]
        clicked_month = None
        if point.get("customdata"):
            clicked_month = normalize_month_value(point["customdata"][0])
        updated_state["month"] = (
            None if current_filter_state.get("month") == clicked_month else clicked_month
        )
        updated_state["month_range"] = None
        updated_state["severity"] = None
        return normalize_filter_state(updated_state)

    if triggered_id.endswith("-choropleth"):
        input_index = CHOROPLETH_IDS.index(triggered_id)
        click_data = input_values[input_index]
        if not click_data or "points" not in click_data:
            return current_filter_state
        point = click_data["points"][0]
        clicked_state = point.get("location")
        clicked_feature, _ = parse_choropleth_id(triggered_id)
        if point.get("customdata"):
            clicked_state = point["customdata"][0] or clicked_state
            clicked_feature = point["customdata"][1] or clicked_feature
        if (
            current_filter_state.get("state") == clicked_state
            and current_filter_state.get("feature") == clicked_feature
        ):
            updated_state["state"] = None
            updated_state["feature"] = None
            updated_state["severity"] = None
        else:
            updated_state["state"] = clicked_state
            updated_state["feature"] = clicked_feature
            updated_state["severity"] = None
        return normalize_filter_state(updated_state)

    return current_filter_state


@app.callback(
    Output("accident-map", "figure"),
    Input("filter-state", "data"),
    Input("reset-button", "n_clicks"),
)
def update_map_figure(filter_state, reset_clicks):
    normalized_filters = (
        normalize_filter_state(DEFAULT_FILTER_STATE)
        if ctx.triggered_id == "reset-button"
        else normalize_filter_state(filter_state)
    )
    map_data = get_map_points(normalized_filters)
    return create_accident_scatter_map(
        pd.DataFrame(map_data["records"]),
        normalized_filters,
        map_revision=reset_clicks or 0,
    )


@app.callback(
    Output("map-viewport", "data"),
    Input("accident-map", "relayoutData"),
    Input("reset-button", "n_clicks"),
    State("map-viewport", "data"),
    prevent_initial_call=True,
)
def update_map_viewport(relayout_data, _reset_clicks, current_view):
    if ctx.triggered_id == "reset-button":
        return create_default_map_view_state()
    return update_map_view_state(relayout_data, current_view)


@app.callback(
    Output("map-minimap", "figure"),
    Output("minimap-container", "style"),
    Input("filter-state", "data"),
    Input("map-viewport", "data"),
    Input("minimap-toggle", "value"),
)
def update_map_minimap(filter_state, map_view, minimap_toggle):
    normalized_filters = normalize_filter_state(filter_state)
    minimap_style = {} if "show" in (minimap_toggle or []) else {"display": "none"}
    map_view = map_view or create_default_map_view_state()
    center = map_view["center"]
    view_bounds = map_view["bounds"]
    if ctx.triggered_id == "filter-state" and normalized_filters.get("state"):
        map_data = get_map_points(normalized_filters)
        center = {"lat": map_data["center"][0], "lon": map_data["center"][1]}
        view_bounds = create_map_view_bounds(center, FILTERED_MAP_ZOOM)
    return (
        create_map_minimap(
            normalized_filters,
            center,
            view_bounds,
            show_viewport=should_show_minimap_viewport(view_bounds),
        ),
        minimap_style,
    )


@app.callback(
    Output("conditions-visualization", "figure"),
    Output("conditions-description", "children"),
    Output("conditions-render-key", "data"),
    Input("filter-state", "data"),
    Input("conditions-view-mode", "value"),
    State("conditions-render-key", "data"),
)
def update_conditions_chart(filter_state, view_mode, rendered_key):
    normalized_filters = normalize_filter_state(filter_state)
    if view_mode == "bars":
        return (
            create_weather_severity_stacked_bar(
                get_weather_severity_agg(normalized_filters)
            ),
            "Click a bar segment to filter by weather group and severity.",
            None,
        )

    description = "Drag along any axis to filter every visualization by one or more ranges."
    # Brushing an axis feeds condition_ranges straight back into this callback. Pushing
    # a rebuilt Parcoords figure in response re-renders the trace the user is still
    # dragging, which is what made the axis handles stutter and jump back. The lines
    # themselves do not depend on condition_ranges (the data cache ignores that key),
    # so when nothing else changed there is nothing to redraw - leave the figure alone.
    render_key = make_filter_cache_key(
        normalized_filters, scope="conditions-figure", ignore_keys={"condition_ranges"}
    )
    if rendered_key is not None and render_key == rendered_key:
        return no_update, description, no_update

    return (
        create_conditions_parallel_coordinates(
            get_condition_lines(normalized_filters),
            normalized_filters,
        ),
        description,
        render_key,
    )


@app.callback(
    Output("monthly-severity-line-chart", "figure"),
    Output("trend-help", "children"),
    Input("filter-state", "data"),
    Input("trend-window", "value"),
)
def update_monthly_chart(filter_state, smoothing_window):
    normalized_filters = normalize_filter_state(filter_state)
    aggregated = get_monthly_severity_agg(normalized_filters)
    smoothing_window = max(int(smoothing_window or 1), 1)
    help_text = (
        "Monthly values show every short-term change. Drag the overview below the chart to focus on a period."
        if smoothing_window == 1
        else f"Showing a {smoothing_window}-month moving average to reduce short-term noise. Drag the overview below the chart to focus on a period."
    )
    month_revision = normalized_filters.get("month_range") or normalized_filters.get("month") or "all"
    return (
        create_monthly_severity_line_chart(aggregated, smoothing_window, month_revision),
        help_text,
    )


@app.callback(
    [Output(f"{feature}-{severity}-choropleth", "figure") for feature in features for severity in severity_levels]
    + [Output(f"road-grid-scale-{severity}", "children") for severity in severity_levels],
    Input("filter-state", "data"),
)
def update_choropleth_grid(filter_state):
    figures, severity_cmax = build_choropleth_bundle(normalize_filter_state(filter_state))
    scale_legends = [
        create_road_grid_scale_legend(severity, severity_cmax[severity])
        for severity in severity_levels
    ]
    return list(figures) + scale_legends


@app.callback(
    Output("active-filter-summary", "children"),
    Input("filter-state", "data"),
)
def update_active_filter_summary(filter_state):
    return create_active_filter_summary(filter_state)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
