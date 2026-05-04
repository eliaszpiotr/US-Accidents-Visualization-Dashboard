import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import orjson
from dash import Dash, Input, Output, State, ctx, dcc, html
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
}
DEFAULT_MAP_CENTER = [39.8283, -98.5795]
DEFAULT_MAP_ZOOM = 4
FILTERED_MAP_ZOOM = 6
MAX_MAP_POINTS = 10000
CHOROPLETH_IDS = [f"{feature}-{severity}-choropleth" for feature in features for severity in severity_levels]

PAGE_STYLE = {
    "width": "100%",
    "margin": "0 auto",
    "backgroundColor": "#ffffff",
    "minHeight": "100vh",
}
HEADER_STYLE = {
    "width": "100%",
    "height": "48px",
    "backgroundColor": "#000000",
    "color": "white",
    "fontFamily": "Arial",
    "fontSize": "20px",
    "fontWeight": "700",
    "letterSpacing": "0.2px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
}
CONTROL_BAR_STYLE = {
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
    "padding": "16px 20px 8px 20px",
    "gap": "16px",
}
GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(560px, 1fr))",
    "gap": "20px",
    "padding": "0 20px 20px 20px",
    "alignItems": "stretch",
}
CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "none",
    "boxShadow": "none",
    "borderRadius": "0",
    "padding": "14px",
    "boxSizing": "border-box",
    "height": "100%",
    "display": "flex",
    "flexDirection": "column",
}
CARD_TITLE_STYLE = {
    "fontFamily": "Arial",
    "fontSize": "18px",
    "fontWeight": "bold",
    "color": "#111111",
    "marginBottom": "6px",
}
CARD_DESCRIPTION_STYLE = {
    "fontFamily": "Arial",
    "fontSize": "13px",
    "color": "#555555",
    "marginBottom": "12px",
}
BUTTON_STYLE = {
    "fontFamily": "Arial",
    "padding": "8px 14px",
    "border": "none",
    "borderRadius": "4px",
    "backgroundColor": "#000000",
    "color": "white",
    "fontSize": "14px",
    "cursor": "pointer",
}
FOOTER_STYLE = {
    "width": "100%",
    "height": "36px",
    "backgroundColor": "#000000",
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

ACCIDENTS_PL = load_accident_data_polars()
if "Accident_ID" not in ACCIDENTS_PL.columns:
    ACCIDENTS_PL = ACCIDENTS_PL.with_row_count("Accident_ID")

app = Dash(__name__, assets_folder=str(ASSETS_DIR))
CACHE_DIR = GENERATED_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
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


def create_accident_scatter_map(df, filter_state=None):
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

    fig = px.scatter_mapbox(
        map_df,
        lat="Start_Lat",
        lon="Start_Lng",
        color="Severity_Label",
        category_orders={"Severity_Label": [str(level) for level in severity_levels]},
        color_discrete_map={str(level): color for level, color in SEVERITY_COLORS.items()},
        custom_data=["Accident_ID", "State", "Severity", "Weather_Group", "Month_Key", "City", "Street"],
        zoom=zoom,
        center=center,
        height=520,
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        selected=dict(marker=dict(opacity=0.95, size=10)),
        unselected=dict(marker=dict(opacity=0.25)),
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
        mapbox=dict(style="carto-positron", center=center, zoom=zoom),
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="Severity",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        clickmode="event+select",
        dragmode="lasso",
        uirevision="accident-map",
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


def create_monthly_severity_line_chart(agg_df):
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

    fig = px.line(
        chart_data.sort_values("Month_Start"),
        x="Month_Start",
        y="Accident_Count",
        color="Severity_Label",
        markers=True,
        category_orders={"Severity_Label": [str(level) for level in severity_levels]},
        color_discrete_map={str(level): color for level, color in SEVERITY_COLORS.items()},
        custom_data=["Month_Key", "Severity", "Accident_Count"],
    )
    fig.update_traces(
        hovertemplate=(
            "Month: %{customdata[0]}<br>"
            "Severity: %{customdata[1]}<br>"
            "Accident count: %{customdata[2]}<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        clickmode="event+select",
        dragmode="zoom",
        title=None,
        xaxis_title="Month",
        yaxis_title="Number of accidents",
        legend_title_text="Severity",
        height=520,
        margin=dict(l=50, r=30, t=30, b=70),
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
    fig.update_xaxes(rangeslider=dict(visible=False))
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


def build_initial_state():
    initial_filters = normalize_filter_state(DEFAULT_FILTER_STATE)
    initial_map_data = get_map_points(initial_filters)
    initial_map_fig = create_accident_scatter_map(
        pd.DataFrame(initial_map_data["records"]),
        initial_filters,
    )
    weather_fig = create_weather_severity_stacked_bar(get_weather_severity_agg(initial_filters))
    monthly_fig = create_monthly_severity_line_chart(get_monthly_severity_agg(initial_filters))
    choropleth_figures, severity_cmax = build_choropleth_bundle(initial_filters)
    return initial_map_fig, weather_fig, monthly_fig, choropleth_figures, severity_cmax


(
    initial_map_fig,
    initial_weather_fig,
    initial_monthly_fig,
    initial_choropleth_figures,
    initial_severity_cmax,
) = build_initial_state()

app.layout = html.Div(
    [
        dcc.Store(id="filter-state", data=DEFAULT_FILTER_STATE),
        html.Div("Visualization of US Accidents (2016–2023)", style=HEADER_STYLE),
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
                                html.Div("Accident Map", style=CARD_TITLE_STYLE),
                                html.Div(
                                    "Sampled accident locations colored by severity.",
                                    style=CARD_DESCRIPTION_STYLE,
                                ),
                                dcc.Graph(
                                    id="accident-map",
                                    figure=initial_map_fig,
                                    config={"displayModeBar": "hover", "responsive": True},
                                    style={"width": "100%", "height": "520px"},
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
                            style={**CARD_STYLE, "minHeight": "620px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Severity Distribution by Weather Conditions",
                                    style=CARD_TITLE_STYLE,
                                ),
                                html.Div(
                                    "Percentage distribution of severity levels within each weather group.",
                                    style=CARD_DESCRIPTION_STYLE,
                                ),
                                dcc.Graph(
                                    id="weather-visualization",
                                    figure=initial_weather_fig,
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
                                    style={"width": "100%", "height": "650px"},
                                ),
                            ],
                            style={**CARD_STYLE, "minHeight": "720px"},
                        ),
                    ],
                    style=GRID_STYLE,
                ),
            ],
            style={
                "marginTop": "48px",
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
        Input("weather-visualization", "clickData"),
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

    if triggered_id == "weather-visualization":
        click_data = input_values[len(CHOROPLETH_IDS) + 2]
        if not click_data or "points" not in click_data:
            return current_filter_state
        point = click_data["points"][0]
        clicked_weather = None
        if point.get("customdata"):
            clicked_weather = point["customdata"][0]
        updated_state["weather_group"] = (
            None if current_filter_state.get("weather_group") == clicked_weather else clicked_weather
        )
        updated_state["severity"] = None
        return normalize_filter_state(updated_state)

    if triggered_prop.endswith("relayoutData") and triggered_id == "monthly-severity-line-chart":
        relayout_data = input_values[len(CHOROPLETH_IDS) + 4]
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
        click_data = input_values[len(CHOROPLETH_IDS) + 3]
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
)
def update_map_figure(filter_state):
    map_data = get_map_points(normalize_filter_state(filter_state))
    return create_accident_scatter_map(
        pd.DataFrame(map_data["records"]),
        normalize_filter_state(filter_state),
    )


@app.callback(
    Output("weather-visualization", "figure"),
    Input("filter-state", "data"),
)
def update_weather_chart(filter_state):
    aggregated = get_weather_severity_agg(normalize_filter_state(filter_state))
    return create_weather_severity_stacked_bar(aggregated)


@app.callback(
    Output("monthly-severity-line-chart", "figure"),
    Input("filter-state", "data"),
)
def update_monthly_chart(filter_state):
    aggregated = get_monthly_severity_agg(normalize_filter_state(filter_state))
    return create_monthly_severity_line_chart(aggregated)


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
