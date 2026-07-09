"""
Data Center Sustainability & Efficiency Command Center
========================================================
A Streamlit dashboard for exploring the global data-center operations panel
(2019-2025): efficiency (PUE/WUE), consumption trends, geography, water-stress
risk, correlations/drivers, and a facility-level explorer.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import base64

# Add this line right after your imports to lift the restriction:
pd.set_option("styler.render.max_elements", 2000000)

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Center Sustainability & Efficiency Command Center",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ============================================================================
# 2. PASTE ALL THE BACKGROUND IMAGE CODE RIGHT HERE
# ============================================================================
def set_background_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    css = f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{encoded_string});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp > header {{
        background-color: transparent;
    }}
    .stApp .main .block-container {{
        background-color: rgba(255, 255, 255, 0.85); /* Adjust this for readability */
        padding: 2rem;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 3. CALL THE FUNCTION HERE (Replace with your actual image file name)
set_background_local("dashboard_background.png")
DATA_PATH = "data_center_hybrid.csv"

# ----------------------------------------------------------------------------
# COLOR PALETTE  (defined once, reused everywhere)
# ----------------------------------------------------------------------------
COLORS = {
    "excellent": "#0E7C61",   # deep teal-green   -> efficient / good
    "good": "#4FB286",        # mid green
    "poor": "#D96C2B",        # amber/orange      -> inefficient / risk
    "critical": "#B23A3A",    # red               -> high risk
    "primary": "#0E7C61",
    "secondary": "#1B6E8C",   # teal-blue accent
    "neutral": "#7C8B94",     # slate grey
    "bg_accent": "#EAF4F1",
    "low_stress": "#4FB286",
    "medium_stress": "#E3A02A",
    "high_stress": "#B23A3A",
}

FACILITY_TYPE_COLORS = {
    "Enterprise/Standard": "#1B6E8C",
    "Colocation": "#5A8F7B",
    "Hyperscale/AI": "#D96C2B",
}

COOLING_COLORS = {
    "Air Cooled": "#7C8B94",
    "Evaporative": "#1B6E8C",
    "Liquid Cooled": "#0E7C61",
}

STRESS_COLORS = {
    "Low": COLORS["low_stress"],
    "Medium": COLORS["medium_stress"],
    "High": COLORS["high_stress"],
}

SEQ_SCALE = ["#0E7C61", "#4FB286", "#E3A02A", "#D96C2B", "#B23A3A"]  # good -> bad
CONTINUOUS_SCALE = [[0, "#0E7C61"], [0.5, "#E3A02A"], [1, "#B23A3A"]]

PLOTLY_TEMPLATE = "plotly_white"


# ----------------------------------------------------------------------------
# DATA LOADING & PREPARATION
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data center dataset...")
def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and attach derived analytical columns."""
    df = pd.read_csv(path)

    # --- Efficiency tiers based on PUE (lower PUE = more efficient) --------
    df["PUE_Tier"] = pd.cut(
        df["PUE"],
        bins=[-np.inf, 1.3, 1.6, np.inf],
        labels=["Excellent", "Good", "Poor"],
    )

    # --- Numeric water-stress score (used in the priority score) -----------
    stress_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Water_Stress_Score"] = df["Surrounding_Water_Stress_Tier"].map(stress_map)

    # --- Clean Country Data -------------------------------------------------
    # Force as string and strip whitespace
    df["Country"] = df["Country"].astype(str).str.strip()
    
    # Create a mask that excludes "Unknown" AND excludes any row containing digits/numbers
    is_valid_country = (
        df["Country"].ne("Unknown") & 
        ~df["Country"].str.contains(r'\d', regex=True) & 
        (df["Country"].str.len() > 1)
    )

    df["Has_Known_Country"] = is_valid_country
    df["Has_Known_City"] = df["City"].astype(str).str.strip().ne("Unknown")

    return df


def compute_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Sustainability Priority Score.

    Each row is ranked (percentile rank, ascending) on four "bad-if-high"
    metrics: PUE, daily electricity usage, daily water usage, and the numeric
    water-stress tier. A HIGH score therefore means the facility has
    relatively high PUE + high water stress + high consumption -> it is a
    strong candidate for sustainability improvement.

    Computed on whatever slice of data is passed in, so the ranking always
    reflects the currently filtered population.
    """
    out = df.copy()
    if out.empty:
        out["Sustainability_Priority_Score"] = np.nan
        return out

    rank_pue = out["PUE"].rank(ascending=True, pct=True)
    rank_elec = out["Daily_Electricity_Usage_MWh"].rank(ascending=True, pct=True)
    rank_water = out["Daily_Water_Usage_Gallons"].rank(ascending=True, pct=True)
    rank_stress = out["Water_Stress_Score"].rank(ascending=True, pct=True)

    # Scale to 0-100 for readability (equal-weighted composite of 4 rank pcts)
    out["Sustainability_Priority_Score"] = (
        (rank_pue + rank_elec + rank_water + rank_stress) / 4 * 100
    )
    return out


# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
def build_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.title("🔎 Filters")
    st.sidebar.caption("Leave a filter empty to include ALL values.")

    year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
    cap_min, cap_max = float(df["Estimated_Capacity_MW"].min()), float(df["Estimated_Capacity_MW"].max())
    pue_min, pue_max = float(df["PUE"].min()), float(df["PUE"].max())

    all_countries = sorted(df.loc[df["Has_Known_Country"], "Country"].unique().tolist())
    all_facility_types = sorted(df["Facility_Type"].dropna().unique().tolist())
    all_cooling_types = sorted(df["Cooling_System_Type"].dropna().unique().tolist())
    all_stress_tiers = ["Low", "Medium", "High"]

    # Set all categorical defaults to empty lists []
    defaults = {
        "year_range": (year_min, year_max),
        "countries": [],
        "facility_types": [],
        "cooling_types": [],
        "stress_tiers": [],
        "capacity_range": (cap_min, cap_max),
        "pue_range": (
            float(np.floor(pue_min * 100) / 100),
            float(np.ceil(pue_max * 100) / 100),
        ),
    }

    if "filters_initialized" not in st.session_state:
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state["filters_initialized"] = True

    if st.sidebar.button("↺ Reset filters", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.sidebar.markdown("---")

    year_range = st.sidebar.slider(
        "Year range", min_value=year_min, max_value=year_max, key="year_range"
    )

    countries = st.sidebar.multiselect(
        "Country", options=all_countries, key="countries",
        help="Leave empty to include all countries.",
    )

    facility_types = st.sidebar.multiselect(
        "Facility Type", options=all_facility_types, key="facility_types",
        help="Leave empty to include all facility types."
    )

    cooling_types = st.sidebar.multiselect(
        "Cooling System Type", options=all_cooling_types, key="cooling_types",
        help="Leave empty to include all cooling systems."
    )

    stress_tiers = st.sidebar.multiselect(
        "Surrounding Water Stress Tier", options=all_stress_tiers, key="stress_tiers",
        help="Leave empty to include all stress tiers."
    )

    capacity_range = st.sidebar.slider(
        "Estimated Capacity (MW)",
        min_value=float(np.floor(cap_min)),
        max_value=float(np.ceil(cap_max)),
        key="capacity_range",
    )

    pue_range = st.sidebar.slider(
        "PUE",
        min_value=float(np.floor(pue_min * 100) / 100),
        max_value=float(np.ceil(pue_max * 100) / 100),
        step=0.01, key="pue_range",
    )

    return {
        "year_range": year_range,
        "countries": countries,
        "facility_types": facility_types,
        "cooling_types": cooling_types,
        "stress_tiers": stress_tiers,
        "capacity_range": capacity_range,
        "pue_range": pue_range,
    }

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the sidebar filter selections. If a multiselect is empty, it does not filter that column."""
    
    # 1. Apply baseline continuous filters (sliders)
    mask = (
        df["Year"].between(filters["year_range"][0], filters["year_range"][1])
        & df["Estimated_Capacity_MW"].between(*filters["capacity_range"])
        & df["PUE"].between(*filters["pue_range"])
    )
    
    # 2. Only apply categorical filters IF the user selected something
    if filters["facility_types"]:
        mask &= df["Facility_Type"].isin(filters["facility_types"])
        
    if filters["cooling_types"]:
        mask &= df["Cooling_System_Type"].isin(filters["cooling_types"])
        
    if filters["stress_tiers"]:
        mask &= df["Surrounding_Water_Stress_Tier"].isin(filters["stress_tiers"])
        
    if filters["countries"]:
        mask &= df["Country"].isin(filters["countries"])

    out = df.loc[mask].copy()
    return out

# ----------------------------------------------------------------------------
# SMALL FORMATTING HELPERS
# ----------------------------------------------------------------------------
def _value_to_rgb(val, vmin, vmax):
    """Map a value to a green->amber->red RGB string without needing matplotlib.
    Low values (efficient) -> green, high values (inefficient) -> red."""
    if pd.isna(val) or vmax == vmin:
        return ""
    t = (val - vmin) / (vmax - vmin)
    t = min(max(t, 0), 1)
    stops = [(14, 124, 97), (227, 160, 42), (178, 58, 58)]  # green -> amber -> red
    if t < 0.5:
        t2 = t / 0.5
        c0, c1 = stops[0], stops[1]
    else:
        t2 = (t - 0.5) / 0.5
        c0, c1 = stops[1], stops[2]
    r = int(c0[0] + (c1[0] - c0[0]) * t2)
    g = int(c0[1] + (c1[1] - c0[1]) * t2)
    b = int(c0[2] + (c1[2] - c0[2]) * t2)
    return f"background-color: rgba({r},{g},{b},0.35)"


def style_gradient_column(series: pd.Series):
    """Pandas Styler-compatible function: color a numeric column low(green)->high(red).
    Implemented with plain numpy/python so it works without a matplotlib dependency."""
    vmin, vmax = series.min(), series.max()
    return [_value_to_rgb(v, vmin, vmax) for v in series]


def fmt_num(x, decimals=0):
    if pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


def fmt_mwh(x):
    if pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:,.2f}M MWh"
    if abs(x) >= 1_000:
        return f"{x/1_000:,.1f}K MWh"
    return f"{x:,.0f} MWh"


def fmt_gallons(x):
    if pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:,.2f}B gal"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:,.1f}M gal"
    if abs(x) >= 1_000:
        return f"{x/1_000:,.1f}K gal"
    return f"{x:,.0f} gal"


def safe_delta(current, previous):
    """Return a formatted percentage delta string, or None if not computable."""
    if previous in (None, 0) or pd.isna(previous) or pd.isna(current):
        return None
    pct = (current - previous) / previous * 100
    return f"{pct:+.1f}%"


# ----------------------------------------------------------------------------
# KPI COMPUTATION
# ----------------------------------------------------------------------------
def compute_kpis(filtered: pd.DataFrame, full: pd.DataFrame, filters: dict) -> dict:
    """
    Compute headline KPIs for the currently filtered data, plus a YoY delta
    that compares the latest selected year against the immediately preceding
    calendar year (holding all other filters constant), when that prior year
    exists in the full dataset.
    """
    kpis = {
        "avg_pue": filtered["PUE"].mean(),
        "avg_wue": filtered["WUE_L_per_kWh"].mean(),
        "total_elec": filtered["Daily_Electricity_Usage_MWh"].sum(),
        "total_water": filtered["Daily_Water_Usage_Gallons"].sum(),
        "n_facilities": filtered["Facility_ID"].nunique(),
        "n_countries": filtered.loc[filtered["Has_Known_Country"], "Country"].nunique(),
    }

    latest_year = filters["year_range"][1]
    prior_year = latest_year - 1

    # Build a comparable "prior year" slice using identical non-year filters
    prior_filters = dict(filters)
    prior_filters["year_range"] = (prior_year, prior_year)
    this_year_filters = dict(filters)
    this_year_filters["year_range"] = (latest_year, latest_year)

    prior_slice = apply_filters(full, prior_filters)
    this_slice = apply_filters(full, this_year_filters)

    deltas = {}
    if not prior_slice.empty and not this_slice.empty:
        deltas["avg_pue"] = safe_delta(this_slice["PUE"].mean(), prior_slice["PUE"].mean())
        deltas["avg_wue"] = safe_delta(this_slice["WUE_L_per_kWh"].mean(), prior_slice["WUE_L_per_kWh"].mean())
        deltas["total_elec"] = safe_delta(
            this_slice["Daily_Electricity_Usage_MWh"].sum(), prior_slice["Daily_Electricity_Usage_MWh"].sum()
        )
        deltas["total_water"] = safe_delta(
            this_slice["Daily_Water_Usage_Gallons"].sum(), prior_slice["Daily_Water_Usage_Gallons"].sum()
        )
        deltas["n_facilities"] = safe_delta(
            this_slice["Facility_ID"].nunique(), prior_slice["Facility_ID"].nunique()
        )
        deltas["n_countries"] = safe_delta(
            this_slice.loc[this_slice["Has_Known_Country"], "Country"].nunique(),
            prior_slice.loc[prior_slice["Has_Known_Country"], "Country"].nunique(),
        )
    else:
        deltas = {k: None for k in kpis}

    return {"values": kpis, "deltas": deltas}

def render_kpis(kpi_data: dict):
    v, d = kpi_data["values"], kpi_data["deltas"]
    
    # Row 1 of the Matrix
    r1_col1, r1_col2, r1_col3 = st.columns(3)
    r1_col1.metric("Avg PUE", fmt_num(v["avg_pue"], 3), d["avg_pue"], delta_color="inverse")
    r1_col2.metric("Avg WUE (L/kWh)", fmt_num(v["avg_wue"], 3), d["avg_wue"], delta_color="inverse")
    r1_col3.metric("Total Electricity", fmt_mwh(v["total_elec"]), d["total_elec"])
    
    # Add a little vertical spacing between matrix rows
    st.write("") 
    
    # Row 2 of the Matrix
    r2_col1, r2_col2, r2_col3 = st.columns(3)
    r2_col1.metric("Total Water Use", fmt_gallons(v["total_water"]), d["total_water"])
    r2_col2.metric("Facilities", fmt_num(v["n_facilities"]), d["n_facilities"])
    r2_col3.metric("Countries", fmt_num(v["n_countries"]), d["n_countries"])


# ----------------------------------------------------------------------------
# KEY FINDINGS PANEL
# ----------------------------------------------------------------------------
def render_key_findings(df: pd.DataFrame):
    st.subheader("🧭 Key Findings")
    if df.empty:
        st.info("No data matches the current filters — adjust filters to see findings.")
        return

    bullets = []

    # 1. Best cooling system
    cooling_pue = df.groupby("Cooling_System_Type", observed=True)["PUE"].mean().sort_values()
    if not cooling_pue.empty:
        best_cool, best_val = cooling_pue.index[0], cooling_pue.iloc[0]
        worst_cool, worst_val = cooling_pue.index[-1], cooling_pue.iloc[-1]
        if worst_val > 0:
            pct_better = (worst_val - best_val) / worst_val * 100
            bullets.append(
                f"**{best_cool}** systems run most efficiently at an average PUE of **{best_val:.3f}**, "
                f"{pct_better:.1f}% better than the least efficient method, **{worst_cool}** ({worst_val:.3f})."
            )

    # 2. Most efficient facility type
    ft_pue = df.groupby("Facility_Type", observed=True)["PUE"].mean().sort_values()
    if not ft_pue.empty:
        bullets.append(
            f"Among facility types, **{ft_pue.index[0]}** facilities are the most efficient "
            f"(avg PUE **{ft_pue.iloc[0]:.3f}**), while **{ft_pue.index[-1]}** trails at **{ft_pue.iloc[-1]:.3f}**."
        )

    # 3. Water stress vs usage
    stress_water = df.groupby("Surrounding_Water_Stress_Tier", observed=True)["Daily_Water_Usage_Gallons"].mean()
    if {"Low", "High"}.issubset(stress_water.index) and stress_water.get("Low", 0) > 0:
        ratio = stress_water["High"] / stress_water["Low"]
        bullets.append(
            f"Facilities in **High** water-stress regions consume on average **{ratio:.2f}x** the daily water "
            f"of facilities in **Low**-stress regions — a compounding risk factor."
        )

    # 4. Trend
    yearly = df.groupby("Year")["Daily_Electricity_Usage_MWh"].sum()
    if len(yearly) >= 2:
        first_y, last_y = yearly.index.min(), yearly.index.max()
        change = (yearly.loc[last_y] - yearly.loc[first_y]) / yearly.loc[first_y] * 100 if yearly.loc[first_y] else None
        if change is not None:
            direction = "risen" if change > 0 else "fallen"
            bullets.append(
                f"Total electricity usage has **{direction} {abs(change):.1f}%** from {int(first_y)} to {int(last_y)} "
                f"across the filtered facilities."
            )

    # 5. Top concentration
    if df["Has_Known_Country"].any():
        top_country = (
            df.loc[df["Has_Known_Country"]].groupby("Country")["Daily_Electricity_Usage_MWh"].sum().idxmax()
        )
        bullets.append(f"**{top_country}** is the single largest contributor to total electricity consumption in the current view.")

    for b in bullets[:5]:
        st.markdown(f"- {b}")


# ----------------------------------------------------------------------------
# NO-DATA GUARD
# ----------------------------------------------------------------------------
def guard_empty(df: pd.DataFrame) -> bool:
    """Show a friendly message and return True if there's nothing to plot."""
    if df.empty:
        st.warning("🚫 No facilities match the current filter combination. Try widening your filters.")
        return True
    return False


# ----------------------------------------------------------------------------
# CHART BUILDERS  (each returns a Plotly figure; pure functions, no st.* calls)
# ----------------------------------------------------------------------------
def chart_electricity_trend(df: pd.DataFrame, split_by_type: bool) -> go.Figure:
    if split_by_type:
        agg = df.groupby(["Year", "Facility_Type"], observed=True)["Daily_Electricity_Usage_MWh"].sum().reset_index()
        fig = px.line(
            agg, x="Year", y="Daily_Electricity_Usage_MWh", color="Facility_Type",
            color_discrete_map=FACILITY_TYPE_COLORS, markers=True,
            labels={"Daily_Electricity_Usage_MWh": "Total Electricity Usage (MWh)"},
            title="Electricity Usage Trend by Facility Type (2019–2025)",
        )
    else:
        agg = df.groupby("Year")["Daily_Electricity_Usage_MWh"].sum().reset_index()
        fig = px.line(
            agg, x="Year", y="Daily_Electricity_Usage_MWh", markers=True,
            labels={"Daily_Electricity_Usage_MWh": "Total Electricity Usage (MWh)"},
            title="Electricity Usage Trend (2019–2025)",
        )
        fig.update_traces(line_color=COLORS["primary"])
    fig.update_layout(template=PLOTLY_TEMPLATE, hovermode="x unified")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def chart_pue_trend(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby("Year")["PUE"].mean().reset_index()
    fig = px.line(
        agg, x="Year", y="PUE", markers=True,
        title="Average PUE Trend (2019–2025)",
        labels={"PUE": "Average PUE"},
    )
    fig.update_traces(line_color=COLORS["secondary"])
    fig.update_layout(template=PLOTLY_TEMPLATE, hovermode="x unified")
    return fig


def chart_top_n_bar(df: pd.DataFrame, group_col: str, value_col: str, n: int, title: str,
                     x_label: str, horizontal: bool = True) -> go.Figure:
    agg = df.groupby(group_col, observed=True)[value_col].sum().sort_values(ascending=False).head(n).reset_index()
    if horizontal:
        agg = agg.sort_values(value_col)
        fig = px.bar(
            agg, x=value_col, y=group_col, orientation="h",
            title=title, labels={value_col: x_label, group_col: ""},
            color=value_col, color_continuous_scale=[COLORS["secondary"], COLORS["primary"]],
        )
    else:
        fig = px.bar(
            agg, x=group_col, y=value_col, title=title,
            labels={value_col: x_label, group_col: ""},
            color=value_col, color_continuous_scale=[COLORS["secondary"], COLORS["primary"]],
        )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False)
    return fig


def chart_avg_pue_by(df: pd.DataFrame, group_col: str, title: str, color_map: dict) -> go.Figure:
    agg = df.groupby(group_col, observed=True)["PUE"].mean().sort_values().reset_index()
    fig = px.bar(
        agg, x=group_col, y="PUE", title=title, text_auto=".3f",
        labels={"PUE": "Average PUE", group_col: ""},
        color=group_col, color_discrete_map=color_map,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
    return fig


def chart_top_cities(df: pd.DataFrame, n: int = 10) -> go.Figure:
    known = df[df["Has_Known_City"]]
    agg = known.groupby("City")["Facility_ID"].nunique().sort_values(ascending=False).head(n).reset_index()
    agg.columns = ["City", "Facility_Count"]
    agg = agg.sort_values("Facility_Count")
    fig = px.bar(
        agg, x="Facility_Count", y="City", orientation="h",
        title=f"Top {n} Cities by Number of Data Centers",
        labels={"Facility_Count": "Number of Facilities", "City": ""},
        color="Facility_Count", color_continuous_scale=[COLORS["secondary"], COLORS["primary"]],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False)
    return fig


def chart_box(df: pd.DataFrame, group_col: str, value_col: str, title: str, color_map: dict,
              y_label: str) -> go.Figure:
    fig = px.box(
        df, x=group_col, y=value_col, color=group_col, color_discrete_map=color_map,
        title=title, labels={value_col: y_label, group_col: ""},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
    return fig


def chart_scatter_capacity_vs_electricity(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(n=min(len(df), 6000), random_state=42) if len(df) > 6000 else df
    fig = px.scatter(
        sample, x="Estimated_Capacity_MW", y="Daily_Electricity_Usage_MWh", color="Facility_Type",
        color_discrete_map=FACILITY_TYPE_COLORS, opacity=0.55,
        labels={"Estimated_Capacity_MW": "Estimated Capacity (MW)", "Daily_Electricity_Usage_MWh": "Daily Electricity Usage (MWh)"},
        title="Capacity vs. Electricity Usage",
    )
    # Manual OLS trendline (numpy only, no statsmodels dependency)
    x, y = df["Estimated_Capacity_MW"].values, df["Daily_Electricity_Usage_MWh"].values
    if len(x) > 1 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = slope * x_line + intercept
        corr = np.corrcoef(x, y)[0, 1]
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Trend (OLS)",
                                  line=dict(color=COLORS["critical"], dash="dash", width=2)))
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
            text=f"Pearson r = {corr:.3f}", bgcolor="white",
            bordercolor=COLORS["neutral"], borderwidth=1, align="left",
        )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_scatter_pue_vs_electricity(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(n=min(len(df), 6000), random_state=42) if len(df) > 6000 else df
    fig = px.scatter(
        sample, x="PUE", y="Daily_Electricity_Usage_MWh", color="Surrounding_Water_Stress_Tier",
        size="Estimated_Capacity_MW", color_discrete_map=STRESS_COLORS, opacity=0.6,
        category_orders={"Surrounding_Water_Stress_Tier": ["Low", "Medium", "High"]},
        labels={"PUE": "PUE", "Daily_Electricity_Usage_MWh": "Daily Electricity Usage (MWh)",
                "Surrounding_Water_Stress_Tier": "Water Stress"},
        title="Efficiency vs. Consumption Risk Map (size = capacity)",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = ["PUE", "WUE_L_per_kWh", "Estimated_Capacity_MW", "Daily_Electricity_Usage_MWh", "Daily_Water_Usage_Gallons"]
    labels = ["PUE", "WUE", "Capacity (MW)", "Electricity (MWh)", "Water (Gal)"]
    corr = df[cols].corr()
    fig = px.imshow(
        corr, x=labels, y=labels, text_auto=".2f", color_continuous_scale="RdYlGn_r",
        zmin=-1, zmax=1, title="Correlation Heatmap — Key Numeric Drivers",
        aspect="auto",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_pue_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="PUE", nbins=40, title="Distribution of PUE (with Efficiency Tier Bands)",
        labels={"PUE": "PUE"}, color_discrete_sequence=[COLORS["secondary"]],
    )
    fig.add_vrect(x0=df["PUE"].min() - 0.05, x1=1.3, fillcolor=COLORS["excellent"], opacity=0.12, line_width=0,
                  annotation_text="Excellent", annotation_position="top left")
    fig.add_vrect(x0=1.3, x1=1.6, fillcolor=COLORS["good"], opacity=0.12, line_width=0,
                  annotation_text="Good", annotation_position="top left")
    fig.add_vrect(x0=1.6, x1=df["PUE"].max() + 0.05, fillcolor=COLORS["poor"], opacity=0.12, line_width=0,
                  annotation_text="Poor", annotation_position="top left")
    fig.update_layout(template=PLOTLY_TEMPLATE, bargap=0.02)
    return fig


def chart_histogram(df: pd.DataFrame, col: str, title: str, x_label: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40, title=title, labels={col: x_label},
                        color_discrete_sequence=[COLORS["primary"]])
    fig.update_layout(template=PLOTLY_TEMPLATE, bargap=0.02)
    return fig


def chart_geo_map(df: pd.DataFrame, metric: str) -> go.Figure:
    known = df[df["Has_Known_Country"]]
    if metric == "Electricity Usage":
        agg = known.groupby("Country")["Daily_Electricity_Usage_MWh"].sum().reset_index()
        value_col, label = "Daily_Electricity_Usage_MWh", "Total Electricity Usage (MWh)"
    else:
        agg = known.groupby("Country")["Facility_ID"].nunique().reset_index()
        agg.columns = ["Country", "Facility_Count"]
        value_col, label = "Facility_Count", "Number of Facilities"

    fig = px.choropleth(
        agg, locations="Country", locationmode="country names", color=value_col,
        color_continuous_scale=[COLORS["bg_accent"], COLORS["secondary"], COLORS["primary"]],
        labels={value_col: label},
        title=f"{label} by Country",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, geo=dict(showframe=False, showcoastlines=True))
    return fig


def chart_priority_ranking(df_scored: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df_scored.sort_values("Sustainability_Priority_Score", ascending=False).head(n)
    top = top.sort_values("Sustainability_Priority_Score")
    label = top["Facility_Name"].fillna("") + " (" + top["Owner_Company"].fillna("") + ")"
    fig = px.bar(
        top, x="Sustainability_Priority_Score", y=label, orientation="h",
        title=f"Top {n} Facilities by Sustainability Priority Score",
        labels={"Sustainability_Priority_Score": "Priority Score (0–100, higher = more urgent)", "y": ""},
        color="Sustainability_Priority_Score", color_continuous_scale=[COLORS["good"], COLORS["poor"], COLORS["critical"]],
        hover_data={"PUE": ":.3f", "Daily_Electricity_Usage_MWh": ":,.0f", "Surrounding_Water_Stress_Tier": True},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False, yaxis_title="")
    return fig


def chart_treemap(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby(["Surrounding_Water_Stress_Tier", "Facility_Type"], observed=True).size().reset_index(name="Count")
    fig = px.treemap(
        agg, path=["Surrounding_Water_Stress_Tier", "Facility_Type"], values="Count",
        color="Surrounding_Water_Stress_Tier", color_discrete_map=STRESS_COLORS,
        title="Facility Type Mix by Water Stress Tier",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_cooling_donut(df: pd.DataFrame) -> go.Figure:
    agg = df["Cooling_System_Type"].value_counts().reset_index()
    agg.columns = ["Cooling_System_Type", "Count"]
    fig = px.pie(
        agg, names="Cooling_System_Type", values="Count", hole=0.5,
        color="Cooling_System_Type", color_discrete_map=COOLING_COLORS,
        title="Cooling System Type Share",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def chart_outlier_box(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    fig = px.box(df, y=col, points="outliers", title=title, labels={col: col.replace("_", " ")},
                 color_discrete_sequence=[COLORS["secondary"]])
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


# ----------------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------------
def main():
    df_raw = load_data(DATA_PATH)
    filters = build_sidebar(df_raw)
    filtered = apply_filters(df_raw, filters)

    st.title("🌎 Data Center Sustainability & Efficiency Command Center")
    st.caption(
        "Explore PUE/WUE efficiency, consumption trends, geography, water-stress risk, "
        "and improvement priorities across a global panel of data centers (2019–2025)."
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("Rows matching filters", f"{len(filtered):,} / {len(df_raw):,}")

   

    # --- NEW: Initialize the view state ---
    if "current_view" not in st.session_state:
        st.session_state.current_view = "summary"

    # ==========================================
    # VIEW 1: SUMMARY PAGE
    # ==========================================
    if st.session_state.current_view == "summary":
        kpi_data = compute_kpis(filtered, df_raw, filters)
        render_kpis(kpi_data)
        st.divider()
        
        # Use columns to align the Key Findings on the left and the button on the right
        col_main, col_btn = st.columns([0.85, 0.15])
        
        with col_main:
            render_key_findings(filtered)
            
        with col_btn:
            st.write("") # Add a little vertical spacing
            st.write("")
            # Clicking this changes the state and re-runs the app to show the tabs
            if st.button("Next ➡️", use_container_width=True, type="primary"):
                st.session_state.current_view = "tabs"
                st.rerun()

    # ==========================================
    # VIEW 2: TABS PAGE
    # ==========================================
    elif st.session_state.current_view == "tabs":
        
        # Add a back button so users don't get trapped on the tabs page
        if st.button("⬅️ Back to Summary"):
            st.session_state.current_view = "summary"
            st.rerun()
            
        st.divider()

        # 1. Define your tab names explicitly
        tab_names = [
            "📊 Overview", "⚙️ Efficiency (PUE/WUE)", "🔌 Consumption & Trends",
            "🗺️ Geography", "🚨 Sustainability Risk & Priority",
            "🔗 Correlations & Drivers", "🔍 Facility Explorer"
        ]

        # 2. Initialize the session state for tracking the active tab
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = tab_names[0]

        # 4. Use a horizontal radio button to emulate the tab UI
        selected_tab = st.radio(
            "Navigation",
            options=tab_names,
            horizontal=True,
            label_visibility="collapsed",
            key="active_tab" 
        )

        # ---------------------------- OVERVIEW ----------------------------
        if selected_tab == tab_names[0]:
            st.subheader("Portfolio Composition")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(chart_cooling_donut(filtered), use_container_width=True)
            with c2:
                st.plotly_chart(chart_treemap(filtered), use_container_width=True)

            cooling_share = filtered["Cooling_System_Type"].value_counts(normalize=True)
            if not cooling_share.empty:
                st.markdown(
                    f"💡 **{cooling_share.index[0]}** is the dominant cooling approach "
                    f"({cooling_share.iloc[0]*100:.1f}% of filtered facilities). "
                    f"The treemap shows how facility type mix shifts across water-stress tiers — "
                    f"watch for concentrations of **Hyperscale/AI** capacity in **High**-stress regions."
                )

        # ---------------------------- EFFICIENCY ----------------------------
        elif selected_tab == tab_names[1]:
            st.subheader("PUE Trend & Drivers")
            st.plotly_chart(chart_pue_trend(filtered), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(chart_avg_pue_by(filtered, "Cooling_System_Type", "Average PUE by Cooling System Type", COOLING_COLORS), use_container_width=True)
            with c2:
                st.plotly_chart(chart_avg_pue_by(filtered, "Facility_Type", "Average PUE by Facility Type", FACILITY_TYPE_COLORS), use_container_width=True)

            cooling_pue = filtered.groupby("Cooling_System_Type", observed=True)["PUE"].mean().sort_values()
            ft_pue = filtered.groupby("Facility_Type", observed=True)["PUE"].mean().sort_values()
            if not cooling_pue.empty and not ft_pue.empty:
                st.markdown(
                    f"💡 **{cooling_pue.index[0]}** cooling delivers the lowest average PUE "
                    f"(**{cooling_pue.iloc[0]:.3f}**) in the current selection. Among facility types, "
                    f"**{ft_pue.index[0]}** is most efficient (**{ft_pue.iloc[0]:.3f}**), while "
                    f"**{ft_pue.index[-1]}** runs hottest (**{ft_pue.iloc[-1]:.3f}**)."
                )

            st.plotly_chart(chart_pue_histogram(filtered), use_container_width=True)
            excellent_pct = (filtered["PUE_Tier"] == "Excellent").mean() * 100
            poor_pct = (filtered["PUE_Tier"] == "Poor").mean() * 100
            st.markdown(
                f"💡 **{excellent_pct:.1f}%** of filtered facility-years fall in the **Excellent** PUE tier (≤1.30), "
                f"while **{poor_pct:.1f}%** fall in the **Poor** tier (>1.60)."
            )

            st.subheader("Distributions by Cooling & Water Stress")
            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(chart_box(filtered, "Cooling_System_Type", "PUE", "PUE Distribution by Cooling System Type", COOLING_COLORS, "PUE"), use_container_width=True)
            with c4:
                st.plotly_chart(chart_box(filtered, "Surrounding_Water_Stress_Tier", "Daily_Water_Usage_Gallons", "Water Usage Distribution by Water Stress Tier", STRESS_COLORS, "Daily Water Usage (Gallons)"), use_container_width=True)

            stress_water = filtered.groupby("Surrounding_Water_Stress_Tier", observed=True)["Daily_Water_Usage_Gallons"].median()
            if {"Low", "High"}.issubset(stress_water.index):
                st.markdown(
                    f"💡 Median daily water usage in **High**-stress regions is "
                    f"**{fmt_gallons(stress_water['High'])}** vs **{fmt_gallons(stress_water['Low'])}** in **Low**-stress regions."
                )

        # ---------------------------- CONSUMPTION & TRENDS ----------------------------
        elif selected_tab == tab_names[2]:
            split = st.toggle("Split electricity trend by Facility Type", value=False)
            st.plotly_chart(chart_electricity_trend(filtered, split), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    chart_top_n_bar(filtered.loc[filtered["Has_Known_Country"]], "Country", "Daily_Electricity_Usage_MWh", 10,
                                    "Top 10 Countries by Electricity Consumption", "Total Electricity Usage (MWh)"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    chart_top_n_bar(filtered, "Owner_Company", "Daily_Electricity_Usage_MWh", 10,
                                    "Top 10 Owner Companies by Electricity Consumption", "Total Electricity Usage (MWh)"),
                    use_container_width=True,
                )

            yearly = filtered.groupby("Year")["Daily_Electricity_Usage_MWh"].sum()
            if len(yearly) >= 2 and yearly.iloc[0] != 0:
                pct = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100
                st.markdown(
                    f"💡 Electricity usage has changed **{pct:+.1f}%** from {int(yearly.index.min())} to {int(yearly.index.max())} "
                    f"across the filtered facilities."
                )

            st.subheader("Capacity vs. Consumption")
            st.plotly_chart(chart_scatter_capacity_vs_electricity(filtered), use_container_width=True)
            corr_ce = filtered["Estimated_Capacity_MW"].corr(filtered["Daily_Electricity_Usage_MWh"])
            st.markdown(f"💡 Capacity and daily electricity usage have a Pearson correlation of **{corr_ce:.3f}** in the current view.")

            st.subheader("Distributions")
            dist_choice = st.radio("Distribution metric", ["Estimated Capacity (MW)", "Daily Electricity Usage (MWh)"], horizontal=True)
            if dist_choice.startswith("Estimated"):
                st.plotly_chart(chart_histogram(filtered, "Estimated_Capacity_MW", "Distribution of Estimated Capacity", "Estimated Capacity (MW)"), use_container_width=True)
            else:
                st.plotly_chart(chart_histogram(filtered, "Daily_Electricity_Usage_MWh", "Distribution of Daily Electricity Usage", "Daily Electricity Usage (MWh)"), use_container_width=True)

        # ---------------------------- GEOGRAPHY ----------------------------
        elif selected_tab == tab_names[3]:
            geo_metric = st.radio("Map metric", ["Electricity Usage", "Facility Count"], horizontal=True)
            st.plotly_chart(chart_geo_map(filtered, geo_metric), use_container_width=True)
            st.plotly_chart(chart_top_cities(filtered, 10), use_container_width=True)

            known = filtered[filtered["Has_Known_City"]]
            if not known.empty:
                top_city = known.groupby("City")["Facility_ID"].nunique().idxmax()
                st.markdown(f"💡 **{top_city}** has the highest concentration of data centers among facilities with known city data.")
            st.caption("Note: rows with 'Unknown' city/country are excluded from geography visuals but remain in KPI totals.")

        # ---------------------------- SUSTAINABILITY RISK & PRIORITY ----------------------------
        elif selected_tab == tab_names[4]:
            st.subheader("Efficiency vs. Consumption Risk Map")
            st.plotly_chart(chart_scatter_pue_vs_electricity(filtered), use_container_width=True)
            st.markdown(
                "💡 Facilities in the upper-right with **High** water-stress coloring and larger bubbles combine "
                "poor efficiency, high consumption, and elevated water risk — the strongest candidates for intervention."
            )

            st.subheader("Sustainability Priority Ranking")
            scored = compute_priority_score(filtered)
            top_n = st.slider("Number of facilities to rank", 5, 30, 15)
            st.plotly_chart(chart_priority_ranking(scored, top_n), use_container_width=True)
            st.caption(
                "Priority Score (0–100) equally weights percentile rank on PUE, daily electricity usage, "
                "daily water usage, and water-stress tier. Higher = more urgent for sustainability investment."
            )

            top_table = (
                scored.sort_values("Sustainability_Priority_Score", ascending=False)
                .head(top_n)[["Facility_ID", "Facility_Name", "Owner_Company", "Country", "City",
                              "PUE", "Daily_Electricity_Usage_MWh", "Daily_Water_Usage_Gallons",
                              "Surrounding_Water_Stress_Tier", "Sustainability_Priority_Score"]]
            )
            st.dataframe(top_table, use_container_width=True, hide_index=True)

        # ---------------------------- CORRELATIONS & DRIVERS ----------------------------
        elif selected_tab == tab_names[5]:
            st.subheader("Correlation Heatmap")
            st.plotly_chart(chart_corr_heatmap(filtered), use_container_width=True)

            corr_matrix = filtered[["PUE", "WUE_L_per_kWh", "Estimated_Capacity_MW",
                                     "Daily_Electricity_Usage_MWh", "Daily_Water_Usage_Gallons"]].corr()
            pue_drivers = corr_matrix["PUE"].drop("PUE").abs().sort_values(ascending=False)
            if not pue_drivers.empty:
                st.markdown(
                    f"💡 The strongest linear driver of PUE in this view is **{pue_drivers.index[0].replace('_', ' ')}** "
                    f"(|r| = {pue_drivers.iloc[0]:.3f})."
                )

            st.subheader("Distribution & Outlier Analysis")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(chart_outlier_box(filtered, "PUE", "PUE Outliers"), use_container_width=True)
            with c2:
                st.plotly_chart(chart_outlier_box(filtered, "Estimated_Capacity_MW", "Capacity Outliers (MW)"), use_container_width=True)
            with c3:
                st.plotly_chart(chart_outlier_box(filtered, "Daily_Electricity_Usage_MWh", "Electricity Usage Outliers (MWh)"), use_container_width=True)

            def iqr_outlier_count(series: pd.Series) -> int:
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

            n_pue_out = iqr_outlier_count(filtered["PUE"])
            n_elec_out = iqr_outlier_count(filtered["Daily_Electricity_Usage_MWh"])
            st.markdown(
                f"💡 Using the 1.5×IQR rule, **{n_pue_out:,}** facility-years are PUE outliers and "
                f"**{n_elec_out:,}** are electricity-usage outliers in the current selection."
            )

        # ---------------------------- FACILITY EXPLORER ----------------------------
        elif selected_tab == tab_names[6]:
            st.subheader("Facility-Level Explorer")
            search = st.text_input("Search by Facility Name, Owner Company, City, or Country")

            explorer_df = filtered.copy()
            if search:
                s = search.lower()
                explorer_df = explorer_df[
                    explorer_df["Facility_Name"].str.lower().str.contains(s, na=False)
                    | explorer_df["Owner_Company"].str.lower().str.contains(s, na=False)
                    | explorer_df["City"].str.lower().str.contains(s, na=False)
                    | explorer_df["Country"].str.lower().str.contains(s, na=False)
                ]

            st.caption(f"Showing {len(explorer_df):,} of {len(filtered):,} filtered rows.")

            display_cols = [
                "Year", "Facility_ID", "Facility_Name", "Owner_Company", "City", "Country",
                "Facility_Type", "Estimated_Capacity_MW", "PUE", "PUE_Tier", "Cooling_System_Type",
                "WUE_L_per_kWh", "Daily_Electricity_Usage_MWh", "Daily_Water_Usage_Gallons",
                "Surrounding_Water_Stress_Tier",
            ]

            if not explorer_df.empty:
                # Color-code PUE / WUE (green = efficient, red = inefficient) without a
                # matplotlib dependency, using a small custom RGB-interpolation helper.
                styled = (
                    explorer_df[display_cols]
                    .style.apply(style_gradient_column, subset=["PUE"])
                    .apply(style_gradient_column, subset=["WUE_L_per_kWh"])
                    .format({
                        "Estimated_Capacity_MW": "{:.2f}", "PUE": "{:.3f}", "WUE_L_per_kWh": "{:.3f}",
                        "Daily_Electricity_Usage_MWh": "{:,.2f}", "Daily_Water_Usage_Gallons": "{:,.2f}",
                    })
                )
                st.dataframe(styled, use_container_width=True, height=450)

                csv_bytes = explorer_df[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download filtered data as CSV", data=csv_bytes,
                    file_name="data_center_filtered.csv", mime="text/csv",
                )
            else:
                st.info("No rows match your search within the current filters.")


if __name__ == "__main__":
    main()