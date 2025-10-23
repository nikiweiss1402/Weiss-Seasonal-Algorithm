"""Weiss Seasonal Algo"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

import streamlit as st
import yfinance as yf

ANNUALIZATION_FACTOR = 252

THEME_STYLE = """
<style>
body, .stApp {
    background-color: #040506;
    color: #f39c12;
}
.stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #f39c12 !important;
}
.stDataFrame, .stTable {
    color: #f39c12 !important;
}
</style>
"""

PLOT_THEME = dict(
    paper_bgcolor="#040506",
    plot_bgcolor="#040506",
    font=dict(color="#f39c12"),
)

LIGHT_PLOT_THEME = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#111111"),
)


@st.cache_data(show_spinner=False)
def load_ohlc(symbol: str, start: date, end: date) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    raw = yf.download(symbol, start=start_ts, end=end_ts, progress=False)
    if raw.empty:
        return pd.DataFrame()

    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(symbol, axis=1, level=-1)
        except (KeyError, IndexError):
            data.columns = data.columns.get_level_values(0)

    columns = list(data.columns)

    def match_column(target: str) -> str | None:
        target = target.lower()
        for col in columns:
            name = str(col).lower().replace(" ", "_")
            if name == target:
                return col
            if name.endswith(f"_{target}"):
                return col
            parts = name.replace("-", "_").split("_")
            if target in parts:
                return col
        return None

    open_col = match_column("open")
    close_col = match_column("close")
    if open_col is None or close_col is None:
        return pd.DataFrame()

    subset = data[[open_col, close_col]].copy()
    subset.rename(columns={open_col: "Open", close_col: "Close"}, inplace=True)
    return subset.dropna()


def compute_sharpe(returns: pd.Series) -> float:
    if returns.empty:
        return math.nan
    vol = returns.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return math.nan
    return math.sqrt(ANNUALIZATION_FACTOR) * returns.mean() / vol


def compute_sortino(returns: pd.Series) -> float:
    if returns.empty:
        return math.nan
    downside = returns[returns < 0]
    if downside.empty:
        return math.inf
    d_vol = downside.std(ddof=1)
    if d_vol == 0 or np.isnan(d_vol):
        return math.nan
    return math.sqrt(ANNUALIZATION_FACTOR) * returns.mean() / d_vol


def make_light_figure(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#111111")
    ax.grid(color="#d0d0d0", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_color("#111111")
    return fig, ax


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def inverse_normal_cdf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924
    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857
    c1 = -0.00778489400243029
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878
    d1 = 0.00778469570904146
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        numerator = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denominator = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        return numerator / denominator
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        numerator = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denominator = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        return -numerator / denominator
    q = p - 0.5
    r = q * q
    numerator = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    denominator = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    return numerator / denominator


def binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return float("nan")
    probs = [math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    observed = probs[k]
    cumulative = sum(prob for prob in probs if prob <= observed)
    return min(1.0, 2.0 * cumulative)


def compute_deflated_sharpe(returns: pd.Series, num_trials: int) -> tuple[float, float, float, float]:
    array = np.asarray(returns, dtype=float)
    n = len(array)
    if n < 2 or np.allclose(array.std(ddof=1), 0):
        return float("nan"), float("nan"), float("nan"), float("nan")
    sr = array.mean() / array.std(ddof=1)
    skew = pd.Series(array).skew()
    excess_kurt = pd.Series(array).kurtosis()
    sr_sigma = math.sqrt((1 - skew * sr + ((excess_kurt + 3 - 1) / 4) * sr ** 2) / (n - 1))
    trials = max(1, num_trials)
    if trials <= 1:
        sr_star = 0.0
    else:
        sr_star = math.sqrt(2 * math.log(trials))
        correction = (math.log(math.log(trials)) + math.log(4 * math.pi)) / (2 * math.sqrt(2 * math.log(trials)))
        sr_star -= correction
    z_value = (sr - sr_star) / sr_sigma
    dsr = normal_cdf(z_value)
    return dsr, sr, sr_star, sr_sigma


def compute_log_seasonal_curve(raw: pd.DataFrame, years: int) -> pd.DataFrame:
    if raw.empty or "Close" not in raw:
        return pd.DataFrame()

    df = raw.copy().dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    df["day_of_year"] = df.index.dayofyear

    years_available = sorted(df["year"].unique())
    if not years_available:
        return pd.DataFrame()
    if len(years_available) > years:
        df = df[df["year"].isin(years_available[-years:])]

    df["log_price"] = np.log(df["Close"])
    df["log_return"] = df.groupby("year")["log_price"].diff()
    df.dropna(subset=["log_return"], inplace=True)

    mean_daily_log_return = (
        df.groupby("day_of_year")["log_return"].mean().sort_index()
    )
    if mean_daily_log_return.empty:
        return pd.DataFrame()

    mean_daily_log_return = mean_daily_log_return.reindex(range(1, 367)).fillna(0.0)
    cumulative_log = mean_daily_log_return.cumsum()
    seasonal_curve = np.exp(cumulative_log) - 1.0

    return seasonal_curve.to_frame(name="Seasonality")


def day_of_year_to_label(day: int) -> str:
    base = datetime(2000, 1, 1) + timedelta(days=day - 1)
    return base.strftime("%b %d")


def parse_month_day(value: str) -> tuple[int, int]:
    month_str, day_str = value.strip().split("-")
    month = int(month_str)
    day = int(day_str)
    datetime(2000, month, day)  # validation
    return month, day


def month_day_to_day_of_year(month: int, day: int) -> int:
    return (datetime(2000, month, day) - datetime(2000, 1, 1)).days + 1


def ordered_days_for_window(start_day: int, end_day: int) -> list[int]:
    if start_day <= end_day:
        return list(range(start_day, end_day + 1))
    return list(range(start_day, 367)) + list(range(1, end_day + 1))


def aggregate_window_trades(
    df: pd.DataFrame, day_window: list[int], direction_multiplier: int
) -> pd.DataFrame:
    if not day_window:
        return pd.DataFrame()

    subset = df[df["day_of_year"].isin(day_window)]
    if subset.empty:
        return pd.DataFrame()

    trades: list[dict[str, object]] = []
    for year, group in subset.groupby("year"):
        group = group.sort_index()
        entry_row = group.iloc[0]
        exit_row = group.iloc[-1]
        entry_price = float(entry_row["Open"])
        exit_price = float(exit_row["Close"])
        ret = direction_multiplier * ((exit_price - entry_price) / entry_price)
        trades.append(
            {
                "year": int(year),
                "entry_date": entry_row.name.date(),
                "exit_date": exit_row.name.date(),
                "return": float(ret),
                "win": int(ret > 0),
            }
        )

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)


def evaluate_windows(
    df: pd.DataFrame,
    min_len: int,
    max_len: int,
    allowed_days: list[int],
    direction_multiplier: int,
) -> pd.DataFrame:
    if not allowed_days:
        return pd.DataFrame()

    total_days = len(allowed_days)
    effective_max = min(max_len, total_days)
    extended_days = allowed_days + allowed_days

    results: list[dict[str, object]] = []
    for start_idx in range(total_days):
        for window_len in range(min_len, effective_max + 1):
            window_days = extended_days[start_idx : start_idx + window_len]
            trades = aggregate_window_trades(df, window_days, direction_multiplier)
            if trades.empty:
                continue

            start_day = window_days[0]
            end_day = window_days[-1]
            returns = trades["return"]
            avg_return = returns.mean()
            total_return = float((returns + 1.0).prod() - 1.0)
            win_rate = float(trades["win"].mean())
            sharpe = compute_sharpe(returns)
            sortino = compute_sortino(returns)

            results.append(
                {
                    "start_day": start_day,
                    "end_day": end_day,
                    "window_length": window_len,
                    "range_label": f"{day_of_year_to_label(start_day)} → {day_of_year_to_label(end_day)}",
                    "win_rate": win_rate,
                    "avg_return": float(avg_return),
                    "total_return": total_return,
                    "sharpe": float(sharpe),
                    "sortino": float(sortino),
                    "trades": len(trades),
                }
            )

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results.drop_duplicates(
        subset=["start_day", "window_length"], keep="first", inplace=True
    )
    df_results.sort_values(
        by=["win_rate", "avg_return", "total_return"],
        ascending=[False, False, False],
        inplace=True,
    )
    df_results.reset_index(drop=True, inplace=True)
    return df_results


def build_cumulative_curve(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    returns = trades["return"].reset_index(drop=True)
    cumulative = (returns + 1.0).cumprod() - 1.0
    return pd.DataFrame({"Cumulative": cumulative.values})


def main() -> None:
    st.set_page_config(page_title="Weiss Seasonal Algo", layout="wide")
    st.markdown(THEME_STYLE, unsafe_allow_html=True)

    st.title("Weiss Seasonal Algo")
    st.caption("Quantitative view on seasonal trade setups.")

    today = date.today()
    default_start = datetime(2000, 1, 1)
    default_end = datetime(2000, 12, 31)

    with st.sidebar:
        st.header("Parameters")
        symbol = st.text_input("Asset / Ticker", value="EURUSD=X")
        history_years = int(
            st.number_input("History (years)", min_value=1, max_value=50, value=10, step=1)
        )
        direction = st.radio("Trade direction", options=("Long", "Short"), horizontal=True)
        start_label = st.text_input("Season start (MM-DD)", value="09-10")
        end_label = st.text_input("Season end (MM-DD)", value="10-15")
        min_len = int(
            st.number_input("Minimum window (days)", min_value=1, max_value=366, value=10, step=1)
        )
        max_len = int(
            st.number_input("Maximum window (days)", min_value=1, max_value=366, value=30, step=1)
        )
        seasonality_type = st.selectbox(
            "Seasonality type",
            options=(
                "Standard seasonality",
                "Election cycle year 1",
                "Election cycle year 2",
                "Election cycle year 3",
                "Election cycle year 4",
            ),
        )

    try:
        start_month, start_day = parse_month_day(start_label)
        end_month, end_day = parse_month_day(end_label)
    except Exception:
        st.error("Please provide valid dates in MM-DD format.")
        st.stop()

    if min_len > max_len:
        st.error("Minimum window must not exceed maximum window.")
        st.stop()

    run_clicked = st.sidebar.button("Run analysis", type="primary")
    if not run_clicked:
        st.stop()

    window_start_day = month_day_to_day_of_year(start_month, start_day)
    window_end_day = month_day_to_day_of_year(end_month, end_day)
    direction_multiplier = 1 if direction == "Long" else -1

    end_date = today
    start_date = today - timedelta(days=int(history_years * 365.25))
    split_date = start_date

    if start_date >= end_date:
        st.error("The start date must be earlier than the end date.")
        st.stop()

    with st.spinner("Loading price data..."):
        raw_prices = load_ohlc(symbol, start_date, end_date)

    if raw_prices.empty:
        st.error("No data found. Please check symbol and date range.")
        st.stop()

    seasonal_curve = compute_log_seasonal_curve(raw_prices, history_years)

    seasonal_fig: go.Figure | None = None
    if not seasonal_curve.empty:
        seasonal_values = seasonal_curve["Seasonality"]
        hover_labels = [
            (datetime(2000, 1, 1) + timedelta(days=int(day) - 1)).strftime("%b %d")
            for day in seasonal_curve.index
        ]
        seasonal_fig = go.Figure()
        seasonal_fig.add_trace(
            go.Scatter(
                x=seasonal_curve.index,
                y=seasonal_values,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                name="Seasonal profile",
                customdata=hover_labels,
                hovertemplate="Day: %{customdata}<br>Return: %{y:.2%}<extra></extra>",
            )
        )
        y_min = seasonal_values.min()
        y_max = seasonal_values.max()
        if window_start_day <= window_end_day:
            seasonal_fig.add_shape(
                type="rect",
                x0=window_start_day,
                x1=window_end_day,
                y0=y_min,
                y1=y_max,
                xref="x",
                yref="y",
                fillcolor="rgba(41, 128, 185, 0.10)",
                line=dict(width=0),
            )
        else:
            seasonal_fig.add_shape(
                type="rect",
                x0=window_start_day,
                x1=366,
                y0=y_min,
                y1=y_max,
                xref="x",
                yref="y",
                fillcolor="rgba(41, 128, 185, 0.10)",
                line=dict(width=0),
            )
            seasonal_fig.add_shape(
                type="rect",
                x0=1,
                x1=window_end_day,
                y0=y_min,
                y1=y_max,
                xref="x",
                yref="y",
                fillcolor="rgba(41, 128, 185, 0.10)",
                line=dict(width=0),
            )
        for boundary in (window_start_day, window_end_day):
            seasonal_fig.add_shape(
                type="line",
                x0=boundary,
                x1=boundary,
                y0=y_min,
                y1=y_max,
                xref="x",
                yref="y",
                line=dict(color="#9a9a9a", width=2),
            )
        seasonal_fig.update_layout(PLOT_THEME)
        seasonal_fig.update_xaxes(
            tickmode="array",
            tickvals=[month_day_to_day_of_year(m, 1) for m in range(1, 13)],
            ticktext=[datetime(2000, m, 1).strftime("%b") for m in range(1, 13)],
            range=[1, 366],
        )
        seasonal_fig.update_yaxes(tickformat=".2%")
    else:
        st.info("Not enough data for the seasonal curve.")

    prices = raw_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)
    asset_returns = (prices["Close"] - prices["Open"]) / prices["Open"]
    if direction_multiplier == -1:
        strategy_returns = -asset_returns
    else:
        strategy_returns = asset_returns

    prices = prices.assign(
        strategy_return=strategy_returns,
        asset_return=asset_returns,
        win=(strategy_returns > 0).astype(int),
        year=prices.index.year,
        day_of_year=prices.index.dayofyear,
        cycle_year=prices.index.year.map(lambda year: ((year - 2020) % 4) + 1),
    )

    if seasonality_type == "Standard seasonality":
        filtered_df = prices
    else:
        target_cycle = int(seasonality_type.split()[-1])
        filtered_df = prices[prices["cycle_year"] == target_cycle]

    if filtered_df.empty:
        st.error("No data available for the selected seasonality.")
        return

    allowed_days = ordered_days_for_window(window_start_day, window_end_day)
    if not allowed_days:
        st.error("No trading days found inside the selected window.")
        return

    if min_len > len(allowed_days):
        st.error("Minimum window length exceeds the available window.")
        return

    base_trades = aggregate_window_trades(filtered_df, allowed_days, direction_multiplier)
    if base_trades.empty:
        st.error("No yearly trades found inside the selected window.")
        return

    st.success(
        f"Loaded {len(base_trades)} yearly trades in window (last {history_years} years)."
    )
    caption = (
        f"Window {start_label} to {end_label} | "
        f"Min {min_len} | Max {min(max_len, len(allowed_days))} calendar days"
    )
    st.caption(caption)

    results_df = evaluate_windows(
        filtered_df,
        min_len,
        max_len,
        allowed_days,
        direction_multiplier,
    )
    if results_df.empty:
        st.warning("No windows found. Please adjust the parameters.")
        return

    top_results = results_df.head(10).copy()
    if top_results.empty:
        st.warning("No windows available for display.")
        return

    def format_results_table(frame: pd.DataFrame) -> pd.DataFrame:
        display = frame[
            [
                "range_label",
                "window_length",
                "win_rate",
                "avg_return",
                "total_return",
            ]
        ].rename(
            columns={
                "range_label": "Window",
                "window_length": "Length",
                "win_rate": "Win rate",
                "avg_return": "Average return",
                "total_return": "Total return",
            }
        )
        display["Win rate"] = display["Win rate"].map(lambda x: f"{x:.1%}")
        display["Average return"] = display["Average return"].map(lambda x: f"{x:.3%}")
        display["Total return"] = display["Total return"].map(lambda x: f"{x:.1%}")
        return display

    ranking_display = format_results_table(top_results)

    left_col, right_col = st.columns([3, 7])
    with left_col:
        st.subheader("Top 10 windows (win rate)")
        option_labels = [
            f"{idx + 1}. {row['range_label']} | Win rate {row['win_rate']:.1%} | Trades {row['trades']}"
            for idx, row in top_results.iterrows()
        ]
        st.dataframe(ranking_display, use_container_width=True)


    with right_col:
        st.subheader(f"Seasonal curve ({history_years} years)")
        if seasonal_fig:
            st.plotly_chart(seasonal_fig, use_container_width=True)
        else:
            st.info("Seasonal curve unavailable.")

    selected_row = top_results.iloc[0]

    offset_map = {day: idx for idx, day in enumerate(allowed_days)}
    heatmap_lengths = sorted(results_df["window_length"].unique())
    heatmap_matrix = pd.DataFrame(
        np.nan,
        index=heatmap_lengths,
        columns=allowed_days,
    )
    for _, row in results_df.iterrows():
        start_day = int(row["start_day"])
        if start_day not in offset_map:
            continue
        length = int(row["window_length"])
        value = float(row["win_rate"]) * 100.0
        current = heatmap_matrix.at[length, start_day]
        if np.isnan(current):
            heatmap_matrix.at[length, start_day] = value
        else:
            heatmap_matrix.at[length, start_day] = (current + value) / 2
    heatmap_matrix.sort_index(inplace=True)
    heatmap_date_labels = [day_of_year_to_label(day) for day in heatmap_matrix.columns]

    selected_start_day = int(selected_row["start_day"])
    selected_length = int(selected_row["window_length"])
    start_position = allowed_days.index(selected_start_day)
    extended_days = allowed_days + allowed_days
    selected_window_days = extended_days[start_position : start_position + selected_length]

    selected_trades = aggregate_window_trades(
        filtered_df, selected_window_days, direction_multiplier
    )

    if selected_trades.empty:
        yearly_returns = pd.Series(dtype=float)
    else:
        yearly_returns = selected_trades.groupby("year")["return"].sum().sort_index()

    forward_start_ts = pd.Timestamp(split_date)
    if selected_trades.empty:
        forward_trades = pd.DataFrame(columns=["entry_date", "return"])
    else:
        forward_trades = selected_trades[
            pd.to_datetime(selected_trades["entry_date"]) >= forward_start_ts
        ]
    oos_winrate = float(forward_trades["return"].gt(0).mean()) if not forward_trades.empty else float("nan")

    with st.expander("Data", expanded=False):
        st.subheader("Annual window returns")
        yearly_df = yearly_returns.reset_index().rename(columns={"year": "Year", "return": "Return"})
        if yearly_df.empty:
            st.info("No yearly returns available for the selected window.")
        else:
            fig_year, ax_year = make_light_figure((12, 3))
            colors = ["#4db6ff" if val >= 0 else "#ff7043" for val in yearly_df["Return"]]
            ax_year.bar(yearly_df["Year"].astype(str), yearly_df["Return"], color=colors)
            ax_year.axhline(0, color="#999999", linewidth=0.8, alpha=0.6)
            ax_year.set_ylabel("Return", color="#111111")
            ax_year.set_title("Yearly window returns", color="#111111")
            st.pyplot(fig_year, use_container_width=True)
            plt.close(fig_year)
            st.caption(f"Quelle: {symbol}")
            st.download_button(
                "Export as CSV",
                yearly_df.to_csv(index=False).encode("utf-8"),
                file_name=f"yearly_returns_{symbol.replace('/', '_')}.csv",
                mime="text/csv",
                key="yearly_returns_csv",
            )

        heat_col, oos_col = st.columns(2)

        with heat_col:
            st.subheader("Winrate Heatmap")
            heatmap_csv = heatmap_matrix.copy()
            heatmap_csv.columns = heatmap_date_labels
            st.download_button(
                "Export as CSV",
                heatmap_csv.to_csv().encode("utf-8"),
                file_name=f"heatmap_{symbol.replace('/', '_')}.csv",
                mime="text/csv",
                key="heatmap_csv_top",
            )
            if heatmap_matrix.dropna(how="all").dropna(axis=1, how="all").empty:
                st.info("Heatmap unavailable for the current configuration.")
            else:
                fig_hm, ax_hm = make_light_figure((6, 4))
                matrix = np.ma.masked_invalid(heatmap_matrix.values)
                im = ax_hm.imshow(
                    matrix,
                    cmap="viridis",
                    origin="lower",
                    vmin=0,
                    vmax=100,
                    aspect="auto",
                )
                ax_hm.set_xlabel("Start date", color="#111111")
                ax_hm.set_ylabel("Length (days)", color="#111111")
                ax_hm.set_title("Winrate Heatmap", color="#111111", fontsize=14)
                step_x = max(1, len(heatmap_date_labels) // 10)
                tick_positions = list(range(0, len(heatmap_date_labels), step_x))
                ax_hm.set_xticks(tick_positions)
                ax_hm.set_xticklabels([heatmap_date_labels[i] for i in tick_positions], rotation=45, ha="right", color="#111111")
                step_y = max(1, len(heatmap_lengths) // 10)
                ax_hm.set_yticks(range(0, len(heatmap_lengths), step_y))
                ax_hm.set_yticklabels([str(heatmap_lengths[i]) for i in range(0, len(heatmap_lengths), step_y)], color="#111111")
                cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel("Win rate (%)", color="#111111")
                cbar.ax.tick_params(colors="#111111")
                st.pyplot(fig_hm, use_container_width=True)
                plt.close(fig_hm)
            st.caption(f"Quelle: {symbol}")
            st.download_button(
                "Export as CSV",
                heatmap_csv.to_csv().encode("utf-8"),
                file_name=f"heatmap_{symbol.replace('/', '_')}.csv",
                mime="text/csv",
                key="heatmap_csv_bottom",
            )

        with oos_col:
            st.subheader("Walk-Forward (OOS)")
            st.caption(f"Walk-Forward Winrate: {oos_winrate:.2%} - Trades: {len(forward_trades)}")
            if forward_trades.empty:
                st.info("No forward trades after the chosen date.")
                oos_df = pd.DataFrame(columns=["Trade", "Return", "Cumulative"])
            else:
                cumulative_moves = forward_trades["return"].cumsum()
                oos_df = pd.DataFrame(
                    {
                        "Trade": np.arange(1, len(cumulative_moves) + 1),
                        "Return": forward_trades["return"].values,
                        "Cumulative": cumulative_moves.values,
                    }
                )
                fig_oos, ax_oos = make_light_figure((6, 4))
                ax_oos.plot(oos_df["Trade"], oos_df["Cumulative"], color="#4db6ff", linewidth=2)
                ax_oos.set_xlabel("Trades", color="#111111")
                ax_oos.set_ylabel("Summe der Moves", color="#111111")
                ax_oos.set_title("Walk-Forward (OOS)", color="#111111", fontsize=14)
                ax_oos.grid(color="#cccccc", alpha=0.4)
                st.pyplot(fig_oos, use_container_width=True)
                plt.close(fig_oos)
            st.caption(f"Quelle: {symbol}")
            st.download_button(
                "Export as CSV",
                oos_df.to_csv(index=False).encode("utf-8"),
                file_name=f"walk_forward_oos_{symbol.replace('/', '_')}.csv",
                mime="text/csv",
            )



if __name__ == "__main__":
    main()







