import os
import io
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from statsmodels.tsa.stattools import grangercausalitytests


# -----------------------------
# 0) Basic Config / Styling
# -----------------------------
st.set_page_config(page_title="Sentiment Alpha (FinBERT)", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stMetric"]{
  background:#f6f8fa;
  padding:12px;
  border-radius:14px;
  border:1px solid #e5e7eb;
}
.block-container{padding-top: 1.8rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 FinBERT 情绪因子 · Lead–Lag 分析 · 择时策略回测")
st.caption("Pipeline: Tokenization → FinBERT Inference → Daily Aggregation → Time Alignment → Granger Causality (multi-lag) → Backtest")


def beautify_fig(fig, title=None, ytitle=None):
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=title,
        yaxis_title=ytitle,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


# -----------------------------
# 1) Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("🧪 实验控制台")

    # (Optional) China mirror to improve HF downloads
    use_hf_mirror = st.toggle("使用 HF 国内镜像（可选）", value=True)
    if use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    uploaded_file = st.file_uploader("1) 上传新闻数据（CSV / Excel）", type=["csv", "xlsx", "xls"])

    ticker = st.text_input("2) 股票代码（Yahoo Finance）", value="AAPL").strip().upper()

    max_lag = st.slider("3) Granger 最大滞后阶（1..N）", 1, 12, 5)

    agg_method = st.selectbox("4) 日度聚合方法", ["mean", "median"], index=0)
    align_mode = st.selectbox("5) 新闻日期对齐到交易日", ["next_trading_day（周末/节假日 → 下一个交易日）", "same_day（仅保留重叠日期）"], index=0)

    sentiment_threshold = st.slider("6) 情绪阈值（>阈值持有）", -0.5, 0.5, 0.0, 0.01)

    include_cost = st.toggle("7) 交易成本（可选）", value=False)
    cost_bps = st.slider("   单边成本（bps）", 0, 50, 5, 1) if include_cost else 0

    rf_annual = st.slider("8) 无风险利率（年化，用于 Sharpe）", 0.0, 0.10, 0.02, 0.005)

    finbert_batch = st.slider("9) FinBERT batch size（越大越快，越大越占内存）", 8, 64, 32, 8)

    run_btn = st.button("🚀 运行全流程", type="primary")


# -----------------------------
# 2) Data Loading / Parsing
# -----------------------------
def _read_uploaded_file(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()

    # read bytes once
    raw = file.read()
    bio = io.BytesIO(raw)

    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(bio)
    else:
        df = pd.read_excel(bio)

    return df


def _standardize_news_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept common schemas:
      - title/date/stock
      - headline/date/ticker
      - Headline/Date/Ticker
    Output columns: Date, Headline, Ticker (optional)
    """
    if df.empty:
        return df

    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    lower_map = {c.lower(): c for c in df.columns}

    # headline/title
    headline_col = None
    for k in ["headline", "title", "text", "news", "content"]:
        if k in lower_map:
            headline_col = lower_map[k]
            break

    # date/time
    date_col = None
    for k in ["date", "datetime", "time", "published", "timestamp"]:
        if k in lower_map:
            date_col = lower_map[k]
            break

    # ticker/stock
    ticker_col = None
    for k in ["ticker", "stock", "symbol"]:
        if k in lower_map:
            ticker_col = lower_map[k]
            break

    if headline_col is None or date_col is None:
        raise ValueError("新闻数据必须包含日期列（date）和标题列（headline/title）。请检查你的 CSV/Excel 列名。")

    out = pd.DataFrame()
    out["Headline"] = df[headline_col].astype(str)
    out["Date"] = pd.to_datetime(df[date_col], errors="coerce")

    if ticker_col is not None:
        out["Ticker"] = df[ticker_col].astype(str).str.upper().str.strip()
    else:
        out["Ticker"] = np.nan

    out = out.dropna(subset=["Date", "Headline"])
    out["Headline"] = out["Headline"].replace("nan", np.nan).dropna()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out["NewsDate"] = out["Date"].dt.normalize()  # midnight
    return out[["NewsDate", "Headline", "Ticker"]]


@st.cache_data(show_spinner=False)
def load_news_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(bio)
    else:
        df = pd.read_excel(bio)
    return df


@st.cache_data(show_spinner=False)
def get_market_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Download daily market data and compute returns (simple + log).
    Returns: Date, Close, Return, Log_Return
    """
    df = yf.download(ticker, start=start.date(), end=(end + pd.Timedelta(days=1)).date(), progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    # MultiIndex handling
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    close = df[col]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")

    out = pd.DataFrame({"Date": pd.to_datetime(df["Date"]).dt.normalize(), "Close": close})
    out = out.dropna().sort_values("Date").reset_index(drop=True)

    out["Return"] = out["Close"].pct_change()
    out["Log_Return"] = np.log(out["Close"] / out["Close"].shift(1))
    out = out.dropna().reset_index(drop=True)
    return out


# -----------------------------
# 3) FinBERT: Load Model + Batch Inference
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_finbert() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, int, int]:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    # Robustly map label indices
    id2label = {int(k): v for k, v in model.config.id2label.items()} if hasattr(model.config, "id2label") else {}
    label2id = {v.lower(): k for k, v in id2label.items()}

    def find_idx(target: str) -> int:
        # Try exact label
        for k, v in id2label.items():
            if target in v.lower():
                return k
        # fallback typical order [positive, negative, neutral] for ProsusAI/finbert
        if target == "positive":
            return 0
        if target == "negative":
            return 1
        return 2

    pos_idx = find_idx("positive")
    neg_idx = find_idx("negative")
    return tokenizer, model, pos_idx, neg_idx


def finbert_infer_scores(
    texts: list,
    tokenizer,
    model,
    pos_idx: int,
    neg_idx: int,
    batch_size: int = 32,
) -> np.ndarray:
    device = torch.device("cpu")
    model.to(device)

    scores = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        s = probs[:, pos_idx] - probs[:, neg_idx]  # continuous factor in [-1, 1]
        scores.extend(s.tolist())
    return np.array(scores, dtype=np.float32)


# -----------------------------
# 4) Alignment: News → Trading Day
# -----------------------------
def build_daily_sentiment(news_df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Aggregate headline-level sentiment to daily sentiment factor.
    """
    if news_df.empty:
        return pd.DataFrame()
    g = news_df.groupby("NewsDate")["Sentiment"]
    daily = g.mean() if method == "mean" else g.median()
    out = daily.reset_index().rename(columns={"NewsDate": "Date", "Sentiment": "Sentiment_Factor"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out.sort_values("Date").reset_index(drop=True)


def align_sentiment_to_market(
    daily_sent: pd.DataFrame,
    market: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """
    Align sentiment factor to market trading days.
    mode:
      - next_trading_day: map sentiment date -> next available market date (forward asof)
      - same_day: inner join on exact date
    """
    if daily_sent.empty or market.empty:
        return pd.DataFrame()

    mkt = market[["Date"]].sort_values("Date").reset_index(drop=True)
    sent = daily_sent.copy().sort_values("Date").reset_index(drop=True)

    if mode.startswith("next_trading_day"):
        sent = sent.rename(columns={"Date": "NewsDate"})
        aligned = pd.merge_asof(
            sent.sort_values("NewsDate"),
            mkt.sort_values("Date"),
            left_on="NewsDate",
            right_on="Date",
            direction="forward",
            allow_exact_matches=True,
        )
        aligned = aligned.rename(columns={"Date": "TradeDate"})
        aligned = aligned.dropna(subset=["TradeDate"])
        # multiple news days may map to same trade day => re-aggregate
        aligned = (
            aligned.groupby("TradeDate")["Sentiment_Factor"]
            .mean()
            .reset_index()
            .rename(columns={"TradeDate": "Date"})
        )
        aligned["Date"] = pd.to_datetime(aligned["Date"]).dt.normalize()
        return aligned.sort_values("Date").reset_index(drop=True)

    # same_day
    aligned = pd.merge(daily_sent, market[["Date"]], on="Date", how="inner")
    return aligned[["Date", "Sentiment_Factor"]].sort_values("Date").reset_index(drop=True)


# -----------------------------
# 5) Granger: multi-lag + both directions
# -----------------------------
@dataclass
class GrangerResultRow:
    lag: int
    p_sent_to_ret: float
    p_ret_to_sent: float


def run_granger_multi_lag(merged: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """
    For statsmodels.grangercausalitytests:
      Passing array with columns [y, x] tests whether x Granger-causes y.
    We test:
      - Sentiment -> Return: y=Return, x=Sentiment
      - Return -> Sentiment: y=Sentiment, x=Return
    """
    df = merged[["Return", "Sentiment_Factor"]].dropna().copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Need enough samples: roughly > max_lag + 10 to be safe
    if len(df) < (max_lag + 12):
        raise ValueError(f"样本太少：需要至少 ~{max_lag+12} 行，当前只有 {len(df)} 行。请扩大日期范围或提供更多新闻。")

    # Sentiment -> Return
    ts_sr = df[["Return", "Sentiment_Factor"]].to_numpy()
    res_sr = grangercausalitytests(ts_sr, maxlag=max_lag, verbose=False)

    # Return -> Sentiment
    ts_rs = df[["Sentiment_Factor", "Return"]].to_numpy()
    res_rs = grangercausalitytests(ts_rs, maxlag=max_lag, verbose=False)

    rows = []
    for lag in range(1, max_lag + 1):
        # choose a common test: ssr_ftest p-value
        p1 = float(res_sr[lag][0]["ssr_ftest"][1])
        p2 = float(res_rs[lag][0]["ssr_ftest"][1])
        rows.append(GrangerResultRow(lag=lag, p_sent_to_ret=p1, p_ret_to_sent=p2))

    out = pd.DataFrame([r.__dict__ for r in rows])
    out = out.rename(
        columns={
            "lag": "Lag",
            "p_sent_to_ret": "P-value (Sentiment → Return)",
            "p_ret_to_sent": "P-value (Return → Sentiment)",
        }
    )
    return out


# -----------------------------
# 6) Backtest: timing policy + risk metrics
# -----------------------------
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())  # negative number


def sharpe_ratio(daily_ret: pd.Series, rf_annual: float = 0.02) -> float:
    rf_daily = rf_annual / 252.0
    excess = daily_ret - rf_daily
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * math.sqrt(252))


def run_timing_backtest(
    merged: pd.DataFrame,
    threshold: float = 0.0,
    cost_bps: int = 0,
    rf_annual: float = 0.02,
) -> Tuple[pd.DataFrame, dict]:
    """
    Policy:
      position[t] = 1 if sentiment[t-1] > threshold else 0
      strategy_ret[t] = position[t] * market_ret[t] - cost * |position[t]-position[t-1]|
    """
    df = merged[["Date", "Close", "Return", "Sentiment_Factor"]].dropna().copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["Position"] = (df["Sentiment_Factor"].shift(1) > threshold).astype(int)
    df["Position"] = df["Position"].fillna(0).astype(int)

    # turnover: trade when position changes
    df["Trade"] = df["Position"].diff().abs().fillna(0)
    tc = (cost_bps / 10000.0) * df["Trade"]  # proportion cost on trade days

    df["Strategy_Return"] = df["Position"] * df["Return"] - tc
    df["Benchmark_Return"] = df["Return"]

    # equity curves
    df["Equity_Strategy"] = (1.0 + df["Strategy_Return"]).cumprod()
    df["Equity_Benchmark"] = (1.0 + df["Benchmark_Return"]).cumprod()

    # drawdown series
    df["DD_Strategy"] = df["Equity_Strategy"] / df["Equity_Strategy"].cummax() - 1.0
    df["DD_Benchmark"] = df["Equity_Benchmark"] / df["Equity_Benchmark"].cummax() - 1.0

    # metrics
    strat_cum = float(df["Equity_Strategy"].iloc[-1] - 1.0)
    bench_cum = float(df["Equity_Benchmark"].iloc[-1] - 1.0)

    strat_sharpe = sharpe_ratio(df["Strategy_Return"], rf_annual=rf_annual)
    bench_sharpe = sharpe_ratio(df["Benchmark_Return"], rf_annual=rf_annual)

    strat_mdd = max_drawdown(df["Equity_Strategy"])
    bench_mdd = max_drawdown(df["Equity_Benchmark"])

    strat_vol = float(df["Strategy_Return"].std() * math.sqrt(252))
    bench_vol = float(df["Benchmark_Return"].std() * math.sqrt(252))

    n_trades = int(df["Trade"].sum())
    exposure = float(df["Position"].mean())

    metrics = {
        "Strategy Cumulative Return": strat_cum,
        "Benchmark Cumulative Return": bench_cum,
        "Alpha (Strategy - Benchmark)": strat_cum - bench_cum,
        "Strategy Sharpe": strat_sharpe,
        "Benchmark Sharpe": bench_sharpe,
        "Strategy Max Drawdown": strat_mdd,
        "Benchmark Max Drawdown": bench_mdd,
        "Strategy Vol (ann.)": strat_vol,
        "Benchmark Vol (ann.)": bench_vol,
        "Trades": n_trades,
        "Exposure": exposure,
        "Transaction Cost (bps)": cost_bps,
        "Sentiment Threshold": threshold,
    }
    return df, metrics


# -----------------------------
# 7) Main Run
# -----------------------------
tabs = st.tabs(["① 数据与管线", "② 情绪因子", "③ Lead–Lag（Granger）", "④ 策略回测", "⑤ 导出"])


if not run_btn:
    with tabs[0]:
        st.info("👈 先在左侧上传新闻数据并设置参数，然后点击 **运行全流程**。")
        st.markdown(
            """
**你这份 App 将展示：**
- 新闻标题 → FinBERT → 连续情绪因子（日度聚合）
- 情绪因子与收益序列对齐（支持周末新闻映射到下一交易日）
- Granger 因果检验（1..N 阶，输出 p-values）
- 择时策略回测（昨日情绪>阈值持有，否则空仓）+ 风险调整指标
"""
        )
    st.stop()


if uploaded_file is None:
    st.error("请先上传新闻 CSV/Excel 文件。")
    st.stop()

# Read file bytes for caching
file_bytes = uploaded_file.getvalue()
raw_df = load_news_cached(file_bytes, uploaded_file.name)

try:
    news = _standardize_news_columns(raw_df)
except Exception as e:
    st.error(str(e))
    st.stop()

# Filter by ticker if ticker column exists with values
if news["Ticker"].notna().any():
    if ticker in set(news["Ticker"].dropna().unique()):
        news = news[news["Ticker"] == ticker].copy()
    else:
        st.warning(f"新闻文件中未发现 Ticker={ticker} 的记录，将对全部新闻做情绪计算（你也可以换个 ticker 或检查文件）。")

if news.empty:
    st.error("清洗/筛选后新闻为空。请检查数据文件。")
    st.stop()

# Determine date range from news
min_news_date = pd.to_datetime(news["NewsDate"].min()).normalize()
max_news_date = pd.to_datetime(news["NewsDate"].max()).normalize()
start = min_news_date - pd.Timedelta(days=5)
end = max_news_date + pd.Timedelta(days=10)

# Market data
with st.spinner(f"下载 {ticker} 市场数据并计算收益序列..."):
    market = get_market_data(ticker, start=start, end=end)

if market.empty:
    st.error("无法获取市场数据（Yahoo Finance）。请检查 ticker 或网络。")
    st.stop()

# FinBERT
with st.spinner("加载 FinBERT 模型..."):
    tokenizer, finbert_model, pos_idx, neg_idx = load_finbert()

# Sentiment inference (batched)
with st.spinner("FinBERT 推理：将新闻标题转换为连续情绪因子..."):
    headlines = news["Headline"].astype(str).tolist()
    prog = st.progress(0.0)
    scores = []

    # batch loop with progress
    n = len(headlines)
    step = max(finbert_batch, 1)
    for i in range(0, n, step):
        batch = headlines[i : i + step]
        batch_scores = finbert_infer_scores(batch, tokenizer, finbert_model, pos_idx, neg_idx, batch_size=len(batch))
        scores.extend(batch_scores.tolist())
        prog.progress(min(1.0, (i + step) / n))
    prog.empty()

news = news.reset_index(drop=True)
news["Sentiment"] = np.array(scores, dtype=np.float32)

# Daily aggregate (NLP → factor)
daily_sent = build_daily_sentiment(news, method=agg_method)

# Align to market trading day (time alignment)
aligned_sent = align_sentiment_to_market(
    daily_sent,
    market,
    mode=align_mode,
)

# Merge aligned sentiment with market returns
merged = pd.merge(market, aligned_sent, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

if len(merged) < 30:
    st.warning(f"合并后的有效样本较少（{len(merged)} 行）。可能导致 Granger 检验不稳定。建议扩大新闻日期覆盖或换更长时间窗口。")

# -----------------------------
# Tab ①: Data & Pipeline
# -----------------------------
with tabs[0]:
    st.subheader("① 数据与管线概览（NLP → 因子 → 对齐）")

    c1, c2, c3 = st.columns(3)
    c1.metric("新闻条数（headline-level）", f"{len(news)}")
    c2.metric("情绪日度点数（daily factor）", f"{len(daily_sent)}")
    c3.metric("合并后样本（对齐到交易日）", f"{len(merged)}")

    left, right = st.columns(2)

    with left:
        st.markdown("**新闻数据（清洗后）**")
        st.dataframe(news[["NewsDate", "Headline", "Sentiment"]].head(10), use_container_width=True, height=280)

    with right:
        st.markdown("**市场数据（收益序列）**")
        st.dataframe(market.head(10), use_container_width=True, height=280)

    st.markdown("**对齐后的数据（用于因果检验与回测）**")
    st.dataframe(merged.head(15), use_container_width=True, height=240)

# -----------------------------
# Tab ②: Sentiment Factor Visualization
# -----------------------------
with tabs[1]:
    st.subheader("② 情绪因子（FinBERT）与价格走势")

    # Factor distribution
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=news["Sentiment"], nbinsx=50, name="Headline Sentiment"))
    beautify_fig(fig_hist, title="Headline-level Sentiment Distribution", ytitle="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Price + sentiment factor subplot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=merged["Date"], y=merged["Close"], name=f"{ticker} Close", mode="lines", line=dict(width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=merged["Date"], y=merged["Sentiment_Factor"], name="Daily Sentiment Factor", opacity=0.55),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Factor", secondary_y=True)
    beautify_fig(fig, title="Aligned Sentiment Factor vs Price")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab ③: Lead–Lag (Granger) across multiple lags
# -----------------------------
with tabs[2]:
    st.subheader("③ Lead–Lag 结构检验：Granger Causality（多滞后阶）")
    st.caption("同时报告 Sentiment→Return 与 Return→Sentiment 的 p-values（1..MaxLag）。")

    try:
        gr_df = run_granger_multi_lag(merged, max_lag=max_lag)
        st.dataframe(gr_df.style.format({c: "{:.4f}" for c in gr_df.columns if "P-value" in c}), use_container_width=True)

        # Plot p-values across lags
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=gr_df["Lag"], y=gr_df["P-value (Sentiment → Return)"], mode="lines+markers", name="Sentiment → Return"))
        fig_p.add_trace(go.Scatter(x=gr_df["Lag"], y=gr_df["P-value (Return → Sentiment)"], mode="lines+markers", name="Return → Sentiment"))
        fig_p.add_hline(y=0.05, line_dash="dash", annotation_text="0.05", annotation_position="top left")
        beautify_fig(fig_p, title="Granger p-values across lag orders", ytitle="p-value")
        st.plotly_chart(fig_p, use_container_width=True)

        sig_sr = (gr_df["P-value (Sentiment → Return)"] < 0.05).any()
        best_lag = int(gr_df.loc[gr_df["P-value (Sentiment → Return)"].idxmin(), "Lag"])
        best_p = float(gr_df["P-value (Sentiment → Return)"].min())

        if sig_sr:
            st.success(f"✅ 检测到 **Sentiment → Return** 在某些滞后阶上显著（p<0.05）。最小 p-value 出现在 lag={best_lag}（p={best_p:.4f}）。")
        else:
            st.info(f"当前样本下未检测到显著的 Sentiment → Return（p<0.05）。最小 p-value：lag={best_lag}（p={best_p:.4f}）。")

    except Exception as e:
        st.warning(f"Granger 检验无法执行：{e}")

# -----------------------------
# Tab ④: Backtest Timing Policy + Risk-Adjusted Metrics
# -----------------------------
with tabs[3]:
    st.subheader("④ 择时策略回测（Timing Policy）")
    st.markdown("策略规则：**昨日情绪因子 > 阈值 → 今日持有；否则空仓**。")

    bt_df, metrics = run_timing_backtest(
        merged,
        threshold=sentiment_threshold,
        cost_bps=cost_bps,
        rf_annual=rf_annual,
    )

    # Metrics cards
    c1, c2, c3 = st.columns(3)
    c1.metric("策略累计收益", f"{metrics['Strategy Cumulative Return']*100:.2f}%", delta=f"vs 基准 {metrics['Benchmark Cumulative Return']*100:.2f}%")
    c2.metric("策略 Sharpe", f"{metrics['Strategy Sharpe']:.2f}", delta=f"vs 基准 {metrics['Benchmark Sharpe']:.2f}")
    c3.metric("策略最大回撤", f"{metrics['Strategy Max Drawdown']*100:.2f}%", delta=f"vs 基准 {metrics['Benchmark Max Drawdown']*100:.2f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Alpha（策略-基准）", f"{metrics['Alpha (Strategy - Benchmark)']*100:.2f}%")
    c5.metric("交易次数（换仓）", f"{metrics['Trades']}")
    c6.metric("暴露度（持仓比例）", f"{metrics['Exposure']*100:.1f}%")

    # Equity curve plot
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["Equity_Strategy"], name="Strategy Equity", mode="lines", line=dict(width=3)))
    fig_eq.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["Equity_Benchmark"], name="Buy&Hold Equity", mode="lines", line=dict(dash="dash")))
    beautify_fig(fig_eq, title="Equity Curve: Strategy vs Benchmark", ytitle="Equity")
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown plot
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["DD_Strategy"], name="Strategy Drawdown", mode="lines"))
    fig_dd.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["DD_Benchmark"], name="Benchmark Drawdown", mode="lines", line=dict(dash="dash")))
    beautify_fig(fig_dd, title="Drawdown: Strategy vs Benchmark", ytitle="Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Signal & Sentiment visualization
    fig_sig = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sig.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["Sentiment_Factor"], name="Sentiment Factor", mode="lines"), secondary_y=True)
    fig_sig.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["Position"], name="Position (0/1)", mode="lines", line=dict(width=2)), secondary_y=False)
    fig_sig.update_yaxes(title_text="Position", secondary_y=False)
    fig_sig.update_yaxes(title_text="Sentiment", secondary_y=True)
    beautify_fig(fig_sig, title="Signal Construction: Prior-day Sentiment → Position")
    st.plotly_chart(fig_sig, use_container_width=True)

    st.markdown("**回测明细（前 30 行）**")
    st.dataframe(bt_df.head(30), use_container_width=True, height=260)

# -----------------------------
# Tab ⑤: Export / Download
# -----------------------------
with tabs[4]:
    st.subheader("⑤ 导出（可复现实验）")
    st.caption("下载对齐后的数据、Granger 结果、回测明细，方便你在 notebook / 报告中复现与绘图。")

    # merged data download
    merged_csv = merged.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载对齐后的数据（news-factor-market aligned）", merged_csv, file_name=f"{ticker}_aligned_data.csv", mime="text/csv")

    # granger result download
    try:
        gr_df = run_granger_multi_lag(merged, max_lag=max_lag)
        gr_csv = gr_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ 下载 Granger 结果（multi-lag p-values）", gr_csv, file_name=f"{ticker}_granger_pvalues.csv", mime="text/csv")
    except Exception:
        st.info("Granger 结果不可用（样本不足或检验失败）。")

    # backtest detail download
    bt_df, metrics = run_timing_backtest(
        merged,
        threshold=sentiment_threshold,
        cost_bps=cost_bps,
        rf_annual=rf_annual,
    )
    bt_csv = bt_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载回测明细（positions/returns/equity）", bt_csv, file_name=f"{ticker}_backtest_detail.csv", mime="text/csv")

    # metrics download
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载指标汇总（metrics）", metrics_csv, file_name=f"{ticker}_metrics.csv", mime="text/csv")
