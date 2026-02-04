import os
import io
import math
import time
import uuid
import queue
import threading
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

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

st.title("📈 FinBERT 情绪因子 · Lead–Lag 分析 · 择时策略回测（长文本 + 异步推理）")
st.caption("Pipeline: Tokenization → FinBERT (sliding window/pooling) → Daily Aggregation → Time Alignment → Granger (multi-lag) → Backtest")


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


def _rerun():
    # 兼容不同 streamlit 版本
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


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
    align_mode = st.selectbox(
        "5) 新闻日期对齐到交易日",
        ["next_trading_day（周末/节假日 → 下一个交易日）", "same_day（仅保留重叠日期）"],
        index=0,
    )

    sentiment_threshold = st.slider("6) 情绪阈值（>阈值持有）", -0.5, 0.5, 0.0, 0.01)

    include_cost = st.toggle("7) 交易成本（可选）", value=False)
    cost_bps = st.slider("   单边成本（bps）", 0, 50, 5, 1) if include_cost else 0

    rf_annual = st.slider("8) 无风险利率（年化，用于 Sharpe）", 0.0, 0.10, 0.02, 0.005)

    # ---------- 长文本（sliding window）控制 ----------
    st.markdown("---")
    st.subheader("🧩 长文本支持（sliding window）")

    longtext_mode = st.selectbox(
        "文本长度处理模式",
        ["auto（超过512 tokens才切分）", "always（强制切分）", "off（不切分/直接截断）"],
        index=0,
        help="auto：标题短文本不会切分；长文档会按窗口切分并聚合。always：所有文本都切分。",
    )
    window_max_len = st.selectbox("窗口 max_length", [128, 256, 512], index=2, help="BERT/FinBERT 最大通常为 512 tokens。长文本请用 512。")
    window_stride = st.slider("滑窗 stride（重叠）", 0, 256, 128, 16, help="stride 越大重叠越多，越能保留跨段信息，但更耗时。")
    pooling = st.selectbox("窗口聚合 pooling", ["mean", "max"], index=0, help="mean：更平滑；max：更敏感（抓强情绪片段）。")

    # ---------- 异步推理 ----------
    st.markdown("---")
    st.subheader("⚡ 异步推理（不卡 UI）")
    enable_async = st.toggle("启用异步推理（推荐）", value=True, help="启用后：推理在后台线程跑，界面不会被卡住。")

    finbert_batch = st.slider("FinBERT batch size（越大越快，越大越占内存）", 8, 128, 32, 8)

    run_btn = st.button("🚀 运行全流程", type="primary")


# -----------------------------
# 2) Data Loading / Parsing
# -----------------------------
@st.cache_data(show_spinner=False)
def load_news_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
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
    Output columns: NewsDate, Headline, Ticker (optional)
    """
    if df.empty:
        return df

    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    lower_map = {c.lower(): c for c in df.columns}

    headline_col = None
    for k in ["headline", "title", "text", "news", "content"]:
        if k in lower_map:
            headline_col = lower_map[k]
            break

    date_col = None
    for k in ["date", "datetime", "time", "published", "timestamp"]:
        if k in lower_map:
            date_col = lower_map[k]
            break

    ticker_col = None
    for k in ["ticker", "stock", "symbol"]:
        if k in lower_map:
            ticker_col = lower_map[k]
            break

    if headline_col is None or date_col is None:
        raise ValueError("新闻数据必须包含日期列（date）和标题/正文列（headline/title/text/content）。请检查你的 CSV/Excel 列名。")

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
def get_market_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Download daily market data and compute returns (simple + log).
    Returns: Date, Close, Return, Log_Return
    """
    df = yf.download(ticker, start=start.date(), end=(end + pd.Timedelta(days=1)).date(), progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
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
# 3) FinBERT: Load Model
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_finbert() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, int, int]:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()} if hasattr(model.config, "id2label") else {}

    def find_idx(target: str) -> int:
        for k, v in id2label.items():
            if target in v.lower():
                return k
        # fallback typical order
        if target == "positive":
            return 0
        if target == "negative":
            return 1
        return 2

    pos_idx = find_idx("positive")
    neg_idx = find_idx("negative")
    return tokenizer, model, pos_idx, neg_idx


def _pick_device() -> torch.device:
    # 允许自动用 GPU（如果部署环境有 CUDA）
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def finbert_scores_sliding_window(
    texts: List[str],
    tokenizer,
    model,
    pos_idx: int,
    neg_idx: int,
    batch_size: int = 32,
    max_length: int = 512,
    stride: int = 128,
    mode: str = "auto",
    pooling: str = "mean",
    progress_cb=None,  # progress_cb(done_chunks, total_chunks)
) -> np.ndarray:
    """
    长文本支持：
      - mode=off: 直接截断 max_length
      - mode=auto: 只有超长文本才启用 overflow sliding window
      - mode=always: 所有文本都启用 sliding window
    聚合：
      - pooling=mean: 对一个文档的多个窗口分数取平均
      - pooling=max: 取最大（抓强情绪片段）
    """
    if len(texts) == 0:
        return np.array([], dtype=np.float32)

    device = _pick_device()
    model = model.to(device)

    # 决定是否启用 overflow
    use_overflow = mode in ["always", "auto"]

    # 为了实现 auto，我们需要知道每条文本是否超长
    # 这里用 tokenizer 不截断地先做一次长度估计（只取 input_ids 长度）
    # 注意：这一步相对轻量，比跑模型便宜
    lengths = []
    if mode == "auto":
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=True, truncation=False)
            lengths.append(len(ids))
    else:
        lengths = [max_length + 1] * len(texts)  # 强制认为超长，走 overflow（always）

    # 对每条文本判断：是否需要切分
    need_chunk = [(l > max_length) for l in lengths] if mode == "auto" else ([True] * len(texts) if mode == "always" else [False] * len(texts))

    # 如果 mode=off：直接截断推理（批量快）
    if mode == "off" or (mode == "auto" and not any(need_chunk)):
        scores = []
        n = len(texts)
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            s = probs[:, pos_idx] - probs[:, neg_idx]
            scores.extend(s.tolist())
            if progress_cb:
                progress_cb(min(i + batch_size, n), n)
        return np.array(scores, dtype=np.float32)

    # ---------- 混合：短文本直接截断，长文本 sliding window ----------
    # 为了简单可靠：我们把所有文本都用 overflow tokenizer 一次性展开为 chunks
    # 但对于不需要切分的文本，我们可以让 stride=0 并保持单块；这里直接统一走 overflow
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        stride=stride,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    overflow_map = enc["overflow_to_sample_mapping"]  # 每个chunk属于哪个原始样本

    total_chunks = int(input_ids.shape[0])
    # 安全阈值：避免极端长文本导致 chunk 数爆炸
    if total_chunks > 6000:
        raise RuntimeError(f"文本过长导致窗口数量过多（{total_chunks} chunks）。建议减小数据量或缩短文本/增大 stride/降低 max_length。")

    # 对 chunk 批量跑模型
    chunk_scores = np.zeros((total_chunks,), dtype=np.float32)
    done = 0
    for i in range(0, total_chunks, batch_size):
        ids = input_ids[i : i + batch_size].to(device)
        msk = attention_mask[i : i + batch_size].to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=msk).logits
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        s = probs[:, pos_idx] - probs[:, neg_idx]
        chunk_scores[i : i + len(s)] = s.astype(np.float32)
        done = i + len(s)
        if progress_cb:
            progress_cb(done, total_chunks)

    # 聚合回 doc-level 分数
    doc_scores: List[List[float]] = [[] for _ in range(len(texts))]
    for c_idx, doc_idx in enumerate(overflow_map.tolist()):
        doc_scores[doc_idx].append(float(chunk_scores[c_idx]))

    out = np.zeros((len(texts),), dtype=np.float32)
    for i, arr in enumerate(doc_scores):
        if not arr:
            out[i] = 0.0
        else:
            if pooling == "max":
                out[i] = float(np.max(arr))
            else:
                out[i] = float(np.mean(arr))
    return out


# -----------------------------
# 3.5) Async Inference Worker (Producer-Consumer)
# -----------------------------
@dataclass
class InferenceJob:
    job_id: str
    status: str  # queued/running/done/error
    created_at: float
    progress: float
    message: str
    scores: Optional[np.ndarray] = None
    error: Optional[str] = None


class AsyncFinBERTWorker:
    """
    后台线程：消费队列里的推理任务
    - 解耦 ingestion/UI 与 推理
    - 支持 long text sliding window + pooling
    """

    def __init__(self, tokenizer, model, pos_idx: int, neg_idx: int):
        self.tokenizer = tokenizer
        self.model = model
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx

        self.q: "queue.Queue[Tuple[str, List[str], dict]]" = queue.Queue()
        self.jobs: Dict[str, InferenceJob] = {}
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, texts: List[str], params: dict) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self.jobs[job_id] = InferenceJob(
                job_id=job_id,
                status="queued",
                created_at=time.time(),
                progress=0.0,
                message="Queued",
            )
        self.q.put((job_id, texts, params))
        return job_id

    def get(self, job_id: str) -> Optional[InferenceJob]:
        with self._lock:
            return self.jobs.get(job_id)

    def _update(self, job_id: str, **kwargs):
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)

    def _loop(self):
        while True:
            job_id, texts, params = self.q.get()
            try:
                self._update(job_id, status="running", message="Running", progress=0.0)

                def cb(done, total):
                    p = 0.0 if total == 0 else float(done) / float(total)
                    self._update(job_id, progress=p, message=f"Running ({done}/{total})")

                scores = finbert_scores_sliding_window(
                    texts=texts,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    pos_idx=self.pos_idx,
                    neg_idx=self.neg_idx,
                    batch_size=int(params["batch_size"]),
                    max_length=int(params["max_length"]),
                    stride=int(params["stride"]),
                    mode=str(params["mode"]),
                    pooling=str(params["pooling"]),
                    progress_cb=cb,
                )
                self._update(job_id, status="done", message="Done", progress=1.0, scores=scores)

            except Exception as e:
                self._update(job_id, status="error", message="Error", error=str(e), progress=0.0)

            finally:
                self.q.task_done()


def get_worker():
    """
    在 session 中持久化 worker（每个用户会话一个），避免每次 rerun 重建线程。
    """
    if "finbert_worker" not in st.session_state:
        tokenizer, finbert_model, pos_idx, neg_idx = load_finbert()
        st.session_state.finbert_worker = AsyncFinBERTWorker(tokenizer, finbert_model, pos_idx, neg_idx)
    return st.session_state.finbert_worker


# -----------------------------
# 4) Alignment: News → Trading Day
# -----------------------------
def build_daily_sentiment(news_df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame()
    g = news_df.groupby("NewsDate")["Sentiment"]
    daily = g.mean() if method == "mean" else g.median()
    out = daily.reset_index().rename(columns={"NewsDate": "Date", "Sentiment": "Sentiment_Factor"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out.sort_values("Date").reset_index(drop=True)


def align_sentiment_to_market(daily_sent: pd.DataFrame, market: pd.DataFrame, mode: str) -> pd.DataFrame:
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
        aligned = aligned.rename(columns={"Date": "TradeDate"}).dropna(subset=["TradeDate"])
        aligned = (
            aligned.groupby("TradeDate")["Sentiment_Factor"]
            .mean()
            .reset_index()
            .rename(columns={"TradeDate": "Date"})
        )
        aligned["Date"] = pd.to_datetime(aligned["Date"]).dt.normalize()
        return aligned.sort_values("Date").reset_index(drop=True)

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
    df = merged[["Return", "Sentiment_Factor"]].dropna().copy()
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < (max_lag + 12):
        raise ValueError(f"样本太少：需要至少 ~{max_lag+12} 行，当前只有 {len(df)} 行。请扩大日期范围或提供更多新闻。")

    ts_sr = df[["Return", "Sentiment_Factor"]].to_numpy()
    res_sr = grangercausalitytests(ts_sr, maxlag=max_lag, verbose=False)

    ts_rs = df[["Sentiment_Factor", "Return"]].to_numpy()
    res_rs = grangercausalitytests(ts_rs, maxlag=max_lag, verbose=False)

    rows = []
    for lag in range(1, max_lag + 1):
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
    return float(dd.min())


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
    df = merged[["Date", "Close", "Return", "Sentiment_Factor"]].dropna().copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["Position"] = (df["Sentiment_Factor"].shift(1) > threshold).astype(int)
    df["Position"] = df["Position"].fillna(0).astype(int)

    df["Trade"] = df["Position"].diff().abs().fillna(0)
    tc = (cost_bps / 10000.0) * df["Trade"]

    df["Strategy_Return"] = df["Position"] * df["Return"] - tc
    df["Benchmark_Return"] = df["Return"]

    df["Equity_Strategy"] = (1.0 + df["Strategy_Return"]).cumprod()
    df["Equity_Benchmark"] = (1.0 + df["Benchmark_Return"]).cumprod()

    df["DD_Strategy"] = df["Equity_Strategy"] / df["Equity_Strategy"].cummax() - 1.0
    df["DD_Benchmark"] = df["Equity_Benchmark"] / df["Equity_Benchmark"].cummax() - 1.0

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
# 7) Main Run + Tabs
# -----------------------------
tabs = st.tabs(["① 数据与管线", "② 情绪因子", "③ Lead–Lag（Granger）", "④ 策略回测", "⑤ 导出"])

if not run_btn:
    with tabs[0]:
        st.info("👈 先在左侧上传新闻数据并设置参数，然后点击 **运行全流程**。")
        st.markdown(
            """
**你这份 App 将展示：**
- 新闻标题/正文 → FinBERT（支持长文本滑窗）→ 连续情绪因子（日度聚合）
- 情绪因子与收益序列对齐（支持周末新闻映射到下一交易日）
- Granger 因果检验（1..N 阶，输出 p-values）
- 择时策略回测（昨日情绪>阈值持有，否则空仓）+ 风险调整指标
- 异步推理：推理在后台跑，UI 不被卡住（适合新闻 spikes）
"""
        )
    st.stop()

if uploaded_file is None:
    st.error("请先上传新闻 CSV/Excel 文件。")
    st.stop()

# 读入数据
file_bytes = uploaded_file.getvalue()
raw_df = load_news_cached(file_bytes, uploaded_file.name)

try:
    news = _standardize_news_columns(raw_df)
except Exception as e:
    st.error(str(e))
    st.stop()

# ticker 过滤
if news["Ticker"].notna().any():
    if ticker in set(news["Ticker"].dropna().unique()):
        news = news[news["Ticker"] == ticker].copy()
    else:
        st.warning(f"新闻文件中未发现 Ticker={ticker} 的记录，将对全部新闻做情绪计算。")

if news.empty:
    st.error("清洗/筛选后新闻为空。请检查数据文件。")
    st.stop()

# 日期范围 -> 市场数据
min_news_date = pd.to_datetime(news["NewsDate"].min()).normalize()
max_news_date = pd.to_datetime(news["NewsDate"].max()).normalize()
start = min_news_date - pd.Timedelta(days=5)
end = max_news_date + pd.Timedelta(days=10)

with st.spinner(f"下载 {ticker} 市场数据并计算收益序列..."):
    market = get_market_data(ticker, start=start, end=end)

if market.empty:
    st.error("无法获取市场数据（Yahoo Finance）。请检查 ticker 或网络。")
    st.stop()

# ========== 异步推理：提交任务 ==========
# 用参数签名避免重复提交（用户反复切 tab 或 rerun 时）
job_params = dict(
    batch_size=int(finbert_batch),
    max_length=int(window_max_len),
    stride=int(window_stride),
    mode="auto" if longtext_mode.startswith("auto") else ("always" if longtext_mode.startswith("always") else "off"),
    pooling=str(pooling),
)

headlines = news["Headline"].astype(str).tolist()

# 用 session_state 管理 job 生命周期
if "infer_job_id" not in st.session_state:
    st.session_state.infer_job_id = None
    st.session_state.infer_job_sig = None

job_sig = (len(headlines), ticker, tuple(sorted(job_params.items())), float(min_news_date.value), float(max_news_date.value))

if enable_async:
    worker = get_worker()
    if st.session_state.infer_job_sig != job_sig:
        # 新参数/新数据 => 新 job
        st.session_state.infer_job_id = worker.submit(headlines, job_params)
        st.session_state.infer_job_sig = job_sig
else:
    # 同步模式：直接算（会卡 UI）
    st.session_state.infer_job_id = "__sync__"
    st.session_state.infer_job_sig = job_sig

# 取结果（如果 ready）
scores = None
infer_status = None
infer_msg = ""
infer_progress = 0.0
infer_error = None

if enable_async:
    job = worker.get(st.session_state.infer_job_id) if st.session_state.infer_job_id else None
    if job:
        infer_status = job.status
        infer_msg = job.message
        infer_progress = job.progress
        infer_error = job.error
        if job.status == "done":
            scores = job.scores
else:
    # 同步计算
    with st.spinner("FinBERT 推理中（同步模式会卡住 UI）..."):
        tokenizer, finbert_model, pos_idx, neg_idx = load_finbert()
        pbar = st.progress(0.0)

        def cb(done, total):
            pbar.progress(0.0 if total == 0 else float(done) / float(total))

        scores = finbert_scores_sliding_window(
            texts=headlines,
            tokenizer=tokenizer,
            model=finbert_model,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            batch_size=int(finbert_batch),
            max_length=int(window_max_len),
            stride=int(window_stride),
            mode=job_params["mode"],
            pooling=job_params["pooling"],
            progress_cb=cb,
        )
        pbar.empty()
        infer_status = "done"
        infer_progress = 1.0

# -----------------------------
# Tab ①: Data & Pipeline (可先看，不卡)
# -----------------------------
with tabs[0]:
    st.subheader("① 数据与管线概览（NLP → 因子 → 对齐）")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("新闻条数（headline/text）", f"{len(news)}")
    c2.metric("市场交易日样本", f"{len(market)}")
    c3.metric("长文本模式", longtext_mode.split("（")[0])
    c4.metric("异步推理", "ON" if enable_async else "OFF")

    # 推理状态区（关键：UI 不阻塞）
    st.markdown("### ⚡ 推理状态（后台运行）" if enable_async else "### 🐢 推理状态（同步运行）")
    if enable_async:
        st.progress(float(infer_progress))
        if infer_status in ["queued", "running"]:
            st.info(f"FinBERT 推理中：{infer_status} · {infer_msg}")
            colx1, colx2 = st.columns([1, 3])
            with colx1:
                if st.button("🔄 刷新状态/继续管线"):
                    _rerun()
            with colx2:
                st.caption("提示：异步推理不会卡住界面。你可以先检查数据预览；推理完成后点击刷新进入后续分析。")
        elif infer_status == "error":
            st.error(f"推理失败：{infer_error}")
        elif infer_status == "done":
            st.success("✅ 推理完成！你可以切换到其他 Tab 查看情绪/Granger/回测结果。")
    else:
        st.success("✅ 推理完成（同步）。")

    left, right = st.columns(2)
    with left:
        st.markdown("**新闻数据（清洗后）**")
        st.dataframe(news[["NewsDate", "Headline"]].head(10), use_container_width=True, height=260)
    with right:
        st.markdown("**市场数据（收益序列）**")
        st.dataframe(market.head(10), use_container_width=True, height=260)

    st.caption("说明：该 Tab 在推理未完成时也可正常查看（解耦 ingestion 与 inference）。")

# 如果还没推理完：后续 tabs 给占位提示（不报错）
if scores is None:
    with tabs[1]:
        st.info("⏳ 情绪推理尚未完成。请回到 ① Tab 点击“刷新状态/继续管线”。")
    with tabs[2]:
        st.info("⏳ 情绪推理尚未完成。请回到 ① Tab 点击“刷新状态/继续管线”。")
    with tabs[3]:
        st.info("⏳ 情绪推理尚未完成。请回到 ① Tab 点击“刷新状态/继续管线”。")
    with tabs[4]:
        st.info("⏳ 情绪推理尚未完成。请回到 ① Tab 点击“刷新状态/继续管线”。")
    st.stop()

# 推理完成：写入 Sentiment
news = news.reset_index(drop=True)
news["Sentiment"] = np.array(scores, dtype=np.float32)

# Daily factor
daily_sent = build_daily_sentiment(news, method=agg_method)

# Align to trading days
aligned_sent = align_sentiment_to_market(daily_sent, market, mode=align_mode)

# Merge aligned sentiment with market returns
merged = pd.merge(market, aligned_sent, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

if len(merged) < 30:
    st.warning(f"合并后的有效样本较少（{len(merged)} 行）。可能导致 Granger 检验不稳定。建议扩大新闻日期覆盖或换更长时间窗口。")


# -----------------------------
# Tab ②: Sentiment Factor Visualization
# -----------------------------
with tabs[1]:
    st.subheader("② 情绪因子（FinBERT）与价格走势")

    # Factor distribution
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=news["Sentiment"], nbinsx=50, name="Headline/Text Sentiment"))
    beautify_fig(fig_hist, title="Text-level Sentiment Distribution", ytitle="Count")
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

    st.caption(f"长文本模式：{longtext_mode} · pooling={pooling} · max_length={window_max_len} · stride={window_stride}")


# -----------------------------
# Tab ③: Lead–Lag (Granger) across multiple lags
# -----------------------------
with tabs[2]:
    st.subheader("③ Lead–Lag 结构检验：Granger Causality（多滞后阶）")
    st.caption("同时报告 Sentiment→Return 与 Return→Sentiment 的 p-values（1..MaxLag）。")

    try:
        gr_df = run_granger_multi_lag(merged, max_lag=max_lag)
        st.dataframe(gr_df.style.format({c: "{:.4f}" for c in gr_df.columns if "P-value" in c}), use_container_width=True)

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=gr_df["Lag"], y=gr_df["P-value (Sentiment → Return)"], mode="lines+markers", name="Sentiment → Return"))
        fig_p.add_trace(go.Scatter(x=gr_df["Lag"], y=gr_df["P-value (Return → Sentiment)"], mode="lines+markers", name="Return → Sentiment"))
        fig_p.add_hline(y=0.05, line_dash="dash", annotation_text="0.05", annotation_position="top left")
        beautify_fig(fig_p, title="Granger p-values across lag orders", ytitle="p-value")
        st.plotly_chart(fig_p, use_container_width=True)

        best_lag = int(gr_df.loc[gr_df["P-value (Sentiment → Return)"].idxmin(), "Lag"])
        best_p = float(gr_df["P-value (Sentiment → Return)"].min())
        sig_sr = (gr_df["P-value (Sentiment → Return)"] < 0.05).any()

        if sig_sr:
            st.success(f"✅ 检测到 **Sentiment → Return** 在某些滞后阶上显著（p<0.05）。最小 p-value：lag={best_lag}（p={best_p:.4f}）。")
        else:
            st.info(f"未检测到显著的 **Sentiment → Return**（p<0.05）。最小 p-value：lag={best_lag}（p={best_p:.4f}）。")

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

    c1, c2, c3 = st.columns(3)
    c1.metric("策略累计收益", f"{metrics['Strategy Cumulative Return']*100:.2f}%", delta=f"vs 基准 {metrics['Benchmark Cumulative Return']*100:.2f}%")
    c2.metric("策略 Sharpe", f"{metrics['Strategy Sharpe']:.2f}", delta=f"vs 基准 {metrics['Benchmark Sharpe']:.2f}")
    c3.metric("策略最大回撤", f"{metrics['Strategy Max Drawdown']*100:.2f}%", delta=f"vs 基准 {metrics['Benchmark Max Drawdown']*100:.2f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric("Alpha（策略-基准）", f"{metrics['Alpha (Strategy - Benchmark)']*100:.2f}%")
    c5.metric("交易次数（换仓）", f"{metrics['Trades']}")
    c6.metric("暴露度（持仓比例）", f"{metrics['Exposure']*100:.1f}%")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["Equity_Strategy"], name="Strategy Equity", mode="lines", line=dict(width=3)))
    fig_eq.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["Equity_Benchmark"], name="Buy&Hold Equity", mode="lines", line=dict(dash="dash")))
    beautify_fig(fig_eq, title="Equity Curve: Strategy vs Benchmark", ytitle="Equity")
    st.plotly_chart(fig_eq, use_container_width=True)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["DD_Strategy"], name="Strategy Drawdown", mode="lines"))
    fig_dd.add_trace(go.Scatter(x=bt_df["Date"], y=bt_df["DD_Benchmark"], name="Benchmark Drawdown", mode="lines", line=dict(dash="dash")))
    beautify_fig(fig_dd, title="Drawdown: Strategy vs Benchmark", ytitle="Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

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

    merged_csv = merged.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载对齐后的数据（news-factor-market aligned）", merged_csv, file_name=f"{ticker}_aligned_data.csv", mime="text/csv")

    try:
        gr_df = run_granger_multi_lag(merged, max_lag=max_lag)
        gr_csv = gr_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ 下载 Granger 结果（multi-lag p-values）", gr_csv, file_name=f"{ticker}_granger_pvalues.csv", mime="text/csv")
    except Exception:
        st.info("Granger 结果不可用（样本不足或检验失败）。")

    bt_df, metrics = run_timing_backtest(
        merged,
        threshold=sentiment_threshold,
        cost_bps=cost_bps,
        rf_annual=rf_annual,
    )
    bt_csv = bt_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载回测明细（positions/returns/equity）", bt_csv, file_name=f"{ticker}_backtest_detail.csv", mime="text/csv")

    metrics_df = pd.DataFrame([metrics])
    metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载指标汇总（metrics）", metrics_csv, file_name=f"{ticker}_metrics.csv", mime="text/csv")
