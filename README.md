# 📈 FinBERT-Based Sentiment and Stock Lead–Lag Analysis

An end-to-end pipeline that converts financial headlines (or long-form news text) into a **continuous sentiment factor** using **FinBERT**, aligns it with daily stock returns, tests **lead–lag predictive structure** via **Granger causality** across multiple lag orders, and validates a **sentiment timing policy** through backtesting with risk-adjusted metrics.

This repo additionally addresses real-world engineering bottlenecks:
- **Long-context documents beyond BERT’s 512-token limit** via **sliding windows + pooling**
- **High-latency inference during news spikes** via an **asynchronous producer–consumer pipeline** (UI stays responsive)
- **Auto-refresh** while inference is running (no manual refresh needed)
 
**Tech Stack:** Python (PyTorch, Transformers, Statsmodels), Streamlit, Plotly, yfinance

**Live Demo:** https://finbert-7zfu3euxvd5ogfgq69fsjm.streamlit.app/  
**Repo:** https://github.com/jolyne525/FinBERT

---

## Quick Start

### 1) Clone Repository
```bash
git clone https://github.com/jolyne525/FinBERT.git
cd FinBERT
````

### 2) Install Dependencies

> Recommended: use a virtual environment to avoid dependency conflicts.

**Option A: venv**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Option B: conda**

```bash
conda create -n finbert-alpha python=3.10 -y
conda activate finbert-alpha
pip install -r requirements.txt
```

### 3) Run the Streamlit App

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually: [http://localhost:8501](http://localhost:8501)).

---

## 🖼️ Dashboard Preview

<p align="center">
  <img width="1536" height="1024" alt="Dashboard Preview" src="https://github.com/user-attachments/assets/74111373-c644-4181-8e40-67aee9b9122d" />
</p>

> Tip: You can also store screenshots under `assets/` and reference them as `assets/your_image.png`.

---

## ✨ What This Project Demonstrates 

- **Coupled time-series modeling (sentiment + market):**  
  Converts news → continuous sentiment factor (FinBERT) and prices → daily return signals, enabling lead–lag analysis and signal construction.

- **Reproducible NLP→factor→analysis pipeline:**  
  Tokenization → FinBERT inference (batched) → daily aggregation → time alignment to trading days.

- **Long-context handling (512-token limit):**  
  Uses **sliding windows (stride overlap)** and **pooling (mean/max)** to expand model coverage for long documents while keeping output as a single continuous factor.

- **Lead–lag testing with Granger causality:**  
  Runs Granger causality across **lag = 1..N**, reporting **p-values** for both directions: Sentiment→Return and Return→Sentiment.

- **Sentiment timing policy + backtest validation:**  
  Strategy: **hold when prior-day sentiment > threshold, otherwise stay out**, validated with **cumulative return, Sharpe ratio, max drawdown, volatility, trades/exposure** vs Buy & Hold.

- **Asynchronous pipeline for real-time responsiveness:**  
  Decouples ingestion/UI from inference using a background worker + queue, preventing UI blocking during heavy inference loads.

---

## 🔧 How It Works (Pipeline)

1. **Input News (CSV/Excel)**  
2. **FinBERT Sentiment Inference**
   - Score = P(positive) − P(negative)
   - **Long-text support:** sliding window + pooling (mean/max)
3. **Daily Aggregation** (mean/median)  
4. **Time Alignment**
   - Option A: map non-trading days → **next trading day**
   - Option B: **same-day inner join** (keep only overlap)
5. **Lead–Lag Test (Granger)**
   - Multi-lag (1..N) p-value table + p-value curves
6. **Backtest Timing Policy**
   - Position[t] = 1 if sentiment[t−1] > threshold else 0
   - Optional transaction cost (bps)
7. **Export**
   - Download aligned dataset + Granger results + backtest details (CSV)

---

## 📁 Expected News Data Format

Your CSV/Excel should include at least:
- `date` (or `Date`, `datetime`, `published`, `timestamp`, etc.)
- `headline` (or `title`, `text`, `content`)

Optional:
- `ticker` / `stock` / `symbol` (if provided, the app can filter by ticker)

Example CSV:

```csv
date,headline,ticker
2024-01-03,"Company beats earnings expectations",AAPL
2024-01-04,"Guidance revised downward amid macro uncertainty",AAPL
```

---

## ⚙️ Key App Controls (UI)

* **Long-text mode**

  * `auto`: only split when text exceeds 512 tokens
  * `always`: always use sliding windows
  * `off`: truncate (fastest, but loses context)
* **Window max_length / stride**

  * `max_length` typically 512 for BERT-family
  * `stride` controls overlap (higher overlap → better continuity but slower)
* **Pooling**

  * `mean`: stable factor (recommended)
  * `max`: captures extreme sentiment segments
* **Async inference**

  * Runs inference in a background worker (UI stays responsive)
  * Useful when you ingest many headlines / long documents

---

## ⚡ Async Inference + Auto Refresh

This project uses an async worker queue to prevent inference from blocking Streamlit’s UI.

Auto-refresh while inference is running is powered by:

* `streamlit-autorefresh`

Make sure it is included in `requirements.txt`:

```txt
streamlit-autorefresh
```

If you deploy to Streamlit Cloud, dependencies must be listed in `requirements.txt`, otherwise auto-refresh will fall back to manual refresh.

---

## 📊 Metrics Reported

* Cumulative return (strategy vs benchmark)
* Sharpe ratio (risk-adjusted)
* Max drawdown
* Annualized volatility
* Trades / turnover proxy
* Exposure (average position)

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push these files to GitHub:

   * `impr_fin.py` (main Streamlit app)
   * `requirements.txt`
   * `README.md`
   * (optional) `assets/` screenshots

2. Go to Streamlit Community Cloud → **New app**

3. Select your repo + branch

4. Set **Main file path** to:

   * `impr_fin.py`

5. Click **Deploy**


---

## 📎 Notes / Limitations

* Granger tests require sufficient aligned samples (larger max lag needs more observations).
* Results depend heavily on news coverage and time range.
* This project is for research/education only — **not financial advice**.

---

## 📜 License

MIT

---
