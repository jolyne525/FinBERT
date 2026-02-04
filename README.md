# 📈 FinBERT-Based Sentiment and Stock Lead–Lag Analysis

An end-to-end pipeline that converts financial headlines into a **continuous sentiment factor** using **FinBERT**, aligns it with daily stock returns, tests **lead–lag predictive structure** via **Granger causality** across multiple lag orders, and validates a **sentiment timing policy** through backtesting with risk-adjusted metrics.

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

## ✨ What This Project Demonstrates (Resume-Aligned)

* **Coupled time-series modeling (sentiment + market):**
  Converts headlines → continuous sentiment factor (FinBERT) and prices → return signals, enabling lead–lag analysis and signal construction.

* **Reproducible NLP-to-factor-to-analysis pipeline:**
  Tokenization → FinBERT inference (batched) → daily aggregation → time alignment to trading days.

* **Lead–lag testing with Granger causality:**
  Runs Granger causality across **lag = 1..N**, reporting **p-values** (and directionality: Sentiment→Return and Return→Sentiment).

* **Sentiment timing policy + backtest validation:**
  Strategy: **hold when prior-day sentiment > threshold, otherwise stay out**, validated with **cumulative return, Sharpe ratio, max drawdown, volatility, turnover/exposure** vs Buy & Hold.

---

## 🔧 How It Works (Pipeline)

1. **Input News (CSV/Excel)**
2. **FinBERT Sentiment Inference**

   * Score = P(positive) − P(negative)
3. **Daily Aggregation** (mean/median)
4. **Time Alignment**

   * Option A: map non-trading days → **next trading day**
   * Option B: use **same-day inner join**
5. **Lead–Lag Test (Granger)**

   * Multi-lag (1..N) p-value table and p-value curve
6. **Backtest Timing Policy**

   * Position[t] = 1 if sentiment[t−1] > threshold else 0
   * Optional transaction cost (bps)
7. **Export**

   * Download aligned dataset + Granger results + backtest details (CSV)

---

## 📁 Expected News Data Format

Your CSV/Excel should include at least:

* `date` (or `Date`, `datetime`, `published`, etc.)
* `headline` (or `title`)

Optional:

* `ticker` / `stock` / `symbol` (if provided, the app can filter by ticker)

Example CSV:

```csv
date,headline,ticker
2024-01-03,"Company beats earnings expectations",AAPL
2024-01-04,"Guidance revised downward amid macro uncertainty",AAPL
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push these files to GitHub:

   * `impr_fin.py`
   * `requirements.txt`
   * `README.md`
   * (optional) `assets/` screenshots

2. Go to Streamlit Community Cloud → **New app**

3. Select your repo + branch

4. Set **Main file path** to: `impr_fin.py`

5. Click **Deploy**

---

## 📊 Metrics Reported

* Cumulative return (strategy vs benchmark)
* Sharpe ratio (risk-adjusted)
* Max drawdown
* Annualized volatility
* Trades / turnover proxy
* Exposure (average position)

---

## 📎 Notes / Limitations

* Granger tests require sufficient aligned samples (larger max lag needs more observations).
* Results depend heavily on news coverage and time range.
* For research/education only — not financial advice.

---

## 📜 License

MIT
