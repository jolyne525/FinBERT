\# FinBERT-Alpha: Sentiment-Driven Causal Inference System



[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)](https://streamlit.io/)

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)


\## Project Overview



This project implements a \*\*Natural Language Processing (NLP)\*\* pipeline to analyze the causal relationship between financial news sentiment and stock price movements.



Using \*\*FinBERT\*\* (a BERT model pre-trained on financial texts), the system extracts sentiment signals from unstructured news headlines and performs \*\*Granger Causality Tests\*\* to validate whether sentiment creates a statistically significant lead on stock returns. Finally, it executes a backtest to evaluate the "Sentiment Alpha."



\##  Key Features



\* \*\*Transformer-based NLP\*\*: Utilizes `ProsusAI/finbert` for state-of-the-art financial sentiment classification.

\* \*\*Causal Inference\*\*: Implements \*\*Granger Causality Tests\*\* (via `statsmodels`) to statistically prove the predictive power of news.

\* \*\*Automated Backtesting\*\*: Simulates a long-short strategy based on sentiment signals vs. Buy \& Hold benchmark.

\* \*\*Region Optimized\*\*: Configured with HF-Mirror to ensure stable model downloading in restricted network environments.

\* \*\*Interactive Visualization\*\*: Dynamic plotting of Equity Curves and Sentiment Distributions using Plotly.



\##  Tech Stack



\* \*\*Core Logic\*\*: Python, Pandas, NumPy

\* \*\*Deep Learning\*\*: PyTorch, HuggingFace Transformers

\* \*\*Statistics\*\*: Statsmodels (Granger Causality)

\* \*\*Data Source\*\*: Yahoo Finance API (`yfinance`)

\* \*\*UI/UX\*\*: Streamlit



\##  Data Requirements



To run the analysis, upload a CSV or Excel file containing historical news. The system expects the following columns (headers are auto-renamed in the app, but ensure roughly these contents):



| title (Headline) | date | stock (Ticker) |

| :--- | :--- | :--- |

| Agilent announces new Q3 results... | 2023-01-15 | A |

| Market hits record high... | 2023-01-16 | A |



\*(Note: The system includes a robust fallback mechanism to generate simulated bullish market data if Yahoo Finance connectivity fails.)\*



\##  Quick Start



\### 1. Clone Repository

```bash

git clone https://github.com/jolyne525/FinBERT.git
cd FinBERT
