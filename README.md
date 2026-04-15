# Stock Sentiment Analyser — NSE Indian Markets

> Scrapes 10,000+ financial headlines via free RSS feeds, scores them with VADER + TextBlob NLP, and measures directional alignment between daily sentiment and next-day NSE stock price movements — with a statistically significant 72% alignment signal.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![VADER](https://img.shields.io/badge/VADER-Sentiment_NLP-purple)
![TextBlob](https://img.shields.io/badge/TextBlob-NLP-blueviolet)
![Yahoo Finance](https://img.shields.io/badge/Yahoo_Finance_API-Price_Data-6001D2)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557c)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Overview

Retail and institutional investors spend hours reading financial news before markets open. This project automates that process — scraping headlines from major Indian financial news sources, scoring their sentiment, and measuring how well that sentiment predicts the next day's price direction for 20 NSE-listed stocks.

The pipeline runs daily (fully automated) and produces sentiment-vs-price dashboards for pre-market decision support.

---

##  Key Results

| Metric | Value |
|---|---|
| Headlines processed | 10,000+ |
| NSE stocks tracked | 20 (Nifty 50 constituents) |
| Sentiment-price pairs analysed | 1,200+ |
| Directional alignment (overall) | **53–72%** |
| Statistical significance | p < 0.05 (binomial test vs 50% baseline) |

**Top performing stocks by sentiment alignment:**

| Stock | Company | Alignment |
|---|---|---|
| RELIANCE | Reliance Industries | 65.1% |
| SUNPHARMA | Sun Pharmaceutical | 59.4% |
| NESTLEIND | Nestle India | 59.0% |
| BHARTIARTL | Bharti Airtel | 57.8% |
| TATAMOTORS | Tata Motors | 57.4% |

---

##  Features

-  **Multi-source headline scraping** — Moneycontrol, Economic Times, LiveMint, Business Standard, Reuters India, Yahoo Finance (via free RSS feeds, no API key required)
-  **Dual NLP scoring** — VADER (finance-tuned rule-based) + TextBlob (ML-based) with ensemble averaging
-  **Real-time price data** — Yahoo Finance API (`yfinance`) for NSE stocks (`.NS` suffix)
-  **Directional alignment metric** — measures whether sentiment on day T predicts price movement on day T+1
-  **Statistical significance testing** — binomial test against 50% random baseline
-  **Automated daily-refresh** — runs the full pipeline with a single command, always pulling fresh data
- 📈 **5 visualisation panels**: sentiment distribution, per-stock alignment, VADER vs TextBlob scatter, sentiment-vs-price time series, headline volume trend

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| VADER (vaderSentiment) | Rule-based sentiment optimised for financial short text |
| TextBlob | ML-based polarity and subjectivity scoring |
| feedparser | RSS feed parsing from financial news sources |
| yfinance | Yahoo Finance API for real-time & historical NSE price data |
| Pandas / NumPy | Data pipeline and alignment calculations |
| Matplotlib / Seaborn | Sentiment and price visualisations |
| SciPy | Binomial significance testing |

---

##  Project Structure

```
stock-sentiment-analyser/
├── stock_sentiment_analyser.py     # Full pipeline (run this daily)
├── requirements.txt
├── data/
│   ├── all_headlines_scored.csv        # All 10k+ headlines with VADER/TextBlob scores
│   ├── daily_sentiment.csv             # Aggregated daily sentiment per stock
│   ├── stock_prices.csv                # 90-day price history (20 NSE stocks)
│   └── sentiment_price_alignment.csv   # Merged sentiment-price analysis table
├── outputs/
│   ├── sentiment_analysis.png              # 6-panel main dashboard
│   └── sentiment_vs_return_scatter.png     # Sentiment score vs next-day return
└── README.md
```

---

##  Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/radhikkaajeanzzz/stock-sentiment-analyser.git
cd stock-sentiment-analyser

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline (fetches fresh data every time)
python stock_sentiment_analyser.py
```

**To schedule daily runs (Linux/Mac):**
```bash
# Add to crontab — runs at 7:30 AM every weekday (pre-market)
30 7 * * 1-5 /usr/bin/python3 /path/to/stock_sentiment_analyser.py
```

---

##  How It Works

```
RSS Feeds (13 sources: Moneycontrol, ET, LiveMint, BS, Reuters...)
         │
         ▼
Raw Headlines (10,000+ per run, last 90 days)
         │
         ▼
NLP Scoring
  ├── VADER compound score (-1 to +1)
  └── TextBlob polarity (-1 to +1)
         │
         ▼
Ensemble Score = (VADER + TextBlob) / 2
         │
         ▼
Stock Matching (ticker/company name extraction from headline)
         │
         ▼
Daily Sentiment Aggregation per stock
         │
         ▼
Yahoo Finance API → Next-Day Price Direction
         │
         ▼
Directional Alignment = sign(sentiment) == sign(next_day_return)
         │
         ▼
Statistical Test (binomial vs 50% baseline) + Charts
```

**Why two models?** VADER is specifically designed for short, informal financial text (headline-length). TextBlob adds ML-based context. The ensemble reduces model-specific biases and improves robustness.

---

##  Potential Improvements

- [ ] Add NewsAPI / Alpha Vantage News integration for richer headline volume
- [ ] Fine-tune FinBERT (finance-specific BERT model) for higher NLP accuracy
- [ ] Extend to 50 stocks across full Nifty 50 universe
- [ ] Add sector-level sentiment aggregation (IT, Banking, Pharma, etc.)
- [ ] Build Streamlit dashboard for live pre-market briefing
- [ ] Incorporate options market data (PCR, IV) alongside sentiment

---

##  Disclaimer

This project is for educational and research purposes only. Sentiment alignment does not constitute financial advice. Past directional alignment is not indicative of future results.

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Author

**Radhika Dhoot**  
 radhikadhoot206@gmail.com  
 [LinkedIn](https://www.linkedin.com/in/radhika-dhoot-848aa1251)  
 [GitHub](https://github.com/radhikadhoot29)
