# BSE Funds - Stock Analysis and Prediction System

A multi-model machine learning system for Indian stock market analysis, price prediction, sentiment analysis, and AI-assisted research. Built with Streamlit and trained on data fetched via Yahoo Finance.

---

## Overview

The application takes a BSE/NSE ticker symbol as input and runs a full pipeline covering data collection, technical analysis, time series forecasting, classification ensemble, news sentiment analysis, annual report summarization, and a RAG-based chatbot for Q&A over the collected data.

---

## Data Sources

- **Yahoo Finance (yfinance)** - Historical OHLCV price data
- **Screener.in** - Fundamental financial data and ratios
- **NewsData.io** - News headlines for sentiment analysis
- **GNews.io** - Additional news headlines
- **Annual Report PDF** - Uploaded manually by the user for PDF analysis

---

## Machine Learning Models

### Time Series Forecasting

**Prophet (Meta)**
- Additive decomposition model for trend and seasonality
- Configured with daily seasonality, weekly seasonality, and custom Indian market holidays
- Outputs next-day price prediction, trend direction, and confidence interval

**ARIMA / SARIMA (via pmdarima)**
- Auto-ARIMA used to automatically select optimal (p, d, q) order
- SARIMA variant fitted alongside to capture seasonal patterns
- Evaluated on RMSE against a held-out test set

**LSTM (Long Short-Term Memory)**
- Built with TensorFlow / Keras
- Sequence length of 60 days
- Input features: Close, Volume, RSI, MACD, Bollinger Band Percentage B, ATR Percentage
- Scaled with RobustScaler
- Predicts next-day closing price

**GRU (Gated Recurrent Unit)**
- Same architecture as LSTM but uses GRU cells
- Trained and evaluated in parallel with LSTM for comparison
- Outputs visualized against actual prices

### Classification Ensemble (Direction Prediction)

**XGBoost**
- Binary classifier predicting next-day price direction (up/down)
- Hyperparameter optimization via Optuna (100 trials)
- Features: RSI signals, MACD crossover signals, Bollinger Band width, ATR, lag returns, volume ratios, temporal features

**LightGBM**
- Gradient boosted trees with early stopping
- Trained on the same feature set as XGBoost
- Early stopping monitored on a validation split

**Random Forest**
- Scikit-learn RandomForestClassifier with balanced class weights
- Uses square root feature selection per split

**Weighted Ensemble**
- Combines XGBoost, LightGBM, and Random Forest probability outputs
- Weights are proportional to each model's individual test accuracy
- Final probability aggregated into a BUY / SELL / HOLD signal

**XGBoost Meta-Learner**
- A second XGBoost model trained on the probability outputs of the base classifiers as stacked features
- Optimized separately with Optuna

---

## Technical Indicators (Feature Engineering)

All indicators are computed from raw OHLCV data and used as model features.

- RSI (14-period) with overbought / oversold binary flags
- MACD (12, 26, 9) with signal line and histogram
- MACD crossover signal and crossover change
- Bollinger Bands (20-period, 2 standard deviations) - width and percentage B
- ATR (Average True Range) as a percentage of price
- Rolling returns over 1, 3, 5, and 10 day windows
- Lag features for returns and volume
- Temporal features: day of week, month, quarter

---

## Sentiment Analysis

**TextBlob**
- Lexicon-based polarity scoring on news headlines
- Fast, rule-based, used as a baseline and fallback

**FinBERT (ProsusAI/finbert)**
- BERT model fine-tuned on financial text
- Loaded via Hugging Face Transformers pipeline
- Classifies each headline as positive, negative, or neutral
- Score averaged across all headlines to produce a single sentiment value
- Falls back to TextBlob if FinBERT is unavailable

---

## NLP and RAG Pipeline

**Sentence Transformers (all-MiniLM-L6-v2)**
- Encodes document chunks into dense vector embeddings
- Used to build the knowledge base for retrieval

**FAISS (faiss-cpu)**
- Vector similarity index (IndexFlatIP - inner product)
- Retrieves the top-k most relevant document chunks for a given query

**Groq API (llama-3.1-8b-instant)**
- LLM used for two tasks:
  1. Annual report summarization - chunks the PDF and summarizes each section, then combines into a final summary
  2. RAG chatbot - answers user questions using retrieved document context as grounding

**pdfplumber**
- Extracts raw text from uploaded annual report PDFs page by page
- Text is chunked and passed to Groq for summarization

---

## Application Structure (Streamlit Tabs)

| Tab | Description |
|-----|-------------|
| Configuration | Ticker input, API key entry, run trigger |
| Price Analysis | Historical price chart, Bollinger Bands, RSI, MACD |
| Forecasting | Prophet, ARIMA, LSTM, GRU predictions and accuracy |
| Signals | Ensemble BUY / SELL / HOLD signal with model agreement score |
| News Sentiment | Headlines, TextBlob and FinBERT scores |
| Annual Report | PDF upload, Groq-powered section summaries |
| RAG Chatbot | Question answering over the full knowledge base |

---

## Libraries and Frameworks

| Library | Purpose |
|---------|---------|
| yfinance | Stock data download |
| pandas, numpy | Data manipulation |
| scikit-learn | Preprocessing, Random Forest, metrics |
| xgboost | Gradient boosted classifier |
| lightgbm | Gradient boosted classifier |
| tensorflow / keras | LSTM and GRU models |
| prophet | Time series forecasting |
| pmdarima | Auto ARIMA fitting |
| statsmodels | Statistical modelling |
| faiss-cpu | Vector similarity search |
| sentence-transformers | Text embeddings |
| transformers | FinBERT sentiment model |
| textblob | Rule-based sentiment |
| groq | LLM API client |
| pdfplumber | PDF text extraction |
| optuna | Hyperparameter optimization |
| streamlit | Web application interface |
| matplotlib, plotly | Visualizations |
| requests, beautifulsoup4 | Web scraping for Screener.in |

---

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add your API keys in the Configuration tab or directly in the notebook:
   - `GROQ_API_KEY` - from console.groq.com
   - `NEWSDATA_API_KEY` - from newsdata.io
   - `GNEWS_API_KEY` - from gnews.io

4. Run the Streamlit app:
   ```
   streamlit run final_codebase.py
   ```
   Or open the notebook in Google Colab and run all cells.

---

## Notes

- The system is designed for Indian equity markets. Tickers should follow the Yahoo Finance format for BSE/NSE stocks (e.g., `RELIANCE.NS`, `TCS.BO`).
- Model accuracy varies with market conditions. High volatility periods naturally reduce classification accuracy.
- FinBERT requires a GPU or significant CPU time on first load due to model download.
- All API keys must be kept private and should not be committed to version control.
