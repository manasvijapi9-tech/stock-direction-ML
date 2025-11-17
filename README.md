# Stock Direction Prediction for TRENT (TRENT.NS)

This project develops a complete machine learning pipeline to predict the next-day price direction (up or down) of **TRENT (TRENT.NS)** using technical indicators and supervised classification models.

The workflow demonstrates practical data science and financial modeling skills: data acquisition, preprocessing, feature engineering, model development, evaluation, and interpretation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DJpfRFVMSgv8Bu2YUDOgoGNaWbM3Fnb8#scrollTo=23L6dG3M5ulu)

---

## 1. Project Overview

The goal of this project is to forecast whether TRENT’s closing price will move upward the following day.  
The pipeline collects historical market data, computes technical indicators, prepares a machine learning dataset, trains classification models, and evaluates predictive performance.

The project demonstrates:

- Handling financial time-series data
- Engineering technical indicators
- Constructing supervised learning targets from price data
- Comparing Logistic Regression and Random Forest models
- Evaluating model performance and reliability
- Interpreting feature importance for TRENT’s price behavior
- Building modular, production-style Python code

---

## 2. Workflow Summary

1. **Data Acquisition**  
   Historical price data for TRENT (TRENT.NS) is fetched directly from Yahoo Finance.

2. **Data Cleaning**  
   Price data is standardized, numerical columns are fixed, and invalid rows are removed.

3. **Feature Engineering**  
   Technical indicators computed:
   - RSI  
   - SMA10  
   - SMA20  
   - MACD  
   - Historical volatility  

4. **Dataset Construction**  
   - Features (**X**) are built from indicators  
   - Target (**y**) marks whether the next day’s close is higher than the current day's  
   - Final dataset is saved for modeling

5. **Model Training**  
   - Logistic Regression  
   - Random Forest Classifier  
   Both models are trained and evaluated on TRENT’s historical data.

6. **Evaluation & Interpretation**  
   - Confusion matrix  
   - Prediction probability distribution  
   - Random Forest feature importance plot  

---

## 3. Key Results

### Confusion Matrix
<img src="plots/confusion_matrix.png" width="420">

### Prediction Probability Distribution
<img src="plots/probability_distribution.png" width="420">

### Feature Importance (Random Forest)
<img src="plots/feature_importance.png" width="420">

---

## 4. How to Run the Project in Google Colab

Clone the repository and install dependencies:

```bash
!git clone https://github.com/manasvijapi9-tech/stock-direction-ML.git
%cd stock-direction-ML
!pip install -r requirements.txt
```

### Step 1 — Fetch TRENT Data
```bash
!python src/fetch_data.py
```

### Step 2 — Clean the Raw Data
```python
import pandas as pd

df = pd.read_csv("data/TRENT_NS_raw.csv")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df.dropna(inplace=True)
df.to_csv("data/TRENT_NS_raw_clean.csv", index=False)
```

### Step 3 — Compute Technical Indicators
```python
from src.indicators import add_all_indicators
import pandas as pd

df = pd.read_csv("data/TRENT_NS_raw_clean.csv")
df2 = add_all_indicators(df)
df2.to_csv("data/data_with_indicators.csv", index=False)
```

### Step 4 — Prepare ML Dataset
```bash
!python src/prepare_data.py
```

### Step 5 — Train the Models
```bash
!python src/train_model.py
```

### Step 6 — Evaluate Model Performance
```bash
!python src/evaluate.py
```

All resulting plots will be saved in the **plots/** directory.

---

## 5. Key Learnings

- Building full ML pipelines around real financial data  
- Best practices for time-series preprocessing  
- Constructing supervised learning labels for market forecasting  
- Comparing linear vs ensemble classifiers  
- Interpreting feature importance for TRENT’s technical indicators  
- Designing modular, maintainable project architecture  

---

## 6. Repository Structure

```
stock-direction-ML/
│
├── data/                   # Raw, cleaned, and engineered datasets
├── models/                 # Trained Logistic Regression and Random Forest models
├── plots/                  # Evaluation and interpretation visuals
├── src/                    # Modular pipeline scripts
│   ├── fetch_data.py
│   ├── indicators.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── evaluate.py
│
├── stock_direction_pipeline.ipynb   # Complete Colab workflow
├── requirements.txt
└── README.md
```

---

## 7. Author

**Manasvi Japi**  
Student interested in Data Science, Quantitative Finance, and Applied Machine Learning.


