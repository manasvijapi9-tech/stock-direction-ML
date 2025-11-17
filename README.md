# Stock Direction Prediction using Machine Learning

An end-to-end machine learning pipeline that predicts next-day stock direction (up or down) using technical indicators, classification models, and a fully reproducible workflow.  
The project includes data collection, cleaning, feature engineering, model training, evaluation, and visualization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manasvijapi9-tech/stock-direction-ML/blob/main/stock_direction_pipeline.ipynb)

---

## 1. Project Overview

This project builds a supervised ML model to predict whether a stock’s closing price will increase the next day.  
It uses financial technical indicators as features and evaluates both Logistic Regression and Random Forest classifiers.

The goal is to demonstrate practical understanding of:
- Time-series preprocessing  
- Technical indicator engineering  
- Binary classification for financial prediction  
- Model evaluation using real market data  
- Building modular, production-style ML code  

---

## 2. Project Pipeline

The project follows a clear, structured sequence:

1. **Data Fetching**  
   - Downloads historical stock data directly from Yahoo Finance.

2. **Data Cleaning**  
   - Fixes numeric inconsistencies, removes invalid rows, and formats price data.

3. **Feature Engineering**  
   - Computes technical indicators such as RSI, SMA10, SMA20, MACD, and volatility.

4. **Dataset Preparation**  
   - Creates feature matrix **X** and binary target variable **y** indicating next-day movement.

5. **Model Training**  
   - Logistic Regression  
   - Random Forest Classifier  
   - Feature importance visualization

6. **Model Evaluation**  
   - Confusion matrix  
   - Classification metrics  
   - Probability distribution analysis

---

## 3. Visual Results

### Confusion Matrix
<img src="plots/confusion_matrix.png" width="420">

### Prediction Probability Distribution
<img src="plots/probability_distribution.png" width="420">

### Feature Importance (Random Forest)
<img src="plots/feature_importance.png" width="420">

---

## 4. How to Run (Google Colab)

Clone the repository:

```bash
!git clone https://github.com/manasvijapi9-tech/stock-direction-ML.git
%cd stock-direction-ML
!pip install -r requirements.txt
```

### Step 1 — Fetch Raw Data
```bash
!python src/fetch_data.py
```

### Step 2 — Clean the Data
```python
import pandas as pd
df = pd.read_csv("data/TCS_NS_raw.csv")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df.dropna(inplace=True)
df.to_csv("data/TCS_NS_raw_clean.csv", index=False)
```

### Step 3 — Add Technical Indicators
```python
from src.indicators import add_all_indicators
import pandas as pd

df = pd.read_csv("data/TCS_NS_raw_clean.csv")
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

### Step 6 — Evaluate the Models
```bash
!python src/evaluate.py
```

All generated plots are saved in the **plots/** directory.

---

## 5. Key Learnings

- Constructing reproducible machine learning pipelines  
- Handling and preprocessing financial time-series data  
- Designing target variables for predictive modeling  
- Evaluating classification models in imbalanced financial settings  
- Understanding feature importance in Random Forest models  
- Building modular, maintainable project structure  

---

## 6. Repository Structure

```
stock-direction-ML/
│
├── data/                   # Raw, cleaned, engineered, and model-ready data
├── models/                 # Saved ML models
├── plots/                  # Confusion matrix, feature importance, probability plot
├── src/                    # All project scripts
│   ├── fetch_data.py
│   ├── indicators.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── evaluate.py
│
├── stock_direction_pipeline.ipynb   # Full Google Colab notebook
├── requirements.txt
└── README.md
```

---

## 7. Author

**Manasvi Japi**  
Aspiring Data Scientist & Quantitative Finance Student  
Focused on applied machine learning and financial modeling.

