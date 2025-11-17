# Stock Direction Prediction using Machine Learning

An end-to-end machine learning pipeline that predicts next-day stock direction (up or down) using technical indicators and supervised learning models.  
The project demonstrates practical data science skills through structured preprocessing, feature engineering, model training, evaluation, and visualization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DJpfRFVMSgv8Bu2YUDOgoGNaWbM3Fnb8#scrollTo=23L6dG3M5ulu)


## 1. Project Overview

This project predicts whether a stock’s closing price will increase the following day.  
It incorporates technical analysis signals and applies machine learning techniques for binary classification.

The workflow demonstrates competency in:

- Financial data preprocessing  
- Technical indicator computation  
- Feature and target engineering  
- Model development (Logistic Regression, Random Forest)  
- Model evaluation and interpretation  
- Modular, production-style Python code

---

## 2. Project Pipeline

1. **Data Fetching**  
   Historical price data is pulled from Yahoo Finance.

2. **Data Cleaning**  
   Handles numeric inconsistencies and invalid rows.

3. **Feature Engineering**  
   Computes indicators such as RSI, SMA10, SMA20, MACD, and volatility.

4. **Dataset Preparation**  
   Creates supervised ML labels (up/down), feature matrix (X), and target vector (y).

5. **Model Training**  
   Trains Logistic Regression and Random Forest classifiers.

6. **Evaluation**  
   Generates a confusion matrix, probability distribution, and classification metrics.

7. **Interpretation**  
   Produces a feature importance plot to illustrate which technical signals matter most.

---

## 3. Visual Results

### Confusion Matrix
<img width="700" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/522529d3-4b81-4c90-8547-042bb136baff" />

### Prediction Probability Distribution
<img width="800" height="500" alt="probability_distribution" src="https://github.com/user-attachments/assets/8a542a29-861f-4510-b4fb-b6e13832d4f3" />


### Feature Importance (Random Forest)
<img width="800" height="500" alt="feature_importance" src="https://github.com/user-attachments/assets/fa0775ee-b7b6-41b8-84cb-392484f65901" />


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

Plots generated will be saved in the `plots/` directory.

---

## 5. Key Learnings

- Construction of an end-to-end ML pipeline  
- Handling time-series financial data correctly  
- Designing binary labels for predictive modeling  
- Training and comparing classification models  
- Evaluating performance using appropriate visual tools  
- Identifying informative technical indicators  
- Maintaining clean and modular project structure  

---

## 6. Repository Structure

```
stock-direction-ML/
│
├── data/                   # Raw, cleaned, and engineered datasets
├── models/                 # Saved logistic & random forest models
├── plots/                  # Confusion matrix, feature importance, probability plot
├── src/                    # Modular pipeline scripts
│   ├── fetch_data.py
│   ├── indicators.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── evaluate.py
│
├── stock_direction_pipeline.ipynb   # Complete Colab notebook
├── requirements.txt
└── README.md
```

---

## 7. Author

**Manasvi Japi**  
Aspiring Data Scientist and Quantitative Finance Student  
Focused on applied machine learning and financial modeling.



