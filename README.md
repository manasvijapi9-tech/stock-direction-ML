📈 Stock Direction Prediction (Machine Learning)

A full end-to-end ML pipeline for predicting if a stock will go UP or DOWN the next day.

🔍 Overview

This project predicts next-day stock direction (Up/Down) using technical indicators and supervised machine learning models.
It includes data collection → cleaning → feature engineering → ML training → evaluation, all fully automated.

Built using:
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

🧩 What This Project Does (Simple Explanation)
Stage	Description
1. Fetch Data	Downloads historical stock prices from Yahoo Finance
2. Clean Data	Removes bad values + fixes datatype issues
3. Add Indicators	RSI, SMA10, SMA20, MACD, Volatility
4. Prepare Dataset	Creates features (X) & labels (y)
5. Train Models	Logistic Regression + Random Forest
6. Evaluate	Confusion matrix + probability plot
7. Feature Importance	Shows which technical indicators matter most

📊 Results
Confusion Matrix
<img src="plots/confusion_matrix.png" width="450">
Prediction Probability Distribution
<img src="plots/probability_distribution.png" width="450">
Feature Importance
<img src="plots/feature_importance.png" width="450">
⚙️ How to Run (Google Colab)
!git clone https://github.com/manasvijapi9-tech/stock-direction-ML.git
%cd stock-direction-ML
!pip install -r requirements.txt

1️⃣ Fetch Data
!python src/fetch_data.py

2️⃣ Clean Data
import pandas as pd
df = pd.read_csv("data/TCS_NS_raw.csv")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df.dropna(inplace=True)
df.to_csv("data/TCS_NS_raw_clean.csv", index=False)

3️⃣ Add Indicators
from src.indicators import add_all_indicators
import pandas as pd
df = pd.read_csv("data/TCS_NS_raw_clean.csv")
df2 = add_all_indicators(df)
df2.to_csv("data/data_with_indicators.csv", index=False)

4️⃣ Prepare Dataset
!python src/prepare_data.py

5️⃣ Train Models
!python src/train_model.py

6️⃣ Evaluate
!python src/evaluate.py


Your plots appear in:

plots/confusion_matrix.png  
plots/feature_importance.png  
plots/probability_distribution.png  

🧠 What I Learned

How to engineer financial indicators

How ML models behave on time-series data

How to build modular Python pipelines

How to evaluate classification performance

How to make production-style ML code and documentation

👤 Author

Manasvi Japi
Interested in: Quantitative Finance, Data Science, and Business Analytics



