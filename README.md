# demo_pro2

🏡 House Price Prediction – Machine Learning Project

A complete end-to-end machine learning pipeline for predicting house prices using structured data.
This project preprocesses numeric & categorical features, trains a regression model, evaluates performance, and generates visual insights.

📌 Table of Contents

Overview

Features

Dataset

ML Pipeline

Visualizations

Project Structure

Installation

How to Run

Results

Future Improvements

License

📘 Overview

This project implements a Linear Regression model wrapped inside a Scikit-Learn pipeline.
It automatically preprocesses data using:

StandardScaler for numeric features

OneHotEncoder for categorical features

The model is then trained and evaluated using RMSE and R² Score.
Several plots are generated to analyze price distribution and model performance.

⭐ Features

✔️ Automated preprocessing (scaling + encoding)
✔️ Train/test split for evaluation
✔️ Linear Regression model
✔️ Model performance metrics
✔️ Multiple visualizations
✔️ Fully reproducible workflow

📊 Dataset

The dataset used is:

House_Price_India.csv


Target variable: Price
Inputs: Mixture of numeric and categorical housing attributes.

🧠 ML Pipeline

The pipeline consists of:

ColumnTransformer

Numeric → StandardScaler

Categorical → OneHotEncoder

Model

Linear Regression

Train/Test Split

80% training

20% testing

Metrics

RMSE

R² Score

🖼️ Visualizations

Save your generated plot images inside an images/ folder.
Here is how they will appear in the README:

📌 1. Distribution of House Prices

📌 2. Correlation Heatmap

📌 3. Actual vs Predicted Prices

📌 4. Residual Plot

📁 Project Structure
House Price Prediction Project
│
├── images/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── House_Price_India.csv
├── house_price_prediction.py       # (optional) Python script version
└── README.md

⚙️ Installation

Install dependencies with:

pip install pandas numpy scikit-learn matplotlib seaborn

▶️ How to Run

Place the dataset House_Price_India.csv in the project folder.

Run the script:

python house_price_prediction.py


—or paste the code into a Jupyter Notebook or Google Colab cell.

📈 Results

After training the model, the script prints:

RMSE – Indicates prediction error

R² Score – Indicates goodness of fit

Example:

RMSE: <value>
R² Score: <value>


Visualizations further help evaluate performance:

Price distribution
![LinkedIn Dashboard](/images/img1.png)

Feature correlations
![LinkedIn Dashboard](/images/img4.png)
![LinkedIn Dashboard](/images/img5.png)

Actual vs predicted values
![LinkedIn Dashboard](/images/img2.png)

Residual behavior
![LinkedIn Dashboard](/images/img3.png)

🔮 Future Improvements

🚀 Try more advanced models (Random Forest, XGBoost, LightGBM)
⚙️ Add hyperparameter tuning
📉 Handle outliers
🧩 Add feature engineering
🏗️ Deploy model using Flask/Streamlit
