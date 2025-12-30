# 🌍 AtmoSense: Air Quality Prediction System

AtmoSense is an end-to-end Machine Learning–based Air Quality Prediction and Classification System designed to analyze, predict, and categorize air pollution levels using real-world environmental data.
The project demonstrates a complete data science pipeline, from raw data handling to model deployment-ready predictions.
# 👩‍🎓 Author


Mahmuda Sultana

Department of Computer Science & Engineering

Metropolitan University, Sylhet
# 🗂️ Dataset Information:


Dataset: Air Quality UCI Dataset

Source: UCI Machine Learning Repository

Type: Time-series environmental data

Records Used: 9,357

Key Features:

CO(GT), NOx(GT), NO2(GT), O3

Temperature, Humidity

Date & Time (converted to DateTime index)

# 📌 Project Overview:

Air pollution is a major environmental and public health challenge worldwide. Accurate prediction and classification of air quality can help governments, researchers, and citizens make informed decisions.

AtmoSense leverages Machine Learning, statistical analysis, and data visualization to:

-Predict pollutant concentration levels (Regression)

-Classify air quality into meaningful categories

-Analyze temporal patterns and correlations

-Provide a reusable real-time prediction system

-Save trained models for future deployment
# 🎯 Objectives:

-Perform detailed Exploratory Data Analysis (EDA) on air quality data

-Clean and preprocess raw environmental datasets

-Engineer advanced temporal and statistical features

-Train and evaluate regression models to predict CO concentration

-Train and evaluate classification models to categorize air quality

-Compare model performance visually and numerically

-Apply dimensionality reduction (PCA)

-Build an interactive prediction system

-Export trained models and results

# 🛠️ Technologies Used
## Programming Language:

Python 3

## Libraries & Tools:

-Data Processing: pandas, numpy

-Visualization: matplotlib, seaborn, plotly, missingno

-Machine Learning: scikit-learn

-Advanced Models: XGBoost, LightGBM, CatBoost

--Interpretability: SHAP

-Forecasting: Prophet

-Model Saving: joblib  

# 🔄 Project Workflow:
Data Loading
   ↓
Exploratory Data Analysis (EDA)
   ↓
Data Cleaning & Imputation
   ↓
Outlier Handling
   ↓
Feature Engineering
   ↓
Train–Test Split
   ↓
Feature Scaling
   ↓
Regression Models
   ↓
Classification Models
   ↓
Model Comparison
   ↓
PCA Analysis
   ↓
Real-Time Prediction
   ↓
Model Saving & Export

# 📊 Exploratory Data Analysis (EDA):
-Missing value analysis using MissingNo

-Correlation analysis with heatmaps

-Time-series visualization of pollutants

-Distribution plots and boxplots for outlier detection

-Temporal trend analysis (daily, monthly, seasonal)

# 🧹 Data Cleaning & Feature Engineering:
##Data Cleaning:

-Conversion of decimal separators (, → .)

-Date & Time merged into DateTime index

-KNN-based missing value imputation

-IQR-based outlier capping

-Removal of columns with excessive missing values

##Feature Engineering:

-Temporal features: Hour, Day, Month, Week, Quarter

-Cyclical encoding using sine & cosine functions

-Binary indicators: Weekend, Night, Working Hours

-Air Quality categories derived from CO levels 

# 📈 Regression Models (CO Prediction):
##Models Implemented:
-Linear Regression (Scikit-learn)

-Linear Regression from Scratch (Gradient Descent)
## Evaluation Metrics:

-Mean Squared Error (MSE)

-R² Score

## Visual Analysis:

-Actual vs Predicted plots

-Residual analysis

-Loss convergence graph (Gradient Descent)

# 🧪 Classification Models (Air Quality Category):

## Categories

Good: CO < 1

Moderate: 1 ≤ CO < 3

Bad: CO ≥ 3

## Models Used:

-K-Nearest Neighbors (KNN)

-Logistic Regression

## Evaluation Metrics:

-Accuracy

-Precision, Recall, F1-score

-Confusion Matrix

-Cross-validation for optimal K (KNN)

#🏆 Model Comparison:

-Regression models compared using R² Score

-Classification models compared using Accuracy

-Bar chart visualization included for clarity   
# 📌 Results Summary:

-Linear Regression produced reliable CO predictions

-Gradient Descent validated theoretical concepts

-KNN achieved strong classification accuracy

-Logistic Regression provided interpretable results
