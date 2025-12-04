# 🛒 Customer Re-Purchase Prediction  
Predicting if a customer will buy again the following month

This repository contains two Jupyter notebooks that form a complete pipeline to predict whether a customer will make a repeat purchase in the next month.  
The project includes **data cleaning**, **feature engineering**, **aggregation**, **model evaluation**, and **machine learning experimentation**.

---

## 📁 Project Structure
```
--------------------

📂 Customer-Rebuy-Prediction
│
├── Notebook_data_cleaning.ipynb
│      → Data cleaning, preprocessing, feature engineering, customer aggregation
│
├── Notebook_prediction.ipynb
│      → Testing of ML models, evaluation, visualizations
│
├── models/
│      ├── models.py
│      │     → Training functions for:
│      │           - Ensemble models (Random Forest)
│      │           - XGBoost
│      │           - Logistic Regression
│      │
│      └── MLP.py
│            → PyTorch nn.Module class + training
│
├── utils/
│      └── evaluation.py
│            → Evaluation utilities:
│                  - Confusion matrix
│                  - ROC / AUC
│                  - Precision, Recall, F1-score
│                  - Probability & threshold analysis
│
└── README.md
       → Documentation of the full project

```
---

## 1️⃣ Notebook: Data Cleaning & Feature Engineering  
**File:** `Notebook_data_cleaning.ipynb`

This notebook focuses on building a **clean, consistent, and machine-learning-ready dataset**.

### 🔧 Main steps:
- **Data Import & Inspection**
  - Exploration of raw transactional data  
  - Detection of missing values, duplicate rows, inconsistent entries

- **Data Cleaning**
  - Handling missing dates and extreme values  
  - Standardizing formats (dates, integers, categorical fields)

- **Feature Engineering**
  - Creation of customer-level metrics such as:
    - Number of visits during previous months
    - RFM indicator (Recence, Frequency, Monetary)
    - Total and average spend  
    - Monthly activity profile  

- **Aggregation**
  - Grouping data by customer ID to build a single row per customer 
  - Grouping data by customoer ID and Month (+Year) to create features about monthly behavior
  - Combining transactional history into meaningful features

- **Final Dataset Export**
  - Saving the cleaned dataset for modeling in the following notebook !  :D

---

## 2️⃣ Notebook: Model Training & Prediction  
**File:** `Notebook_prediction.ipynb`

This notebook evaluates several machine learning models to classify whether a customer will re-purchase next month.

### 🤖 Models Tested
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Simple Neural Network (PyTorch / TensorFlow depending on setup)**
- Additional experiments on thresholds, scaling, and dealing with class imbalance

### 📊 Evaluation & Metrics
- Train/validation/test split  
- Performance metrics:
  - Accuracy  
  - Precision / Recall  
  - F1-score  
  - Confusion matrix  
- Analysis of **false negatives** (customers predicted as non-rebuyers but actually rebuy)

### 🚀 Model Selection
Comparison of multiple models to select the best one for production usage.

---

## 🧠 Goal of the Project  
The goal is to help marketing teams identify **which customers are likely to repurchase next month**, allowing actions such as:
- Retargeting
- Personalized campaigns
- Incentives for likely churners

---

## 📌 Requirements
Recommended environment:
- Python 3.10+
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- matplotlib
- PyTorch

# 📝 Author

Project developed by Lucas MIedzyrzecki.

# 📄 License

This project is open-source. Feel free to reuse and adapt the code. The dataset is not included due to confidentiality agreements.