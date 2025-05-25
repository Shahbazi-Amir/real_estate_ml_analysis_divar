# 🏠 Real Estate Price Prediction – Final Report

## 🎯 Objective

The objective of this project is to predict real estate prices in Tehran using data collected from Divar.ir, by applying various machine learning algorithms.

## 🧹 Data Preprocessing

- Handling missing values and outliers  
- Encoding categorical variables  
- Normalizing numerical features  

## 🧠 Models Used

- **Linear Regression:** Baseline for comparison  
- **Lasso Regression:** For feature selection  
- **Ridge Regression:** For multicollinearity control  
- **Random Forest:** A nonlinear ensemble method  
- **XGBoost:** Boosted tree-based model with top performance  

## 📊 Model Results

| Model             | R² Score | RMSE   |
|-------------------|----------|--------|
| Linear Regression | 0.65     | 1.20   |
| Lasso Regression  | 0.66     | 1.18   |
| Ridge Regression  | 0.67     | 1.15   |
| Random Forest     | 0.75     | 1.00   |
| XGBoost           | 0.78     | 0.95   |

## ✅ Conclusion

XGBoost achieved the best performance with an R² score of 0.78, making it the most accurate model for predicting real estate prices. Data preprocessing and feature selection had a significant impact on the final results.
