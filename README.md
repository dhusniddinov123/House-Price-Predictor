# 🏠 House Price Predictor

A Multiple Linear Regression project that demonstrates advanced pre-processing techniques to predict residential property values. This project moves beyond simple "straight-line" math to handle categorical data and non-linear relationships.

## 🚀 Key Features & Concepts
- **Categorical Encoding:** Utilized **One-Hot Encoding** to transform geographic locations into machine-readable numeric data.
- **Polynomial Features:** Implemented degree-2 features to capture non-linear trends in house pricing (e.g., how price explodes as square footage increases).
- **Feature Scaling:** Applied **StandardScaler** to ensure all variables (sqft vs. bedrooms) are treated equally by the gradient descent algorithm.
- **Regularization (Ridge):** Used **L2 Regularization** (Alpha=10) to prevent the model from overfitting on a small dataset, ensuring more stable predictions for unseen data.
- **Error Metrics:** Evaluated performance using both **R-Squared** (variance explanation) and **Mean Absolute Error** (dollar-value accuracy).

## 🛠️ Technology Stack
- **Python**
- **Pandas:** Data manipulation and encoding.
- **Scikit-Learn:** Modeling, scaling, and evaluation.

## 📂 Project Structure
- `main.py`: The complete pipeline from raw data to prediction.
- `README.md`: Documentation of the ML concepts used.

## 📈 Learning Reflection
The primary challenge of this project was balancing **Bias and Variance**. While a standard Polynomial model achieved a high R² score, it suffered from "extreme" coefficients. By introducing **Ridge Regression**, I penalized model complexity to achieve a more reliable "Honest Score" suitable for real-world application.

---
*Created as part of my **ML Internship Journey**. This project demonstrates foundational skills in supervised learning and data pre-processing.*