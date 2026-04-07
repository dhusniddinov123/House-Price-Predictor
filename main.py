import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

#DATA PREPARATION
data = {
    'sqft': [1500, 1800, 2400, 3000, 3500, 1200, 5000, 2100, 1100, 4200, 1500, 1600],
    'bedrooms': [3, 3, 4, 4, 5, 2, 6, 3, 2, 5, 3, 3],
    'age': [10, 15, 5, 2, 1, 20, 0, 8, 25, 3, 12, 14],
    'location': ['Downtown', 'Suburb', 'Rural', 'Downtown', 'Suburb', 'Rural', 'Downtown', 'Suburb', 'Rural', 'Downtown', 'Suburb', 'Rural'],
    'price': [400000, 350000, 250000, 550000, 480000, 150000, 950000, 310000, 110000, 820000, 380000, 240000]
}
df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=['location'])

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# 2. FEATURE ENGINEERING (Polynomials + Scaling)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 3. MODELING (The Strict Teacher - Ridge)
# We use alpha=10 as you found it worked best!
model = Ridge(alpha=10)

# 4. CROSS-VALIDATION (The "Trust" Test)
# We test the model 5 different times on different parts of the data
cv_scores = cross_val_score(model, X_scaled, y, cv=3) # Using 3-fold because data is tiny

print(f"--- Model Reliability Report ---")
print(f"Individual CV Scores: {cv_scores}")
print(f"Average 'Honest' R2 Score: {np.mean(cv_scores):.4f}")

# 5. FINAL TRAINING & PREDICTION
model.fit(X_scaled, y)
print(f"\nPipeline Complete. Model is ready for production.")