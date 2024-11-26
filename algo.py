import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load the dataset
medical_df = pd.read_csv('../new/insurance.csv')

# Check for missing values
if medical_df.isnull().sum().any():
    medical_df = medical_df.dropna()  # Optionally, handle with imputation if needed

# Feature Engineering: Creating BMI categories and age groups
medical_df['bmi_category'] = pd.cut(
    medical_df['bmi'], bins=[0, 18.5, 24.9, 29.9, np.inf],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# Update the categorical and numerical columns list after feature engineering
categorical_cols = ["sex", "smoker", "region", "bmi_category"]
numerical_cols = ["age", "bmi", "children"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)  # Drop first to avoid dummy variable trap
    ]
)

# Transform the data
X = medical_df.drop(columns=["charges"])
y = medical_df["charges"]

# Polynomial Features to capture non-linearity
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(preprocessor.fit_transform(X))

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_poly)

# Print the number of components selected by PCA
print(f"PCA selected {X_pca.shape[1]} components out of {X_poly.shape[1]} original features.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)  # Regularization parameter can be tuned
lasso_model = Lasso(alpha=0.1)  # Regularization parameter can be tuned
random_forest_model = RandomForestRegressor(random_state=42)
knn_model = KNeighborsRegressor(n_neighbors=5)
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Fit the models
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
xgboost_model.fit(X_train, y_train)

# Predictions for each model
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)
y_pred_xgb = xgboost_model.predict(X_test)

# Metrics for each model
models_metrics = {
    "Linear Regression": (mean_squared_error(y_test, y_pred_linear), r2_score(y_test, y_pred_linear)),
    "Ridge Regression": (mean_squared_error(y_test, y_pred_ridge), r2_score(y_test, y_pred_ridge)),
    "Lasso Regression": (mean_squared_error(y_test, y_pred_lasso), r2_score(y_test, y_pred_lasso)),
    "Random Forest Regressor": (mean_squared_error(y_test, y_pred_rf), r2_score(y_test, y_pred_rf)),
    "KNN Regressor": (mean_squared_error(y_test, y_pred_knn), r2_score(y_test, y_pred_knn)),
    "XGBoost Regressor": (mean_squared_error(y_test, y_pred_xgb), r2_score(y_test, y_pred_xgb))
}

# Print metrics for each model
for model_name, (mse, r2) in models_metrics.items():
    print(f"{model_name} -> Mean Squared Error (MSE): {mse:.2f}, R2 Score: {r2:.2f}")

# Stacking Regressor: Combine multiple models
stacking_regressor = StackingRegressor(
    estimators=[
        ('ridge', ridge_model),
        ('rf', random_forest_model),
        ('knn', knn_model),
        ('xgb', xgboost_model)
    ],
    final_estimator=RandomForestRegressor(random_state=42)
)
stacking_regressor.fit(X_train, y_train)
y_pred_stacking = stacking_regressor.predict(X_test)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)
print(f"Stacking Regressor -> Mean Squared Error (MSE): {mse_stacking:.2f}, R2 Score: {r2_stacking:.2f}")

# Cross-validation to validate the best model
cv_scores = cross_val_score(random_forest_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
avg_mse_cv = -np.mean(cv_scores)
print(f"Random Forest Average MSE with 5-Fold CV: {avg_mse_cv:.2f}")

# Determine the best model based on MSE
best_model_name = min(models_metrics, key=lambda x: models_metrics[x][0])
best_model_metrics = models_metrics[best_model_name]

print(f"\nBest Model after Enhancement: {best_model_name}")
print(f"Metrics -> Mean Squared Error (MSE): {best_model_metrics[0]:.2f}, R2 Score: {best_model_metrics[1]:.2f}")


