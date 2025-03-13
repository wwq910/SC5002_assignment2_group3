import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset Selection
# Fetch the Ames Housing dataset from openML
ames = fetch_openml(name='house_prices', version=1, as_frame=True)
df = ames.frame
print("Original Dataset shape:", df.shape)


# 2. Data Preprocessing with a Selected Subset of Features
# Select 7 features plus the target
selected_features = [
    'LotArea',      
    'OverallQual',   
    'GrLivArea',     
    'TotalBsmtSF',   
    'YearBuilt',     
    'GarageCars',    
    'Neighborhood'   
]
target_column = 'SalePrice'  # house sale price is our target
df_selected = df[selected_features + [target_column]]
print("Selected Data shape:", df_selected.shape)

# 2.1 Handle Missing Values
# For numeric columns, fill missing values with the median.
# For categorical columns, fill missing values with "Missing".
numeric_cols = ['LotArea', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'GarageCars']
categorical_cols = ['Neighborhood']

for col in numeric_cols:
    df_selected[col].fillna(df_selected[col].median(), inplace=True)
for col in categorical_cols:
    df_selected[col].fillna("Missing", inplace=True)

print("\nMissing values after imputation:", df_selected.isnull().sum().sum())

# 2.2 Feature Scaling and Encoding
# Separate features (X) and target (y)
X = df_selected.drop(columns=[target_column])
y = df_selected[target_column]

# Scale numeric features so they have zero mean and unit variance.
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X[numeric_cols])
# Convert scaled array back to DataFrame with original column names
X_numeric = pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=X.index)

# One-Hot Encode the categorical column "Neighborhood"
X_categorical = pd.get_dummies(X[categorical_cols], drop_first=True)

# Combine the scaled numeric features with the one-hot encoded categorical features.
X_final = pd.concat([X_numeric, X_categorical], axis=1)
print("Shape after scaling and encoding:", X_final.shape)

# 2.3 Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)


# 3. Model Training and Evaluation
# 3.1 Linear Regression with K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
lin_reg = LinearRegression()

# Perform cross-validation on the training set
cv_mse_scores_lin = cross_val_score(
    lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold
)
cv_r2_scores_lin = cross_val_score(
    lin_reg, X_train, y_train, scoring='r2', cv=kfold
)

mean_mse_lin = -np.mean(cv_mse_scores_lin)  # Convert to positive MSE
mean_r2_lin = np.mean(cv_r2_scores_lin)

print("\nLinear Regression (CV) - MSE:", mean_mse_lin)
print("Linear Regression (CV) - R^2:", mean_r2_lin)

# Train on entire training set and evaluate on the test set
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

test_mse_lin = mean_squared_error(y_test, y_pred_lin)
test_r2_lin = r2_score(y_test, y_pred_lin)

print("\nLinear Regression (Test) - MSE:", test_mse_lin)
print("Linear Regression (Test) - R^2:", test_r2_lin)

# 3.2 Ridge Regression with Different Alpha Values
alpha_values = [0.01, 0.1, 1, 10, 100]
ridge_cv_results = []

for alpha in alpha_values:
    ridge_reg = Ridge(alpha=alpha, random_state=42)
    cv_mse_scores_ridge = cross_val_score(
        ridge_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold
    )
    cv_r2_scores_ridge = cross_val_score(
        ridge_reg, X_train, y_train, scoring='r2', cv=kfold
    )
    
    mean_mse_ridge = -np.mean(cv_mse_scores_ridge)
    mean_r2_ridge = np.mean(cv_r2_scores_ridge)
    ridge_cv_results.append((alpha, mean_mse_ridge, mean_r2_ridge))

print("\nRidge Regression Cross-Validation Results:")
for alpha, mse_val, r2_val in ridge_cv_results:
    print(f"Alpha={alpha}, CV MSE={mse_val:.2f}, CV R^2={r2_val:.4f}")

# Select the best alpha (based on lowest CV MSE)
best_alpha = sorted(ridge_cv_results, key=lambda x: x[1])[0][0]
print(f"\nBest alpha based on CV MSE: {best_alpha}")

# Retrain Ridge Regression with the best alpha on the full training set and test it.
best_ridge = Ridge(alpha=best_alpha, random_state=42)
best_ridge.fit(X_train, y_train)
y_pred_ridge = best_ridge.predict(X_test)

test_mse_ridge = mean_squared_error(y_test, y_pred_ridge)
test_r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"\nRidge Regression (Test) with alpha={best_alpha} - MSE:", test_mse_ridge)
print(f"Ridge Regression (Test) with alpha={best_alpha} - R^2:", test_r2_ridge)


# 4. Analysis and Comparison
print("\n--- Final Comparison ---")
print(f"Linear Regression - Test MSE: {test_mse_lin:.2f}, R^2: {test_r2_lin:.4f}")
print(f"Ridge Regression (alpha={best_alpha}) - Test MSE: {test_mse_ridge:.2f}, R^2: {test_r2_ridge:.4f}")


# 5. Suggestions for Improving Model Performance
print("\nFurther improvements could include:")
print("- Trying more alpha values for tuning Ridge.")
print("- Outlier handling or transforming the target (e.g., log-transform SalePrice).")
print("- Experimenting with additional features or further feature engineering.")
print("- Scaling additional features if you decide to expand the feature set.")


