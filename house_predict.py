# =========================
# HOUSING PRICE PREDICTION
# =========================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. LOAD DATASET
# Replace 'housing.csv' with your Kaggle dataset file name if needed
data = pd.read_csv("housing.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nBasic info:")
print(data.info())

print("\nMissing values in each column:")
print(data.isnull().sum())


# 3. SEPARATE FEATURES AND TARGET
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]


# 4. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("\nNumerical columns:", list(num_cols))
print("Categorical columns:", list(cat_cols))


# 5. PREPROCESSING
# Numerical pipeline:
# - fill missing values using median
# - scale values
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline:
# - fill missing values using most frequent category
# - convert categories to numbers using OneHotEncoder
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


# 6. SPLIT DATA INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# ==========================================
# 7. MODEL 1 - LINEAR REGRESSION
# ==========================================
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

y_pred_lr = linear_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n===== Linear Regression Results =====")
print("MAE :", mae_lr)
print("MSE :", mse_lr)
print("RMSE:", rmse_lr)
print("R2  :", r2_lr)


# ==========================================
# 8. MODEL 2 - DECISION TREE REGRESSOR
# ==========================================
tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42))
])

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("\n===== Decision Tree Results =====")
print("MAE :", mae_tree)
print("MSE :", mse_tree)
print("RMSE:", rmse_tree)
print("R2  :", r2_tree)

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n===== Random Forest Results =====")
print("MAE :", mae_rf)
print("MSE :", mse_rf)
print("RMSE:", rmse_rf)
print("R2  :", r2_rf)


results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [mae_lr, mae_tree, mae_rf],
    "RMSE": [rmse_lr, rmse_tree, rmse_rf],
    "R2 Score": [r2_lr, r2_tree, r2_rf]
})

print("\n===== Model Comparison =====")
print(results)


# 11. CHOOSE BEST MODEL
best_model = rf_model   # usually Random Forest performs better
y_pred_best = y_pred_rf


# 12. PLOT ACTUAL VS PREDICTED
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices (Best Model)")
plt.show()


# 13. FEATURE IMPORTANCE (for Random Forest)
# Need transformed feature names

# Get one-hot encoded category names
ohe = rf_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)

all_feature_names = list(num_cols) + list(cat_feature_names)

importances = rf_model.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n===== Top 10 Important Features =====")
print(feature_importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(data=feature_importance_df.head(10), x="Importance", y="Feature")
plt.title("Top 10 Important Features")
plt.show()


# 14. MAKE A NEW PREDICTION ON FIRST TEST SAMPLE
sample = X_test.iloc[[0]]
sample_actual = y_test.iloc[0]
sample_pred = best_model.predict(sample)[0]

print("\n===== Example Prediction =====")
print("Actual Price   :", sample_actual)
print("Predicted Price:", sample_pred)