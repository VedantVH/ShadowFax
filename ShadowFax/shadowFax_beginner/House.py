# Boston_HousePrice_Enhanced.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---------------- 1) Load dataset ----------------
def load_boston_dataframe(csv_path='boston.csv'):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset from {csv_path}. Shape: {df.shape}")
        return df
    else:
        from sklearn.datasets import fetch_openml
        boston = fetch_openml(name='boston', version=1, as_frame=True)
        df = boston.frame.copy()
        if 'MEDV' not in df.columns and 'target' in df.columns:
            df.rename(columns={'target':'MEDV'}, inplace=True)
        print("Loaded Boston dataset from OpenML. Shape:", df.shape)
        return df

df = load_boston_dataframe()

# ---------------- 2) EDA & Data Analysis ----------------
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Basic Statistics ---")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Target distribution
plt.figure(figsize=(6,4))
sns.histplot(df['MEDV'], kde=True, bins=30)
plt.title("MEDV Distribution")
plt.show()

# ---------------- 3) Preprocessing ----------------
target_col = 'MEDV'
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Outlier removal for numeric features (optional, z-score method)
from scipy.stats import zscore
X_numeric = X[numeric_cols]
z_scores = np.abs(zscore(X_numeric))
outliers = (z_scores > 3).any(axis=1)
print(f"Outliers detected: {outliers.sum()} rows")


# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson', standardize=False)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# ---------------- 4) Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- 5) Define regression models ----------------
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(C=1.0, epsilon=0.2),
    'MLPRegressor': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42)
}

if XGB_AVAILABLE:
    models['XGBoost'] = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
if LGB_AVAILABLE:
    models['LightGBM'] = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=-1, random_state=42, n_jobs=-1)
if CB_AVAILABLE:
    models['CatBoost'] = cb.CatBoostRegressor(n_estimators=200, learning_rate=0.1, depth=4, verbose=0, random_state=42)

# ---------------- 6) Train, Predict, Evaluate ----------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[
        ('preproc', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}")
    return pipeline, y_pred

trained_models = {}
predictions = {}
results_summary = []

for name, model in models.items():
    pipe, y_pred = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    trained_models[name] = pipe
    predictions[name] = y_pred
    results_summary.append({
        'Model': name,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    })

results_df = pd.DataFrame(results_summary).sort_values(by='RMSE')
print("\n--- Model Comparison ---")
print(results_df)

# ---------------- 7) Residual Analysis ----------------
best_model_name = results_df.iloc[0]['Model']
y_pred_best = predictions[best_model_name]
residuals = y_test - y_pred_best

plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred_best, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title(f"Residuals vs Predicted ({best_model_name})")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title(f"Residual Distribution ({best_model_name})")
plt.show()

# ---------------- 8) Feature Importance / SHAP ----------------
tree_models = ['RandomForest','GradientBoosting','XGBoost','LightGBM','CatBoost']
for name in tree_models:
    if name in trained_models:
        model = trained_models[name].named_steps['model']
        preproc_fit = trained_models[name].named_steps['preproc']
        feature_names = numeric_cols.copy()
        if categorical_cols:
            ohe = preproc_fit.named_transformers_['cat'].named_steps['onehot']
            ohe_names = list(ohe.get_feature_names_out(categorical_cols))
            feature_names += ohe_names
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print(f"\nTop features for {name}:")
        print(feat_imp.head(10))
        plt.figure(figsize=(8,4))
        feat_imp.head(10).plot(kind='bar')
        plt.title(f"Top 10 Feature Importances ({name})")
        plt.tight_layout()
        plt.show()

# Optional: SHAP summary (if installed)
if SHAP_AVAILABLE and best_model_name in tree_models:
    model = trained_models[best_model_name].named_steps['model']
    preproc_fit = trained_models[best_model_name].named_steps['preproc']
    X_transformed = preproc_fit.transform(X_test)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)

# ---------------- 9) Save best model ----------------
import joblib
joblib.dump(trained_models[best_model_name], f'best_boston_model_{best_model_name}.joblib')
print(f"\nSaved best model: best_boston_model_{best_model_name}.joblib")
