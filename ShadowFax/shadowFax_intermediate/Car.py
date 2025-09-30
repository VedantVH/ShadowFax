# ==========================
# Car Selling Price Prediction
# ==========================

# --------------------------
# 1. Import Libraries
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, request, jsonify
import os

# --------------------------
# 2. Load Dataset
# --------------------------
file_path = 'car_data.csv'  # replace with your actual file path

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please check the path.")

df = pd.read_csv(file_path)

# Strip any extra spaces from column names
df.columns = df.columns.str.strip()

# Preview data
print(df.head())
print(df.info())
print(df.describe())

# --------------------------
# 3. Data Preprocessing
# --------------------------

# Handle missing values
df = df.dropna()  # or use df.fillna() for advanced imputation

# Derive new feature: Car Age
if 'Year' in df.columns:
    df['Car_Age'] = 2025 - df['Year']  # replace 2025 with current year
    df = df.drop(['Year'], axis=1)

# --------------------------
# 4. Define Features
# --------------------------
numerical_features = [col for col in ['Kilometers_Driven', 'Owner', 'Present_Price', 'Car_Age'] if col in df.columns]
categorical_features = [col for col in ['Fuel_Type', 'Seller_Type', 'Transmission'] if col in df.columns]

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# --------------------------
# 5. Exploratory Data Analysis (EDA)
# --------------------------

# Numeric correlation heatmap
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')  # saved instead of show
plt.close()

# Pairplot for numeric features
sns.pairplot(numeric_df)
plt.savefig('pairplot.png')
plt.close()

# Distribution of target variable
sns.histplot(df['Selling_Price'], kde=True)
plt.title('Selling Price Distribution')
plt.savefig('selling_price_distribution.png')
plt.close()

# Categorical vs Target
for cat in categorical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=cat, y='Selling_Price', data=df)
    plt.title(f'{cat} vs Selling Price')
    plt.savefig(f'{cat}_vs_selling_price.png')
    plt.close()

# --------------------------
# 6. Feature Engineering / Preprocessing Pipeline
# --------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# --------------------------
# 7. Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 8. Model Selection & Hyperparameter Tuning
# --------------------------
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt']  # removed 'auto' to avoid warnings
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', search)
])

# --------------------------
# 9. Model Training
# --------------------------
pipeline.fit(X_train, y_train)

# --------------------------
# 10. Model Evaluation
# --------------------------
y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

# Feature importance
best_model = pipeline.named_steps['model'].best_estimator_
feature_names = numerical_features + list(
    pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
)
importances = best_model.feature_importances_
feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12,6))
feat_importance.plot(kind='bar')
plt.title('Feature Importances')
plt.savefig('feature_importance.png')
plt.close()

# --------------------------
# 11. Deployment using Flask
# --------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df_input = pd.DataFrame([data])
        prediction = pipeline.predict(df_input)
        return jsonify({'Predicted_Selling_Price': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
