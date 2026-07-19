import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "laptops_clean.csv"))

feature_cols = ['Company', 'Type', 'Screen_Size_Inches', 'Touchscreen', 'IPS_Panel',
                'CPU_Brand', 'CPU_Speed_GHz', 'RAM_GB', 'Storage_GB', 'Storage_Type',
                'GPU_Brand', 'Weight_KG']

if 'Is_Hybrid_Storage' in df.columns:
    df['Is_Hybrid_Storage'] = df['Is_Hybrid_Storage'].astype(int)
    feature_cols.append('Is_Hybrid_Storage')

target = 'Price_Euros'

df = df.dropna(subset=feature_cols + [target])

categorical_cols = ['Company', 'Type', 'CPU_Brand', 'Storage_Type', 'GPU_Brand']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[feature_cols].copy()
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
numeric_cols = ['Screen_Size_Inches', 'CPU_Speed_GHz', 'RAM_GB', 'Storage_GB', 'Weight_KG']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'Random Forest': RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)
}

cv_folds = 3

results = {}
best_model = None
best_score = -np.inf

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
    results[name] = {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std()
    }
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

results_df = pd.DataFrame(results).T
print("=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)
print(results_df.to_string())
print(f"\nBest model: {best_name} with R² = {best_score:.4f}")

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODEL_DIR, "laptop_price_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
joblib.dump(results_df, os.path.join(MODEL_DIR, "results.pkl"))

print(f"\nFeature Importance ({best_name}):")
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(fi.to_string())
else:
    coef = pd.Series(best_model.coef_, index=feature_cols).sort_values(ascending=False)
    print(coef.to_string())

print("\nModel and artifacts saved to 'model/' directory.")
