import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# ===============================
# 1. Load and Inspect Dataset
# ===============================
df = pd.read_csv("air_quality.csv")

print("\n✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("\nSample Data:\n", df.head())
print("\nColumns in dataset:\n", df.columns.tolist())

# Handle missing values
df = df.dropna()   # or use df.fillna(method="ffill")

# ===============================
# 2. Feature Selection
# ===============================
X = df[['year', 'month', 'day', 'hour', 'PM10', 'AT']]  # Features
y = df['PM2.5']  # Target

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Train Model
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# 5. Predictions & Evaluation
# ===============================
y_pred = model.predict(X_test)

print("\n📊 Model Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ===============================
# 6. Feature Importance
# ===============================
coeffs = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\n📌 Feature Importance:\n", coeffs)

# ===============================
# 7. Visualization
# ===============================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue", label="Predictions")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5")
plt.legend()
plt.show()

# ===============================
# 8. Save Model for Future Use
# ===============================
joblib.dump(model, "pm25_model.pkl")
print("\n💾 Model saved as pm25_model.pkl")

# ===============================
# 9. Example: Load & Predict New Data
# ===============================
loaded_model = joblib.load("pm25_model.pkl")
example_input = [[2025, 9, 8, 14, 120, 28]]  # year, month, day, hour, PM10, AT
new_pred = loaded_model.predict(example_input)
print("\n🔮 Predicted PM2.5 for new data:", new_pred[0])
