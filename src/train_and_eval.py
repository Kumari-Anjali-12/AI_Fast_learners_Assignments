# src/train_and_eval.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
import os

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "study_scores.csv")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_PATH)
X = df[['Hours']].values
y = df['Score'].values

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("Model coefficients:")
print("  Slope (m):", model.coef_[0])
print("  Intercept (c):", model.intercept_)
print()
print("Evaluation on test set:")
print(f"  MAE: {mae:.3f}")
print(f"  MSE: {mse:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  R2: {r2:.3f}")

# 6. Save model
model_path = os.path.join(OUTPUT_DIR, "model.joblib")
joblib.dump(model, model_path)
print(f"Saved model to {model_path}")

# 7. Save predictions CSV (test rows + predicted)
pred_df = pd.DataFrame({
    'Hours': X_test.flatten(),
    'ActualScore': y_test,
    'PredictedScore': y_pred
}).sort_values('Hours')
pred_csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"Saved predictions to {pred_csv_path}")

# 8. Plot (scatter + regression line using whole dataset for line)
plt.figure(figsize=(8,6))
plt.scatter(X, y, label='Data points')
# regression line across range
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, label='Regression line')
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Student Score Prediction from Study Hours")
plt.legend()
plot_path = os.path.join(OUTPUT_DIR, "score_plot.png")
plt.savefig(plot_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved plot to {plot_path}")

# 9. Example usage: predict for a new value
new_hours = [[7.25]]  # change as needed
pred_example = model.predict(new_hours)[0]
print(f"Example: predicted score for {new_hours[0][0]} hours = {pred_example:.2f}")
