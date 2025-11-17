import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "student_scores.csv")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA)

# 2. Select features and target
X = df[['StudyHours', 'Attendance']]
y = df['MathScore']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R2:", r2)

# 7. Save model
joblib.dump(model, os.path.join(OUT, "math_score_model.joblib"))

# 8. Save predictions
pred_df = pd.DataFrame({
    "StudyHours": X_test['StudyHours'],
    "Attendance": X_test['Attendance'],
    "ActualMathScore": y_test,
    "PredictedMathScore": y_pred
})
pred_df.to_csv(os.path.join(OUT, "predictions.csv"), index=False)

# 9. Plot
plt.scatter(df['StudyHours'], df['MathScore'], label='Data')
plt.xlabel("Study Hours")
plt.ylabel("Math Score")
plt.title("Study Hours vs Math Score")
plt.savefig(os.path.join(OUT, "studyhours_mathscore.png"), dpi=150)
plt.close()
