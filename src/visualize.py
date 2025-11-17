import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ============================
#   CREATE OUTPUT FOLDER
# ============================
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ============================
#   LOAD DATA
# ============================
df = pd.read_csv("data/student_scores.csv")
df["AverageScore"] = df[["MathScore", "ScienceScore", "ReadingScore"]].mean(axis=1)

X = df[["StudyHours", "Attendance"]]
y = df["AverageScore"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ============================
# 1. Study Hours vs Average Score
# ============================
plt.figure(figsize=(8,5))
plt.scatter(df["StudyHours"], df["AverageScore"])
plt.xlabel("Study Hours")
plt.ylabel("Average Score")
plt.title("Study Hours vs Average Score")

z = np.polyfit(df["StudyHours"], df["AverageScore"], 1)
p = np.poly1d(z)
plt.plot(df["StudyHours"], p(df["StudyHours"]))

plt.savefig(f"{output_dir}/studyhours_vs_avgscore.png", dpi=300)
plt.show()

# ============================
# 2. Attendance vs Average Score
# ============================
plt.figure(figsize=(8,5))
plt.scatter(df["Attendance"], df["AverageScore"])
plt.xlabel("Attendance (%)")
plt.ylabel("Average Score")
plt.title("Attendance vs Average Score")

z = np.polyfit(df["Attendance"], df["AverageScore"], 1)
p = np.poly1d(z)
plt.plot(df["Attendance"], p(df["Attendance"]))

plt.savefig(f"{output_dir}/attendance_vs_avgscore.png", dpi=300)
plt.show()

# ============================
# 3. Actual vs Predicted Scores
# ============================
plt.figure(figsize=(8,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Score")
plt.title("Actual vs Predicted Scores")
plt.legend()

plt.savefig(f"{output_dir}/actual_vs_predicted.png", dpi=300)
plt.show()

# ============================
# 4. Residual Plot
# ============================
residuals = y_test.values - y_pred

plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Score")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.savefig(f"{output_dir}/residual_plot.png", dpi=300)
plt.show()
