# 📘 AI Mini Project — Student Performance Prediction using Linear Regression

This project analyzes how **Study Hours** and **Attendance** influence a student’s **Average Score** using a Multiple Linear Regression model.  
It is a lightweight, beginner-friendly AI/ML project — clean, interpretable, and perfect for academic submissions.

---

## ✨ Project Overview

The goal of this project is to develop a predictive model that estimates a student's performance based on two key academic behaviors:

- **Study Hours**
- **Attendance (%)**

The model outputs:
- Predicted Average Score  
- Performance graphs  
- Error analysis (Residuals)

This project demonstrates the complete ML pipeline:
1. Data creation / loading  
2. Preprocessing  
3. Model training  
4. Model evaluation  
5. Visualization  

---

## 🗂 Project Structure
AI_FAST_LEARNERS_ASSIG...

data/

student_scores.csv

outputs/

actual_vs_predicted.png

attendance_vs_avgscore.png

math_score_model.joblib

predictions.csv

residual_plot.png

studyhours_mathscore.png

studyhours_vs_avgscore.png

report/

STUDENT PERFORMANCE PREDICTIO...

src/

train_and_eval.py

train_model.py

visualize.py

venv/

README.txt

requirements.txt

---

###🚀 How to Run the Project

### **1. Clone the Repository**
```bash
git clone <your-repo-link>
cd <your-repo-name>

2. Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Train the Model
python src/train_model.py

5. Generate Visualizations
python src/visualize.py

📊 Visual Outputs

The project generates the following graphs:

Study Hours vs Average Score

Attendance vs Average Score

Actual vs Predicted Scores

Residual Plot

These graphs are stored in:
outputs/

🧠 Model Performance
Metric	Value
Coefficients	StudyHours ≈ 3.54, Attendance ≈ 0.63
Intercept	6.31
RMSE	2.15
R² Score	0.938

The high R² score indicates that the model explains 93.8% of the variance — strong performance for a simple linear model.

🎯 Key Learning Outcomes

Understanding Multiple Linear Regression

Evaluating model accuracy

Visualizing relationships between academic factors

Using Python (pandas, sklearn, matplotlib)

Packaging and publishing a complete ML project on GitHub

📝 Requirements
pandas
matplotlib
scikit-learn
numpy

👤 Author

Anjali- MLTA57
AI Mini Project
2025-2026
