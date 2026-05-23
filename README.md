# AI-Powered-Employee-Retention-and-Workforce-Analytics-System
AI-based employee attrition prediction system using Machine Learning and data analytics. The project analyzes HR data to identify employees likely to leave an organization using classification algorithms, predictive analytics, visualization dashboards, and performance evaluation techniques.

## What the app does

- Trains Logistic Regression and Random Forest attrition models
- Compares model metrics such as accuracy, precision, recall, F1-score, and ROC-AUC
- Shows attrition trends, role-wise risk, overtime patterns, and tenure patterns
- Scores a full employee dataset in batch
- Predicts attrition risk for a single employee from an interactive form

## How to run

1. Install dependencies with `pip install -r requirements.txt`
2. Place the IBM HR Attrition CSV in `data/` or upload it in the app sidebar
3. Train the model artifacts with `python train.py --data data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
4. Launch the dashboard with `streamlit run app.py`

If the saved model artifacts are missing, the Streamlit app can retrain from the currently loaded dataset using the sidebar button.
