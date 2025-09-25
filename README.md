# 📧 Email Campaign Optimizer

This project develops a **machine learning solution to optimize email marketing campaigns** by predicting user engagement and recommending the best email configuration (time, version, text) for each user. The goal is to maximize **click-through rates (CTR)** and provide actionable insights for marketers.

---

## 📂 Project Structure

- **Email Optimizer.ipynb** — Main Jupyter notebook containing:
  - Data preprocessing
  - Exploratory Data Analysis (EDA)
  - Model training and evaluation
  - Recommendation logic
- **merged_email_data.csv** — Merged dataset with engineered features for modeling.
- **ensemble_model.pkl** — Saved ensemble model for predicting email clicks.
- **click_prediction_results.csv** — Contains actual and predicted click likelihoods for test data.

---

## 🔄 Workflow

### 1. Data Preprocessing
- Loaded raw email datasets.
- Created binary features such as **email_opened** and **link_clicked**.
- Engineered features like **user_past_purchases, email_text, email_version**.
- Merged datasets into `merged_email_data.csv`.

### 2. Exploratory Data Analysis (EDA)
Conducted extensive **EDA and data visualization** to uncover patterns:
- Opening the email is the strongest indicator of a subsequent click.
- Short emails are more likely to be opened than long ones.
- Personalized emails consistently outperform generic ones.
- Engagement peaks during **business hours** on weekdays.
- Customers with **>13 purchases** show higher CTR, indicating higher value.
- **UK and US** users have higher CTR compared to other countries.

Visualizations helped identify key behavioral trends, guiding feature engineering and model design.

### 3. Model Training
- Addressed **class imbalance (~2% clicks)** using:
  - **Stratified sampling** for train-test split.
  - `class_weight="balanced"` during model training.
- Trained multiple classification models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost (best individual performance)
- Combined models using a **soft-voting ensemble** for improved balanced results.
- Saved the final model as `ensemble_model.pkl`.

### 4. Prediction & Evaluation
- Predicted click probabilities on test data.
- Evaluated using **Precision, Recall, F1-score, ROC-AUC**.
- Saved results in `click_prediction_results.csv`.

### 5. Recommendation Engine
- Implemented `recommend_best_email` function.
- Suggests the **optimal hour, email format, and personalization strategy** for each user based on model predictions.

---

## 🚀 Usage

1. Open **Email Optimizer.ipynb** in Jupyter Notebook or VS Code.
2. Run all cells to:
   - Preprocess data
   - Conduct EDA
   - Train models
   - Evaluate results
   - Generate recommendations
3. Review output files for processed data, predictions, and model artifacts.

---

## 📦 Requirements

- Python 3.x  
- pandas  
- scikit-learn  
- joblib  
- matplotlib  

Install dependencies with:

```bash
pip install pandas scikit-learn joblib matplotlib
