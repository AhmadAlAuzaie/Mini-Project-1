
# Machine Learning Pipeline Project

This project demonstrates a full ML workflow from exploratory data analysis (EDA) to preprocessing, modeling, and hyperparameter tuning using Python and scikit-learn.

## 📂 Project Structure

```
root/
├── eda.ipynb                  # Exploratory analysis
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── data/
│   └── data.csv               # Raw dataset
├── src/
│   ├── data_preparation.py    # Data loading, cleaning, feature setup
│   ├── model_training.py      # Preprocessing, modeling, evaluation
│   └── config.yaml            # Model and pipeline config settings
└── main.py                    # Entrypoint for pipeline execution
```

## ✅ Features Covered

1. **Exploratory Data Analysis (EDA)**  
   Conducted in `eda.ipynb` with basic insights, distributions, and correlations.

2. **Data Preprocessing**  
   Encapsulated in `src/data_preparation.py` with feature engineering and cleaning.

3. **Modeling**  
   Logistic Regression, Ridge Regression, and Lasso Regression pipelines in `src/model_training.py`.

4. **Hyperparameter Tuning**  
   GridSearchCV is implemented to tune a classifier and the regressors using parameters from `src/config.yaml`.

## 📊 Requirements

```bash
pip install -r requirements.txt
```

## 🚀 Run Instructions

```bash
python main.py
```
