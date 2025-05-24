import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from math import sqrt
from src.data_preparation import load_and_prepare_data
from src.model_training import train_models
import yaml

file_path = "/mnt/data/mini_project_1_data.csv"

if __name__ == "__main__":
    x, y = load_and_prepare_data(file_path)
    train_models(x, y)

# Write all files
with open("/mnt/data/src/data_preparation.py", "w") as f:
    f.write(data_prep_py)

with open("/mnt/data/src/model_training.py", "w") as f:
    f.write(model_train_py)

with open("/mnt/data/src/config.yaml", "w") as f:
    f.write(config_yaml)

with open("/mnt/data/main.py", "w") as f:
    f.write(main_py)

def evaluate_cls(name, y_true, y_pred):
    print(f"{name} Classification Report:\n", classification_report(y_true, y_pred))
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}\n")

def evaluate_reg(name, y_true, y_pred):
    print(f"{name} Regression Metrics:")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred):.4f}\n")

# Load data
df = pd.read_csv('mini_project_1_data.csv')
df['high_share'] = (df['shares'] > df['shares'].median()).astype(int)
df['weekday'] = df['weekday'].str.strip().str.lower().str.capitalize()
df['data_channel'] = df['data_channel'].fillna('Unknown').str.strip().str.lower().str.replace('_', ' ').str.capitalize()

x = df.drop(columns=['shares', 'ID', 'URL'])
y = df['high_share']
x = x.copy()
x = x.dropna()
y = y.loc[x.index]

num_features = x.select_dtypes(include='number').columns.tolist()
cat_features = x.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

cls_pipeline = Pipeline([('Preprocessor', preprocessor), ('Classification', LogisticRegression(max_iter=1000))])
ridge_pipeline = Pipeline([('Preprocessor', preprocessor), ('RidgeRegression', Ridge(alpha=1))])
lasso_pipeline = Pipeline([('Preprocessor', preprocessor), ('LassoRegression', Lasso(alpha=1))])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit untuned
cls_pipeline.fit(x_train, y_train)
ridge_pipeline.fit(x_train, y_train)
lasso_pipeline.fit(x_train, y_train)

# Evaluate untuned
evaluate_cls("Logistic (Untuned)", y_test, cls_pipeline.predict(x_test))
evaluate_reg("Ridge (Untuned)", y_test, ridge_pipeline.predict(x_test))
evaluate_reg("Lasso (Untuned)", y_test, lasso_pipeline.predict(x_test))

# Grid Search (Tuned)
cls_grid = GridSearchCV(cls_pipeline, {
    'Classification__penalty': ['l2'],
    'Classification__C': [0.1, 1, 10],
    'Classification__fit_intercept': [True, False],
    'Classification__solver': ['lbfgs']
}, cv=5, scoring='accuracy')

ridge_grid = GridSearchCV(ridge_pipeline, {
    'RidgeRegression__alpha': [0.1, 1, 10],
    'RidgeRegression__fit_intercept': [True, False]
}, cv=5, scoring='r2')

lasso_grid = GridSearchCV(lasso_pipeline, {
    'LassoRegression__alpha': [0.1, 1, 10],
    'LassoRegression__fit_intercept': [True, False]
}, cv=5, scoring='r2')

cls_grid.fit(x_train, y_train)
ridge_grid.fit(x_train, y_train)
lasso_grid.fit(x_train, y_train)

# Evaluate tuned
evaluate_cls("Logistic (Tuned)", y_test, cls_grid.best_estimator_.predict(x_test))
evaluate_reg("Ridge (Tuned)", y_test, ridge_grid.best_estimator_.predict(x_test))
evaluate_reg("Lasso (Tuned)", y_test, lasso_grid.best_estimator_.predict(x_test))

# Confusion matrix
cm = confusion_matrix(y_test, cls_grid.best_estimator_.predict(x_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
