from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import yaml

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

# Display basic information about the dataset
df.info(), df.head()

# Check for missing values
df.isnull().sum()

# Summary statistics of the dataset
df.describe().T

# Check for duplicate entries
df.duplicated().sum()
duplicates = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))

# Drop duplicate entries
df.drop_duplicates(inplace=True)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

def build_preprocessor(x):
    num_features = x.select_dtypes(include='number').columns.tolist()
    cat_features = x.select_dtypes(include='object').columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    # Retrieve feature names after preprocessing
    preprocessor.fit(x)
    feature_names = num_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features))
    
    return preprocessor, num_features, cat_features, feature_names

# Evaluation function
def evaluate_cls(name, y_true, y_pred):
    print(f"{name} Classification Report:", classification_report(y_true, y_pred))
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

def train_models(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config['test_size'], random_state=config['random_state'])
    
    preprocessor, _, _ = build_preprocessor(x)

    pipelines = {
        'Logistic': Pipeline([('Preprocessor', preprocessor), ('Classification', LogisticRegression(max_iter=1000))]),
        'Linear': Pipeline([('Preprocessor', preprocessor), ('LinearRegression', LinearRegression())]),
        'Ridge': Pipeline([('Preprocessor', preprocessor), ('RidgeRegression', Ridge(alpha=1))]),
        'Lasso': Pipeline([('Preprocessor', preprocessor), ('LassoRegression', Lasso(alpha=1))])
    }

    for name, pipe in pipelines.items():
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        if name == 'logistic':
            evaluate_cls(f"{name.title()}", y_test, y_pred)
        else:
            evaluate_reg(f"{name.title()}", y_test, y_pred)        
    
    best_pipelines = {
        'Best Logistic': (Pipeline([('Preprocessor', preprocessor), ('Classification', LogisticRegression(max_iter=1000))]), config['cls_params']),
        'Best Ridge': (Pipeline([('Preprocessor', preprocessor), ('RidgeRegression', Ridge())]), config['ridge_params']),
        'Best Lasso': (Pipeline([('Preprocessor', preprocessor), ('LassoRegression', Lasso())]), config['lasso_params'])
    }
    
    for name, (pipe, params) in best_pipelines.items():
        grid = GridSearchCV(pipe, {f"{pipe.steps[-1][0]}__{k}": v for k, v in params.items()}, 
                            cv=5, scoring='r2' if "Regression" in name else 'accuracy')
        grid.fit(x_train, y_train)
        y_pred = grid.best_estimator_.predict(x_test)
        if "Logistic" in name:
            evaluate_cls(name, y_test, y_pred)
        else:
            evaluate_reg(name, y_test, y_pred)