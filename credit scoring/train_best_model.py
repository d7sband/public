import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, log_loss

def main():
    print("Loading data...")
    url = "https://raw.githubusercontent.com/PersDep/data-mining-intro-2021/main/hw03-EDA-data/german_credit.csv"
    data = pd.read_csv(url, na_values='none')

    # Preprocessing
    print("Preprocessing data...")
    X = data.drop('credit_risk', axis=1)
    y = data['credit_risk'].map({'bad': 0, 'good': 1})

    num_cols = ['duration', 'amount', 'age']
    cat_cols = [i for i in X.columns if i not in num_cols]

    # Get indices of categorical columns for SMOTENC
    cat_cols_indices = [X.columns.get_loc(c) for c in cat_cols]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, stratify=y, random_state=42, test_size=0.2
    )

    # Define Transformers
    # Numerical: Impute median -> Scale
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute constant -> OHE
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ]
    )

    # Model with best hyperparameters
    # n_estimators=281, learning_rate=0.0929920030595642, max_depth=10, min_child_weight=1, 
    # subsample=0.8752945550232539, colsample_bytree=0.7686511209987077, gamma=0, reg_lambda=7, reg_alpha=3
    clf = XGBClassifier(
        n_estimators=281,
        learning_rate=0.0929920030595642,
        max_depth=10,
        min_child_weight=1,
        subsample=0.8752945550232539,
        colsample_bytree=0.7686511209987077,
        gamma=0,
        reg_lambda=7,
        reg_alpha=3,
        booster='gbtree',
        n_jobs=-1,
        random_state=42
    )

    # Full Pipeline with SMOTENC
    # Note: SMOTENC must be applied before the preprocessor transforms categorical cols to OHE, 
    # BUT SMOTENC needs numeric data or raw data. 
    # In the original notebook, how was SMOTENC applied?
    # Usually SMOTENC is applied on the raw data (or partially processed).
    # Let's check the notebook structure logic if possible, but standard is:
    # ImbPipeline([('smotenc', SMOTENC(...)), ('preprocessor', ...), ('clf', ...)])
    # However, SMOTENC needs to know which cols are categorical. 
    # If we pass raw dataframe to SMOTENC, it works if we give indices.
    
    pipeline = ImbPipeline([
        ('smotenc', SMOTENC(categorical_features=cat_cols_indices, random_state=42)),
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

    # Training
    print("Training model...")
    pipeline.fit(x_train, y_train)

    # Evaluation
    print("Evaluating model...")
    y_pred = pipeline.predict(x_test)
    y_pred_proba = pipeline.predict_proba(x_test)[:, 1]

    f1 = f1_score(y_test, y_pred, average='weighted')
    loss = log_loss(y_test, y_pred_proba)

    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Log Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
