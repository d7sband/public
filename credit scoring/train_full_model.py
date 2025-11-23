import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
import joblib

def main():
    print("Loading full dataset...")
    url = "https://raw.githubusercontent.com/PersDep/data-mining-intro-2021/main/hw03-EDA-data/german_credit.csv"
    data = pd.read_csv(url, na_values='none')

    # Preprocessing
    print(f"Data shape: {data.shape}")
    X = data.drop('credit_risk', axis=1)
    y = data['credit_risk'].map({'bad': 0, 'good': 1})

    num_cols = ['duration', 'amount', 'age']
    cat_cols = [i for i in X.columns if i not in num_cols]

    # Get indices of categorical columns for SMOTENC
    cat_cols_indices = [X.columns.get_loc(c) for c in cat_cols]

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
    pipeline = ImbPipeline([
        ('smotenc', SMOTENC(categorical_features=cat_cols_indices, random_state=42)),
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

    # Training on FULL dataset
    print("Training model on the FULL dataset (100% of data)...")
    pipeline.fit(X, y)
    print("Training complete.")

    # Save model
    output_file = 'production_xgboost_model.pkl'
    print(f"Saving production model to '{output_file}'...")
    joblib.dump(pipeline, output_file)
    print("Done.")

if __name__ == "__main__":
    main()
