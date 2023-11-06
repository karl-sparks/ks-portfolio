import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from scipy.stats import randint, uniform

from data_funs import clean_log_data
from custom_trans import KMeansTransformer
import config

pipe = make_pipeline(
    StandardScaler(),
    KMeansTransformer(n_clusters=5, random_seed=2023),
    SelectPercentile(percentile=95),
    XGBRegressor(random_state=2023),
)

param_dist = {
    # Parameters for StandardScaler if needed; usually it does not need hyperparameter tuning
    # If you have parameters to tune for KMeansTransformer, they can be added here
    "kmeanstransformer__n_clusters": randint(2, 10),  # Example for KMeans clusters
    "kmeanstransformer__random_seed": randint(
        2023, 2040
    ),  # If you have a random_seed parameter
    # Parameters for SelectPercentile
    "selectpercentile__percentile": randint(1, 100),  # Varying the percentile
    # Parameters for XGBRegressor
    "xgbregressor__n_estimators": randint(50, 500),  # Number of trees in XGB
    "xgbregressor__max_depth": randint(3, 10),  # Depth of trees in XGB
    "xgbregressor__learning_rate": uniform(0.01, 0.3),  # Learning rate for XGB
    "xgbregressor__subsample": uniform(
        0.5, 0.5
    ),  # Subsample ratio of the training instance
    "xgbregressor__colsample_bytree": uniform(0.5, 0.5),  # Subsample ratio of columns
    "xgbregressor__min_child_weight": randint(
        1, 10
    ),  # Minimum sum of instance weight (hessian) needed in a child
    "xgbregressor__gamma": uniform(
        0, 5
    ),  # Minimum loss reduction required to make a further partition
    "xgbregressor__reg_alpha": uniform(0, 1),  # L1 regularization term on weights
    "xgbregressor__reg_lambda": uniform(1, 4),  # L2 regularization term on weights
}

if __name__ == "__main__":
    raw_train_logs = pd.read_csv(f"{config.DATA_DIR_PATH}/train_logs.csv")

    raw_train_scores = pd.read_csv(f"{config.DATA_DIR_PATH}/train_scores.csv")

    grouped_df = clean_log_data(raw_train_logs)

    training_data = pd.merge(grouped_df, raw_train_scores, on="id")

    train_X, test_X, train_y, test_y = train_test_split(
        training_data.drop(["id", "score"], axis=1),
        training_data["score"],
        train_size=0.8,
        stratify=training_data["score"],
        random_state=2023,
    )

    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=2000,  # Number of parameter settings that are sampled
        cv=3,  # Number of folds in cross-validation
        verbose=2,
        random_state=42,  # Seed for reproducibility
        n_jobs=-1,  # Number of jobs to run in parallel
        scoring="neg_mean_squared_error",
    )

    print("Starting Fit")
    random_search.fit(train_X, train_y)

    print(random_search.best_score_)
    print("=" * 20)
    print(random_search.best_params_)
    print("=" * 20)
    print(random_search.best_estimator_)

    preds = random_search.predict(test_X)

    score = mean_squared_error(test_y, preds, squared=False)

    print(f"Final test score: {score}")
