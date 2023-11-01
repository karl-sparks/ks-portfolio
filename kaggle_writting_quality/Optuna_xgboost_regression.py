import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
import optuna
import logging

logging.basicConfig(level=logging.DEBUG)


class CONFIG:
    DATA_DIR_PATH = "D:\dev\linking-writing-processes-to-writing-quality"
    COLS_TO_DROP = ["id", "score"]
    RANDOM_SEED = 2023
    TEST_SIZE = 0.2
    ESTIMATORS = [("clf", XGBRegressor(random_state=RANDOM_SEED))]
    CV = 4
    N_ITER = 200


class objective:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __call__(self, trial):
        train_x, valid_x, train_y, valid_y = train_test_split(
            self.data,
            self.target,
            stratify=self.target,
            test_size=0.25,
            random_state=CONFIG.RANDOM_SEED,
        )

        param = {
            "verbosity": 0,
            "objective": trial.suggest_categorical(
                "objective",
                [
                    "reg:squarederror",
                    "reg:squaredlogerror",
                ],
            ),
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 12)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        bst = xgb.XGBRegressor(**param, random_state=CONFIG.RANDOM_SEED)
        bst.fit(train_x, train_y)
        preds = bst.predict(valid_x)
        pred_labels = preds

        root_mean_squared_error = np.sqrt(mean_squared_error(valid_y, pred_labels))
        return root_mean_squared_error


if __name__ == "__main__":
    raw_train_logs = pd.read_csv(f"{CONFIG.DATA_DIR_PATH}/train_logs.csv")

    raw_train_scores = pd.read_csv(f"{CONFIG.DATA_DIR_PATH}/train_scores.csv")

    raw_train_scores["score"] = ((raw_train_scores["score"] * 2) - 1).astype(int)

    def num_unique_text(x):
        return len(set(x))

    agg_dict = {
        "event_id": "count",
        "action_time": ["mean", "sum"],
        "down_event": num_unique_text,
        "up_event": num_unique_text,
        "text_change": num_unique_text,
        "word_count": ["mean", "sum"],
    }

    grouped_df = raw_train_logs.groupby("id").agg(agg_dict)

    grouped_df.columns = ["_".join(col).strip() for col in grouped_df.columns.values]

    training_data = pd.merge(grouped_df, raw_train_scores, on="id")

    study = optuna.create_study(
        direction="minimize", storage="sqlite:///kaggle_xg_study.db"
    )

    study.optimize(
        objective(
            training_data.drop(columns=CONFIG.COLS_TO_DROP), training_data["score"]
        ),
        n_trials=CONFIG.N_ITER,
        timeout=600,
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
