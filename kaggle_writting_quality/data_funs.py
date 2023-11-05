import numpy as np
import logging

import config


def clean_log_data(log_data):
    logging.info("Cleaning log data")

    def num_unique_text(x):
        return len(set(x))

    def share_of_input(x):
        return np.sum(x == "Input") / len(x)

    def share_of_deletions(x):
        return np.sum(x == "Remove/Cut") / len(x)

    def first_quartile(series):
        return series.quantile(0.25)

    def third_quartile(series):
        return series.quantile(0.75)

    def production_rate(x):
        return (x.iloc[-1] - x.iloc[0]) * 60 / (x.index[-1] - x.index[0])

    def pause_proportion(x):
        return (x > config.PAUSE_THRESHOLD).sum() / len(x)

    def revision_rate(x):
        return (x.isin(["Remove/Cut", "Replace"])).sum() / len(x)

    def movement_rate(x):
        return x.str.startswith("Move From").sum() / len(x)

    def paste_rate(x):
        return (x == "Paste").sum() / len(x)

    agg_dict = {
        "event_id": "count",
        "activity": [
            share_of_input,
            share_of_deletions,
            revision_rate,
            movement_rate,
            paste_rate,
        ],
        "action_time": [
            "mean",
            "sum",
            "std",
            "min",
            "max",
            first_quartile,
            third_quartile,
        ],
        "down_event": num_unique_text,
        "up_event": num_unique_text,
        "text_change": num_unique_text,
        "word_count": [
            "mean",
            "sum",
            "std",
            "min",
            "max",
            first_quartile,
            third_quartile,
            production_rate,
        ],
        "IKI": [pause_proportion, "mean", "sum", "std", "min", "max"],
    }

    log_data["IKI"] = (
        log_data.groupby("id")["down_time"].shift(-1) - log_data["up_time"]
    )

    grouped_df = log_data.groupby("id").agg(agg_dict)

    grouped_df.columns = ["_".join(col).strip() for col in grouped_df.columns.values]

    logging.info("Finished cleaning log data")

    return grouped_df
