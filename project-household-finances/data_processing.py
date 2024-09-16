"""
This module contains the code needed to process raw files into consumption layer automatically
"""

import polars as pl
import os
from datetime import datetime

import logging

import config as cfg


def validate_up_data(up_transaction_data: pl.DataFrame) -> None:
    """
    Function to validate that the provided DataFrame is consistent with the UP Bank transaction DataFrame
    """

    if not isinstance(up_transaction_data, pl.DataFrame):
        raise TypeError(
            f"Expected a Polars DataFrame, but got {type(up_transaction_data).__name__} instead."
        )

    if (
        len(
            cfg.UP_BANK_TRANSACTION_DATAFRAME_COLUMNS - set(up_transaction_data.columns)
        )
        != 0
    ):
        raise ValueError(f"Data not consistent with UP Transaction data")


def add_fy_to_data_frame(data_frame: pl.DataFrame) -> pl.DataFrame:
    if "created_at" not in data_frame.columns:
        raise ValueError("created_at column missing; unable to create FY column")

    return (
        data_frame.with_columns(pl.col("created_at").dt.year().alias("year"))
        .with_columns(
            pl.when(pl.col("created_at").dt.quarter() <= 2)
            .then(pl.col("year"))
            .otherwise(pl.col("year") + 1)
            .alias("FY"),
        )
        .with_columns(
            (pl.lit("FY") + (pl.col("FY") - 2000).cast(pl.Utf8)).alias(
                "FY"
            )  # Convert to str FY + year end number to make clear its a FY
        )
    )


def generate_savings_data(up_transaction_data: pl.DataFrame) -> pl.DataFrame:
    """
    Function to process the up transaction data, to generate per FY savings data.

    Parameters:
    up_transaction_data (pl.DataFrame): Input Polars DataFrame with up transaction data.

    Returns:
    pl.DataFrame: A Polars DataFrame containing the savings data.
    """

    validate_up_data(up_transaction_data)

    results = (
        add_fy_to_data_frame(up_transaction_data)
        .filter(
            pl.col("transaction_id")
            != "3bb51c9d-5893-43ee-824c-41256803d5c3"  # Remove home loan transaction because its oneside in the data
        )
        .group_by("FY")
        .agg([pl.lit("Savings").alias("name"), pl.sum("amount").alias("value")])
        .sort("FY")
        .with_columns(pl.lit(datetime.now()))
    )

    logging.info(f"Created savings data with shape: {results.shape}")

    return results


def generate_employment_income_data(up_transaction_data: pl.DataFrame) -> pl.DataFrame:
    """
    Function to process the up transaction data, to generate per FY savings data.

    Parameters:
    up_transaction_data (pl.DataFrame): Input Polars DataFrame with up transaction data.

    Returns:
    pl.DataFrame: A Polars DataFrame containing the savings data.
    """

    validate_up_data(up_transaction_data)

    up_transaction_data.with_columns(
        pl.when(
            pl.col("description").is_in(karl_employers)
            | pl.col("description").is_in(flora_employers)
        )
        .then(pl.col("description"))
        .otherwise(None)
        .alias("employer")
    ).filter(pl.col("employer").is_not_null()).select(
        pl.col("created_at").cast(pl.Date).alias("date"),
        pl.col("employer").alias("name"),
        pl.col("amount").alias("value"),
    )
