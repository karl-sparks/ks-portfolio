import polars as pl
from upbankapi import Client
from dotenv import load_dotenv

from datetime import datetime
from zoneinfo import ZoneInfo
import os

import config as cfg
import google_drive as gd


pl.Config(tbl_cols=18)


def check_file_update(file_path):
    if not os.path.exists(file_path):
        return False

    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

    time_diff = datetime.now() - mtime

    return time_diff.days == 0


def load_data_from_up_bank(limit=None):
    if check_file_update(cfg.ACCOUNTS_FILEPATH) and check_file_update(
        cfg.TRANSACTIONS_FILEPATH
    ):
        print("Data already exists")
        return

    print("Reloading from Up Bank API")

    load_dotenv()

    # implicitly use the environment variable UP_TOKEN
    client = Client()
    accounts = client.accounts()

    account_df = pl.DataFrame()
    transaction_df = pl.DataFrame()

    # list accounts
    for account in accounts:
        account_df.vstack(
            pl.DataFrame(
                {
                    "account_id": account.id,
                    "name": account.name,
                    "balance": account.balance,
                    "balance_in_base_units": account.balance_in_base_units,
                    "balance_as_at": datetime.now(tz=ZoneInfo(cfg.DEFAULT_TIMEZONE)),
                    "currency": account.currency,
                    "type": str(account.type),
                    "ownership_type": str(account.ownership_type),
                    "created_at": account.created_at,
                }
            ),
            in_place=True,
        )

        # list transactions for account
        num_trans = 0
        print(f"Start load of transactions from {account.name}")
        for transaction in account.transactions(limit=limit, page_size=100):
            num_trans += 1
            if num_trans % 1000 == 0:
                print(f"Loaded {num_trans} from {account.name}")
            transaction_df = pl.concat(
                [
                    transaction_df,
                    pl.DataFrame(
                        {
                            "transaction_id": transaction.id,
                            "account_id": account.id,
                            "amount": transaction.amount,
                            "amount_in_base_units": transaction.amount_in_base_units,
                            "card_purchase_method": str(
                                transaction.card_purchase_method
                            ),
                            "cashback": str(transaction.cashback),
                            "category": str(transaction.category),
                            "created_at": transaction.created_at,
                            "description": transaction.description,
                            "foreign_amount": str(transaction.foreign_amount),
                            "hold_info": str(transaction.hold_info),
                            "message": transaction.message,
                            "raw_text": transaction.raw_text,
                            "round_up": str(transaction.round_up),
                            "status": transaction.status,
                            "settled_at": transaction.settled_at,
                            "tags": "".join(str(transaction.tags)),
                            "updated_at": datetime.now(
                                tz=ZoneInfo(cfg.DEFAULT_TIMEZONE)
                            ),
                        }
                    ),
                ],
                how="diagonal_relaxed",
            )

        print(
            f"Loaded {transaction_df.select(pl.len()).item()} transactions from {account_df.select(pl.len()).item()} accounts"
        )

    account_df = account_df.with_columns(
        [
            pl.col("balance_as_at").dt.convert_time_zone(cfg.DEFAULT_TIMEZONE),
            pl.col("created_at").dt.convert_time_zone(cfg.DEFAULT_TIMEZONE),
        ]
    )

    transaction_df = transaction_df.with_columns(
        [
            pl.col("created_at").dt.convert_time_zone(cfg.DEFAULT_TIMEZONE),
            pl.col("settled_at").dt.convert_time_zone(cfg.DEFAULT_TIMEZONE),
            pl.col("updated_at").dt.convert_time_zone(cfg.DEFAULT_TIMEZONE),
        ]
    )

    print(
        f"Saving {transaction_df.select(pl.len()).item()} transactions in {account_df.select(pl.len()).item()} accounts"
    )

    account_df.write_parquet(file=cfg.ACCOUNTS_FILEPATH)
    transaction_df.write_parquet(file=cfg.TRANSACTIONS_FILEPATH)


def load_data_from_gsheets():
    sheets_df = pl.DataFrame(gd.get_sheet(cfg.BALANCE_SHEET_ID, cfg.BALANCE_RANGE))

    sheets_df.write_parquet(file=cfg.BALANCE_FILEPATH)


if __name__ == "__main__":
    load_data_from_up_bank()
    load_data_from_gsheets()
