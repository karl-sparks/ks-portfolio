DEFAULT_TIMEZONE = "Australia/Sydney"
DATA_FOLDER = "/workspaces/ks-portfolio/project-household-finances/data"
ACCOUNTS_FILEPATH = f"{DATA_FOLDER}/accounts.parquet"
TRANSACTIONS_FILEPATH = f"{DATA_FOLDER}/transactions.parquet"
BALANCE_FILEPATH = f"{DATA_FOLDER}/balance_data.parquet"
CBA_FILEPATH = f"{DATA_FOLDER}/cba_transaction_data.parquet"
CBA_CSV_FILEPATH = f"{DATA_FOLDER}/CBA_transaction_data.csv"
CBA_COLUMNS = ["DATE", "AMOUNT", "DESCRIPTION", "BALANCE"]

DATA_FOLDER_ID = "1UxF8qeTtoabXGuqO2S4MT68BKT9XtlXp"
ACCOUNTS_ID = "1x-3dmBxPxDTZapygRE0pcS_Vs80y18NH"
TRANSACTIONS_ID = "1E6CNPzF4c7Z5tb_OZJMG2Ha1ih-yRpjU"

BALANCE_SHEET_ID = "1j5xljQ4lCOwVIiwveX2Tx1PVsGhIpZN1DY-8ZJCnXkk"
BALANCE_DATA_ID = "1sHyXG0PQG1bjGzsIXK0O7CUR3Y1KY9LT"
BALANCE_RANGE = "balance_data!A2:G"

CBA_ID = "1KnrNbYMv4gaO7OMiVnUqHlofZb-hv3zQ"

UP_BANK_TRANSACTION_DATAFRAME_COLUMNS = {
    "account_id",
    "amount",
    "amount_in_base_units",
    "card_purchase_method",
    "cashback",
    "category",
    "created_at",
    "description",
    "foreign_amount",
    "hold_info",
    "message",
    "raw_text",
    "round_up",
    "settled_at",
    "status",
    "tags",
    "transaction_id",
    "updated_at",
}
