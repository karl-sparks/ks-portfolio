import load_data as ld
import google_drive as gd
import cba_data as cd
import config as cfg


def main():
    ld.load_data_from_up_bank()
    ld.load_data_from_gsheets()
    cd.process_cba_transactions(
        input_filepath=cfg.CBA_CSV_FILEPATH,
        output_filepath=cfg.CBA_FILEPATH,
        col_names=cfg.CBA_COLUMNS,
    )

    print("backup data to google drive")
    gd.upload_or_update_file(cfg.ACCOUNTS_FILEPATH, cfg.ACCOUNTS_ID, cfg.DATA_FOLDER_ID)
    gd.upload_or_update_file(
        cfg.TRANSACTIONS_FILEPATH, cfg.TRANSACTIONS_ID, cfg.DATA_FOLDER_ID
    )
    gd.upload_or_update_file(
        cfg.BALANCE_FILEPATH, cfg.BALANCE_DATA_ID, cfg.DATA_FOLDER_ID
    )
    gd.upload_or_update_file(cfg.CBA_FILEPATH, cfg.CBA_ID, cfg.DATA_FOLDER_ID)


if __name__ == "__main__":
    main()
