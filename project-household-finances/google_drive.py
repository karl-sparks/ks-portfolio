import os.path

from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

import config as cfg
from gdrive_data_models import gdrive_list_item, gsheets_balance_data

# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def get_gservice(service_type: str) -> Resource:
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build(
            service_type, "v3" if service_type == "drive" else "v4", credentials=creds
        )
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f"An error occurred: {error}")

    return service


def upload_or_update_file(
    upload_data_path: str, file_id: Optional[str], folder_id: Optional[str]
) -> bool:
    if file_id:
        print("Trying file update", os.path.basename(upload_data_path), file_id)
        return update_file(upload_data_path, file_id)
    elif folder_id:
        print("Trying file upload", os.path.basename(upload_data_path), folder_id)
        return upload_file(upload_data_path, folder_id)
    print("Must supply either folder_id or file_id to upload or update a file")
    return False


def upload_file(upload_data_path: str, folder_id: str) -> bool:
    if not os.path.exists(upload_data_path):
        print(f"Upload file not found: {upload_data_path}")
        return False

    service = get_gservice("drive")

    # first, define file metadata, such as the name and the parent folder ID
    file_metadata = {"name": os.path.basename(upload_data_path), "parents": [folder_id]}
    media = MediaFileUpload(upload_data_path)

    try:
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
    except HttpError as error:
        print(f"An error occured: {error}")
        return False

    print("File created, id:", file.get("id"))
    return True


def update_file(upload_data_path: str, file_id: str) -> bool:
    if not os.path.exists(upload_data_path):
        print(f"Upload file not found: {upload_data_path}")
        return False

    service = get_gservice("drive")

    media = MediaFileUpload(upload_data_path)

    try:
        file = (
            service.files()
            .update(fileId=file_id, media_body=media, fields="id")
            .execute()
        )
    except HttpError as error:
        print(f"An error occured: {error}")
        return False

    print("File updated, id:", file.get("id"))
    return True


def list_files(folder_id: str) -> List[gdrive_list_item]:
    service = get_gservice("drive")

    results = (
        service.files()
        .list(
            pageSize=10,
            q=f"'{folder_id}' in parents",
            fields="nextPageToken, files(id, name)",
        )
        .execute()
    )

    items = results.get("files", [])

    return_items = []

    if not items:
        print("No files found.")
        return return_items

    for item in items:
        return_items.append(gdrive_list_item(id=item["id"], name=item["name"]))

    print(f"Found {len(return_items)} items")
    return return_items


def get_sheet(sheet_id: str, sheet_range: str) -> List[gsheets_balance_data]:
    service = get_gservice("sheets")

    result = (
        service.spreadsheets()
        .values()
        .get(
            spreadsheetId=sheet_id,
            range=sheet_range,
        )
        .execute()
    )

    values = result.get("values", [])

    return_items = []

    for record_id, type, holder, date, balance, currency, balance_aud in values:
        row_item = gsheets_balance_data(
            RECORD_ID=record_id,
            TYPE=type,
            HOLDER=holder,
            DATE=date,
            BALANCE=balance,
            CURRENCY=currency,
            BALANCE_AUD=balance_aud,
        )
        return_items.append(row_item)

    print(f"Extracted {len(return_items)} rows from gsheet")
    return return_items


if __name__ == "__main__":
    files = list_files(cfg.DATA_FOLDER_ID)

    for file in files:
        print(file)
