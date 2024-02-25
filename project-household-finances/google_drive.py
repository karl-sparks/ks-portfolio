import os.path

from typing import List
from dataclasses import dataclass

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from dotenv import load_dotenv

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive"]


@dataclass
class gdrive_list_item:
    id: str
    name: str

    def __str__(self):
        return f"{self.name} - id: {self.id}"


def get_gdrive() -> Resource:
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
        service = build("drive", "v3", credentials=creds)
        print(f"Successfully connected to drive")
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f"An error occurred: {error}")

    return service


def upload_file(upload_data_path: str, folder_id: str) -> bool:
    if not os.path.exists(upload_data_path):
        print(f"File not found: {upload_data_path}")
        return False

    service = get_gdrive()

    # first, define file metadata, such as the name and the parent folder ID
    file_metadata = {"name": upload_data_path, "parents": [folder_id]}
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


def list_files(folder_id: str) -> List[gdrive_list_item]:
    service = get_gdrive()

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


if __name__ == "__main__":
    load_dotenv()
    files = list_files(os.getenv("DATA_FOLDER"))

    for file in files:
        print(file)
