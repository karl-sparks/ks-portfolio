from pydantic import BaseModel, Field
from enum import Enum
from typing import List
from datetime import date


class CurrencyEnum(str, Enum):
    AUD = "AUD"


class gdrive_list_item(BaseModel):
    id: str
    name: str

    def __str__(self):
        return f"{self.name} - id: {self.id}"


class gsheets_balance_data(BaseModel):
    RECORD_ID: int
    TYPE: str
    HOLDER: str
    DATE: date
    BALANCE: float
    CURRENCY: CurrencyEnum
    BALANCE_AUD: float
