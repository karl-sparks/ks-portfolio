#Script used to scrap data from the Domain API

import domainAPIAccess as domain
import pandas as pd


postcodes = pd.read_csv(r'C:\dev\data\NSW_post_codes.csv', dtype={'post_code':'str', '0': 'int'})['post_code']

for p in postcodes:
    domain.query_postcode(p)
