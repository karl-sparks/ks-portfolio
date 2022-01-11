#Script used to scrap data from the Domain API

import domainAPIAccess as domain
import pandas as pd

def inital_query():
    postcodes = pd.read_csv(r'C:\dev\data\NSW_post_codes.csv', dtype={'post_code':'str', '0': 'int'})['post_code']

    for p in postcodes:
        domain.query_postcode(p)


def update_state(state = 'NSW'):
    last_week = pd.Timestamp.today() - pd.Timedelta(2, 'days')
    domain.query_API(state, listedSince=last_week.isoformat())
    

states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT']

for state in states:
    update_state(state) 

#update minimum each time it runs out of requests each day.
VIC_postcodes = [f'{i:04}' for i in range(3322, 4000)]

for pc in VIC_postcodes:
    domain.query_API(postcode=pc)