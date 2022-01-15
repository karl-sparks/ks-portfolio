#Script used to scrap data from the Domain API each day

import domainAPIAccess as domain
import pandas as pd
import os

wd_path = r'C:\dev\data\daily_scrap'
main_data_path = r'C:\dev\data\main_data\domain_data_cleaned.feather'
postcode_path = r'C:\dev\data\checked_postcodes.feather'

# First check each state for newly listed properties 
states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT']

for state in states:
    try:
        domain.update_state(wd_path, state)
    except ConnectionRefusedError as err:
        print(f'Unable to query {state} for new listings')
        print(err)
        break

# Next check all postcodes one at a time.
pc_df = pd.read_feather(postcode_path)
unchecked_postcodes = pc_df[~pc_df['Checked']].postcode
print(f'Checking {len(pc_df) - pc_df["Checked"].sum()} postcodes: starting with postcode {unchecked_postcodes[0]}')
for pc in unchecked_postcodes:
    try:
        domain.query_API(wd_path, postcode=pc)
        pc_df.loc[pc_df['postcode'] == pc, 'Checked'] = True # Since this is only reached if the pc is queried successfully, by default its False.
    except ConnectionRefusedError as err:
        print(f'Ran out of daily queries, {len(pc_df) - pc_df["Checked"].sum()} remaining postcodes for checking.')
        print(err)
        break
    finally: 
        pc_df.to_feather(postcode_path)

print('Cleaning and saving raw data. This part usually takes > 20 seconds.')
df = domain.clean_data(wd_path)
df.to_feather(main_data_path)
print(f'{len(df):,} properties succesfully saved to: {main_data_path}')