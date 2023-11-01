#This contains functions to access domain's API
import re
import requests
import pandas as pd
import numpy as np
import time
import os

#Data cleaning configurations. 
cleaning_config = {
    'fillnavalues': { # Columns and values to replace NAs
    'propertyDetails.bathrooms': 0,
    'propertyDetails.bedrooms': 0,
    'propertyDetails.carspaces': 0},
    'min_price': 25000, # Minimum and maximum price values to allow. If outside range will change price to NaN.
    'max_price': 10000000 # This check is to remove some extreme outliners for data quality. Largely due to odd listings.
    }


def send_query(state, postcode, pageNumber, listingType, listedSince):


    url = 'https://api.domain.com.au/v1/listings/residential/_search'
    headers = {
        'X-API-Key': 'key_f3f6f4e711c05ddbb78dac3c37858393',
        'accept': 'application/json'
    }

    post_request = {
        "listingType":listingType,
        "pageSize":100,
        "pageNumber": pageNumber,
        "propertyTypes":"",
        "minBedrooms":"",
        "maxBedrooms":"",
        "minBathrooms":"",
        "maxBathrooms":"",
        "locations":[
            {
                "state": state,
                "region":"",
                "area":"",
                "suburb":"",
                "postCode":postcode,
                "includeSurroundingSuburbs":False
            }
        ],
        "listedSince": listedSince
    }

    return requests.post(url=url, json=post_request, headers=headers)

def extract_listings(json_response):


    listings = []
    
    for l in json_response:
        if l['type'] == 'Project':
            for t in l['listings']:
                listings.append(t)
        else:
            listings.append(l['listing'])

    return listings


def has_remaining_quota(headers):
    daily_remaining = 0

    if 'X-Quota-PerDay-Remaining' in headers:
        daily_remaining = headers['X-Quota-PerDay-Remaining']

    return daily_remaining > 0

def quota_wait(headers):
    """
    Returns number of seconds to sleep before retrying API. 

    headers: 429 response headers object   
    """
    if 'Retry-After' in headers:
        return int(headers['Retry-After'])
    else:
        raise KeyError(f'Can not find Retry-After key in header: {headers.keys()}')

def check_response_status(response):
    if 'X-Quota-PerDay-Remaining' in response.headers:
        return response.headers["X-Quota-PerDay-Remaining"]
    elif 'Retry-After' in response.headers: 
        seconds_to_sleep = quota_wait(response.headers)
        raise ConnectionRefusedError(f'Reached Quota Maximum. Required wait period is {seconds_to_sleep/60/60:.2f} hours')
    else:
        raise(ConnectionError(f'Status Code is {response.status_code} \n Headers contain: \n {response.headers}')) 


def write_output(json_data, id, wd_path):
    df = pd.json_normalize(json_data)

    today_str = str(pd.Timestamp.today().strftime('%Y_%m_%d_raw_data'))
    output_folder = os.path.join(wd_path, today_str)

    if os.path.exists(output_folder):
        df.to_feather(os.path.join(output_folder, id + '_raw_data.feather'))
    else:
        os.mkdir(output_folder)
        df.to_feather(os.path.join(output_folder, id + '_raw_data.feather'))

def query_API(wd_path, state = "", postcode = "", listingType = "Sale", listedSince = ""):
    initial_response = send_query(state, postcode, 1, listingType, listedSince)

    remaining_quota = check_response_status(initial_response)

    if 'X-Total-Count' in initial_response.headers:
        total_listings = int(initial_response.headers['X-Total-Count'])
    else:
        total_listings = 0
    
    number_of_pages = total_listings // 100 + (total_listings % 100 > 1)

    if total_listings == 0:
        print(f'No listings for {state} - {postcode} - {listingType} - {listedSince} : {remaining_quota} remaining quota today.')
        return None
    else:
        print(f'Querying {state} - {postcode} - {listingType} - {listedSince}: {remaining_quota} remaining quota today.')

    if number_of_pages > 10:
        print('Warning! more than 10 pages of results.')
        print('Will miss properties listed on pages > 10')
        number_of_pages = 10

    print(f'Total number of listings: {total_listings}, number of pages: {number_of_pages}')

    write_output(extract_listings(initial_response.json()), f'{state}_{postcode}_{listingType}_{pd.Timestamp.today().strftime("%Y_%m_%d")}_pg_1', wd_path)

    for i in range(2, number_of_pages + 1):
        loop_response = send_query(state, postcode, i, listingType, listedSince)

        remaining_quota = check_response_status(loop_response)
        print(f'Querying page {i} of {number_of_pages}, {remaining_quota} remaining quota today.')

        write_output(extract_listings(initial_response.json()), f'{state}_{postcode}_{listingType}_{pd.Timestamp.today().strftime("%Y_%m_%d")}_pg_{i}', wd_path)


def concat_raw_data(rawpath):
    dfs = []
    for file in os.listdir(rawpath):
        df = pd.read_feather(os.path.join(rawpath, file))
        df['query'] = file
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def convert_price_to_dollars(price):
    if type(price) != str:
        print('Expected str, instead got ' + str(type(price)) + ' - ' + str(price))
        return np.nan

    if not re.search('\$', price):
        return np.nan

    
    if re.search('[\d\s][kK](?=[\s,\.]|$)', price):
        clean_price = remove_non_numeric(price)
        return_price = pd.to_numeric(clean_price, errors='coerce') * 1000
    elif re.search('[\d\s][mM](?![^iI\s,\.]|IN|in|In|iN)', price):
        clean_price = remove_non_numeric(price)
        return_price = pd.to_numeric(clean_price, errors='coerce') * 1000000
    else:
        clean_price = remove_non_numeric(price)
        return_price = pd.to_numeric(clean_price, errors='coerce')

    return return_price

def remove_non_numeric(price):
    first_match = re.search('\$[\d\.,\s]+', price)
    if first_match:
        return re.sub('[\$,\s]', '', first_match.group())
    else:
        return np.nan


def create_price_column(act_price, infer_price):
    act_price.fillna(infer_price)

def load_all_raw_data(wd_path):
    folders_raw_data = [os.path.join(wd_path, folder) for folder in os.listdir(wd_path)]

    return [concat_raw_data(folder) for folder in folders_raw_data]

def clean_data(wd_path: str) -> pd.DataFrame:
    df = pd.concat(load_all_raw_data(wd_path), ignore_index=True).drop_duplicates('id')

    df['converted_price'] = df['priceDetails.displayPrice'].apply(convert_price_to_dollars)

    df['price'] = df['priceDetails.price'].fillna(df['converted_price'])
    
    df['dateListed'] = pd.to_datetime(df['dateListed'])

    df.fillna(cleaning_config['fillnavalues'], inplace=True)

    df.loc[(df['price'] < cleaning_config['min_price']) | (df['price'] > cleaning_config['max_price']), 'price'] = np.nan

    return df.reset_index(drop=True)

def update_state(wd_path, state = 'NSW'):
    last_week = pd.Timestamp.today() - pd.Timedelta(2, 'days')
    query_API(wd_path, state=state, listedSince=last_week.isoformat())