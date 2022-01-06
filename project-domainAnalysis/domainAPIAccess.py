#This contains functions to access domain's API
import requests
import pandas as pd
import time
import os

def query_domain_api(postcode, pageNumber, listingType):


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
                "state":"",
                "region":"",
                "area":"",
                "suburb":"",
                "postCode":postcode,
                "includeSurroundingSuburbs":False
            }
        ]
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
    if response.status_code == 429:
        seconds_to_sleep = quota_wait(response.headers)
        print(f'Reached Quota Maximum. Sleeping for {seconds_to_sleep/60/60:.2f} hours')
        time.sleep(seconds_to_sleep)
    elif response.status_code != 200:
        raise(ConnectionError(f'Status Code is {response.status_code}'))
    else:
        return response.headers["X-Quota-PerDay-Remaining"]

def write_output(json_data, id):
    df = pd.json_normalize(json_data)

    today_str = str(pd.Timestamp.today().strftime('%Y_%m_%d_raw_data'))
    output_folder = os.path.join(r'C:\dev\data', today_str)

    if os.path.exists(output_folder):
        df.to_feather(os.path.join(output_folder, id + '_raw_data.feather'))
    else:
        os.mkdir(output_folder)
        df.to_feather(os.path.join(output_folder, id + '_raw_data.feather'))

def query_postcode(postcode, listingType = 'Sale'):
    initial_response = query_domain_api(postcode, 1, listingType)

    remaining_quota = check_response_status(initial_response)

    total_listings = int(initial_response.headers['X-Total-Count'])
    number_of_pages = total_listings // 100 + (total_listings % 100 > 1)

    if total_listings == 0:
        print(f'No listings for postcode: {postcode}, {remaining_quota} remaining quota today.')
        return None
    else:
        print(f'Querying postcode: {postcode}, listingType: {listingType}, {remaining_quota} remaining quota today.')

    if number_of_pages > 10:
        print('Warning! more than 10 pages of results.')
        print('Will miss properties listed on pages > 10')
        number_of_pages = 10

    print(f'Total number of listings: {total_listings}, number of pages: {number_of_pages}')

    write_output(extract_listings(initial_response.json()), f'postcode_{postcode}_pg_1')

    for i in range(2, number_of_pages + 1):
        loop_response = query_domain_api(postcode, i, listingType)

        remaining_quota = check_response_status(loop_response)
        print(f'Querying page {i} of {number_of_pages}, {remaining_quota} remaining quota today.')

        write_output(extract_listings(initial_response.json()), f'postcode_{postcode}_pg_{i}')

    
