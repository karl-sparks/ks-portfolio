#This script holds the functions to scrap data.
import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import re
import time
import os

#Configuration and helper funcitons
price_regexp = '\$.*(?=<!)'
feature_regexp = '(?<=>).(?=<!)'
street_address_regexp = '(?<=data-testid="address-line1">).+(?=<!--)'
suburb_address_regexp = '(?<=<span>)[\w\s]+(?=</span>)'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}

scrapped_data_folder = 'scrapped_data' #local folder to store raw data
save_folder = 'clean_data' #s3 bucket to store cleaned data

def scrap_domain_page(soup):
  """
  Scraps a domain.com.au webpage.

  args:
    soup (BeautifulSoup): BeautifulSoup() of webpage to scrap

  returns:
    DataFrame containing scrapped data
  """
  raw_data = {'listed_price': [], 'beds': [], 'baths': [], 'cars': [],\
            'street': [], 'suburb': [], 'state': [], 'post_code': [],\
            'listing_agent': [], 'listing_agency': []}

  for listing in soup.find_all('div', class_=['css-hgk76f', 'css-rxp4mi']):
    
    if listing.find_all('div', class_='css-176pzbt'):
      continue

    re_price = re.search(price_regexp, str(listing.find('p', class_='css-mgq8yx')))
    if re_price:
      price = re_price.group()
    else:
      price = 'NA'

    features = ['NA', 'NA', 'NA']
    for num, feature in enumerate(listing.find_all('span', class_='css-lvv8is')):
      re_features = re.search(feature_regexp, str(feature))
    
      if re_features:
        features[num] = re_features.group()

    address_string = listing.find_all('h2', class_='css-bqbbuf')
    
    re_street_address = re.search(street_address_regexp, str(address_string))
    if re_street_address:
      street_address = re_street_address.group()
    else:
      street_address = 'NA'

    suburb = 'NA'
    state = 'NA'
    post_code = 'NA'
    re_suburb = re.findall(suburb_address_regexp, str(address_string))
    if re_suburb:
      suburb = re_suburb[0]
      if len(re_suburb) > 1:
        state = re_suburb[1]
        if len(re_suburb) > 2:
          post_code = re_suburb[2]

    re_agent = re.findall(suburb_address_regexp, str(listing.find_all('div', class_='css-1gaz9vo')))

    if re_agent:
      agent = re_agent[0]
      if len(re_agent) > 1:
        agency = re_agent[1]
      else:
        agency = 'NA'
    else:
      re_second_agent = re.findall(suburb_address_regexp, str(listing.find_all('div', class_='css-1o2z7mc')))
      
      if re_second_agent:
        agent = re_second_agent[0]
        if len(re_second_agent) > 1:
          agency = re_second_agent[1]
        else:
          agency = 'NA'
      else:
        agent = 'NA'
        agency = 'NA'

    raw_data['listed_price'].append(price)
    raw_data['beds'].append(features[0])
    raw_data['baths'].append(features[1])
    raw_data['cars'].append(features[2])
    raw_data['street'].append(street_address)
    raw_data['suburb'].append(suburb)
    raw_data['state'].append(state)
    raw_data['post_code'].append(post_code)
    raw_data['listing_agent'].append(agent)
    raw_data['listing_agency'].append(agency)

  df = pd.DataFrame(raw_data)
  df['date_scrapped'] = pd.Timestamp.today()

  return df

def scrap_domain_url(url, delay = 1, save_url = 'scrapped_data'):
  """
  Scrap a domain webpage from URL. Will check for next page of results and
  continue scrapping until there are no remaining pages.

  args:
  url (str): url of initial domain webpage.
  delay (float): number of seconds to delay between webpages. This is to avoid DDOS blocking.

  returns: df containing scrapped data
  """
  r = requests.get(url, headers = headers, timeout = 5)

  soup = BeautifulSoup(r.text, 'html.parser')

  if soup is None:
    return None

  df = scrap_domain_page(soup)
  
  url_id = re.search('(?<=&).+', url).group()

  print('scrapped ' + str(len(df)) + ' properties from ' + url)

  df.to_feather(os.path.join(scrapped_data_folder, str(pd.Timestamp.today().strftime("%Y_%m_%d_")) + url_id + '.feather'))

  next_url_match = soup.find('link', rel='next')
  if next_url_match:
    next_url = next_url_match['href']
    time.sleep(delay)
    scrap_domain_url(next_url)


def read_scrapped_data():
  dfs = []
  for file in os.listdir(scrapped_data_folder):
    df = pd.read_feather(os.path.join(scrapped_data_folder, file))
    dfs.append(df)

  return pd.concat(dfs, ignore_index=True)

def clean_write_data(raw_data, numeric_columns = ['beds', 'baths', 'cars']):
  """
  Clean and write raw_data.
  """
  raw_data['price'] = raw_data['listed_price'].apply(convert_price_to_dollars)

  for column in numeric_columns:
    raw_data.loc[raw_data[column] == "−", column] = '0'
    raw_data.loc[raw_data[column] == "NA", column] = '0'
    raw_data[column] = pd.to_numeric(raw_data[column])

  raw_data['has_price'] = raw_data['listed_price'] != 'NA'

  raw_data = raw_data[raw_data['street'] != 'NA']

  final_data = raw_data.drop_duplicates(['street', 'beds', 'post_code']).sort_values(['street', 'post_code']).reset_index(drop=True)

  save_file_name = os.path.join(save_folder, 'cleaned_data.feather')
  print('Saved cleaned data to ' + save_file_name)
  display(final_data)
  final_data.to_feather(save_file_name)

def convert_price_to_dollars(price):
  if price == 'NA':
    return pd.to_numeric('NA', errors='coerce')

  if re.search('[\d\s][mM]', price):
    clean_price = remove_non_numeric(price)
    return_price = pd.to_numeric(clean_price, errors='coerce') * 1000000
  elif re.search('[\d\s][kK]', price):
    clean_price = remove_non_numeric(price)
    return_price = pd.to_numeric(clean_price, errors='coerce') * 1000
  else:
    clean_price = remove_non_numeric(price)
    return_price = pd.to_numeric(clean_price, errors='coerce')

  return return_price

def remove_non_numeric(price):
  first_match = re.search('[\d.,]+', price)
  if first_match:
    return re.sub('[^\d]', '', first_match.group())
  else:
    return 'NA'