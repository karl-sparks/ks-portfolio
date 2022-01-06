import helperFunctions as scrapper
import time


url_base = 'https://www.domain.com.au/sale/?ssubs=0&postcode='

url = url_base + '2000'

NSW_urls = [url_base + f"{i:04}" for i in range(2000, 3000)]

delay_between_get_requests = 0.5

for url in NSW_urls:
  scrapper.scrap_domain_url(url, delay = delay_between_get_requests)
  time.sleep(delay_between_get_requests)

  scrapped_data = scrapper.read_scrapped_data()

scrapper.clean_write_data(scrapped_data)