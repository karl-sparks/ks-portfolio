# Household finances data scripts
This folder contains a series of scripts used to extract data and run analysis on my household fincances

# Load data and update Google Drive backup
Run the `load_data.py` script to load data from source systems and update the Google Drive back up.

## Data extraction sub-scripts
`google_drive.py` has the functions needed to load data from or to Google Drive
`load_data.py` has the functions to load data from Up Bank and process the CBA transaction data.

**Note**: CBA data was a csv export manually grabbed from CBA's Netbank website, not the App.

# Google Drive Setup
To set up access to Google drive you need to:

 1. Set up Google Cloud project and enable the Drive API
 2. Set up your email as a test user.
 3. Create a OAuth 2.0 Client for this script.
 4. Download client secret and name it 'credentials.json'. Add this to you working directory.

 # Up Bank API Key
 Get your Up Bank API key from: [https://developer.up.com.au/]
