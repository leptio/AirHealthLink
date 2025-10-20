"""Retrieves data through requests to the AQS API."""
from time import sleep
from typing import List, Dict, Any
import pandas as pd
import requests
import private_keys

first_write: bool = True
api_key: str = private_keys.api_key
email: str = private_keys.email

#FIPS codes for all states
states: List[str] = [
    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19",
    "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
    "36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53",
    "54","55","56"
]

#PM2.5
pollutant: str = "88101"
start_date: str = "20220101"
end_date: str = "20221231"
base_url: str = "https://aqs.epa.gov/data/api/"
all_data: List[Any] = []

with open("data/county_level_pm25.csv", "w", newline='', encoding="utf-8") as f:
    for i, state in enumerate(states, start=1):
        print(f"Processing state {i}/{len(states)}: {state}")
        #Get list of counties for this state
        county_meta_url: str = f"https://aqs.epa.gov/data/api/list/countiesByState?email={email}&key={api_key}&state={state}"
        county_response: requests.Response = requests.get(county_meta_url)
        #Check if response failed and why
        if county_response.status_code != 200:
            print(county_response.status_code)
            print(county_response.text)
        county_json: Dict[str, List[Dict[str, str]]] = county_response.json()
        print(county_json)
        counties: List[str] = [c['code'] for c in county_json.get('Data',[])]

        for j, county in enumerate(counties, start=1):
            print(f"Processing county {j}/{len(counties)}: {county}")
            url = "https://aqs.epa.gov/data/api/dailyData/byCounty"
            params = {
                "email": email,
                "key": api_key,
                "param": pollutant,
                "bdate": start_date,
                "edate": end_date,
                "state": state,
                "county": county
            }
            response = requests.get(url, params=params)
            #Check if response failed and why
            if response.status_code != 200:
                print(response.status_code)
                print(response.text)
            data = response.json()
            print(f"Response status: {response.status_code}, JSON keys: {list(data.keys())}, Data length: {len(data.get('Data', []))}")
            if "Data" in data and data["Data"]:
                df = pd.DataFrame(data["Data"])
                df.to_csv(f, index=False, header=first_write)
                first_write = False
                f.flush()
            sleep(6)
