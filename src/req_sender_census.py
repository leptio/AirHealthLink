"""Retrieves economic data for all U.S. counties using the U.S. Census API."""
from time import sleep
from typing import List, Dict, Any
import pandas as pd
import requests
import private_keys

# API credentials
api_key: str = private_keys.api_key_census

# Census API base URL
base_url: str = "https://api.census.gov/data/2022/acs/acs5"

# Variables to retrieve: median household income, per capita income, poverty rate
variables: List[str] = ["B19013_001E", "B19301_001E", "B17001_002E"]

# FIPS codes for all states
states: List[str] = [
    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19",
    "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
    "36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53",
    "54","55","56"
]

first_write: bool = True

with open("data/county_level_economic_status.csv", "w", newline='', encoding="utf-8") as f:
    for i, state in enumerate(states, start=1):
        print(f"Processing state {i}/{len(states)}: {state}")
        params = {
            "get": ",".join(["NAME"] + variables),
            "for": "county:*",
            "in": f"state:{state}",
            "key": api_key
        }
        response: requests.Response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(response.status_code)
            print(response.text)
            sleep(2)
            continue
        data = response.json()
        if len(data) < 2:
            print(f"No data for state {state}")
            continue
        df = pd.DataFrame(data[1:], columns=data[0])
        df.to_csv(f, index=False, header=first_write)
        first_write = False
        f.flush()
        sleep(2)
