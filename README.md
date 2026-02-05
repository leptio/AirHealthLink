# AirHealthLink

AirHealthLink is a project dedicated to studying the link between economic condition and air pollution using publically-available levels of common pollutants (PM2.5). 

![AirHealthLink Demo](https://github.com/leptio/AirHealthLink/raw/main/output.gif)
---


## Scope

Data is taken from all available counties on the [AQS API](https://aqs.epa.gov/aqsweb/documents/data_api.html) from January 1, 2022 to December 31, 2022. 

---


## Usage/Replication

1. Create a local clone of the repository on your machine.
2. [Sign up for an AQS API key](https://aqs.epa.gov/aqsweb/documents/data_api.html#signup) and a [U.S. Census API key](https://api.census.gov/data/key_signup.html) and put them in [src/private_keys.py](https://github.com/leptio/AirHealthLink/blob/main/src/private_keys.py), alongside the e-mail used to sign up for them.
3. Run /src/main.py.
