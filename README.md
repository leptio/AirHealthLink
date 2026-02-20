# AirHealthLink

AirHealthLink is a project dedicated to studying the link between economic condition and air pollution using publically-available levels of common pollutants (PM2.5).  It integrates environmental data from the EPA Air Quality System (AQS) and economic data from the U.S. Census Bureau to quantify disparities in air pollution exposure.

# Abstract

Using 615 counties with complete data:

A consistent negative association was observed between PM2.5 and income, showing an estimated decrease of approximately $1,350 in median household income per 1 μg/m³ increase in PM2.5 (p < 0.001).

Lower-income counties experience approximately 0.8 μg/m³ higher PM2.5 concentrations compared to higher-income counties.

These results provide quantitative evidence of economic disparities in air pollution exposure across U.S. counties.

![AirHealthLink Demo](https://github.com/leptio/AirHealthLink/raw/main/output.gif)
---


## Scope

Data is taken from all available counties on the [AQS API](https://aqs.epa.gov/aqsweb/documents/data_api.html) from January 1, 2022 to December 31, 2022. 

---
## Aggregation

Daily PM2.5 data was aggregated to county-level summaries:

Annual mean, median, standard deviation, 90th percentile,

Monthly averages,

Seasonal averages (DJF, MAM, JJA, SON)

Counts of days exceeding:

12 μg/m³,

25 μg/m³,

35 μg/m³,

---
## Statistical Analysis
AirHealthLink performs the following analyses automatically:

Correlation Analysis,

Pearson correlation,

Spearman rank correlation,

Ordinary Least Squares (OLS),

Robust regression (Huber),

Median quantile regression,


For income decile analysis:

ANOVA,

Kruskal–Wallis test,

Spearman rank trend test,

Monthly and seasonal regressions


Comparison of lowest and highest quintiles (extreme bin comparison) using:

Welch’s t-test,

Kolmogorov–Smirnov test,

Bootstrap confidence intervals

---

## Usage/Replication

1. Create a local clone of the repository on your machine.
2. [Sign up for an AQS API key](https://aqs.epa.gov/aqsweb/documents/data_api.html#signup) and a [U.S. Census API key](https://api.census.gov/data/key_signup.html) and put them in [src/private_keys.py](https://github.com/leptio/AirHealthLink/blob/main/src/private_keys.py), alongside the e-mail used to sign up for them.
3. Run /src/main.py.
