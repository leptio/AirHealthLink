# AirHealthLink

**Quantifying the relationship between county-level air pollution and economic status across the United States.**

AirHealthLink pairs daily PM2.5 measurements from the EPA Air Quality System (AQS) with
income and poverty data from the U.S. Census Bureau's American Community Survey, then runs a
battery of statistical tests to measure how pollution exposure varies with economic condition.
It also renders the full year of PM2.5 data as an animated county-level choropleth.

![AirHealthLink Demo](https://github.com/leptio/AirHealthLink/raw/main/output.gif)

---

## Key Findings

Based on **615 U.S. counties** with complete PM2.5 and economic data for calendar year 2022:

| Finding | Estimate |
| --- | --- |
| Change in median household income per 1 μg/m³ increase in PM2.5 | **−$1,350** (p < 0.001) |
| Excess PM2.5 in lower-income vs. higher-income counties | **≈ 0.8 μg/m³** |
| Direction of association | Consistent and negative across every model tested |

The negative association holds under Pearson and Spearman correlation, OLS, robust (Huber)
regression, and median quantile regression, and persists across monthly and seasonal
breakdowns. Taken together these provide quantitative evidence of economic disparity in air
pollution exposure across U.S. counties.

> These are **observational, county-level** results. See [Limitations](#limitations).

---

## How It Works

The project is a four-stage pipeline. Each stage writes files that the next stage reads.

| # | Stage | Script | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 | Fetch pollution data | [`src/req_sender.py`](src/req_sender.py) | AQS API | `data/county_level_pm25.csv` |
| 2 | Fetch economic data | [`src/req_sender_census.py`](src/req_sender_census.py) | Census ACS API | `data/county_level_economic_status.csv` |
| 3 | Statistical analysis | [`src/analysis.py`](src/analysis.py) | both CSVs | `analysis_output_*.csv` |
| 4 | Visualization | [`src/main.py`](src/main.py) → [`src/to_gif.py`](src/to_gif.py) | PM2.5 CSV + shapefile | `plots/frames/*.png` → `output.gif` |

Stages 3 and 4 are independent of each other; both depend on stage 1.

---

## Repository Layout

```
src/
  req_sender.py          AQS client — daily PM2.5 (param 88101) for every county, 2022
  req_sender_census.py   Census ACS 5-year client — income and poverty variables
  analysis.py            PM25EconomicAnalyzer: aggregation, correlation, regression, group tests
  visualization.py       Visualization: per-day county choropleth via GeoPandas + Matplotlib
  main.py                Entry point for the visualization stage
  to_gif.py              Downscales frames and assembles output.gif
  private_keys.py        API credentials (placeholder values — see Setup)
data/
  shapefile/             Census TIGER/Line county boundaries (cb_2022_us_county_5m)
  county_level_economic_status.csv
  output/
analysis_output_*.csv    Analysis results committed at the repo root
output.gif               Rendered demo animation
```

---

## Requirements

Python 3.13 (earlier 3.x versions should work, but the committed bytecode targets 3.13).

```bash
pip install pandas numpy scipy statsmodels requests matplotlib seaborn geopandas shapely imageio pillow
```

GeoPandas pulls in GDAL/PROJ. If installation gives you trouble, conda tends to be smoother:
`conda install -c conda-forge geopandas`.

---

## Setup

1. Clone the repository.

2. Request the two API keys — both are free:
   - [AQS API key](https://aqs.epa.gov/aqsweb/documents/data_api.html#signup)
   - [U.S. Census API key](https://api.census.gov/data/key_signup.html)

3. Fill in [`src/private_keys.py`](src/private_keys.py) with your keys and the email address you
   registered with:

   ```python
   api_key: str = "your-aqs-key"
   api_key_census: str = "your-census-key"
   email: str = "you@example.com"
   ```

   This file is listed in `.gitignore`. It is tracked in the repo only as a placeholder — take
   care not to commit your real keys over it.

---

## Usage

**Run every command from the repository root.** Several scripts resolve data paths relative to
the current working directory, so running them from inside `src/` will fail to find inputs.

### 1. Fetch PM2.5 data

```bash
python src/req_sender.py
```

Walks all 51 state FIPS codes, enumerates each state's counties, and pulls daily PM2.5 for 2022.

> **This stage takes roughly 5–6 hours.** The script sleeps 6 seconds between county requests to
> stay within AQS rate limits, and there are ~3,100 counties. Output is flushed incrementally, so
> an interrupted run leaves a usable partial CSV.

### 2. Fetch economic data

```bash
python src/req_sender_census.py
```

Retrieves median household income (`B19013_001E`), per capita income (`B19301_001E`), and
population below poverty (`B17001_002E`) from the ACS 5-year estimates. Takes about two minutes.

### 3. Run the analysis

```bash
python src/analysis.py
```

Optional flags:

```bash
python src/analysis.py --pm25 data/county_level_pm25.csv \
                       --econ data/county_level_economic_status.csv \
                       --income_var B19013_001E
```

Prints findings to stdout as they are computed and writes three CSVs:

- `analysis_output_pm25_aggregates.csv` — per-county annual, monthly, and seasonal PM2.5 summaries
- `analysis_output_decile_stats.csv` — PM2.5 statistics grouped by income decile
- `data/output/analysis_output_merged_joined.csv` — the merged pollution + economic table

### 4. Render the animation

```bash
mkdir -p plots/frames
python src/main.py
python src/to_gif.py
```

`main.py` writes one PNG per day of 2022 into `plots/frames/`, then `to_gif.py` downscales them
and assembles `output.gif`.

> Frames are rendered at `dpi=3000` and are very large. Budget several GB of free disk space for
> the intermediate `plots/frames/` directory.

---

## Data Scope

- **Pollutant:** PM2.5, AQS parameter code `88101`
- **Period:** January 1 – December 31, 2022
- **Coverage:** all counties reporting to the AQS API across the 50 states plus DC
- **Economic data:** ACS 2022 5-year estimates, county level
- **Geometry:** Census TIGER/Line `cb_2022_us_county_5m` (1:5,000,000 scale)

### Aggregation

County-level daily PM2.5 is summarized into:

- **Annual:** mean, median, standard deviation, 90th percentile
- **Monthly:** per-month averages
- **Seasonal:** DJF, MAM, JJA, SON averages
- **Exceedance counts:** days above 12, 25, and 35 μg/m³

---

## Analytical Methodology

**Correlation and regression**

- Pearson correlation
- Spearman rank correlation
- Ordinary least squares (OLS)
- Robust regression (Huber)
- Median quantile regression

**Income decile evaluation**

- One-way ANOVA across deciles
- Kruskal–Wallis test
- Spearman rank trend test
- Monthly and seasonal regressions

**Extreme quintile comparison**

- Welch's t-test
- Kolmogorov–Smirnov test
- Bootstrap confidence intervals

Column detection is handled adaptively — [`analysis.py`](src/analysis.py) probes a list of
candidate names for date, measurement, state, county, and FIPS fields, so it tolerates
variations in AQS and Census output formats.

---

## Limitations

- **Ecological, not individual.** All results are county-level aggregates. A county-level
  association does not establish that lower-income *individuals* breathe more PM2.5 — inferring
  that from these data would be an ecological fallacy.
- **Correlational.** No causal identification strategy is used. Nothing here separates pollution
  driving down income from low income sorting into polluted areas, or from a confounder driving
  both.
- **Monitor coverage is uneven.** AQS reports only from counties with active monitors, which skew
  urban and more populous. The 615 analyzed counties are not a random sample of U.S. counties.
- **Single year.** 2022 only; no trend or panel analysis.

---

## Data Sources

- [EPA Air Quality System (AQS) API](https://aqs.epa.gov/aqsweb/documents/data_api.html)
- [U.S. Census Bureau ACS 5-Year API](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- [Census TIGER/Line Cartographic Boundary Files](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)
