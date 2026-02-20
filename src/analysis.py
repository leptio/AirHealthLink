"""
pm25_economic_analyzer.py

Class that performs analysis correlating daily PM2.5 (county-level, day-by-day)
with county-level economic status (ACS variables). 

Usage example:
    analyzer = PM25EconomicAnalyzer(pm25_csv="data/county_level_pm25.csv",
                                    econ_csv="data/county_level_economic_status.csv",
                                    econ_income_var="B19013_001E") 
    analyzer.run_all()

"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from datetime import datetime
from pandas import DataFrame, Series
import warnings

warnings.filterwarnings("ignore")


class PM25EconomicAnalyzer:
    """
    Analyze the relationship between PM2.5 (daily, county-level) and county economic status.

    pm25_csv: path to daily PM2.5 CSV (AQS daily style)
    econ_csv: path to county-level economic CSV (Census ACS output)
    econ_income_var: the census variable to use as primary income measure 
    pm25_df: raw pm2.5 dataframe after loading
    econ_df: raw economic dataframe after loading
    joined: merged dataframe with aggregated PM2.5 and economic vars
    results: dict to store analysis outputs and summary tables
    """

    def __init__(
        self,
        pm25_csv: str,
        econ_csv: str,
        econ_income_var: str = "B19013_001E",
        econ_vars: Optional[List[str]] = None,
        date_col_candidates: Optional[List[str]] = None,
        mean_col_candidates: Optional[List[str]] = None,
    ):
        """
        Initialize the analyzer. Only loads file paths; should call load_data() to read files

        pm25_csv: path to the PM2.5 daily CSV (rows = daily measurements per county)
        econ_csv: path to county economic CSV (one row per county)
        econ_income_var: primary economic variable name in econ_csv to correlate with
        econ_vars: additional econ variable names to keep
        date_col_candidates: optional override to specify which columns to look for 
        mean_col_candidates: optional override for which columns contain PM2.5 readings
        """
        self.pm25_csv = pm25_csv
        self.econ_csv = econ_csv
        self.econ_income_var = econ_income_var
        self.econ_vars = econ_vars or [econ_income_var]
        self.date_col_candidates = date_col_candidates or [
            "date_local",
            "date",
            "sample_date",
            "date_observed",
            "Date",
            "datetime",
        ]
        self.mean_col_candidates = mean_col_candidates or [
            "arithmetic_mean",
            "sample_measurement",
            "value",
            "daily_mean",
            "sample_value",
        ]
        self.state_col_candidates = ["state_code", "state", "stateFIPS",
                                     "state_fips", "state_code_new"]
        self.county_col_candidates = ["county_code", "county", "countyFIPS",
                                      "county_fips", "county_code_new"]
        self.fips_col_candidates = ["fips", "FIPS", "countyFIPS", "GEOID",
                                    "geoid", "COUNTYFP"]
        self.pm25_df: Optional[DataFrame] = None
        self.econ_df: Optional[DataFrame] = None
        self.joined: Optional[DataFrame] = None
        self.results: Dict[str, Any] = {}

    def load_data(self) -> None:
        """Load datasets and normalize column names."""
        pm = pd.read_csv(self.pm25_csv, dtype=str, low_memory=False)
        econ = pd.read_csv(self.econ_csv, dtype=str, low_memory=False)

        # detect date column
        date_col = self._detect_column(pm.columns.tolist(), self.date_col_candidates)
        # detect mean column
        mean_col = self._detect_column(pm.columns.tolist(), self.mean_col_candidates)
        # detect state/county columns or fips
        fips_col = self._detect_column(pm.columns.tolist(), self.fips_col_candidates)
        state_col = self._detect_column(pm.columns.tolist(), self.state_col_candidates)
        county_col = self._detect_column(pm.columns.tolist(), self.county_col_candidates)

        print("Detected columns in PM2.5 file:")
        print("  date_col =", date_col)
        print("  mean_col =", mean_col)
        print("  fips_col =", fips_col)
        print("  state_col =", state_col, "; county_col =", county_col)

        pm2 = pm.copy()
        # parse dates
        if date_col:
            pm2["date"] = pd.to_datetime(pm2[date_col], errors="coerce")
        else:
            pm2["date"] = pd.to_datetime(
                pm2.select_dtypes(include=["object"]).apply(
                    lambda x: x.astype(str)), errors="coerce")
            if pm2["date"].isna().all():
                pm2["date"] = pd.NaT

        if mean_col:
            pm2["pm25"] = pd.to_numeric(pm2[mean_col], errors="coerce")
        else:
            # try common numeric columns
            numeric_cols = pm.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                pm2["pm25"] = pd.to_numeric(pm[numeric_cols[0]], errors="coerce")
            else:
                pm2["pm25"] = np.nan

        if fips_col and fips_col in pm2.columns:
            pm2["fips"] = pm2[fips_col].str.zfill(5)
        elif state_col and county_col and state_col in pm2.columns and county_col in pm2.columns:
            pm2["fips"] = pm2[state_col].str.zfill(2) + pm2[county_col].str.zfill(3)
        else:
            # try columns named state_code and county_code in numeric forms
            maybe_state = self._detect_column(pm.columns.tolist(), [
                "state_code", "STATE", "STATE_CODE"])
            maybe_county = self._detect_column(pm.columns.tolist(), [
                "county_code", "COUNTY", "COUNTY_CODE"])
            if maybe_state and maybe_county:
                pm2["fips"] = pm2[maybe_state].str.zfill(2) + pm2[maybe_county].str.zfill(3)
            else:
                # if nothing, add index-based placeholder and warn
                pm2["fips"] = None
                print("WARNING: Could not construct county FIPS for PM2.5 rows.")

        pm_keep = ["fips", "date", "pm25"]
        for c in pm_keep:
            if c not in pm2.columns:
                pm2[c] = None

        econ2 = econ.copy()
        econ_fips_col = self._detect_column(econ2.columns.tolist(), self.fips_col_candidates)
        if econ_fips_col:
            econ2["fips"] = econ2[econ_fips_col].str.zfill(5)
        else:
            # Census outputs standardization: try state + county
            state_col_e = self._detect_column(econ2.columns.tolist(), [
                "state", "STATE", "state_code", "STATEFP", "STATEFIPS"])
            county_col_e = self._detect_column(econ2.columns.tolist(), [
                "county", "COUNTY", "county_code", "COUNTYFP", "COUNTYFIPS"])
            if state_col_e and county_col_e:
                econ2["fips"] = econ2[state_col_e].str.zfill(2) + econ2[county_col_e].str.zfill(3)
            else:
                # try GEOID / NAME
                geoid_c = self._detect_column(econ2.columns.tolist(), ["GEOID", "geoid", "Geoid"])
                if geoid_c:
                    econ2["fips"] = econ2[geoid_c].astype(str).str.zfill(5)
                else:
                    econ2["fips"] = None
                    print("WARNING: Could not construct county FIPS for econ file.")

        # convert selected econ columns to numeric
        for var in self.econ_vars:
            if var in econ2.columns:
                econ2[var] = pd.to_numeric(
                    econ2[var].str.replace(",", "").replace("nan", ""), errors="coerce")
            else:
                # keep column as NaN if not present
                econ2[var] = np.nan

        self.pm25_df = pm2[pm_keep].copy()
        self.econ_df = econ2[["fips"] + self.econ_vars + [
            c for c in econ2.columns if c in ("NAME", "name")]].copy()

        print("After normalization:")
        print("  PM2.5 sample rows:", self.pm25_df.shape)
        print("  Econ sample rows:", self.econ_df.shape)
        print("  Econ vars present:", [v for v in self.econ_vars if v in self.econ_df.columns])

    def _detect_column(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        """
        Find the best matching column name among candidates (case-insensitive substring match).
        Returns the first matching column name or None.
        """
        cols_lower = {c.lower(): c for c in columns}
        for cand in candidates:
            cand_lower = cand.lower()
            if cand in columns:
                return cand
            if cand_lower in cols_lower:
                return cols_lower[cand_lower]
        for col in columns:
            for cand in candidates:
                if cand.lower() in col.lower():
                    return col
        return None

    def aggregate_pm25(self) -> DataFrame:
        """
        Aggregate daily pm25 into multiple summary forms per county:
        annual_mean (mean over entire available date range)
        monthly_means (DataFrame with county x month (1-12) mean pm2.5)
        seasonal_means (seasonal groups (DJF, MAM, JJA, SON) means)
        extremes (per-county count of days above thresholds (12, 25, 35 ug/m3 etc))
        summary stats (std, median, 90th percentile)

        returns DataFrame with a row per county (fips) and aggregated columns
        """
        if self.pm25_df is None:
            raise RuntimeError("PM2.5 data not loaded. Call load_data() first.")

        df = self.pm25_df.copy()
        df = df.dropna(subset=["fips", "pm25", "date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        df["month"] = df["date"].dt.month
        # seasons: DJF (12,1,2), MAM (3,4,5), JJA (6,7,8), SON (9,10,11)
        def month_to_season(m: int) -> str:
            if m in (12, 1, 2):
                return "DJF"
            if m in (3, 4, 5):
                return "MAM"
            if m in (6, 7, 8):
                return "JJA"
            return "SON"

        df["season"] = df["month"].apply(
            lambda m: month_to_season(int(m)) if not pd.isna(m) else None)

        agg = (
            df.groupby("fips")["pm25"]
            .agg(
                annual_mean="mean",
                annual_median="median",
                annual_std="std",
                annual_90th=lambda x: np.percentile(
                    x.dropna(), 90) if x.dropna().size > 0 else np.nan,
                n_days="count",
            )
            .reset_index()
        )

        days_above_35 = (
            df.assign(above_35=df["pm25"] > 35)
            .groupby("fips")["above_35"]
            .sum()
            .rename("days_above_35")
            .reset_index()
        )

        agg = agg.merge(days_above_35, on="fips", how="left")
        agg["days_above_35"] = agg["days_above_35"].fillna(0)

        agg = agg.reset_index()

        monthly = df.groupby(["fips", "month"])["pm25"].mean().reset_index()
        monthly_pivot = monthly.pivot(index="fips", columns="month", values="pm25")

        # rename columns month_1 thru month_12
        monthly_pivot.columns = [f"month_{int(c)}_mean" for c in monthly_pivot.columns]
        monthly_pivot = monthly_pivot.reset_index()

        seasonal = df.groupby(["fips", "season"])["pm25"].mean().reset_index()
        seasonal_pivot = seasonal.pivot(index="fips", columns="season", values="pm25")
        seasonal_pivot = seasonal_pivot.reset_index()

        # extremes: counts above thresholds
        thresholds = [12.0, 25.0, 35.0]  # common reference thresholds (µg/m3)
        extremes = df.groupby("fips").apply(lambda g: pd.Series({
            f"days_above_{thr}": (g["pm25"] > thr).sum() for thr in thresholds
        }))
        extremes = extremes.reset_index()

        # combine all
        merged = agg.merge(monthly_pivot, on="fips", how="left").merge(
            seasonal_pivot, on="fips", how="left").merge(
                extremes, on="fips", how="left")
        
        for col in merged.columns:
            if col != "fips":
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        self.results["pm25_aggregates"] = merged
        print("PM2.5 aggregation complete. Counties aggregated:", merged.shape[0])
        print("  Overall annual mean PM2.5 - median of counties:",
              np.nanmedian(merged["annual_mean"]))
        print("  Counties with >35 µg/m3 days (any):", (merged["days_above_35"] > 0).sum())

        return merged

    def merge_with_econ(self, pm25_agg: Optional[DataFrame] = None) -> DataFrame:
        """
        Merge aggregated PM2.5 with econ_df on fips.
        """
        if pm25_agg is None:
            pm25_agg = self.aggregate_pm25()

        if self.econ_df is None:
            raise RuntimeError("Economic data not loaded - call load_data() first")

        econ = self.econ_df.copy()
        econ = econ.dropna(subset=["fips"])
        # merge
        merged = pm25_agg.merge(econ, on="fips", how="inner")
        if "NAME" in merged.columns:
            merged["county_name"] = merged["NAME"]
        elif "name" in merged.columns:
            merged["county_name"] = merged["name"]
        else:
            merged["county_name"] = merged["fips"]

        for var in self.econ_vars:
            merged[var] = pd.to_numeric(merged[var], errors="coerce")

        print("Merged PM2.5 aggregates with economic table.")
        print(f"  merged shape: {merged.shape}; counties with both PM2.5 & econ: {merged.shape[0]}")
        missing_income = merged[self.econ_vars[0]].isna().sum()
        print(f"  Missing primary econ var ({self.econ_vars[0]}) in merged: {missing_income}")

        self.joined = merged
        return merged

    def run_correlations(self,
                         x_col: str = "annual_mean",
                         econ_col: Optional[str] = None) -> Dict[str, Any]:
        """
        compute Pearson and Spearman correlations between pm2.5 (x_col)
        and an economic variable (econ_col), print/return results
        """
        if self.joined is None:
            raise RuntimeError("Data not merged - call merge_with_econ() first")

        econ_col = econ_col or self.econ_income_var
        df = self.joined[[x_col, econ_col]].dropna()
        x = df[x_col].astype(float)
        y = df[econ_col].astype(float)

        print(f"\nCorrelation analysis: PM2.5 '{x_col}' vs Econ '{econ_col}' (n={len(df)})")

        if len(df) < 10:
            print("Small sample size for correlation - do not use results")

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)

        print(f"  Pearson r={pearson_r:.4f}, p={pearson_p:.4g}")
        print(f"  Spearman rho={spearman_rho:.4f}, p={spearman_p:.4g}")

        results = {
            "pearson": {"r": pearson_r, "p": pearson_p},
            "spearman": {"rho": spearman_rho, "p": spearman_p},
        }
        self.results.setdefault("correlations", {})[f"{x_col}|{econ_col}"] = results
        return results

    def run_regressions(self, econ_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs OLS, RLM, and quantile regression of econ_col on PM2.5 annual mean
        """
        if self.joined is None:
            raise RuntimeError("Data not merged: call merge_with_econ() first.")

        econ_col = econ_col or self.econ_income_var
        df = self.joined.copy().dropna(subset=["annual_mean", econ_col])
        # ensure numeric
        df["annual_mean"] = pd.to_numeric(df["annual_mean"], errors="coerce")
        df[econ_col] = pd.to_numeric(df[econ_col], errors="coerce")
        df = df.dropna(subset=["annual_mean", econ_col])

        print(f"\nRegression analyses (n={len(df)})")

        # decide dependent and independent var: canonical approach 1) income ~ pm25
        dep = econ_col
        indep = "annual_mean"
        df = df[[dep, indep]].copy()
        df = df.dropna()
        df = df.rename(columns={dep: "income", indep: "pm25"})

        results = {}

        # OLS
        X = sm.add_constant(df["pm25"])
        ols_model = sm.OLS(df["income"], X).fit(cov_type="HC1")
        print("OLS regression: income ~ pm25")
        print(ols_model.summary().tables[1])
        results["ols"] = ols_model

        # RLM - Huber
        try:
            rlm_model = sm.RLM(df["income"], X, M=sm.robust.norms.HuberT()).fit()
            print("\nRLM (Huber) regression results (coef, se):")
            print(rlm_model.params)
            print(rlm_model.bse)
            results["rlm"] = rlm_model
        except Exception as e:
            print("RLM failed:", e)

        # Quantile regression (median)
        try:
            qreg = QuantReg(df["income"], X).fit(q=0.5)
            print("\nMedian quantile regression (coef):")
            print(qreg.params)
            results["quantile_median"] = qreg
        except Exception as e:
            print("Quantile regression failed:", e)

        # save summary stats
        self.results.setdefault("regressions", {})["income_on_pm25"] = {
            "ols_summary": ols_model.summary().as_text(),
            "ols_coef": ols_model.params.to_dict(),
            "ols_pvalues": ols_model.pvalues.to_dict(),
        }
        return results

    def decile_group_analysis(self, econ_col: Optional[str] = None) -> DataFrame:
        """
        Creates deciles by econ_col and computes PM2.5 mean and median across deciles, then
        prints differences and ANOVA/Kruskal-Wallis tests
        """
        if self.joined is None:
            raise RuntimeError("Data not merged - call merge_with_econ() first")

        econ_col = econ_col or self.econ_income_var
        df = self.joined.dropna(subset=["annual_mean", econ_col]).copy()
        df[econ_col] = pd.to_numeric(df[econ_col], errors="coerce")
        df = df.dropna(subset=[econ_col])
        if df.empty:
            raise RuntimeError("no joined data for decile analysis")

        df["income_decile"] = pd.qcut(df[econ_col], 10, labels=False, duplicates="drop")
        group_stats = df.groupby(
            "income_decile")["annual_mean"].agg(
                ["count", "mean", "median", "std"]).reset_index()
        print("\nDecile analysis (PM2.5 by income decile):")
        print(group_stats)

        # test for trend across deciles (spearman rank between decile index and mean pm25)
        dec_mean = group_stats.dropna(subset=["mean"])
        if len(dec_mean) >= 3:
            rho, p = stats.spearmanr(
                dec_mean["income_decile"].astype(float), dec_mean["mean"].astype(float))
            print(f"  Spearman between decile index and mean PM2.5: rho={rho:.4f}, p={p:.4g}")
        else:
            print("  Not enough deciles for trend test.")

        # ANOVA and Kruskal-Wallis across deciles
        groups = [g["annual_mean"].values for _, g in df.groupby("income_decile")]
        try:
            f_stat, f_p = stats.f_oneway(*groups)
            print(f"  One-way ANOVA across deciles: F={f_stat:.4f}, p={f_p:.4g}")
        except Exception as e:
            print("  ANOVA failed:", e)
        try:
            kw_stat, kw_p = stats.kruskal(*groups)
            print(f"  Kruskal-Wallis across deciles: H={kw_stat:.4f}, p={kw_p:.4g}")
        except Exception as e:
            print("  Kruskal-Wallis failed:", e)

        self.results["decile_group_stats"] = group_stats
        return group_stats

    def seasonal_monthly_analysis(self, econ_col: Optional[str] = None) -> Dict[str, Any]:
        """
        computes correlation between monthly/seasonal mean PM2.5 and economic variable
        prints results (patterns and strongest/weakest relationships)
        returns dict of correlations (keyed by 'month_#' and season)
        """
        if self.pm25_df is None:
            raise RuntimeError("PM2.5 not loaded.")
        if self.econ_df is None:
            raise RuntimeError("Econ not loaded.")
        econ_col = econ_col or self.econ_income_var
        pm_agg = self.aggregate_pm25()
        merged = self.merge_with_econ(pm_agg)
        corrs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        monthly_cols = [f"month_{m}_mean" for m in range(1, 13)]
        month_results = {}
        for c in monthly_cols:
            if c in merged.columns:
                tmp = merged[[c, econ_col]].dropna()
                if len(tmp) >= 5:
                    r, p = stats.pearsonr(tmp[c].astype(float), tmp[econ_col].astype(float))
                    rho, sp = stats.spearmanr(tmp[c].astype(float), tmp[econ_col].astype(float))
                    month_results[c] = {"pearson_r": r,
                     "pearson_p": p, 
                     "spearman_rho": rho, 
                     "spearman_p": sp, 
                     "n": len(tmp)}
                    msg = f"Month {c}: Pearson r={r:.4f} (p={p:.3g}), " \
                        f"Spearman rho={rho:.4f} (p={sp:.3g}), n={len(tmp)}"
                    print(msg)
                else:
                    print(f"Month {c}: insufficient data (n={len(tmp)})")
        corrs["monthly"] = month_results
        # seasons: DJF (winter), MAM (spring), JJA (summer), SON (fall)
        seasons = ["DJF", "MAM", "JJA", "SON"]
        season_results = {}
        for s in seasons:
            if s in merged.columns:
                tmp = merged[[s, econ_col]].dropna()
                if len(tmp) >= 5:
                    r, p = stats.pearsonr(tmp[s].astype(float), tmp[econ_col].astype(float))
                    rho, sp = stats.spearmanr(tmp[s].astype(float), tmp[econ_col].astype(float))
                    season_results[s] = {
                        "pearson_r": r, 
                        "pearson_p": p, 
                        "spearman_rho": rho, 
                        "spearman_p": sp, 
                        "n": len(tmp)}
                    msg = f"Season {s}: Pearson r={r:.4f} (p={p:.3g}), " \
                        f"Spearman rho={rho:.4f} (p={sp:.3g}), n={len(tmp)}"
                    print(msg)
                else:
                    print(f"Season {s}: insufficient data (n={len(tmp)})")
        corrs["seasonal"] = season_results
        flat = []
        for k, v in month_results.items():
            if v["n"] >= 10:
                flat.append((k, abs(v["pearson_r"]), v["pearson_r"], v["pearson_p"]))
        for k, v in season_results.items():
            if v["n"] >= 10:
                flat.append((k, abs(v["pearson_r"]), v["pearson_r"], v["pearson_p"]))
        if flat:
            flat_sorted = sorted(flat, key=lambda x: x[1], reverse=True)
            top = flat_sorted[0]
            msg = (f"\nStrongest absolute Pearson correlation observed in '{top[0]}': "
                f"r={top[2]:.4f}, p={top[3]:.4g}")
            print(msg)
        else:
            print("\nNo sufficiently large-month/season correlations to highlight "
                "(insufficient n).")

        self.results["temporal_corrs"] = corrs
        return corrs

    def income_bin_extremes_analysis(self, econ_col: Optional[str] = None) -> None:
        """
        compare distributional characteristics of PM2.5 between low-income and high-income bins
        """
        if self.joined is None:
            raise RuntimeError("Data not merged. Call merge_with_econ() first.")
        econ_col = econ_col or self.econ_income_var
        df = self.joined.dropna(subset=["annual_mean", econ_col]).copy()
        df[econ_col] = pd.to_numeric(df[econ_col], errors="coerce")
        df = df.dropna(subset=[econ_col])
        q_low = df[econ_col].quantile(0.2)
        q_high = df[econ_col].quantile(0.8)
        low = df[df[econ_col] <= q_low]["annual_mean"].dropna()
        high = df[df[econ_col] >= q_high]["annual_mean"].dropna()
        print(f"\nComparing extreme income bins (bottom 20% vs top 20%):"
            f"n_low={len(low)}, n_high={len(high)}")
        if len(low) < 5 or len(high) < 5:
            print("do not infer anything: insufficient sample size.")
            return
        mean_low = low.mean()
        mean_high = high.mean()
        diff = mean_low - mean_high
        print(f"  Mean PM2.5 bottom20 = {mean_low:.3f};"
              f"top20 = {mean_high:.3f}; difference = {diff:.3f} (low - high)")

        # welch t-test
        tstat, tp = stats.ttest_ind(low, high, equal_var=False, nan_policy="omit")
        print(f"  Welch t-test: t={tstat:.4f}, p={tp:.4g}")

        # KS test
        ks_stat, ks_p = stats.ks_2samp(low, high)
        print(f"  KS two-sample test: D={ks_stat:.4f}, p={ks_p:.4g}")

        rng = np.random.default_rng(seed=42)
        n_boot = 1000
        boot_diffs = []
        for _ in range(n_boot):
            s_low = rng.choice(np.asarray(low.values, dtype=float), size=len(low), replace=True)
            s_high = rng.choice(np.asarray(high.values, dtype=float), size=len(high), replace=True)
            boot_diffs.append(np.mean(s_low) - np.mean(s_high))
        ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
        print(f"Bootstrap 95% CI for mean difference (low-high): [{ci_lower:.3f}, {ci_upper:.3f}]")

        self.results["income_bin_extremes"] = {
            "mean_low": mean_low,
            "mean_high": mean_high,
            "diff": diff,
            "t_test": {"t": tstat, "p": tp},
            "ks_test": {"D": ks_stat, "p": ks_p},
            "boot_ci": (ci_lower, ci_upper),
        }

    def run_all(self) -> None:
        """
        Performs full analysis
        Prints findings as they are discovered and stores key outputs in self.results
        """
        print("=== START: PM2.5 <-> Economic Status Analysis ===")
        self.load_data()
        pm_agg = self.aggregate_pm25()
        self.merge_with_econ(pm_agg)
        self.run_correlations(x_col="annual_mean", econ_col=self.econ_income_var)
        self.run_regressions()
        self.decile_group_analysis()
        self.seasonal_monthly_analysis()
        self.income_bin_extremes_analysis()
        print("=== END: Analysis. Results stored in analyzer.results ===")


    def save_results_to_csv(self, out_prefix: str = "results") -> None:
        """
        save key result tables to CSV files with the provided prefix
        """
        if "pm25_aggregates" in self.results:
            self.results["pm25_aggregates"].to_csv(f"{out_prefix}_pm25_aggregates.csv", index=False)
        if "decile_group_stats" in self.results:
            self.results["decile_group_stats"].to_csv(f"{out_prefix}_decile_stats.csv", index=False)
        if self.joined is not None:
            self.joined.to_csv(f"data/output/{out_prefix}_merged_joined.csv", index=False)
        print("Saved result CSVs.")

# if executed as a script, run an example pipeline (paths are expected to exist)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze PM2.5 vs Economic Status for US counties.")
    parser.add_argument("--pm25", type=str,
                        default="data/county_level_pm25.csv")
    parser.add_argument("--econ", type=str,
                        default="data/county_level_economic_status.csv")
    parser.add_argument("--income_var", type=str,
                        default="B19013_001E")
    args = parser.parse_args()

    analyzer = PM25EconomicAnalyzer(pm25_csv=args.pm25,
                                    econ_csv=args.econ,
                                    econ_income_var=args.income_var)
    analyzer.run_all()
    analyzer.save_results_to_csv(out_prefix="analysis_output")
