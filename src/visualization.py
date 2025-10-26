"""Module responsible for generating visualization of data"""
from typing import Optional, List, Tuple, Any
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import geopandas as gpd
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon
import imageio
from datetime import date, timedelta
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import os

class Visualization:
    """Class responsible for graphing air quality data."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.setup_plotting_style()

    def setup_plotting_style(self) -> None:
        """Set up consistent plotting style for all visualizations."""
        sns.set_style("darkgrid")


    def plot_cases_on_map(
            self,
            save_path: str,
            title: str="PM2.5 geographic distribution",
            shapefile_path: str = "data/shapefile/cb_2022_us_county_5m.shp") -> None:
        """Create a plot (on a map) of PM2.5 pollution using longitude and latitude."""



        # Load county geometries
        counties = gpd.read_file(shapefile_path)
        valid_states: List[str] = [
            "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19",
            "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35",
            "36","37","38","39","40","41","42","44","45","46","47","48","49","50","51","53",
            "54","55","56"
        ]
        counties = counties[counties['STATEFP'].isin(valid_states)]

        def shift_only_far_east_alaska(geom: Polygon or MultiPolygon) -> Polygon or MultiPolygon:
            if isinstance(geom, Polygon):
                minx, maxx, _, _ = geom.bounds
                # Only shift polygons that are extremely far east (> 0) and west of mainland (> -180)
                if minx > 0:
                    return translate(geom, xoff=-360)
                return geom
            elif isinstance(geom, MultiPolygon):
                parts = []
                for part in geom.geoms:
                    minx, maxx, _, _ = part.bounds
                    if minx > 0:
                        parts.append(translate(part, xoff=-360))
                    else:
                        parts.append(part)
                return MultiPolygon(parts)
            else:
                return geom
        # Apply only to Alaska

        counties.loc[counties['STATEFP']=='02', 'geometry'] = \
            counties.loc[counties['STATEFP']=='02', 'geometry'].apply(shift_only_far_east_alaska)
        # Prepare PM2.5 data: create FIPS code
        self.data['fips'] = self.data['state_code'].astype(str).str.zfill(2) + \
                             self.data['county_code'].astype(str).str.zfill(3)   
        # Prepare counties for merge
        counties['fips'] = counties['STATEFP'] + counties['COUNTYFP']
        start_date = date(2022, 1, 1)
        day_count = 365
        vmin = self.data['arithmetic_mean'].min()
        vmax = self.data['arithmetic_mean'].max()
        for single_date in (start_date + timedelta(n) for n in range(day_count)):
            # Sort data by date
            county_sorted = self.data[self.data['date_local']==single_date.strftime("%Y-%m-%d")]
            # Aggregate PM2.5 by county
            pm25_county = county_sorted.groupby('fips')['arithmetic_mean'].mean().reset_index()
            print(single_date.strftime("%Y-%m-%d"))
            # Merge PM2.5 data with county geometries
            merged = counties.merge(pm25_county, on='fips', how='left')
            # Nonlinear normalization to make red appear earlier
            norm = PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax, clip=True)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 8), dpi=3000)
            merged.boundary.plot(ax=ax, linewidth=0.1, edgecolor='white')
            merged.plot(
                column='arithmetic_mean',
                ax=ax,
                cmap="Reds",
                norm=norm,
                legend=False,
                vmin=vmin,
                vmax=vmax,
                missing_kwds={'color': 'black'},
                linewidth=0
            )
                        
            # Manual colorbar consistent with PowerNorm
            sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
            sm._A = []  # required dummy array
            sm.set_clim(0, vmax)
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label('PM2.5 (µg/m³)', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            # Aesthetic adjustments
            spec_title: str = title + " (" + str(single_date) + ")"
            ax.set_title(spec_title, fontsize=14, color='white')
            ax.axis('off')
            fig.patch.set_facecolor('black')
            #update save path
            save_path = os.path.normpath(save_path)
            nsp: str = save_path + str(single_date) + ".png"

            plt.savefig(nsp, dpi=3000, bbox_inches='tight')
            plt.close()
