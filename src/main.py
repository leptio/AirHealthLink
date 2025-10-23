"""Runs the PM2.5 pollution analysis."""
from typing import Any
import os
import pandas as pd
from analysis import Analyzer # type: ignore
from visualization import Visualization # type: ignore

filepath = os.path.join(os.path.dirname(__file__), '../data/county_level_pm25.csv')

def main() -> None:
    """Main function to implement PM2.5 analysis."""
    # Load data
    df: pd.DataFrame = pd.read_csv(filepath)
    print(df.head())

    # Create visualization
    visualizer: Visualization = Visualization(df)
    local_save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "plots/frames", "geographic_distribution")
    visualizer.plot_cases_on_map(title="PM2.5 geographic distribution",
                                    save_path=local_save_path)

if __name__ == "__main__":
    main()
