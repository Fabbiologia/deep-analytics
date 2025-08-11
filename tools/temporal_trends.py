#!/usr/bin/env python3
"""
Temporal Trends Visualization Module for Ecological Data
Focuses on creating time series plots to analyze ecosystem dynamics.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional, List

def plot_temporal_trend(df: pd.DataFrame, time_col: str, value_col: str, group_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """
    Creates an interactive time series plot for a given metric.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        time_col (str): The name of the column representing time (e.g., 'Year').
        value_col (str): The name of the column with the numeric value to plot.
        group_col (str, optional): The column to group data by, creating separate lines. Defaults to None.
        title (str, optional): The title for the plot. Defaults to a generated title.

    Returns:
        go.Figure: A Plotly figure object.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError("Specified time and value columns must exist in the DataFrame.")

    # Ensure time column is sorted
    df_sorted = df.sort_values(by=time_col)

    # Generate a default title if not provided
    if not title:
        if group_col:
            title = f"Temporal Trend of {value_col} by {group_col}"
        else:
            title = f"Temporal Trend of {value_col}"

    # Create the plot
    fig = px.line(
        df_sorted,
        x=time_col,
        y=value_col,
        color=group_col,
        title=title,
        markers=True,
        labels={
            value_col: value_col.replace('_', ' ').title(),
            time_col: time_col.replace('_', ' ').title()
        }
    )

    # Apply a clean theme and update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title=time_col.replace('_', ' ').title(),
        yaxis_title=value_col.replace('_', ' ').title(),
        legend_title=group_col.replace('_', ' ').title() if group_col else None,
        title_x=0.5
    )

    return fig

# Example usage for standalone testing
if __name__ == '__main__':
    # Create sample data
    sample_data = {
        'Year': [2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021],
        'Region': ['Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo'],
        'Biomass': [1250, 1800, 1350, 1950, 1400, 2100, 1500, 2200],
        'Species_Count': [45, 60, 48, 65, 50, 70, 52, 75]
    }
    df_sample = pd.DataFrame(sample_data)

    print("Generating temporal trend plot for Biomass by Region...")
    
    # Generate the plot
    fig_biomass = plot_temporal_trend(
        df=df_sample, 
        time_col='Year', 
        value_col='Biomass', 
        group_col='Region'
    )

    # To display the plot, you would typically use fig.show()
    # In a script, you might save it to a file:
    # fig_biomass.write_html("temporal_biomass_chart.html")
    
    print("Plot generated. Run `fig.show()` in an interactive environment to view.")

    print("\nGenerating temporal trend plot for Species Count (ungrouped)...")
    fig_species = plot_temporal_trend(
        df=df_sample, 
        time_col='Year', 
        value_col='Species_Count'
    )
    print("Plot generated.")
