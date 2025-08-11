#!/usr/bin/env python3
"""
Streamlit Dashboard for Temporal Trend Analysis

This dashboard uses the temporal_trends module to create interactive
time series visualizations of ecological data.
"""

import streamlit as st
import pandas as pd

# Import the custom visualization module
try:
    from temporal_trends import plot_temporal_trend
    TEMPORAL_TRENDS_AVAILABLE = True
except ImportError:
    st.error("Error: `temporal_trends.py` not found. Please ensure it's in the same directory.")
    TEMPORAL_TRENDS_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Temporal Trend Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Temporal Trend Analysis Dashboard")
st.write("Analyze how ecological metrics change over time. Upload your data or use the sample dataset to get started.")

# --- Data Loading ---

# Sample data for demonstration
@st.cache_data
def get_sample_data():
    sample_data = {
        'Year': [2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022],
        'Region': ['Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo'],
        'MPA_Status': ['Inside', 'Inside', 'Inside', 'Inside', 'Outside', 'Inside', 'Outside', 'Inside', 'Outside', 'Inside'],
        'Biomass': [1250, 1800, 1350, 1950, 1100, 2100, 1200, 2200, 1050, 2300],
        'Species_Count': [45, 60, 48, 65, 42, 70, 40, 75, 38, 78],
        'Trophic_Level': [4.2, 4.5, 4.2, 4.6, 3.9, 4.7, 3.8, 4.7, 3.7, 4.8]
    }
    return pd.DataFrame(sample_data)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = get_sample_data()
        st.sidebar.info("Using sample data instead.")
else:
    df = get_sample_data()
    st.sidebar.info("Using sample LTEM data. Upload a file to analyze your own data.")

# --- Sidebar Controls ---
st.sidebar.header("Chart Configuration")

# Get column names for select boxes
columns = df.columns.tolist()

# Selectbox for time column
time_col = st.sidebar.selectbox(
    "Select Time Column (e.g., Year)",
    options=columns,
    index=columns.index('Year') if 'Year' in columns else 0
)

# Selectbox for value column
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
value_col = st.sidebar.selectbox(
    "Select Value Column to Analyze",
    options=numeric_cols,
    index=numeric_cols.index('Biomass') if 'Biomass' in numeric_cols else 0
)

# Selectbox for grouping column (optional)
categorical_cols = ['None'] + df.select_dtypes(include=['object', 'category']).columns.tolist()
group_col = st.sidebar.selectbox(
    "Select Grouping Column (Optional)",
    options=categorical_cols,
    index=categorical_cols.index('Region') if 'Region' in categorical_cols else 0
)

# --- Main Panel: Chart and Data ---

if TEMPORAL_TRENDS_AVAILABLE:
    st.header(f"Analysis of {value_col} over {time_col}")

    # Handle the 'None' case for the grouping column
    group_col_param = group_col if group_col != 'None' else None

    try:
        # Generate the plot using the imported function
        fig = plot_temporal_trend(
            df=df,
            time_col=time_col,
            value_col=value_col,
            group_col=group_col_param
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Interpretation**: This chart shows the trend of the selected metric over time. Each line represents a different group if a grouping column is selected. Look for upward or downward trends, seasonal patterns, or significant changes.")

    except Exception as e:
        st.error(f"An error occurred while creating the plot: {e}")

# --- Display Raw Data ---
with st.expander("View Raw Data"):
    st.dataframe(df)

st.sidebar.markdown("---")
st.sidebar.info("This dashboard is part of the Phase 2 implementation for the AI-Enhanced Data Exploration Application.")
