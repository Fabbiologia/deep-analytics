#!/usr/bin/env python3
"""
Ecological Analysis Chatbot Mockup
===================================
A chatbot interface demonstration for ecological analysis with pre-prepared responses
for three specific analysis questions:
1. Fish biomass trends in Gulf of California
2. Fish biomass comparisons between regions
3. Invertebrate diversity over time

Author: Ecological Analysis System
Date: 2025-01-08
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import time
import logging
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(123)

# Configure page
st.set_page_config(
    page_title="🌊 Ecological Analysis Assistant",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
        margin-bottom: 2rem;
    }
    .report-section {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


def generate_mock_data(data_type: str, years: int = 10) -> pd.DataFrame:
    """
    Generate mock ecological data for demonstrations.
    
    Parameters:
    -----------
    data_type : str
        Type of data to generate ('biomass', 'diversity')
    years : int
        Number of years of data to generate
    
    Returns:
    --------
    pd.DataFrame
        Mock ecological data
    """
    logger.info(f"Generating mock {data_type} data for {years} years")
    
    base_year = 2015
    dates = pd.date_range(start=f'{base_year}-01-01', 
                          end=f'{base_year + years - 1}-12-31', 
                          freq='M')
    
    if data_type == 'biomass':
        regions = ['North Gulf', 'Central Gulf', 'South Gulf', 'Midriff Islands']
        data = []
        for date in dates:
            for region in regions:
                # Add seasonal and regional variation
                base_biomass = 100 + np.random.randn() * 10
                seasonal_effect = 20 * np.sin(2 * np.pi * date.month / 12)
                regional_multiplier = {'North Gulf': 1.2, 'Central Gulf': 1.0, 
                                      'South Gulf': 0.8, 'Midriff Islands': 1.5}[region]
                trend = (date.year - base_year) * 2  # Slight increasing trend
                
                biomass = base_biomass * regional_multiplier + seasonal_effect + trend + np.random.randn() * 5
                data.append({
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'region': region,
                    'biomass_kg_per_hectare': max(0, biomass),
                    'sample_size': np.random.randint(10, 50),
                    'temperature_c': 20 + seasonal_effect/5 + np.random.randn()
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} biomass records")
        return df
    
    elif data_type == 'diversity':
        sites = ['Site A', 'Site B', 'Site C', 'Site D', 'Site E']
        data = []
        for date in dates:
            for site in sites:
                # Shannon diversity index with temporal variation
                base_diversity = 2.5 + np.random.randn() * 0.3
                temporal_trend = (date.year - base_year) * -0.02  # Slight decline
                seasonal_effect = 0.3 * np.sin(2 * np.pi * date.month / 12)
                
                diversity = base_diversity + temporal_trend + seasonal_effect + np.random.randn() * 0.1
                data.append({
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'site': site,
                    'shannon_diversity': max(0, diversity),
                    'species_richness': np.random.randint(15, 40),
                    'evenness': np.random.uniform(0.6, 0.95),
                    'sample_size': np.random.randint(20, 100)
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} diversity records")
        return df


def create_biomass_trend_visualizations(df: pd.DataFrame) -> Dict:
    """Create visualizations for fish biomass trends."""
    logger.info("Creating biomass trend visualizations")
    figures = {}
    
    # 1. Time series plot
    fig1 = go.Figure()
    for region in df['region'].unique():
        region_data = df[df['region'] == region].groupby('date')['biomass_kg_per_hectare'].mean().reset_index()
        fig1.add_trace(go.Scatter(
            x=region_data['date'],
            y=region_data['biomass_kg_per_hectare'],
            mode='lines+markers',
            name=region,
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    fig1.update_layout(
        title="Fish Biomass Trends in Gulf of California (2015-2024)",
        xaxis_title="Date",
        yaxis_title="Biomass (kg/hectare)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    figures['time_series'] = fig1
    
    # 2. Seasonal pattern
    monthly_avg = df.groupby(['month', 'region'])['biomass_kg_per_hectare'].mean().reset_index()
    fig2 = px.line(monthly_avg, x='month', y='biomass_kg_per_hectare', 
                   color='region', markers=True,
                   title="Seasonal Patterns in Fish Biomass",
                   labels={'month': 'Month', 'biomass_kg_per_hectare': 'Average Biomass (kg/hectare)'},
                   template='plotly_white')
    fig2.update_layout(height=400)
    figures['seasonal'] = fig2
    
    # 3. Annual trend with confidence intervals
    annual_stats = df.groupby(['year', 'region'])['biomass_kg_per_hectare'].agg(['mean', 'std']).reset_index()
    fig3 = go.Figure()
    for region in df['region'].unique():
        region_data = annual_stats[annual_stats['region'] == region]
        fig3.add_trace(go.Scatter(
            x=region_data['year'],
            y=region_data['mean'],
            error_y=dict(type='data', array=region_data['std']),
            mode='lines+markers',
            name=region,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig3.update_layout(
        title="Annual Fish Biomass Trends with Standard Deviation",
        xaxis_title="Year",
        yaxis_title="Mean Biomass (kg/hectare)",
        height=500,
        template='plotly_white'
    )
    figures['annual'] = fig3
    
    logger.info(f"Created {len(figures)} biomass trend visualizations")
    return figures

# -------------------------
# Intent detection helper
# -------------------------
def detect_intent(text: str) -> str:
    """Route user text to one of the three canned analyses or help."""
    q = (text or "").lower()
    if ("trend" in q or "trends" in q) and ("fish" in q and "biomass" in q) and ("gulf" in q):
        return "biomass_trends"
    if ("compare" in q and "biomass" in q and ("region" in q or "regions" in q)):
        return "biomass_compare"
    if ("invertebrate" in q and ("diversity" in q or "over time" in q or "trend" in q)) or ("diversity" in q and "invertebrate" in q):
        return "diversity_time"
    return "help"


def create_biomass_comparison_visualizations(df: pd.DataFrame) -> Dict:
    """Create comparison visualizations for fish biomass between regions."""
    logger.info("Creating biomass comparison visualizations")
    figures = {}
    
    # 1. Box plot comparison
    fig1 = px.box(df, x='region', y='biomass_kg_per_hectare', 
                  color='region',
                  title="Fish Biomass Distribution by Region",
                  labels={'biomass_kg_per_hectare': 'Biomass (kg/hectare)'},
                  template='plotly_white')
    fig1.update_layout(height=500, showlegend=False)
    figures['boxplot'] = fig1
    
    # 2. Bar plot with error bars
    region_stats = df.groupby('region')['biomass_kg_per_hectare'].agg(['mean', 'std', 'count']).reset_index()
    region_stats['se'] = region_stats['std'] / np.sqrt(region_stats['count'])
    
    fig2 = go.Figure(data=[
        go.Bar(x=region_stats['region'], 
               y=region_stats['mean'],
               error_y=dict(type='data', array=region_stats['se']),
               marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ])
    fig2.update_layout(
        title="Mean Fish Biomass by Region (with Standard Error)",
        xaxis_title="Region",
        yaxis_title="Mean Biomass (kg/hectare)",
        height=500,
        template='plotly_white'
    )
    figures['barplot'] = fig2
    
    logger.info(f"Created {len(figures)} biomass comparison visualizations")
    return figures


def create_diversity_visualizations(df: pd.DataFrame) -> Dict:
    """Create visualizations for invertebrate diversity over time."""
    logger.info("Creating diversity visualizations")
    figures = {}
    
    # 1. Shannon diversity time series
    fig1 = go.Figure()
    for site in df['site'].unique():
        site_data = df[df['site'] == site].groupby('date')['shannon_diversity'].mean().reset_index()
        fig1.add_trace(go.Scatter(
            x=site_data['date'],
            y=site_data['shannon_diversity'],
            mode='lines+markers',
            name=site,
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    fig1.update_layout(
        title="Invertebrate Shannon Diversity Over Time",
        xaxis_title="Date",
        yaxis_title="Shannon Diversity Index",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    figures['shannon_time'] = fig1
    
    # 2. Species richness trends
    annual_richness = df.groupby(['year', 'site'])['species_richness'].mean().reset_index()
    fig2 = px.line(annual_richness, x='year', y='species_richness', 
                   color='site', markers=True,
                   title="Species Richness Trends by Site",
                   labels={'species_richness': 'Mean Species Richness'},
                   template='plotly_white')
    fig2.update_layout(height=400)
    figures['richness'] = fig2
    
    logger.info(f"Created {len(figures)} diversity visualizations")
    return figures


def generate_report_section(title: str, content: str) -> None:
    """Display a formatted report section."""
    st.markdown(f'<div class="report-section">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.markdown(content)
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main application function (Chatbot prototype)."""
    # Page uses global set_page_config declared above
    
    # Header bar (minimal)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    
    # Minimal styling and card components to mimic dashboard look
    st.markdown(
        """
        <style>
        /* Overall tone */
        .main-header { font-size: 26px; font-weight: 700; margin-bottom: 12px; }
        
        /* Center chat container */
        .chat-wrapper { max-width: 820px; margin: 0 auto; }
        
        /* Card style */
        .card { background: #ffffff; border: 1px solid #e6e6e6; border-radius: 12px; padding: 16px; }
        .card + .card { margin-top: 12px; }
        .card-title { font-weight: 700; font-size: 14px; color: #222; margin-bottom: 10px; }
        .metric-pill { background:#fafafa; border:1px solid #eee; border-radius: 12px; padding: 14px 16px; text-align:center; }
        .metric-label { color:#555; font-size:12px; }
        .metric-value { font-size:20px; font-weight:700; color:#111; }
        
        /* Chat bubbles */
        .stChatMessage { border: 1px solid #ededed; }
        
        /* Subtle gray background */
        .stApp { background-color: #fbfbfb; }
        
        /* Reduce padding */
        section.main > div { padding-top: 0.5rem; }
        
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "dashboard_ready" not in st.session_state:
        st.session_state.dashboard_ready = False
    # Analysis gating state
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False
    if "analysis_start_time" not in st.session_state:
        st.session_state.analysis_start_time = 0.0
    if "analysis_duration" not in st.session_state:
        st.session_state.analysis_duration = 20
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "pending_intent" not in st.session_state:
        st.session_state.pending_intent = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    
    # Centered chat area
    with st.container():
        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Read user input
    default_prompt = st.session_state.pop("queued_prompt", None)
    if default_prompt:
        # Use the queued quick-prompt directly this turn
        user_query = default_prompt
    else:
        # Show chat input (no default value supported in current Streamlit)
        user_text = st.chat_input("Ask about ecological metrics, trends, or comparisons…")
    
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # Determine intent
        intent = detect_intent(user_text)
        # Set analysis gating and store intent
        st.session_state.pending_intent = intent
        st.session_state.pending_prompt = user_text
        st.session_state.analysis_started = True
        st.session_state.analysis_done = False
        st.session_state.analysis_start_time = time.time()

        # Respond (acknowledgement only before analysis completes)
        with st.container():
            st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
            with st.chat_message("assistant"):
                # small extra delay to feel more natural
                time.sleep(0.3)
                st.markdown("### Working on it…")
                st.write("Starting analysis now. This will take about 20 seconds.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Show timer while analysis runs (no output rendered yet)
    if st.session_state.analysis_started and not st.session_state.analysis_done:
        with st.container():
            st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
            st.info("Running analysis...")
            progress = st.progress(0)
            total = int(st.session_state.analysis_duration)
            for i in range(total + 1):
                progress.progress(i/total)
                time.sleep(1)
            st.success("Analysis complete")
            st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.analysis_done = True
        st.session_state.dashboard_ready = False

    # After timer completes, render the assistant's full analysis, then the dashboard
    if st.session_state.analysis_done:
        intent = st.session_state.pending_intent
        with st.container():
            st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
            with st.chat_message("assistant"):
                if intent == "help":
                    st.markdown("""
                    I can help with:
                    • Fish biomass trends in the Gulf of California
                    • Fish biomass comparison between regions
                    • Invertebrate diversity over time
                    """)
                elif intent == "biomass_trends":
                    st.markdown("### Fish Biomass Trends — Gulf of California")
                    with st.spinner("Generating data and figures…"):
                        df = generate_mock_data('biomass', years=10)
                        figures = create_biomass_trend_visualizations(df)
                        time.sleep(0.5)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(figures["time_series"], use_container_width=True, theme=None)
                    with col2:
                        st.plotly_chart(figures["seasonal"], use_container_width=True, theme=None)
                    initial_biomass = df[df['year'] == df['year'].min()]["biomass_kg_per_hectare"].mean()
                    final_biomass = df[df['year'] == df['year'].max()]["biomass_kg_per_hectare"].mean()
                    percent_change = ((final_biomass - initial_biomass) / initial_biomass) * 100
                    st.markdown("#### Executive Summary")
                    st.write("Overall increasing trend with strong seasonal and regional patterns.")
                    st.markdown("#### Key Results")
                    st.write(f"Mean biomass increased from {initial_biomass:.1f} to {final_biomass:.1f} kg/ha ({percent_change:.1f}%).")
                    st.markdown("#### Discussion")
                    st.write("Patterns align with known upwelling cycles; regional heterogeneity suggests spatial management.")
                    st.markdown("#### Methodology")
                    st.write("Monthly observations were aggregated to annual means per region. A seasonal component was estimated to separate intra-annual variability from the long-term signal, and simple linear trends were summarized at the regional and whole-Gulf levels.")
                    st.markdown("#### Data Notes")
                    st.write("Values are standardized to kg/ha to allow cross-site comparisons. Sites with <3 annual observations were down-weighted in the overall change estimate.")
                    st.markdown("#### Management Implications")
                    st.write("Regions exhibiting consistent increases can serve as source areas for recruitment; adaptive closures can be prioritized around seasons with lowest biomass to minimize impacts.")
                elif intent == "biomass_compare":
                    st.markdown("### Fish Biomass — Regional Comparison")
                    with st.spinner("Generating data and figures…"):
                        df = generate_mock_data('biomass', years=10)
                        figures = create_biomass_comparison_visualizations(df)
                        time.sleep(0.5)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(figures["boxplot"], use_container_width=True, theme=None)
                    with col2:
                        st.plotly_chart(figures["barplot"], use_container_width=True, theme=None)
                    st.markdown("#### Summary")
                    region_stats = df.groupby('region')['biomass_kg_per_hectare'].agg(['mean','std']).round(1)
                    st.dataframe(region_stats, use_container_width=True)
                    st.info("Kruskal-Wallis: H ≈ 45.3, p < 0.001")
                    st.markdown("#### Interpretation")
                    st.write("Differences among regions indicate heterogeneous productivity and/or fishing pressure. The Midriff Islands consistently rank highest, suggesting strong physical forcing (mixing, upwelling) and possible spillover effects.")
                    st.markdown("#### Methods")
                    st.write("We compared distributions of biomass across regions using nonparametric tests, complemented by visualization (boxplots and mean bars with variability). Where relevant, pairwise contrasts can be computed to prioritize area-specific actions.")
                    st.markdown("#### Actionable Next Steps")
                    st.write("Focus monitoring on low-performing regions, evaluate habitat quality, and align catch limits with demonstrated regional capacity.")
                else:
                    st.markdown("### Invertebrate Diversity — Temporal Trends")
                    with st.spinner("Generating data and figures…"):
                        df = generate_mock_data('diversity', years=10)
                        figures = create_diversity_visualizations(df)
                        time.sleep(0.5)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(figures["shannon_time"], use_container_width=True, theme=None)
                    with col2:
                        st.plotly_chart(figures["richness"], use_container_width=True, theme=None)
                    st.markdown("#### Summary")
                    initial_shannon = df[df['year'] == df['year'].min()]["shannon_diversity"].mean()
                    final_shannon = df[df['year'] == df['year'].max()]["shannon_diversity"].mean()
                    change = ((final_shannon - initial_shannon) / initial_shannon) * 100
                    st.write(f"Shannon diversity changed from {initial_shannon:.2f} to {final_shannon:.2f} ({change:.1f}%).")
                    st.markdown("#### Ecological Interpretation")
                    st.write("A decline in Shannon diversity alongside relatively stable evenness suggests losses concentrated in a subset of sensitive taxa. Richness reductions may reflect habitat degradation or thermal stress events.")
                    st.markdown("#### Methods")
                    st.write("Diversity indices were computed per site-year and summarized across space and time. Trends were described using simple slopes and smoothed trajectories to emphasize directionality over short-term noise.")
                    st.markdown("#### Monitoring Recommendations")
                    st.write("Increase sampling frequency at sentinel sites, include habitat covariates (temperature, substrate complexity), and incorporate taxon-specific flags for early warning.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.dashboard_ready = True

    # Render dashboard only after analysis and assistant output complete
    if st.session_state.dashboard_ready:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='main-header'>Welcome to your Environmental Impact Hub</div>", unsafe_allow_html=True)
        st.write("Monitor, understand, and report how your company is contributing to ocean conservation.")

        # Metrics row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1: st.markdown(f"<div class='metric-pill'><div class='metric-value'>12</div><div class='metric-label'>MPAs Supported</div></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-pill'><div class='metric-value'>1,847</div><div class='metric-label'>Media Uploads</div></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='metric-pill'><div class='metric-value'>89</div><div class='metric-label'>Species Observed</div></div>", unsafe_allow_html=True)
        with m4: st.markdown(f"<div class='metric-pill'><div class='metric-value'>8.7</div><div class='metric-label'>Biodiversity Score</div></div>", unsafe_allow_html=True)
        with m5: st.markdown(f"<div class='metric-pill'><div class='metric-value'>2,450</div><div class='metric-label'>Carbon Credits</div></div>", unsafe_allow_html=True)
        with m6: st.markdown(f"<div class='metric-pill'><div class='metric-value'>24</div><div class='metric-label'>Active Sites</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        
        # Two-column mid section
        left, right = st.columns([2, 1])
        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>MPAs Map</div>", unsafe_allow_html=True)
            # Placeholder map container
            st.plotly_chart(create_biomass_trend_visualizations(generate_mock_data('biomass', 3))['annual'], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Recent Data Uploads</div>", unsafe_allow_html=True)
            st.write("• Coral Reef Survey — Sector A (2 hours ago)")
            st.write("• Fish Population Count (5 hours ago)")
            st.write("• Water Quality Measurements (1 day ago)")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        
        # Bottom row: Reports, AI Assistant placeholder card, Plan Management
        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Reports & Downloads</div>", unsafe_allow_html=True)
            st.write("Q4 2024 Impact Report — Generated Dec 15, 2024")
            st.write("Species Data Export — Generated Dec 18, 2024")
            st.button("Generate New Report", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>AI Assistant</div>", unsafe_allow_html=True)
            st.write("Continue the conversation above to update dashboards.")
            st.markdown("</div>", unsafe_allow_html=True)
        with b3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Plan Management</div>", unsafe_allow_html=True)
            st.write("Enterprise Plan — Active")
            st.write("Next billing: Jan 15, 2025 — $2,490/month")
            c1, c2 = st.columns(2)
            with c1: st.button("Upgrade Plan", use_container_width=True)
            with c2: st.button("Contact Support", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
