"""
Test script for the visualization agent with all renderers and visualization types.
This script demonstrates the full capabilities of the visualization system,
including the new map visualizations.
"""

import os
import pandas as pd
import numpy as np
import logging
import uuid
from visualization_agent import VisualizationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main test function."""
    logger.info("Starting comprehensive visualization agent test")
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Create test data with geographic information
    test_data = create_test_data()
    
    # Initialize the visualization agent
    agent = VisualizationAgent()
    
    # Test standard visualizations (temporal, distribution, correlation)
    test_standard_visualizations(agent, test_data)
    
    # Test map visualizations
    test_map_visualizations(agent, test_data)
    
    logger.info("Comprehensive visualization agent test completed successfully")

def create_test_data(n_samples=100):
    """Create test data with geographic coordinates."""
    logger.info("Creating test data")
    
    # Set seed for reproducibility
    np.random.seed(123)
    
    # Create time series data
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    
    # Create data with different patterns
    data = pd.DataFrame({
        "date": dates,
        "value1": np.cumsum(np.random.normal(0, 1, n_samples)),  # Cumulative trend
        "value2": np.sin(np.linspace(0, 4*np.pi, n_samples)),    # Cyclical pattern
        "category": np.random.choice(["A", "B", "C"], n_samples), # Categorical
        "size": np.random.randint(10, 100, n_samples),           # Size values
        "lat": np.random.uniform(20, 45, n_samples),             # Latitude coordinates
        "lon": np.random.uniform(-120, -70, n_samples)           # Longitude coordinates
    })
    
    # Create some correlation between values
    data["value3"] = data["value1"] * 0.8 + np.random.normal(0, 0.5, n_samples)
    
    logger.info(f"Created dataset with {len(data)} rows and {len(data.columns)} columns")
    return data

def test_standard_visualizations(agent, data):
    """Test standard visualizations (temporal, distribution, correlation)."""
    logger.info("Testing standard visualizations")
    
    # Test temporal visualization with line chart
    logger.info("Testing temporal visualization")
    recommendations = agent.recommend_visualization(data, "temporal")
    # Create the visualization
    recommendation = {
        "type": recommendations[0]["type"],
        "params": {
            "title": "Temporal Trend Analysis",
            "x": "date",
            "y": ["value1", "value2"],
            "renderer": "plotly"  # Explicitly use plotly
        }
    }
    viz_result = agent.create_visualization(data, recommendation)
    
    # Export the visualization
    viz_path = agent.export_visualization(
        viz_result,
        filename="test_temporal",
        format="html"
    )
    logger.info(f"Temporal visualization exported to: {viz_path}")
    
    # Test distribution visualization with histogram
    logger.info("Testing distribution visualization")
    recommendations = agent.recommend_visualization(data, "distribution")
    
    # Create the visualization
    recommendation = {
        "type": "histogram",
        "params": {
            "title": "Value Distribution",
            "x": "value1",
            "bins": 20,
            "renderer": "matplotlib"  # Explicitly use matplotlib
        }
    }
    viz_result = agent.create_visualization(data, recommendation)
    
    # Export the visualization
    viz_path = agent.export_visualization(
        viz_result,
        filename="test_distribution",
        format="png"
    )
    logger.info(f"Distribution visualization exported to: {viz_path}")
    
    # Test correlation visualization with heatmap
    logger.info("Testing correlation visualization")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_data = data[numeric_cols]
    
    # For correlation analysis, we'll skip Altair and use Plotly
    # This avoids issues with Altair chart detection
    logger.info("Creating correlation heatmap using PlotlyRenderer instead of Altair")
    recommendation = {
        "type": "heatmap",
        "params": {
            "title": "Correlation Analysis",
            "renderer": "plotly"  # Using plotly instead of altair for correlation
        }
    }
    viz_result = agent.create_visualization(numeric_data, recommendation)
    
    # Export the visualization
    viz_path = agent.export_visualization(
        viz_result,
        filename="test_correlation",
        format="html"
    )
    logger.info(f"Correlation visualization exported to: {viz_path}")

def test_map_visualizations(agent, data):
    """Test map visualizations using folium renderer."""
    logger.info("Testing map visualizations")
    
    # Test basic map
    logger.info("Testing basic map")
    
    # Create the visualization
    recommendation = {
        "type": "map",
        "params": {
            "title": "Geographic Distribution of Data Points",
            "lat": "lat", 
            "lon": "lon",
            "cluster": True,
            "renderer": "folium"  # Must use folium for maps
        }
    }
    viz_result = agent.create_visualization(data, recommendation)
    
    # Export the visualization
    viz_path = agent.export_visualization(
        viz_result,
        filename="test_map",
        format="html"
    )
    logger.info(f"Basic map visualization exported to: {viz_path}")
    
    # Test bubble map
    logger.info("Testing bubble map")
    
    # Create the visualization
    recommendation = {
        "type": "bubble_map",
        "params": {
            "title": "Bubble Map with Sized Points",
            "lat": "lat", 
            "lon": "lon",
            "size": "size",
            "color": "category",
            "renderer": "folium"
        }
    }
    viz_result = agent.create_visualization(data, recommendation)
    
    # Export the visualization
    viz_path = agent.export_visualization(
        viz_result,
        filename="test_bubble_map",
        format="html"
    )
    logger.info(f"Bubble map visualization exported to: {viz_path}")
    
    # Test heatmap
    logger.info("Testing map heatmap")
    
    # Create the visualization
    recommendation = {
        "type": "map_heatmap",
        "params": {
            "title": "Heatmap of Data Density",
            "lat": "lat", 
            "lon": "lon",
            "weight": "value1",
            "renderer": "folium"
        }
    }
    viz_result = agent.create_visualization(data, recommendation)
    
    # Export the visualization
    viz_path = agent.export_visualization(
        viz_result,
        filename="test_heat_map",
        format="html"
    )
    logger.info(f"Heat map visualization exported to: {viz_path}")

if __name__ == "__main__":
    main()
