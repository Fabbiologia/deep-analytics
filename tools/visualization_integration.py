#!/usr/bin/env python3
"""
Visualization Integration Module

This module provides integration functions to add the advanced multi-renderer
visualization system (Matplotlib, Plotly, Altair, Folium) to the main application.

Usage:
1. Import this module in main.py
2. Call integrate_visualization_system() to add the visualization capabilities
3. Add the returned tool to your agent's tools list
"""

import os
import time
import json
import logging
import sys
import pandas as pd
from typing import Dict, Any, List, Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(__file__))

# Import column mapper module
try:
    from column_mapper import map_visualization_columns
    COLUMN_MAPPER_AVAILABLE = True
    logger.info("Column mapper module imported successfully")
except ImportError as e:
    COLUMN_MAPPER_AVAILABLE = False
    logger.info(f"Column mapper module not available: {e}. Column name matching will be limited.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='visualization_integration.log'
)
logger = logging.getLogger('visualization_integration')

# Try to import visualization components
try:
    from visualization_agent import VisualizationAgent
    from visualization_factory import VisualizationFactory
    VISUALIZATION_AGENT_AVAILABLE = True
    logger.info("Visualization Agent and Factory imported successfully")
except ImportError as e:
    VISUALIZATION_AGENT_AVAILABLE = False
    logger.error(f"Failed to import visualization modules: {e}")


def initialize_visualization_system() -> Dict[str, Any]:
    """
    Initialize the visualization system components.
    
    Returns:
        Dict with visualization components and availability status
    """
    components = {
        "available": False,
        "factory": None,
        "agent": None
    }
    
    if VISUALIZATION_AGENT_AVAILABLE:
        try:
            viz_factory = VisualizationFactory()
            viz_agent = VisualizationAgent(visualization_factory=viz_factory)
            components["available"] = True
            components["factory"] = viz_factory
            components["agent"] = viz_agent
            logger.info("Visualization system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize visualization system: {e}")
    
    return components


def create_visualization_tool(components: Dict[str, Any], engine=None):
    """
    Create a LangChain tool for the visualization system.
    
    Args:
        components: Dict with visualization components from initialize_visualization_system()
        engine: SQLAlchemy engine for database connection
        
    Returns:
        A tool function that can be added to the agent
    """
    if not components["available"]:
        # Return a dummy function if components aren't available
        def dummy_visualization_tool(input_str):
            return "Advanced visualization capabilities are not available. Please install the required dependencies."
        return dummy_visualization_tool
    
    viz_agent = components["agent"]
    
    def advanced_visualization_tool(input_str):
        """Function to create visualizations using the VisualizationAgent."""
        try:
            # Parse the input JSON
            viz_params = json.loads(input_str)
            
            # Extract parameters
            data_query = viz_params.get('query')
            viz_type = viz_params.get('viz_type')
            viz_params_dict = viz_params.get('params', {})
            filename = viz_params_dict.get('filename', f"viz_{int(time.time())}")
            
            # Log what we're doing
            logger.info(f"Creating visualization: type={viz_type}, filename={filename}")
            logger.info(f"Using query: {data_query}")
            
            # Get data from database
            if not engine:
                return "Error: Database engine is not available."
                
            data = pd.read_sql(data_query, engine)
            logger.info(f"Query returned DataFrame with shape {data.shape}")
            
            # Apply column mapping to handle column name mismatches
            if COLUMN_MAPPER_AVAILABLE:
                logger.info("Applying column mapping to handle potential column name mismatches")
                mapped_params = map_visualization_columns(data, viz_params_dict)
                logger.info(f"Original params: {viz_params_dict}")
                logger.info(f"Mapped params: {mapped_params}")
                viz_params_dict = mapped_params
            else:
                logger.info("Column mapper not available, using parameters as-is")
                
            # Enhance with proper axis labels
            # If y-axis is biomass-related, ensure it has a proper label regardless of column name
            y_col = viz_params_dict.get('y', '')
            if y_col and isinstance(y_col, str):
                # For biomass measurements, use a standardized label
                if y_col.lower() in ['biomass', 'avgdensity', 'density', 'totalbiomass'] or 'biomas' in y_col.lower():
                    viz_params_dict['ylabel'] = 'Biomass'
                    logger.info(f"Set ylabel to 'Biomass' for better readability")
                # For other ecological measurements
                elif y_col.lower() in ['abundance', 'quantity', 'count', 'totalcount']:
                    viz_params_dict['ylabel'] = y_col.title()
                    logger.info(f"Set ylabel to '{y_col.title()}'")
            
            # Set proper x-axis label too
            x_col = viz_params_dict.get('x', '')
            if x_col and isinstance(x_col, str) and not viz_params_dict.get('xlabel'):
                viz_params_dict['xlabel'] = x_col.title()
                logger.info(f"Set xlabel to '{x_col.title()}'")
                
            # Add proper title if missing
            if not viz_params_dict.get('title') and x_col and y_col:
                if isinstance(y_col, str):
                    viz_params_dict['title'] = f"{y_col.title()} by {x_col.title()}"
                    logger.info(f"Set auto-generated title: {viz_params_dict['title']}")
            
            # Create recommendation dict
            recommendation = {
                "type": viz_type,
                "params": viz_params_dict
            }
            
            # Create and export the visualization
            viz_result = viz_agent.create_visualization(data, recommendation)
            filepath = viz_agent.export_visualization(viz_result, filename=filename)
            
            # Determine what was actually plotted for user feedback
            y_label = viz_params_dict.get('ylabel', y_col if isinstance(y_col, str) else 'Value')
            x_label = viz_params_dict.get('xlabel', x_col if isinstance(x_col, str) else 'Category')
            
            # Enhanced response with what was actually plotted
            return f"Visualization created and saved to {filepath} (plotting {y_label} by {x_label})"
        except Exception as e:
            logger.error(f"Error in advanced_visualization_tool: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error creating visualization: {e}"
    
    return advanced_visualization_tool


def get_visualization_tool_description() -> str:
    """
    Get the description for the visualization tool.
    
    Returns:
        str: Tool description for LangChain
    """
    return """Use this to create advanced visualizations including maps, interactive plots, and more.
    
Input should be a JSON string with these fields:
- query: SQL query to get data
- viz_type: Type of visualization to create (options below)
- params: Dictionary of parameters for the visualization
- params.filename: Optional filename for the output

Available visualization types:
1. Basic charts (all renderers):
   - line: Line chart for time series or trends
   - bar: Bar chart for comparisons
   - scatter: Scatter plot for relationships between variables
   - histogram: Distribution of a single variable
   - heatmap: Correlation or density matrix
   - box_plot: Box plot for distribution and outliers

2. Map visualizations (Folium):
   - map: Basic map with markers
   - bubble_map: Map with sized bubbles for quantitative values
   - map_heatmap: Heatmap overlay on a map

3. Parameters by visualization type:
   - line, bar: x (column name), y (column name or list of columns)
   - scatter: x, y, color (optional), size (optional)
   - histogram: x, bins (optional)
   - heatmap: No required params, will use correlation of numeric columns
   - map, bubble_map, map_heatmap: lat (column name), lon (column name)
   - bubble_map: size (column for bubble size), color (optional)
   - map_heatmap: weight (column for heat intensity)

All visualizations accept:
- title: Chart title
- renderer: Explicitly choose "matplotlib", "plotly", "altair", or "folium"
  (maps require "folium", defaults chosen automatically otherwise)

Example: 
{
  "query": "SELECT Year, AVG(Biomass/Area) as AvgDensity FROM ltem_optimized_regions WHERE Region='Cabo Pulmo' AND Label='PEC' GROUP BY Year",
  "viz_type": "line",
  "params": {
    "title": "Fish Biomass Density in Cabo Pulmo",
    "x": "Year",
    "y": "AvgDensity",
    "filename": "cabo_pulmo_biomass_trend",
    "renderer": "plotly"
  }
}
"""


def integrate_visualization_system(engine=None):
    """
    Integrate the visualization system into the main application.
    
    Args:
        engine: SQLAlchemy engine for database connection
        
    Returns:
        Tuple of (tool_func, tool_description, components)
    """
    # Initialize the visualization system
    components = initialize_visualization_system()
    
    # Create the tool function
    tool_func = create_visualization_tool(components, engine)
    
    # Get the tool description
    tool_description = get_visualization_tool_description()
    
    return tool_func, tool_description, components


# When run directly, test the visualization system
if __name__ == "__main__":
    print("Testing visualization integration...")
    components = initialize_visualization_system()
    if components["available"]:
        print("✅ Visualization system is available")
        print(f"Available renderers: {components['factory'].get_available_renderers()}")
    else:
        print("❌ Visualization system is not available")
