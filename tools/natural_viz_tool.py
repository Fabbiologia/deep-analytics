#!/usr/bin/env python3
"""
Natural Language Visualization Tool

This module provides a natural language interface to the visualization system,
allowing users to request visualizations using plain English commands.

With visualization memory integration, it can now handle follow-up requests
like "Show me the same data as a bar chart" by maintaining context between requests.

Usage:
1. Import this module in main.py
2. Call create_natural_viz_tool() to get a tool function to add to the agent
"""

import json
import logging
from typing import Optional, Dict, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('natural_viz_tool')

try:
    from viz_request_parser import parse_visualization_request, build_tool_input
    VIZ_PARSER_AVAILABLE = True
except ImportError as e:
    VIZ_PARSER_AVAILABLE = False
    logger.error(f"Failed to import viz_request_parser: {e}")
    
try:
    from visualization_memory import (
        is_follow_up_viz_request, 
        get_last_viz_request
    )
    VIZ_MEMORY_AVAILABLE = True
    logger.info("Visualization memory system available for context maintenance.")
except ImportError:
    VIZ_MEMORY_AVAILABLE = False
    logger.error("Visualization memory system not available; follow-up requests may not work properly.")


def create_natural_viz_tool(advanced_viz_tool_func=None):
    """
    Create a tool function that accepts natural language visualization requests.
    
    Args:
        advanced_viz_tool_func: The function from the advanced visualization tool
        
    Returns:
        A tool function that can be added to the agent
    """
    if not VIZ_PARSER_AVAILABLE or not advanced_viz_tool_func:
        # Return a dummy function if dependencies aren't available
        def dummy_natural_viz_tool(query: str) -> str:
            return "Natural language visualization capabilities are not available."
        return dummy_natural_viz_tool
    
    def natural_viz_tool(query: str) -> str:
        """Process a natural language visualization request."""
        try:
            logger.info(f"Received natural language visualization request: {query}")
            
            # Parse the natural language request
            request_dict = parse_visualization_request(query)
            
            # Convert to JSON string for the advanced visualization tool
            tool_input = build_tool_input(request_dict)
            
            # Call the advanced visualization tool
            result = advanced_viz_tool_func(tool_input)
            
            return result
        except Exception as e:
            logger.error(f"Error in natural_viz_tool: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error creating visualization: {str(e)}"
    
    return natural_viz_tool


def get_natural_viz_tool_description() -> str:
    """
    Get the description for the natural language visualization tool.
    
    Returns:
        str: Tool description for LangChain
    """
    return """Create advanced data visualizations using natural language requests.
    
This tool accepts plain English requests and creates appropriate visualizations.
You can request various chart types including maps and interactive visualizations.

Examples of requests:
- "Create a line chart showing fish biomass over time in Cabo Pulmo"
- "Generate a map of sampling locations in La Paz region"
- "Make a bubble map showing fish biomass by location in Cabo Pulmo"
- "Create an interactive scatter plot of size vs biomass for fish in Loreto"
- "Show a correlation heatmap for all numeric variables in invertebrate data"

Follow-up requests are now supported! Examples:
- "Show me the same data as a bar chart"
- "Create a box plot of that same dataset"
- "Make that previous visualization interactive"
- "Show me that data for La Paz region instead"

The tool supports:
1. Chart types: line, bar, scatter, box plots, heatmaps
2. Map visualizations: basic maps, bubble maps, heatmaps on maps
3. Interactive vs static visualizations
4. Filtering by region, taxa (fish vs invertebrates), time periods
5. Custom x and y variables
6. Follow-up requests that reference previous visualizations

Best practices:
- Clearly specify the visualization type you want
- Mention the region of interest
- Specify whether you want data for fish (PEC) or invertebrates (INV)
- Mention if you want an interactive visualization
- For follow-ups, refer to "the same data" or "previous visualization"
"""
