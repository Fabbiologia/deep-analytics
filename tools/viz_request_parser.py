#!/usr/bin/env python3
"""
Visualization Request Parser

This module helps the main agent interpret natural language visualization requests
and convert them to the format required by the advanced visualization tool.

It now integrates with visualization_memory to maintain context between requests,
enabling follow-up visualization queries.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Union

# Import the visualization memory system
try:
    from visualization_memory import (
        save_viz_request,
        get_last_viz_request,
        is_follow_up_viz_request,
        augment_viz_request
    )
    VIZ_MEMORY_AVAILABLE = True
    print("Visualization memory system loaded successfully.")
except ImportError:
    VIZ_MEMORY_AVAILABLE = False
    print("Warning: visualization_memory module not available.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('viz_request_parser')

# Define visualization type mappings
VIZ_TYPE_MAPPINGS = {
    # Basic chart types
    'line': ['line chart', 'time series', 'trend', 'over time', 'across years', 'yearly'],
    'bar': ['bar chart', 'bar graph', 'histogram', 'comparison'],
    'scatter': ['scatter plot', 'scatterplot', 'relationship', 'correlation plot'],
    'box': ['box plot', 'boxplot', 'distribution', 'variation'],
    'heatmap': ['heatmap', 'correlation heatmap', 'correlation matrix'],
    
    # Map visualizations
    'map': ['map', 'locations', 'sites', 'sampling sites', 'survey sites'],
    'bubble_map': ['bubble map', 'sized markers', 'bubbles on map', 'sized points'],
    'map_heatmap': ['heat map', 'density map', 'spatial density', 'hotspot']
}

# Define renderer mappings
RENDERER_MAPPINGS = {
    'matplotlib': ['static', 'simple', 'basic', 'matplotlib'],
    'plotly': ['interactive', 'dynamic', 'clickable', 'plotly', 'hoverable'],
    'altair': ['grammar of graphics', 'altair', 'vega', 'vega-lite'],
    'folium': ['map', 'leaflet', 'folium', 'geographic', 'spatial']
}

def detect_visualization_type(query: str) -> str:
    """
    Detect the visualization type from a natural language query.
    
    Args:
        query: The natural language query
        
    Returns:
        The detected visualization type or 'line' as default
    """
    query = query.lower()
    
    # Check for map visualizations first (they're more specific)
    if any(term in query for term in ['bubble map', 'sized markers']):
        return 'bubble_map'
    elif any(term in query for term in ['heat map on map', 'density map']):
        return 'map_heatmap'
    elif any(term in query for term in ['map', 'locations', 'spatial', 'geographic']):
        return 'map'
    
    # Check for other visualization types
    for viz_type, keywords in VIZ_TYPE_MAPPINGS.items():
        if any(keyword in query for keyword in keywords):
            return viz_type
    
    # Default to line chart
    return 'line'

def detect_renderer(query: str, viz_type: str) -> Optional[str]:
    """
    Detect the preferred renderer from a natural language query.
    
    Args:
        query: The natural language query
        viz_type: The detected visualization type
        
    Returns:
        The detected renderer or None to use default
    """
    query = query.lower()
    
    # Maps always use folium
    if viz_type in ['map', 'bubble_map', 'map_heatmap']:
        return 'folium'
    
    # Check for specific renderer requests
    for renderer, keywords in RENDERER_MAPPINGS.items():
        if any(keyword in query for keyword in keywords):
            return renderer
    
    # Default renderer preferences by visualization type
    renderer_preferences = {
        'line': 'plotly',  # Interactive is good for trends
        'bar': 'plotly',   # Interactive is good for comparisons
        'scatter': 'plotly',  # Interactive for exploring relationships
        'box': 'matplotlib',  # Static works well for boxplots
        'heatmap': 'plotly'   # Interactive heatmaps are informative
    }
    
    return renderer_preferences.get(viz_type)

def detect_region(query: str) -> Optional[str]:
    """
    Detect the region filter from a natural language query.
    
    Args:
        query: The natural language query
        
    Returns:
        The detected region or None
    """
    # Look for regions mentioned in the query
    regions = ['Cabo Pulmo', 'La Paz', 'Loreto']
    
    for region in regions:
        pattern = rf'\b{re.escape(region)}\b'
        if re.search(pattern, query, re.IGNORECASE):
            return region
    
    return None

def detect_taxa_filter(query: str) -> Optional[Dict[str, str]]:
    """
    Detect taxa filters from a natural language query.
    
    Args:
        query: The natural language query
        
    Returns:
        Dict with taxa filters or None
    """
    filters = {}
    
    # Check for fish vs invertebrates
    if any(term in query.lower() for term in ['fish', 'fishes', 'pec']):
        filters['Label'] = 'PEC'
    elif any(term in query.lower() for term in ['invertebrate', 'invertebrates', 'inv']):
        filters['Label'] = 'INV'
    
    return filters if filters else None

def detect_time_filter(query: str) -> Optional[Dict[str, Union[int, List[int]]]]:
    """
    Detect time filters from a natural language query.
    
    Args:
        query: The natural language query
        
    Returns:
        Dict with time filters or None
    """
    # Check for year ranges
    year_range_pattern = r'(\d{4})\s*-\s*(\d{4})'
    match = re.search(year_range_pattern, query)
    if match:
        return {
            'start_year': int(match.group(1)),
            'end_year': int(match.group(2))
        }
    
    # Check for specific years
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
    if years:
        return {'years': [int(year) for year in years]}
    
    return None

def detect_x_y_variables(query: str) -> Dict[str, str]:
    """
    Detect x and y variables from a natural language query.
    
    Args:
        query: The natural language query
        
    Returns:
        Dict with x and y variables
    """
    result = {}
    
    # Common column mappings
    column_mappings = {
        'biomass': 'Biomass',
        'year': 'Year',
        'reef': 'Reef',
        'region': 'Region',
        'depth': 'Depth',
        'size': 'Size',
        'quantity': 'Quantity',
        'area': 'Area',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'trophic level': 'TrophicLevel'
    }
    
    # Try to identify by context clues in the query
    query_lower = query.lower()
    
    # Look for "X by Y" pattern
    by_pattern = r'(\w+)\s+by\s+(\w+)'
    match = re.search(by_pattern, query_lower)
    if match:
        y_var = match.group(1)
        x_var = match.group(2)
        
        # Map to actual column names
        if y_var in column_mappings:
            result['y'] = column_mappings[y_var]
        if x_var in column_mappings:
            result['x'] = column_mappings[x_var]
    
    # Look for "over/across years/time" pattern
    if any(term in query_lower for term in ['over years', 'across years', 'over time', 'across time']):
        result['x'] = 'Year'
    
    # If "biomass" is mentioned, it's likely the y variable
    if 'biomass' in query_lower:
        result['y'] = 'Biomass'
    
    # Default values if not found
    if 'x' not in result:
        result['x'] = 'Year'  # Default x to Year
    
    if 'y' not in result:
        result['y'] = 'Biomass'  # Default y to Biomass
    
    return result

def build_sql_query(viz_type: str, params: Dict[str, Any]) -> str:
    """
    Build an SQL query based on visualization type and parameters.
    
    Args:
        viz_type: The visualization type
        params: The visualization parameters
        
    Returns:
        SQL query string
    """
    # Base query
    select_clause = "SELECT "
    from_clause = " FROM ltem_optimized_regions"
    where_clauses = []
    group_by_clause = ""
    
    # Add region filter if specified
    if 'region' in params:
        where_clauses.append(f"Region = '{params['region']}'")
    
    # Add taxa filter if specified
    if 'label' in params:
        where_clauses.append(f"Label = '{params['label']}'")
    
    # Add time filter if specified
    if 'start_year' in params and 'end_year' in params:
        where_clauses.append(f"Year BETWEEN {params['start_year']} AND {params['end_year']}")
    elif 'years' in params:
        years_str = ', '.join(str(year) for year in params['years'])
        where_clauses.append(f"Year IN ({years_str})")
    
    # Build query based on visualization type
    if viz_type == 'line':
        x_col = params.get('x', 'Year')
        y_col = params.get('y', 'Biomass')
        
        # Use the correct aggregation method: SUM per transect, then AVG across groups
        # For a more accurate ecological biomass calculation
        if y_col.lower() in ['biomass', 'quantity', 'count', 'abundance']:
            select_clause += f"{x_col}, AVG(SUM({y_col})/SUM(Area)) as Biomass"
            logger.info(f"Using ecological aggregation for {y_col} - sum per transect, then average")
        else:
            select_clause += f"{x_col}, AVG({y_col}/Area) as AvgDensity"
            logger.info(f"Using standard density calculation for {y_col}")
            
        group_by_clause = f" GROUP BY {x_col}"
        
    elif viz_type == 'bar':
        x_col = params.get('x', 'Year')
        y_col = params.get('y', 'Biomass')
        
        # Use the correct aggregation method: SUM per transect, then AVG across groups
        # For a more accurate ecological biomass calculation
        if y_col.lower() in ['biomass', 'quantity', 'count', 'abundance']:
            select_clause += f"{x_col}, AVG(SUM({y_col})/SUM(Area)) as Biomass"
            logger.info(f"Using ecological aggregation for {y_col} - sum per transect, then average")
        else:
            select_clause += f"{x_col}, AVG({y_col}/Area) as AvgDensity"
            logger.info(f"Using standard density calculation for {y_col}")
            
        group_by_clause = f" GROUP BY {x_col}"
        
    elif viz_type == 'scatter':
        x_col = params.get('x', 'Size')
        y_col = params.get('y', 'Biomass')
        
        select_clause += f"{x_col}, {y_col}/Area as Density"
        
    elif viz_type == 'heatmap':
        select_clause += "Taxa1, Taxa2, COUNT(*) as Count"
        group_by_clause = " GROUP BY Taxa1, Taxa2"
        
    elif viz_type in ['map', 'bubble_map', 'map_heatmap']:
        if viz_type == 'map':
            select_clause += "AVG(Longitude) as Longitude, AVG(Latitude) as Latitude, Reef, COUNT(*) as SampleCount"
            group_by_clause = " GROUP BY Reef"
        elif viz_type == 'bubble_map':
            select_clause += "AVG(Longitude) as Longitude, AVG(Latitude) as Latitude, Reef, AVG(Biomass/Area) as AvgBiomass"
            group_by_clause = " GROUP BY Reef"
        elif viz_type == 'map_heatmap':
            select_clause += "Longitude, Latitude, Biomass/Area as BiomassPerArea"
    
    # Add where clause if filters are specified
    where_clause = ""
    if where_clauses:
        where_clause = " WHERE " + " AND ".join(where_clauses)
    
    # For advanced ecological calculations that require subqueries
    # We need to handle biomass calculations with proper transect-level aggregation
    # If we detect this is a biomass/quantity calculation and we have a GROUP BY
    y_col = params.get('y', '')
    if (y_col.lower() in ['biomass', 'quantity', 'count', 'abundance']) and group_by_clause:
        # For biomass calculations with a grouping factor, we need a more complex query structure
        # First sum by transect, then average per group (e.g., Region)
        x_col = params.get('x', '')
        if x_col:
            # Use a subquery approach to get proper ecological aggregation
            base_query = f"""WITH TransectSums AS (
                SELECT {x_col}, TransectID, SUM({y_col}) as TotalBiomass, SUM(Area) as TotalArea 
                FROM ltem_optimized_regions
                {where_clause}
                GROUP BY {x_col}, TransectID
            )
            SELECT {x_col}, AVG(TotalBiomass/TotalArea) as Biomass
            FROM TransectSums
            GROUP BY {x_col}
            """
            return base_query
    
    # Standard query building
    query = select_clause + from_clause + where_clause + group_by_clause
    
    return query

def parse_visualization_request(query: str) -> Dict[str, Any]:
    """
    Parse a natural language visualization request into parameters for the visualization tool.
    
    Args:
        query: The natural language query requesting a visualization
        
    Returns:
        Dict with parameters for the advanced visualization tool
    """
    try:
        # Check if this is a follow-up visualization request that should inherit context
        is_follow_up = False
        if VIZ_MEMORY_AVAILABLE and is_follow_up_viz_request(query):
            is_follow_up = True
            logger.info(f"Detected follow-up visualization request: {query}")
        
        # Extract visualization type
        viz_type = detect_visualization_type(query)
        logger.info(f"Detected visualization type: {viz_type}")
        
        # Extract renderer
        renderer = detect_renderer(query, viz_type)
        logger.info(f"Using renderer: {renderer}")
        
        # Extract region filter
        region = detect_region(query)
        logger.info(f"Region filter: {region}")
        
        # Extract taxa filter
        taxa_filters = detect_taxa_filter(query)
        logger.info(f"Taxa filters: {taxa_filters}")
        
        # Extract time filter
        time_filter = detect_time_filter(query)
        logger.info(f"Time filter: {time_filter}")
        
        # Extract x and y variables
        xy_vars = detect_x_y_variables(query)
        logger.info(f"X and Y variables: {xy_vars}")
        
        # Combine parameters
        params = {}
        if region:
            params['region'] = region
        
        if taxa_filters:
            for k, v in taxa_filters.items():
                params[k.lower()] = v
        
        if time_filter:
            params.update({k: v for k, v in time_filter.items()})
        
        if xy_vars:
            params.update({k: v for k, v in xy_vars.items()})
        
        # Add renderer
        if renderer:
            params['renderer'] = renderer
        
        # Add filename
        timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
        params['filename'] = f"{viz_type}_{timestamp}"
        
        # Add title based on the query
        params['title'] = f"{viz_type.replace('_', ' ').title()} - {region}" if region else f"{viz_type.replace('_', ' ').title()} Visualization"
        
        # Build SQL query (only if not a follow-up or if we need a new query)
        sql_query = build_sql_query(viz_type, params) if not is_follow_up else ""
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Build initial result
        result = {
            "query": sql_query,
            "viz_type": viz_type,
            "params": params
        }
        
        # If this is a follow-up request, augment it with context from previous request
        if is_follow_up and VIZ_MEMORY_AVAILABLE:
            result = augment_viz_request(query, result)
            logger.info(f"Augmented request with previous context: {json.dumps(result)}")
        
        # Save this request to memory for future follow-ups
        if VIZ_MEMORY_AVAILABLE:
            save_viz_request(query, result)
            
        return result
        
    except Exception as e:
        logger.error(f"Error parsing visualization request: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return a simple default
        return {
            "query": "SELECT Year, AVG(Biomass/Area) as AvgDensity FROM ltem_optimized_regions GROUP BY Year",
            "viz_type": "line",
            "params": {
                "x": "Year",
                "y": "AvgDensity",
                "title": "Default Visualization",
                "filename": "default_viz"
            }
        }

def build_tool_input(request_dict: Dict[str, Any]) -> str:
    """
    Convert request dict to JSON string for the advanced visualization tool.
    
    Args:
        request_dict: Dict with request parameters
        
    Returns:
        JSON string for the advanced visualization tool
    """
    return json.dumps(request_dict)


if __name__ == "__main__":
    # Test with some example queries
    test_queries = [
        "Create a line chart showing fish biomass over the years in Cabo Pulmo",
        "Show me a map of sampling locations in La Paz region",
        "Generate a bubble map of fish biomass by location in Cabo Pulmo",
        "Create an interactive scatter plot of size vs biomass for fish in Loreto",
        "Make a correlation heatmap of all numeric variables for invertebrates"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = parse_visualization_request(query)
        print(f"Result: {json.dumps(result, indent=2)}")
