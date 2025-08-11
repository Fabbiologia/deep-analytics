#!/usr/bin/env python3
"""
Column Mapper Module

This module provides utilities to map requested column names to actual column names
in DataFrames, handling cases where SQL queries use aliases or calculated columns.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('column_mapper')

class ColumnMapper:
    """
    Utility class to map requested visualization variables to actual DataFrame columns.
    Handles cases where SQL queries create columns with different names than what
    the visualization functions expect.
    """
    
    @staticmethod
    def map_columns(df: pd.DataFrame, viz_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map requested column names to actual columns in the DataFrame.
        
        Args:
            df: Input DataFrame
            viz_params: Visualization parameters including x and y variable names
            
        Returns:
            Updated viz_params with column mappings
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to map_columns")
            return viz_params
            
        updated_params = viz_params.copy()
        available_columns = set(df.columns)
        
        # Log available columns for debugging
        logger.info(f"Available columns in DataFrame: {available_columns}")
        logger.info(f"Requested viz params: {viz_params}")
        
        # Check and map 'x' parameter
        if 'x' in viz_params and viz_params['x'] not in available_columns:
            mapped_x = ColumnMapper._find_matching_column(viz_params['x'], available_columns, df)
            if mapped_x:
                logger.info(f"Mapping requested x '{viz_params['x']}' to available column '{mapped_x}'")
                updated_params['x'] = mapped_x
        
        # Check and map 'y' parameter
        if 'y' in viz_params and viz_params['y'] not in available_columns:
            mapped_y = ColumnMapper._find_matching_column(viz_params['y'], available_columns, df)
            if mapped_y:
                logger.info(f"Mapping requested y '{viz_params['y']}' to available column '{mapped_y}'")
                updated_params['y'] = mapped_y
                
        # Handle other common parameters that might need mapping
        for param in ['color', 'size', 'weight', 'category']:
            if param in viz_params and viz_params[param] not in available_columns:
                mapped_param = ColumnMapper._find_matching_column(viz_params[param], available_columns, df)
                if mapped_param:
                    logger.info(f"Mapping requested {param} '{viz_params[param]}' to '{mapped_param}'")
                    updated_params[param] = mapped_param
        
        return updated_params
    
    @staticmethod
    def _find_matching_column(requested: str, available: set, df: pd.DataFrame) -> Optional[str]:
        """
        Find the best matching column for a requested variable name.
        
        Args:
            requested: Requested column name
            available: Set of available column names
            df: DataFrame containing the data
            
        Returns:
            The name of the best matching column or None if no match found
        """
        # Direct match first
        if requested in available:
            return requested
            
        requested_lower = requested.lower()
        
        # Common mappings for known variables
        common_mappings = {
            'biomass': ['avgdensity', 'avgbiomass', 'biomass_density', 'density'],
            'year': ['years', 'yr', 'year_column'],
            'region': ['regions', 'location', 'site'],
            'species': ['taxa', 'taxa1', 'taxa2', 'species_name'],
            'latitude': ['lat', 'y_coord'],
            'longitude': ['lon', 'long', 'x_coord']
        }
        
        # Check for common mappings
        for key, alternatives in common_mappings.items():
            if requested_lower == key:
                for alt in alternatives:
                    for col in available:
                        if alt.lower() == col.lower():
                            return col
        
        # Check for partial matches in column names
        partial_matches = [col for col in available if requested_lower in col.lower()]
        if partial_matches:
            return partial_matches[0]  # Return the first partial match
            
        # For 'biomass' specifically, look for any density measures as that's often what we want
        if requested_lower == 'biomass' or 'biomass' in requested_lower:
            density_columns = [col for col in available if any(term in col.lower() 
                                                            for term in ['density', 'biomass', 'avg'])]
            if density_columns:
                return density_columns[0]  # Return the first density-related column
                
        # If we have a numeric column and it's the only one, use it (common for y-axis)
        if requested_lower in ['y', 'value', 'biomass', 'density', 'amount', 'count']:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) == 1:
                return numeric_cols[0]
                
        # No match found
        return None


# Function for easy access from other modules
def map_visualization_columns(df: pd.DataFrame, viz_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map visualization parameter column names to actual DataFrame columns.
    
    Args:
        df: DataFrame with the data
        viz_params: Visualization parameters
        
    Returns:
        Updated visualization parameters with mapped column names
    """
    return ColumnMapper.map_columns(df, viz_params)


# Test code
if __name__ == "__main__":
    # Create test data with a column name mismatch scenario
    test_df = pd.DataFrame({
        'Year': [2020, 2021, 2022, 2023],
        'Region': ['Cabo Pulmo', 'La Paz', 'Loreto', 'Cabo Pulmo'],
        'AvgDensity': [2.3, 1.8, 1.9, 2.5]  # This is actually biomass density
    })
    
    # Test with parameters that don't match column names
    test_params = {
        'x': 'Region',
        'y': 'Biomass',  # This doesn't exist, should map to AvgDensity
        'title': 'Average Biomass by Region'
    }
    
    # Map the columns
    updated_params = map_visualization_columns(test_df, test_params)
    
    print("Original DataFrame columns:", test_df.columns.tolist())
    print("Original parameters:", test_params)
    print("Updated parameters:", updated_params)
