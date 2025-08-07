"""
Data validation module for the Ecological Monitoring Agent.
This module provides functions to validate data against the database.
"""

import pandas as pd
from typing import List, Dict, Any, Union, Optional
import re

class DataValidator:
    """
    A class to validate data against the database to prevent hallucination in agent responses.
    """
    
    def __init__(self, engine=None):
        """
        Initialize the DataValidator with a database engine.
        
        Args:
            engine: SQLAlchemy engine for database connection
        """
        self.engine = engine
        self._cached_values = {}  # Cache for database values to reduce repeated queries
    
    def set_engine(self, engine):
        """Set the database engine."""
        self.engine = engine
        self._cached_values = {}  # Reset cache when engine changes
    
    def get_valid_values(self, column_name: str, table_name: str = "ltem_historical_database") -> List[str]:
        """
        Get a list of valid values for a specific column from the database.
        
        Args:
            column_name: Name of the column to get values for
            table_name: Name of the table to query (default: ltem_historical_database)
            
        Returns:
            List of valid values for the column
        """
        cache_key = f"{table_name}.{column_name}"
        
        # Return cached values if available
        if cache_key in self._cached_values:
            return self._cached_values[cache_key]
        
        if self.engine is None:
            return []  # Can't query without an engine
            
        try:
            # Query distinct values for the column
            query = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL"
            df = pd.read_sql(query, self.engine)
            
            # Get the values as a list and cache them
            values = df[column_name].astype(str).tolist()
            self._cached_values[cache_key] = values
            return values
        except Exception as e:
            print(f"Error getting valid values for {column_name}: {e}")
            return []
    
    def validate_regions(self, regions: List[str]) -> List[str]:
        """
        Filter a list of region names to only include valid regions in the database.
        
        Args:
            regions: List of region names to validate
            
        Returns:
            List of valid region names that exist in the database
        """
        valid_regions = self.get_valid_values("Region")
        return [r for r in regions if r in valid_regions]
    
    def validate_reefs(self, reefs: List[str]) -> List[str]:
        """
        Filter a list of reef names to only include valid reefs in the database.
        
        Args:
            reefs: List of reef names to validate
            
        Returns:
            List of valid reef names that exist in the database
        """
        valid_reefs = self.get_valid_values("Reef")
        return [r for r in reefs if r in valid_reefs]
    
    def validate_species(self, species: List[str]) -> List[str]:
        """
        Filter a list of species names to only include valid species in the database.
        
        Args:
            species: List of species names to validate
            
        Returns:
            List of valid species names that exist in the database
        """
        valid_species = self.get_valid_values("Species")
        return [s for s in species if s in valid_species]

    def extract_and_validate_regions(self, text: str) -> Dict[str, bool]:
        """
        Extract mentioned region names from text and validate them against the database.
        
        Args:
            text: Text to extract region names from
            
        Returns:
            Dictionary mapping region names to boolean indicating if they're valid
        """
        valid_regions = self.get_valid_values("Region")
        
        # Create pattern to match region names (case insensitive)
        region_pattern = r'region[s]?[\s:]*([A-Za-z\s]+)'
        
        # Extract potential region names
        matches = re.finditer(region_pattern, text.lower())
        mentioned_regions = {}
        
        for match in matches:
            region_text = match.group(1).strip()
            # Split multiple regions if separated by commas or "and"
            for region in re.split(r',|\sand\s', region_text):
                region = region.strip().title()  # Capitalize for consistency
                if region:
                    mentioned_regions[region] = region in valid_regions
        
        # Also check directly for each valid region in the text
        for valid_region in valid_regions:
            if valid_region.lower() in text.lower() and valid_region not in mentioned_regions:
                mentioned_regions[valid_region] = True
                
        return mentioned_regions

    def sanitize_response(self, response: str) -> str:
        """
        Sanitize an agent response to flag any hallucinated regions or reefs.
        
        Args:
            response: The response text to sanitize
            
        Returns:
            Sanitized response with hallucinated entities flagged
        """
        # Get mentioned regions and their validity
        mentioned_regions = self.extract_and_validate_regions(response)
        
        # Flag invalid regions
        sanitized = response
        for region, is_valid in mentioned_regions.items():
            if not is_valid and len(region) > 3:  # Avoid replacing short strings
                # Flag the invalid region with a warning note
                sanitized = re.sub(
                    fr'\b{re.escape(region)}\b', 
                    f"{region} [WARNING: This region does not exist in the database]", 
                    sanitized
                )
                
        return sanitized
