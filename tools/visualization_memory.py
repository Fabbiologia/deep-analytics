#!/usr/bin/env python3
"""
Visualization Memory Module

This module provides a memory system for the visualization agent to maintain context
between visualization requests, enabling follow-up queries like "show me the same data as a bar chart".
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('viz_memory')

class VisualizationMemory:
    """Class for maintaining memory of previous visualization requests."""
    
    def __init__(self, max_history: int = 5, expiry_minutes: int = 30):
        """
        Initialize the visualization memory system.
        
        Args:
            max_history: Maximum number of previous requests to remember
            expiry_minutes: Number of minutes after which memory entries expire
        """
        self.memory: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.expiry_minutes = expiry_minutes
    
    def save_request(self, query: str, request_dict: Dict[str, Any]) -> None:
        """
        Save a visualization request to memory.
        
        Args:
            query: The original natural language query
            request_dict: The parsed request dictionary
        """
        timestamp = datetime.now()
        
        memory_entry = {
            "timestamp": timestamp,
            "query": query,
            "request": request_dict.copy(),
            "expires": timestamp + timedelta(minutes=self.expiry_minutes)
        }
        
        # Add to memory and maintain max size
        self.memory.append(memory_entry)
        if len(self.memory) > self.max_history:
            self.memory.pop(0)  # Remove oldest entry
        
        logger.info(f"Saved visualization request to memory: {query}")
    
    def get_last_request(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent visualization request.
        
        Returns:
            The most recent request dict or None if no valid requests exist
        """
        # Clean expired entries
        self._clean_expired()
        
        # Return most recent if available
        if self.memory:
            return self.memory[-1]["request"]
        return None
    
    def _clean_expired(self) -> None:
        """Remove expired entries from memory."""
        now = datetime.now()
        self.memory = [entry for entry in self.memory if entry["expires"] > now]
    
    def is_follow_up_request(self, query: str) -> bool:
        """
        Determine if a query is likely a follow-up to a previous visualization.
        
        Args:
            query: The natural language query
        
        Returns:
            True if the query appears to be a follow-up request
        """
        # Clean expired entries
        self._clean_expired()
        
        if not self.memory:
            return False
        
        # Check for common follow-up phrases
        follow_up_phrases = [
            "same data", "same thing", "same variables", "similar", 
            "like before", "previous", "that data", "same dataset",
            "same region", "same analysis", "now show", "instead show",
            "show me the", "same species"
        ]
        
        query_lower = query.lower()
        return any(phrase in query_lower for phrase in follow_up_phrases)
    
    def augment_request(self, query: str, new_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augment a new request with context from previous requests if it's a follow-up.
        
        Args:
            query: The natural language query
            new_request: The new parsed request dictionary
        
        Returns:
            Augmented request dictionary
        """
        if not self.is_follow_up_request(query):
            return new_request
        
        last_request = self.get_last_request()
        if not last_request:
            return new_request
        
        # Preserve new visualization type and renderer if specified
        augmented_request = {**last_request}  # Start with previous request
        
        # Always use the new visualization type if specified
        if "viz_type" in new_request:
            augmented_request["viz_type"] = new_request["viz_type"]
        
        # Always use new parameters if specified
        if "params" in new_request:
            # Start with previous params
            augmented_params = augmented_request.get("params", {}).copy()
            
            # Override with new params
            for key, value in new_request.get("params", {}).items():
                augmented_params[key] = value
                
            # Update filename to reflect the new visualization type
            if "viz_type" in new_request:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                augmented_params["filename"] = f"{new_request['viz_type']}_{timestamp}"
            
            augmented_request["params"] = augmented_params
        
        # Use the new SQL query if provided, otherwise keep the old one
        if "query" in new_request and new_request["query"]:
            augmented_request["query"] = new_request["query"]
        
        logger.info(f"Augmented follow-up request: {json.dumps(augmented_request)}")
        return augmented_request


# Singleton instance for global access
visualization_memory = VisualizationMemory()


# Helper functions for module-level access
def save_viz_request(query: str, request_dict: Dict[str, Any]) -> None:
    """Save a visualization request to the memory system."""
    visualization_memory.save_request(query, request_dict)


def get_last_viz_request() -> Optional[Dict[str, Any]]:
    """Get the most recent visualization request."""
    return visualization_memory.get_last_request()


def is_follow_up_viz_request(query: str) -> bool:
    """Check if a query is a follow-up visualization request."""
    return visualization_memory.is_follow_up_request(query)


def augment_viz_request(query: str, new_request: Dict[str, Any]) -> Dict[str, Any]:
    """Augment a new request with context from previous ones if needed."""
    return visualization_memory.augment_request(query, new_request)
