#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Agent Module

This module provides an agent for generating and managing visualizations
based on data analysis needs and context.

Created: August 2025
"""

import os
import logging
import datetime
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

# Import visualization factory
try:
    from visualization_factory import VisualizationFactory
except ImportError:
    raise ImportError("VisualizationFactory module not found. Please ensure it's installed correctly.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='visualization_agent.log'
)

logger = logging.getLogger('visualization_agent')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


class VisualizationAgent:
    """
    Agent for generating and managing visualizations based on data analysis needs.
    
    This agent can:
    1. Recommend appropriate visualization types based on data characteristics
    2. Generate visualizations using the VisualizationFactory
    3. Export visualizations to various formats
    4. Track visualization history and metadata
    """
    
    def __init__(self, visualization_factory: Optional[VisualizationFactory] = None):
        """Initialize the visualization agent."""
        self.visualization_factory = visualization_factory or VisualizationFactory()
        self.visualization_history = []
        
        # Set seed for reproducibility
        np.random.seed(123)
        
        logger.info("Visualization Agent initialized")
    
    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the input data to determine its characteristics.
        
        Returns a dictionary with data characteristics:
        - num_rows: Number of rows
        - num_cols: Number of columns
        - dtypes: Data types of columns
        - num_categorical: Number of categorical columns
        - num_numerical: Number of numerical columns
        - num_datetime: Number of datetime columns
        - missing_values: Dictionary of missing values by column
        """
        logger.info(f"Analyzing data with shape {data.shape}")
        
        # Basic data characteristics
        analysis = {
            "num_rows": len(data),
            "num_cols": len(data.columns),
            "column_names": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "missing_values": {col: int(data[col].isna().sum()) for col in data.columns},
            "categorical_columns": [],
            "numerical_columns": [],
            "datetime_columns": [],
            "spatial_columns": [],
            "unique_values": {}
        }
        
        # Categorize columns by data type
        for col in data.columns:
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                analysis["datetime_columns"].append(col)
            # Check for numerical
            elif pd.api.types.is_numeric_dtype(data[col]):
                analysis["numerical_columns"].append(col)
            # Check for categorical/text
            else:
                analysis["categorical_columns"].append(col)
                
            # Count unique values for non-numeric columns
            if not pd.api.types.is_numeric_dtype(data[col]):
                unique_count = data[col].nunique()
                analysis["unique_values"][col] = min(unique_count, 10)  # Cap at 10 for performance
        
        # Check for potential spatial data (columns with lat/long in the name)
        spatial_keywords = ['lat', 'latitude', 'lon', 'long', 'longitude', 'geom', 'geometry', 'point']
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in spatial_keywords):
                analysis["spatial_columns"].append(col)
        
        # Count statistics
        analysis["num_categorical"] = len(analysis["categorical_columns"])
        analysis["num_numerical"] = len(analysis["numerical_columns"])
        analysis["num_datetime"] = len(analysis["datetime_columns"])
        
        logger.info(f"Data analysis completed. Found {analysis['num_numerical']} numerical, "
                   f"{analysis['num_categorical']} categorical, and {analysis['num_datetime']} datetime columns")
        
        return analysis
    
    def recommend_visualization(self, data: pd.DataFrame, 
                              analysis_type: Optional[str] = None,
                              columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recommend appropriate visualization types based on data characteristics.
        
        Parameters:
        - data: Input DataFrame
        - analysis_type: Type of analysis (temporal, spatial, categorical, etc.)
        - columns: Optional list of columns to focus on
        
        Returns a list of recommended visualizations with parameters.
        """
        logger.info(f"Recommending visualization for analysis_type={analysis_type}")
        
        # Analyze data characteristics
        data_analysis = self.analyze_data(data)
        recommendations = []
        
        # Filter columns if specified
        available_columns = set(data.columns)
        if columns:
            focus_columns = [col for col in columns if col in available_columns]
            if not focus_columns:
                logger.warning(f"None of the specified columns {columns} found in data")
                focus_columns = list(available_columns)
        else:
            focus_columns = list(available_columns)
        
        # Base recommendations on analysis type
        if analysis_type:
            analysis_type = analysis_type.lower()
            
            # Temporal analysis
            if analysis_type in ["temporal", "time", "time series"]:
                self._recommend_temporal_visualizations(data, data_analysis, focus_columns, recommendations)
            
            # Spatial analysis
            elif analysis_type in ["spatial", "geographic", "geo", "map"]:
                self._recommend_spatial_visualizations(data, data_analysis, focus_columns, recommendations)
            
            # Categorical analysis
            elif analysis_type in ["categorical", "category"]:
                self._recommend_categorical_visualizations(data, data_analysis, focus_columns, recommendations)
            
            # Correlation analysis
            elif analysis_type in ["correlation", "corr", "relationship"]:
                self._recommend_correlation_visualizations(data, data_analysis, focus_columns, recommendations)
            
            # Distribution analysis
            elif analysis_type in ["distribution", "dist"]:
                self._recommend_distribution_visualizations(data, data_analysis, focus_columns, recommendations)
            
            # Comparison analysis
            elif analysis_type in ["comparison", "compare"]:
                self._recommend_comparison_visualizations(data, data_analysis, focus_columns, recommendations)
                
        # If no specific analysis type or no recommendations, provide generic recommendations
        if not recommendations:
            self._recommend_generic_visualizations(data, data_analysis, focus_columns, recommendations)
        
        logger.info(f"Generated {len(recommendations)} visualization recommendations")
        return recommendations
    
    def _recommend_temporal_visualizations(self, data: pd.DataFrame, data_analysis: Dict[str, Any], 
                                         focus_columns: List[str], recommendations: List[Dict[str, Any]]):
        """Generate recommendations for temporal data analysis."""
        datetime_cols = [col for col in data_analysis["datetime_columns"] if col in focus_columns]
        
        # If we have datetime columns
        if datetime_cols:
            time_col = datetime_cols[0]  # Use the first datetime column
            
            # Find numeric columns to plot against time
            for value_col in data_analysis["numerical_columns"]:
                if value_col in focus_columns:
                    # Line chart for time series
                    recommendations.append({
                        "type": "line",
                        "params": {
                            "x": time_col,
                            "y": value_col,
                            "title": f"{value_col} Over Time",
                            "renderer": "plotly"  # Interactive is better for time series
                        },
                        "description": f"Line chart showing {value_col} values over time"
                    })
            
            # If we have categorical columns, suggest grouped analysis
            for cat_col in data_analysis["categorical_columns"]:
                if cat_col in focus_columns and len(data_analysis["numerical_columns"]) > 0:
                    value_col = data_analysis["numerical_columns"][0]
                    recommendations.append({
                        "type": "line",
                        "params": {
                            "x": time_col,
                            "y": value_col,
                            "color": cat_col,
                            "title": f"{value_col} Over Time by {cat_col}",
                            "renderer": "plotly"
                        },
                        "description": f"Line chart showing {value_col} over time, grouped by {cat_col}"
                    })
        else:
            # No datetime columns, see if any columns have date-like names
            time_like_cols = [col for col in focus_columns 
                            if any(kw in col.lower() for kw in ["year", "month", "date", "time", "day"])]            
            
            if time_like_cols and len(data_analysis["numerical_columns"]) > 0:
                # Use the first time-like column as x
                time_col = time_like_cols[0]
                
                # Find numeric columns to plot against time
                for value_col in data_analysis["numerical_columns"]:
                    if value_col in focus_columns:
                        recommendations.append({
                            "type": "line",
                            "params": {
                                "x": time_col,
                                "y": value_col,
                                "title": f"{value_col} Over {time_col}",
                                "renderer": "plotly"
                            },
                            "description": f"Line chart showing {value_col} over {time_col}"
                        })
    
    def _recommend_spatial_visualizations(self, data: pd.DataFrame, data_analysis: Dict[str, Any], 
                                       focus_columns: List[str], recommendations: List[Dict[str, Any]]):
        """Generate recommendations for spatial data analysis."""
        spatial_cols = data_analysis["spatial_columns"]
        
        # Check if we have lat/long pairs
        lat_cols = [col for col in spatial_cols if any(kw in col.lower() for kw in ["lat", "latitude"])]
        lon_cols = [col for col in spatial_cols if any(kw in col.lower() for kw in ["lon", "long", "longitude"])]
        
        if lat_cols and lon_cols and len(lat_cols) > 0 and len(lon_cols) > 0:
            lat_col = lat_cols[0]
            lon_col = lon_cols[0]
            
            # Map visualization
            recommendations.append({
                "type": "map",
                "params": {
                    "lat": lat_col,
                    "lon": lon_col,
                    "title": f"Geographic Distribution",
                    "renderer": "plotly"
                },
                "description": f"Map showing geographic distribution using {lat_col} and {lon_col}"
            })
            
            # If we have numeric columns, suggest bubble map
            if data_analysis["numerical_columns"]:
                value_col = data_analysis["numerical_columns"][0]
                recommendations.append({
                    "type": "bubble_map",
                    "params": {
                        "lat": lat_col,
                        "lon": lon_col,
                        "size": value_col,
                        "title": f"Geographic Distribution of {value_col}",
                        "renderer": "plotly"
                    },
                    "description": f"Bubble map showing geographic distribution of {value_col}"
                })
    
    def _recommend_categorical_visualizations(self, data: pd.DataFrame, data_analysis: Dict[str, Any], 
                                           focus_columns: List[str], recommendations: List[Dict[str, Any]]):
        """Generate recommendations for categorical data analysis."""
        cat_cols = [col for col in data_analysis["categorical_columns"] if col in focus_columns]
        
        if cat_cols:
            cat_col = cat_cols[0]  # Use first categorical column
            
            # Count plot / Bar chart of category counts
            recommendations.append({
                "type": "bar",
                "params": {
                    "x": cat_col,
                    "title": f"Count of {cat_col}",
                    "renderer": "matplotlib"
                },
                "description": f"Bar chart showing counts of each {cat_col} category"
            })
            
            # If we have numeric columns, suggest bar charts with mean values
            if data_analysis["numerical_columns"]:
                for num_col in data_analysis["numerical_columns"][:2]:  # Limit to first 2 numeric columns
                    if num_col in focus_columns:
                        recommendations.append({
                            "type": "bar",
                            "params": {
                                "x": cat_col,
                                "y": num_col,
                                "title": f"{num_col} by {cat_col}",
                                "renderer": "matplotlib"
                            },
                            "description": f"Bar chart showing {num_col} values grouped by {cat_col}"
                        })
            
            # If we have multiple categorical columns, suggest stacked bars or heatmap
            if len(cat_cols) >= 2:
                cat_col2 = cat_cols[1]
                recommendations.append({
                    "type": "heatmap",
                    "params": {
                        "x": cat_col,
                        "y": cat_col2,
                        "title": f"Relationship between {cat_col} and {cat_col2}",
                        "renderer": "matplotlib"
                    },
                    "description": f"Heatmap showing relationship between {cat_col} and {cat_col2}"
                })
    
    def _recommend_correlation_visualizations(self, data: pd.DataFrame, data_analysis: Dict[str, Any], 
                                           focus_columns: List[str], recommendations: List[Dict[str, Any]]):
        """Generate recommendations for correlation analysis."""
        num_cols = [col for col in data_analysis["numerical_columns"] if col in focus_columns]
        
        if len(num_cols) >= 2:
            # Correlation heatmap for all numeric columns
            recommendations.append({
                "type": "heatmap",
                "params": {
                    "title": "Correlation Matrix",
                    "renderer": "plotly"  # Interactive is better for correlation matrices
                },
                "description": "Heatmap showing correlation between all numeric variables"
            })
            
            # Scatter plots for pairs of numeric columns (limit to first few pairs)
            for i in range(min(len(num_cols), 2)):
                for j in range(i+1, min(len(num_cols), 3)):
                    recommendations.append({
                        "type": "scatter",
                        "params": {
                            "x": num_cols[i],
                            "y": num_cols[j],
                            "title": f"Relationship: {num_cols[i]} vs {num_cols[j]}",
                            "renderer": "plotly"
                        },
                        "description": f"Scatter plot showing relationship between {num_cols[i]} and {num_cols[j]}"
                    })
            
            # If we have categorical columns, suggest scatter with color
            if data_analysis["categorical_columns"]:
                cat_col = data_analysis["categorical_columns"][0]
                if cat_col in focus_columns:
                    recommendations.append({
                        "type": "scatter",
                        "params": {
                            "x": num_cols[0],
                            "y": num_cols[1],
                            "color": cat_col,
                            "title": f"Relationship: {num_cols[0]} vs {num_cols[1]} by {cat_col}",
                            "renderer": "plotly"
                        },
                        "description": f"Scatter plot showing relationship between {num_cols[0]} and {num_cols[1]}, colored by {cat_col}"
                    })
    
    def _recommend_distribution_visualizations(self, data: pd.DataFrame, data_analysis: Dict[str, Any], 
                                             focus_columns: List[str], recommendations: List[Dict[str, Any]]):
        """Generate recommendations for distribution analysis."""
        num_cols = [col for col in data_analysis["numerical_columns"] if col in focus_columns]
        
        if num_cols:
            # Histograms for each numeric column (limit to first few)
            for col in num_cols[:3]:  # Limit to first 3 numeric columns
                recommendations.append({
                    "type": "histogram",
                    "params": {
                        "x": col,
                        "title": f"Distribution of {col}",
                        "renderer": "matplotlib"
                    },
                    "description": f"Histogram showing distribution of {col}"
                })
            
            # Box plots if we have categorical columns
            if data_analysis["categorical_columns"]:
                cat_col = data_analysis["categorical_columns"][0]
                if cat_col in focus_columns:
                    for num_col in num_cols[:2]:  # Limit to first 2 numeric columns
                        recommendations.append({
                            "type": "box",
                            "params": {
                                "x": cat_col,
                                "y": num_col,
                                "title": f"Distribution of {num_col} by {cat_col}",
                                "renderer": "matplotlib"
                            },
                            "description": f"Box plot showing distribution of {num_col} across {cat_col} categories"
                        })
    
    def _recommend_comparison_visualizations(self, data: pd.DataFrame, data_analysis: Dict[str, Any], 
                                          focus_columns: List[str], recommendations: List[Dict[str, Any]]):
        """Generate recommendations for comparison analysis."""
        # For numeric comparisons
        num_cols = [col for col in data_analysis["numerical_columns"] if col in focus_columns]
        
        # For categorical grouping
        cat_cols = [col for col in data_analysis["categorical_columns"] if col in focus_columns]
        
        if num_cols and cat_cols:
            cat_col = cat_cols[0]  # Use first categorical column for grouping
            
            # Bar chart for each numeric column grouped by category
            for num_col in num_cols[:3]:  # Limit to first 3 numeric columns
                recommendations.append({
                    "type": "bar",
                    "params": {
                        "x": cat_col,
                        "y": num_col,
                        "title": f"{num_col} by {cat_col}",
                        "renderer": "matplotlib"
                    },
                    "description": f"Bar chart comparing {num_col} across {cat_col} categories"
                })
        
        # Multiple numeric columns comparison
        if len(num_cols) >= 2:
            # Paired bar chart or line chart for comparison
            recommendations.append({
                "type": "bar",
                "params": {
                    "x": data.index.tolist(),
                    "y": num_cols[:3],  # Limit to first 3 numeric columns
                    "title": f"Comparison of {', '.join(num_cols[:3])}",
                    "renderer": "plotly"  # Interactive is better for multiple series
                },
                "description": f"Bar chart comparing values across {', '.join(num_cols[:3])}"
            })
    
    def _recommend_generic_visualizations(self, data: pd.DataFrame, data_analysis: Dict[str, Any], 
                                        focus_columns: List[str], recommendations: List[Dict[str, Any]]):
        """Generate generic visualization recommendations based on data types."""
        # Check for datetime columns for temporal analysis
        if data_analysis["datetime_columns"]:
            self._recommend_temporal_visualizations(data, data_analysis, focus_columns, recommendations)
        
        # Check for spatial columns
        if data_analysis["spatial_columns"]:
            self._recommend_spatial_visualizations(data, data_analysis, focus_columns, recommendations)
        
        # For datasets with numeric columns, suggest correlation analysis
        if len(data_analysis["numerical_columns"]) >= 2:
            self._recommend_correlation_visualizations(data, data_analysis, focus_columns, recommendations)
        
        # For datasets with categorical columns, suggest categorical analysis
        if data_analysis["categorical_columns"]:
            self._recommend_categorical_visualizations(data, data_analysis, focus_columns, recommendations)
        
        # For datasets with numeric columns, suggest distribution analysis
        if data_analysis["numerical_columns"]:
            self._recommend_distribution_visualizations(data, data_analysis, focus_columns, recommendations)
    
    def create_visualization(self, data: pd.DataFrame, recommendation: Dict[str, Any]) -> Any:
        """Create a visualization based on a recommendation."""
        logger.info(f"Creating visualization based on recommendation: {recommendation['type']}")
        
        # Extract visualization parameters from recommendation
        viz_type = recommendation["type"]
        params = recommendation.get("params", {})
        renderer_type = params.pop("renderer", None)  # Extract and remove renderer from params
        
        # Create the visualization using the factory
        try:
            result = self.visualization_factory.create_visualization(
                viz_type=viz_type,
                data=data,
                renderer_type=renderer_type,
                **params
            )
            
            # Record in history
            timestamp = datetime.datetime.now().isoformat()
            history_entry = {
                "timestamp": timestamp,
                "type": viz_type,
                "params": params,
                "renderer": renderer_type,
                "success": True
            }
            self.visualization_history.append(history_entry)
            
            return result
        
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            # Record failure in history
            timestamp = datetime.datetime.now().isoformat()
            history_entry = {
                "timestamp": timestamp,
                "type": viz_type,
                "params": params,
                "renderer": renderer_type,
                "success": False,
                "error": str(e)
            }
            self.visualization_history.append(history_entry)
            raise
    
    def export_visualization(self, viz_result: Any, filename: str = "", 
                         format: str = "png", **kwargs) -> str:
        """Export a visualization to a file."""
        if not filename:
            # Generate a filename based on timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualization_{timestamp}"
        
        # Determine the renderer type from the viz_result
        renderer_type = None
        if isinstance(viz_result, tuple) and len(viz_result) == 2 and hasattr(viz_result[0], "get_figure"):
            renderer_type = "matplotlib"
        elif hasattr(viz_result, "update_layout"):
            renderer_type = "plotly"
        
        # Save using the visualization factory
        try:
            filepath = self.visualization_factory.save_visualization(
                viz_result=viz_result,
                filename=filename,
                format=format,
                renderer_type=renderer_type,
                **kwargs
            )
            logger.info(f"Visualization exported to {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error exporting visualization: {e}")
            raise
    
    def get_visualization_history(self) -> List[Dict[str, Any]]:
        """Get the history of created visualizations."""
        return self.visualization_history
    
    def clear_visualization_history(self):
        """Clear the visualization history."""
        self.visualization_history = []
        logger.info("Visualization history cleared")


# Test code for the visualization agent
if __name__ == "__main__":
    # Set up logging
    logger.info("Running visualization agent test")
    
    try:
        # Create test data
        logger.info("Creating test data")
        np.random.seed(123)
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "value1": np.cumsum(np.random.normal(0, 1, 100)),
            "value2": np.cumsum(np.random.normal(0, 2, 100)),
            "category": np.random.choice(["A", "B", "C"], 100),
            "size": np.random.randint(10, 100, 100),
            "lat": np.random.uniform(20, 40, 100),
            "lon": np.random.uniform(-120, -80, 100)
        })
        
        # Create visualization factory and agent
        from visualization_factory import VisualizationFactory
        viz_factory = VisualizationFactory()
        viz_agent = VisualizationAgent(viz_factory)
        
        # Test visualization recommendations
        logger.info("Testing visualization recommendations")
        recommendations = viz_agent.recommend_visualization(data, analysis_type="temporal")
        print(f"Generated {len(recommendations)} recommendations for temporal analysis")
        for i, rec in enumerate(recommendations[:2]):  # Show just the first 2 recommendations
            print(f"Recommendation {i+1}: {rec['description']}")
            
        # Create and export a visualization
        if recommendations:
            logger.info("Testing visualization creation and export")
            viz_result = viz_agent.create_visualization(data, recommendations[0])
            filepath = viz_agent.export_visualization(viz_result, "test_agent_viz")
            print(f"Visualization exported to: {filepath}")
            
        logger.info("Visualization agent test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in visualization agent test: {e}")
        raise