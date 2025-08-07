#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Factory Module

This module provides a factory for creating visualizations with different backends
(matplotlib, plotly, etc.) following a consistent API.

Created: August 2025
"""

import os
import logging
import uuid
import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import altair as alt
import folium
from folium.plugins import MarkerCluster, HeatMap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='visualization.log'
)

logger = logging.getLogger('visualization_factory')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


class VisualizationRenderer(ABC):
    """Abstract base class for visualization renderers."""
    
    @abstractmethod
    def line_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                  title: str = "", **kwargs) -> Any:
        """Create a line chart."""
        pass
    
    @abstractmethod
    def bar_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                 title: str = "", **kwargs) -> Any:
        """Create a bar chart."""
        pass
    
    @abstractmethod
    def scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                    title: str = "", **kwargs) -> Any:
        """Create a scatter plot."""
        pass
    
    @abstractmethod
    def histogram(self, data: pd.DataFrame, x: str, 
                 title: str = "", **kwargs) -> Any:
        """Create a histogram."""
        pass
    
    @abstractmethod
    def heatmap(self, data: pd.DataFrame, 
               title: str = "", **kwargs) -> Any:
        """Create a heatmap."""
        pass
    
    @abstractmethod
    def box_plot(self, data: pd.DataFrame, x: str, y: str, 
                title: str = "", **kwargs) -> Any:
        """Create a box plot."""
        pass
    
    @abstractmethod
    def save(self, fig: Any, filename: str, format: str = "png", 
            dpi: int = 300, **kwargs) -> str:
        """Save the visualization to a file."""
        pass


class MatplotlibRenderer(VisualizationRenderer):
    """Renderer implementation using Matplotlib."""
    
    def __init__(self):
        """Initialize the matplotlib renderer with appropriate theme."""
        # Set the style according to user rules
        plt.style.use('default')  # Reset any previous style
        self.colorblind_palette = sns.color_palette("colorblind")
        
        # Set seed for reproducibility
        np.random.seed(123)
        
    def _apply_theme(self, fig, ax):
        """Apply theme_bw style as required by user rules."""
        # Apply theme_bw equivalent in matplotlib
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        
        # Apply figure level styles
        fig.patch.set_facecolor('white')
        
        return fig, ax
        
    def line_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                  title: str = "", **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create a line chart using matplotlib."""
        logger.info(f"Creating line chart with x={x}, y={y}")
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Handle single y or multiple y columns
        if isinstance(y, list):
            for col in y:
                ax.plot(data[x], data[col], label=col)
            ax.legend()
        else:
            ax.plot(data[x], data[y])
            
        # Apply labels and title
        ax.set_xlabel(kwargs.get('xlabel', x))
        ax.set_ylabel(kwargs.get('ylabel', y if isinstance(y, str) else "Value"))
        ax.set_title(title)
        
        # Apply theme
        fig, ax = self._apply_theme(fig, ax)
        
        return fig, ax
    
    def bar_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                 title: str = "", **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create a bar chart using matplotlib."""
        logger.info(f"Creating bar chart with x={x}, y={y}")
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Handle single y or multiple y columns
        if isinstance(y, list):
            bar_width = 0.8 / len(y)
            for i, col in enumerate(y):
                offset = i * bar_width - (bar_width * len(y) / 2) + bar_width/2
                x_pos = [i + offset for i in range(len(data))]
                ax.bar(x_pos, data[col], width=bar_width, label=col)
            ax.legend()
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data[x])
        else:
            ax.bar(data[x], data[y])
            
        # Apply labels and title
        ax.set_xlabel(kwargs.get('xlabel', x))
        ax.set_ylabel(kwargs.get('ylabel', y if isinstance(y, str) else "Value"))
        ax.set_title(title)
        
        # Apply theme
        fig, ax = self._apply_theme(fig, ax)
        
        return fig, ax
    
    def scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                    title: str = "", **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create a scatter plot using matplotlib."""
        logger.info(f"Creating scatter plot with x={x}, y={y}")
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Handle color and size parameters if provided
        color = kwargs.get('color', None)
        size = kwargs.get('size', None)
        
        if color is not None and color in data.columns:
            scatter = ax.scatter(data[x], data[y], c=data[color], 
                       s=data[size] if size in data.columns else 30,
                       cmap=kwargs.get('cmap', 'viridis'),
                       alpha=kwargs.get('alpha', 0.7))
            plt.colorbar(scatter, ax=ax, label=color)
        else:
            ax.scatter(data[x], data[y], 
                     s=data[size] if size is not None and size in data.columns else 30,
                     alpha=kwargs.get('alpha', 0.7))
            
        # Apply labels and title
        ax.set_xlabel(kwargs.get('xlabel', x))
        ax.set_ylabel(kwargs.get('ylabel', y))
        ax.set_title(title)
        
        # Apply theme
        fig, ax = self._apply_theme(fig, ax)
        
        return fig, ax
    
    def histogram(self, data: pd.DataFrame, x: str, 
                 title: str = "", **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create a histogram using matplotlib."""
        logger.info(f"Creating histogram with x={x}")
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Create histogram
        ax.hist(data[x], bins=kwargs.get('bins', 10), 
               alpha=kwargs.get('alpha', 0.7),
               color=kwargs.get('color', self.colorblind_palette[0]))
            
        # Apply labels and title
        ax.set_xlabel(kwargs.get('xlabel', x))
        ax.set_ylabel(kwargs.get('ylabel', 'Frequency'))
        ax.set_title(title)
        
        # Apply theme
        fig, ax = self._apply_theme(fig, ax)
        
        return fig, ax
    
    def heatmap(self, data: pd.DataFrame, 
               title: str = "", **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create a heatmap using matplotlib."""
        logger.info("Creating heatmap")
        
        # Calculate correlation matrix if not provided directly
        if 'corr_matrix' in kwargs:
            corr_matrix = kwargs['corr_matrix']
        else:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        
        # Create heatmap using seaborn for better appearance
        sns.heatmap(corr_matrix, annot=kwargs.get('annot', True), 
                   cmap=kwargs.get('cmap', 'viridis'),
                   linewidths=kwargs.get('linewidths', 0.5),
                   ax=ax)
            
        # Apply title
        ax.set_title(title)
        
        return fig, ax
    
    def box_plot(self, data: pd.DataFrame, x: str, y: str, 
                title: str = "", **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create a box plot using matplotlib."""
        logger.info(f"Creating box plot with x={x}, y={y}")
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Create box plot using seaborn for better appearance
        sns.boxplot(x=x, y=y, data=data, ax=ax, 
                   palette=kwargs.get('palette', 'colorblind'))
            
        # Apply labels and title
        ax.set_xlabel(kwargs.get('xlabel', x))
        ax.set_ylabel(kwargs.get('ylabel', y))
        ax.set_title(title)
        
        # Apply theme
        fig, ax = self._apply_theme(fig, ax)
        
        return fig, ax
    
    def save(self, fig: Tuple[plt.Figure, plt.Axes], filename: str, 
            format: str = "png", dpi: int = 300, **kwargs) -> str:
        """Save the matplotlib figure to a file."""
        # Extract the figure from the tuple
        figure, _ = fig
        
        # Get output directory from kwargs, with sensible defaults
        output_dir = kwargs.get('output_dir', None)
        
        # Use user directory if not specified
        if not output_dir:
            # Use current working directory
            output_dir = os.getcwd()
            
            # Create a visualizations folder if it doesn't exist
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            output_dir = vis_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate filepath
        filepath = os.path.join(output_dir, f"{filename}.{format}")
        
        # Save the figure
        figure.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved visualization to {filepath}")
        
        # Close the figure to free memory
        plt.close(figure)
        
        return filepath


class PlotlyRenderer(VisualizationRenderer):
    """Renderer implementation using Plotly."""
    
    def __init__(self):
        """Initialize the plotly renderer with appropriate theme."""
        # Set seed for reproducibility
        np.random.seed(123)
        # Using the 'plotly_white' template which is similar to theme_bw
        self.template = 'plotly_white'
        
    def line_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                  title: str = "", **kwargs) -> go.Figure:
        """Create a line chart using plotly."""
        logger.info(f"Creating line chart with x={x}, y={y}")
        
        # Handle single y or multiple y columns
        if isinstance(y, list):
            fig = go.Figure()
            for col in y:
                fig.add_trace(go.Scatter(x=data[x], y=data[col], mode='lines', name=col))
        else:
            fig = px.line(data, x=x, y=y)
            
        # Apply labels and title
        fig.update_layout(
            title=title,
            xaxis_title=kwargs.get('xlabel', x),
            yaxis_title=kwargs.get('ylabel', y if isinstance(y, str) else "Value"),
            template=self.template
        )
        
        return fig
    
    def bar_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                 title: str = "", **kwargs) -> go.Figure:
        """Create a bar chart using plotly."""
        logger.info(f"Creating bar chart with x={x}, y={y}")
        
        # Handle single y or multiple y columns
        if isinstance(y, list):
            fig = go.Figure()
            for col in y:
                fig.add_trace(go.Bar(x=data[x], y=data[col], name=col))
        else:
            fig = px.bar(data, x=x, y=y)
            
        # Apply labels and title
        fig.update_layout(
            title=title,
            xaxis_title=kwargs.get('xlabel', x),
            yaxis_title=kwargs.get('ylabel', y if isinstance(y, str) else "Value"),
            template=self.template
        )
        
        return fig
    
    def scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                    title: str = "", **kwargs) -> go.Figure:
        """Create a scatter plot using plotly."""
        logger.info(f"Creating scatter plot with x={x}, y={y}")
        
        # Handle color and size parameters if provided
        color = kwargs.get('color', None)
        size = kwargs.get('size', None)
        
        fig = px.scatter(
            data, x=x, y=y, 
            color=color if color in data.columns else None,
            size=size if size in data.columns else None,
            title=title
        )
            
        # Apply labels
        fig.update_layout(
            xaxis_title=kwargs.get('xlabel', x),
            yaxis_title=kwargs.get('ylabel', y),
            template=self.template
        )
        
        return fig
    
    def histogram(self, data: pd.DataFrame, x: str, 
                 title: str = "", **kwargs) -> go.Figure:
        """Create a histogram using plotly."""
        logger.info(f"Creating histogram with x={x}")
        
        fig = px.histogram(
            data, x=x,
            nbins=kwargs.get('bins', 10),
            title=title
        )
            
        # Apply labels
        fig.update_layout(
            xaxis_title=kwargs.get('xlabel', x),
            yaxis_title=kwargs.get('ylabel', 'Frequency'),
            template=self.template
        )
        
        return fig
    
    def heatmap(self, data: pd.DataFrame, 
               title: str = "", **kwargs) -> go.Figure:
        """Create a heatmap using plotly."""
        logger.info("Creating heatmap")
        
        # Calculate correlation matrix if not provided directly
        if 'corr_matrix' in kwargs:
            corr_matrix = kwargs['corr_matrix']
        else:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=kwargs.get('annot', True),
            color_continuous_scale=kwargs.get('color_scale', 'viridis'),
            title=title
        )
        
        # Apply template
        fig.update_layout(template=self.template)
        
        return fig
    
    def box_plot(self, data: pd.DataFrame, x: str, y: str, 
                title: str = "", **kwargs) -> go.Figure:
        """Create a box plot using plotly."""
        logger.info(f"Creating box plot with x={x}, y={y}")
        
        fig = px.box(
            data, x=x, y=y,
            title=title
        )
            
        # Apply labels
        fig.update_layout(
            xaxis_title=kwargs.get('xlabel', x),
            yaxis_title=kwargs.get('ylabel', y),
            template=self.template
        )
        
        return fig
    
    def save(self, fig: go.Figure, filename: str, 
            format: str = "html", dpi: int = 300, **kwargs) -> str:
        """Save the plotly figure to a file."""
        # Get output directory from kwargs, with sensible defaults
        output_dir = kwargs.get('output_dir', None)
        
        # Use user directory if not specified
        if not output_dir:
            # Use current working directory
            output_dir = os.getcwd()
            
            # Create a visualizations folder if it doesn't exist
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            output_dir = vis_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate filepath
        filepath = os.path.join(output_dir, f"{filename}.{format}")
        
        # Save based on format
        if format.lower() == "html":
            fig.write_html(filepath)
        else:
            fig.write_image(filepath, scale=dpi/100)
        
        logger.info(f"Saved visualization to {filepath}")
        return filepath


class AltairRenderer(VisualizationRenderer):
    """Renderer implementation using Altair."""
    
    def __init__(self):
        """Initialize the Altair renderer."""
        # Set seed for reproducibility
        np.random.seed(123)
        
    def line_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                   title: str = "", **kwargs) -> alt.Chart:
        """Create a line chart using Altair."""
        logger.info(f"Creating line chart with x={x}, y={y}")
        
        # Handle single y or multiple y columns
        if isinstance(y, list):
            # Create a long-format dataframe for multiple y columns
            id_vars = [x]
            if 'color' in kwargs and kwargs['color'] in data.columns:
                id_vars.append(kwargs['color'])
                
            long_data = pd.melt(data, 
                              id_vars=id_vars, 
                              value_vars=y, 
                              var_name='variable', 
                              value_name='value')
            
            chart = alt.Chart(long_data).mark_line().encode(
                x=alt.X(x, title=kwargs.get('xlabel', x)),
                y=alt.Y('value', title=kwargs.get('ylabel', 'Value')),
                color='variable',
                tooltip=[x, 'value', 'variable']
            ).properties(
                title=title
            )
        else:
            chart = alt.Chart(data).mark_line().encode(
                x=alt.X(x, title=kwargs.get('xlabel', x)),
                y=alt.Y(y, title=kwargs.get('ylabel', y)),
                tooltip=[x, y]
            ).properties(
                title=title
            )
        
        return chart
    
    def bar_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                  title: str = "", **kwargs) -> alt.Chart:
        """Create a bar chart using Altair."""
        logger.info(f"Creating bar chart with x={x}, y={y}")
        
        # Handle single y or multiple y columns
        if isinstance(y, list):
            # Create a long-format dataframe for multiple y columns
            id_vars = [x]
            if 'color' in kwargs and kwargs['color'] in data.columns:
                id_vars.append(kwargs['color'])
                
            long_data = pd.melt(data, 
                              id_vars=id_vars, 
                              value_vars=y, 
                              var_name='variable', 
                              value_name='value')
            
            chart = alt.Chart(long_data).mark_bar().encode(
                x=alt.X(x, title=kwargs.get('xlabel', x)),
                y=alt.Y('value', title=kwargs.get('ylabel', 'Value')),
                color='variable',
                tooltip=[x, 'value', 'variable']
            ).properties(
                title=title
            )
        else:
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X(x, title=kwargs.get('xlabel', x)),
                y=alt.Y(y, title=kwargs.get('ylabel', y)),
                tooltip=[x, y]
            ).properties(
                title=title
            )
        
        return chart
    
    def scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                     title: str = "", **kwargs) -> alt.Chart:
        """Create a scatter plot using Altair."""
        logger.info(f"Creating scatter plot with x={x}, y={y}")
        
        # Handle color and size parameters if provided
        color = kwargs.get('color', None)
        size = kwargs.get('size', None)
        
        encoding = {
            'x': alt.X(x, title=kwargs.get('xlabel', x)),
            'y': alt.Y(y, title=kwargs.get('ylabel', y)),
            'tooltip': [x, y]
        }
        
        if color is not None and color in data.columns:
            encoding['color'] = color
            encoding['tooltip'].append(color)
            
        if size is not None and size in data.columns:
            encoding['size'] = size
            encoding['tooltip'].append(size)
        
        chart = alt.Chart(data).mark_circle().encode(
            **encoding
        ).properties(
            title=title
        )
        
        return chart
    
    def histogram(self, data: pd.DataFrame, x: str, 
                  title: str = "", **kwargs) -> alt.Chart:
        """Create a histogram using Altair."""
        logger.info(f"Creating histogram with x={x}")
        
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(x, bin=alt.Bin(maxbins=kwargs.get('bins', 10)), title=kwargs.get('xlabel', x)),
            y=alt.Y('count()', title=kwargs.get('ylabel', 'Frequency'))
        ).properties(
            title=title
        )
        
        return chart
    
    def heatmap(self, data: pd.DataFrame, 
                title: str = "", **kwargs) -> alt.Chart:
        """Create a heatmap using Altair."""
        logger.info("Creating heatmap")
        
        # Calculate correlation matrix if not provided directly
        if 'corr_matrix' in kwargs:
            corr_matrix = kwargs['corr_matrix']
        else:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
        
        # Convert to long format for Altair
        corr_data = corr_matrix.reset_index().melt(
            id_vars='index', 
            value_name='correlation'
        )
        
        chart = alt.Chart(corr_data).mark_rect().encode(
            x='index:N',
            y='variable:N',
            color=alt.Color('correlation:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['index', 'variable', 'correlation']
        ).properties(
            title=title
        )
        
        # Add text labels if specified
        if kwargs.get('annot', True):
            text = chart.mark_text(baseline='middle').encode(
                text=alt.Text('correlation:Q', format='.2f'),
                color=alt.condition(
                    alt.datum.correlation > 0.5, 
                    alt.value('white'),
                    alt.value('black')
                )
            )
            chart = chart + text
        
        return chart
    
    def box_plot(self, data: pd.DataFrame, x: str, y: str, 
                 title: str = "", **kwargs) -> alt.Chart:
        """Create a box plot using Altair."""
        logger.info(f"Creating box plot with x={x}, y={y}")
        
        chart = alt.Chart(data).mark_boxplot().encode(
            x=alt.X(x, title=kwargs.get('xlabel', x)),
            y=alt.Y(y, title=kwargs.get('ylabel', y))
        ).properties(
            title=title
        )
        
        return chart
    
    def save(self, fig: alt.Chart, filename: str, 
             format: str = "png", dpi: int = 300, **kwargs) -> str:
        """Save the Altair chart to a file."""
        # Get output directory from kwargs, with sensible defaults
        output_dir = kwargs.get('output_dir', None)
        
        # Use user directory if not specified
        if not output_dir:
            # Use current working directory
            output_dir = os.getcwd()
            
            # Create a visualizations folder if it doesn't exist
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            output_dir = vis_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate filepath based on format
        if format.lower() == "html":
            filepath = os.path.join(output_dir, f"{filename}.html")
            # Save as HTML
            try:
                # Try standard save method
                fig.save(filepath)
            except Exception as e:
                # If that fails, try alternative approaches
                logger.warning(f"Standard save failed: {e}, trying alternative method")
                with open(filepath, 'w') as f:
                    f.write(fig.to_html())
                
        elif format.lower() == "svg":
            filepath = os.path.join(output_dir, f"{filename}.svg")
            # Save as SVG
            try:
                fig.save(filepath)
            except Exception as e:
                logger.error(f"Failed to save as SVG: {e}")
                raise
                
        elif format.lower() == "png":
            filepath = os.path.join(output_dir, f"{filename}.png")
            # Save as PNG
            try:
                # Try to save directly
                fig.save(filepath)
            except Exception as e:
                logger.warning(f"Standard save failed: {e}, trying alternative method")
                # Try using Altair's save method with renderer
                alt.save(fig, filepath)
        else:
            # Default to HTML for unsupported formats
            filepath = os.path.join(output_dir, f"{filename}.html")
            try:
                fig.save(filepath)
            except Exception as e:
                logger.warning(f"Standard save failed: {e}, trying alternative method")
                with open(filepath, 'w') as f:
                    f.write(fig.to_html())
        
        logger.info(f"Saved visualization to {filepath}")
        return filepath


class FoliumRenderer(VisualizationRenderer):
    """Renderer implementation using Folium for maps and geographical visualizations."""
    
    def __init__(self):
        """Initialize the Folium renderer."""
        # Set seed for reproducibility
        np.random.seed(123)
        
    def line_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                   title: str = "", **kwargs) -> Any:
        """Create a line chart - Not directly supported by Folium."""
        logger.warning("Line charts not directly supported by Folium. Use another renderer.")
        raise NotImplementedError("Line charts not supported by Folium renderer")
    
    def bar_chart(self, data: pd.DataFrame, x: str, y: Union[str, List[str]], 
                  title: str = "", **kwargs) -> Any:
        """Create a bar chart - Not directly supported by Folium."""
        logger.warning("Bar charts not directly supported by Folium. Use another renderer.")
        raise NotImplementedError("Bar charts not supported by Folium renderer")
    
    def scatter_plot(self, data: pd.DataFrame, x: str, y: str, 
                     title: str = "", **kwargs) -> Any:
        """Create a scatter plot - Not directly supported by Folium."""
        logger.warning("Scatter plots not directly supported by Folium. Use another renderer.")
        raise NotImplementedError("Scatter plots not supported by Folium renderer")
    
    def histogram(self, data: pd.DataFrame, x: str, 
                  title: str = "", **kwargs) -> Any:
        """Create a histogram - Not directly supported by Folium."""
        logger.warning("Histograms not directly supported by Folium. Use another renderer.")
        raise NotImplementedError("Histograms not supported by Folium renderer")
    
    def heatmap(self, data: pd.DataFrame, 
                title: str = "", **kwargs) -> Any:
        """Create a heatmap using Folium."""
        logger.warning("Standard heatmaps not supported by Folium in this context. Use map_heatmap instead.")
        raise NotImplementedError("Standard heatmaps not supported by Folium renderer")
    
    def box_plot(self, data: pd.DataFrame, x: str, y: str, 
                 title: str = "", **kwargs) -> Any:
        """Create a box plot - Not directly supported by Folium."""
        logger.warning("Box plots not directly supported by Folium. Use another renderer.")
        raise NotImplementedError("Box plots not supported by Folium renderer")
    
    def map(self, data: pd.DataFrame, lat: str, lon: str, 
            title: str = "", **kwargs) -> folium.Map:
        """Create a map using Folium."""
        logger.info(f"Creating map with lat={lat}, lon={lon}")
        
        # Set default map parameters
        zoom_start = kwargs.get('zoom_start', 5)
        color = kwargs.get('color', None)  # For point colors
        cluster = kwargs.get('cluster', True)  # Whether to cluster points
        
        # Calculate map center if not provided
        center = kwargs.get('center', None)
        if center is None:
            center = [data[lat].mean(), data[lon].mean()]
        
        # Create the base map
        m = folium.Map(location=center, 
                      zoom_start=zoom_start,
                      tiles=kwargs.get('tiles', 'OpenStreetMap'))
        
        # Add the title if provided
        if title:
            # Add a title as an HTML div
            title_html = f'''
                <h3 style="text-align:center;margin-bottom:15px;">{title}</h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
        
        # Handle point data
        if cluster:
            # Create a marker cluster
            marker_cluster = MarkerCluster(name="Data Points")
            
            # Add markers to the cluster
            for idx, row in data.iterrows():
                # Skip rows with missing lat/lon
                if pd.isna(row[lat]) or pd.isna(row[lon]):
                    continue
                    
                # Create popup content with all columns
                popup_content = "<table>"
                for col in data.columns:
                    if col not in [lat, lon] or kwargs.get('show_coords', False):
                        popup_content += f"<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>"
                popup_content += "</table>"
                
                # Add marker to cluster
                folium.Marker(
                    location=[row[lat], row[lon]],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='blue' if color is None else row[color] if color in data.columns else color)
                ).add_to(marker_cluster)
                
            marker_cluster.add_to(m)
        else:
            # Add markers directly to the map
            for idx, row in data.iterrows():
                # Skip rows with missing lat/lon
                if pd.isna(row[lat]) or pd.isna(row[lon]):
                    continue
                    
                # Create popup content with all columns
                popup_content = "<table>"
                for col in data.columns:
                    if col not in [lat, lon] or kwargs.get('show_coords', False):
                        popup_content += f"<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>"
                popup_content += "</table>"
                
                # Add marker to map
                folium.Marker(
                    location=[row[lat], row[lon]],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='blue' if color is None else row[color] if color in data.columns else color)
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def map_heatmap(self, data: pd.DataFrame, lat: str, lon: str, 
                   weight: str = None, title: str = "", **kwargs) -> folium.Map:
        """Create a heatmap on a map using Folium."""
        logger.info(f"Creating map heatmap with lat={lat}, lon={lon}, weight={weight}")
        
        # Set default map parameters
        zoom_start = kwargs.get('zoom_start', 5)
        
        # Calculate map center if not provided
        center = kwargs.get('center', None)
        if center is None:
            center = [data[lat].mean(), data[lon].mean()]
        
        # Create the base map
        m = folium.Map(location=center, 
                      zoom_start=zoom_start,
                      tiles=kwargs.get('tiles', 'OpenStreetMap'))
        
        # Add the title if provided
        if title:
            # Add a title as an HTML div
            title_html = f'''
                <h3 style="text-align:center;margin-bottom:15px;">{title}</h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
        
        # Prepare data for heatmap
        heat_data = []
        for idx, row in data.iterrows():
            # Skip rows with missing lat/lon
            if pd.isna(row[lat]) or pd.isna(row[lon]):
                continue
                
            if weight is not None and weight in data.columns:
                # Include weight if provided
                heat_data.append([row[lat], row[lon], row[weight]])
            else:
                heat_data.append([row[lat], row[lon]])
        
        # Add heatmap layer
        HeatMap(
            heat_data,
            min_opacity=kwargs.get('min_opacity', 0.5),
            radius=kwargs.get('radius', 25),
            blur=kwargs.get('blur', 15),
            gradient=kwargs.get('gradient', None)
        ).add_to(m)
        
        return m
    
    def bubble_map(self, data: pd.DataFrame, lat: str, lon: str, 
                  size: str = None, color: str = None,
                  title: str = "", **kwargs) -> folium.Map:
        """Create a bubble map using Folium."""
        logger.info(f"Creating bubble map with lat={lat}, lon={lon}, size={size}, color={color}")
        
        # Set default map parameters
        zoom_start = kwargs.get('zoom_start', 5)
        
        # Calculate map center if not provided
        center = kwargs.get('center', None)
        if center is None:
            center = [data[lat].mean(), data[lon].mean()]
        
        # Create the base map
        m = folium.Map(location=center, 
                      zoom_start=zoom_start,
                      tiles=kwargs.get('tiles', 'OpenStreetMap'))
        
        # Add the title if provided
        if title:
            # Add a title as an HTML div
            title_html = f'''
                <h3 style="text-align:center;margin-bottom:15px;">{title}</h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
        
        # Scale size values if provided
        if size is not None and size in data.columns:
            # Min-max scaling for circle sizes
            min_val = data[size].min()
            max_val = data[size].max()
            size_range = max_val - min_val if max_val != min_val else 1
            
            # Scale to reasonable circle radius range (5-50)
            min_radius = kwargs.get('min_radius', 5)
            max_radius = kwargs.get('max_radius', 50)
            
            # Function to scale values
            def scale_size(val):
                if pd.isna(val):
                    return min_radius
                return min_radius + ((val - min_val) / size_range) * (max_radius - min_radius)
        
        # Add circle markers
        for idx, row in data.iterrows():
            # Skip rows with missing lat/lon
            if pd.isna(row[lat]) or pd.isna(row[lon]):
                continue
                
            # Determine circle radius
            if size is not None and size in data.columns:
                radius = scale_size(row[size])
            else:
                radius = kwargs.get('radius', 10)
                
            # Determine circle color
            if color is not None and color in data.columns:
                # Simplified color mapping (would need more sophisticated handling for continuous values)
                circle_color = str(row[color]) if not pd.isna(row[color]) else 'blue'
            else:
                circle_color = kwargs.get('circle_color', 'blue')
                
            # Create popup content
            popup_content = "<table>"
            for col in data.columns:
                if col not in [lat, lon] or kwargs.get('show_coords', False):
                    popup_content += f"<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>"
            popup_content += "</table>"
            
            # Add circle marker
            folium.CircleMarker(
                location=[row[lat], row[lon]],
                radius=radius,
                color=circle_color,
                fill=True,
                fill_opacity=kwargs.get('fill_opacity', 0.6),
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(m)
        
        return m
    
    def save(self, fig: folium.Map, filename: str, 
             format: str = "html", dpi: int = 300, **kwargs) -> str:
        """Save the Folium map to a file."""
        # Get output directory from kwargs, with sensible defaults
        output_dir = kwargs.get('output_dir', None)
        
        # Use user directory if not specified
        if not output_dir:
            # Use current working directory or user's home directory
            output_dir = os.getcwd()
            
            # Create a visualizations folder if it doesn't exist
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            output_dir = vis_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate filepath - Folium only supports HTML format
        filepath = os.path.join(output_dir, f"{filename}.html")
        
        # Save the map
        fig.save(filepath)
        
        logger.info(f"Saved map visualization to {filepath}")
        return filepath


class VisualizationFactory:
    """Factory for creating visualizations with different backends."""
    
    def __init__(self):
        """Initialize the visualization factory with available renderers."""
        self.renderers = {
            "matplotlib": MatplotlibRenderer(),
            "plotly": PlotlyRenderer(),
            "altair": AltairRenderer(),
            "folium": FoliumRenderer()
        }
        self.default_renderer = "matplotlib"
        logger.info(f"Visualization Factory initialized with renderers: {list(self.renderers.keys())}")
    
    def get_renderer(self, renderer_type: str = None) -> VisualizationRenderer:
        """Get the specified renderer."""
        if renderer_type is None:
            renderer_type = self.default_renderer
        
        renderer = self.renderers.get(renderer_type.lower())
        if renderer is None:
            logger.warning(f"Renderer '{renderer_type}' not found, using default: {self.default_renderer}")
            renderer = self.renderers[self.default_renderer]
        
        return renderer
    
    def set_default_renderer(self, renderer_type: str):
        """Set the default renderer."""
        if renderer_type.lower() in self.renderers:
            self.default_renderer = renderer_type.lower()
            logger.info(f"Default renderer set to: {self.default_renderer}")
        else:
            logger.warning(f"Renderer '{renderer_type}' not found, default unchanged: {self.default_renderer}")
    
    def create_visualization(self, viz_type: str, data: pd.DataFrame, 
                            renderer_type: str = None, **kwargs) -> Any:
        """Create a visualization of the specified type."""
        renderer = self.get_renderer(renderer_type)
        
        start_time = datetime.datetime.now()
        logger.info(f"Creating {viz_type} visualization with {renderer.__class__.__name__}")
        
        # Define standard visualization methods available in all renderers
        standard_viz_methods = {
            "line": renderer.line_chart,
            "bar": renderer.bar_chart,
            "scatter": renderer.scatter_plot,
            "histogram": renderer.histogram,
            "heatmap": renderer.heatmap,
            "box": renderer.box_plot,
            "box_plot": renderer.box_plot  # Adding box_plot as alias for box
        }
        
        # Handle special map visualization types (only available in Folium renderer)
        map_viz_types = ["map", "bubble_map", "map_heatmap"]
        
        if viz_type.lower() in map_viz_types:
            # Force the renderer to Folium for map visualizations
            if not isinstance(renderer, FoliumRenderer):
                logger.warning(f"{viz_type} visualization requires Folium renderer, switching from {renderer.__class__.__name__}")
                renderer = self.renderers.get("folium")
                if renderer is None:
                    logger.error("Folium renderer not found but required for map visualizations")
                    raise ValueError("Folium renderer not found but required for map visualizations")
            
            # Map the visualization type to the appropriate method
            if viz_type.lower() == "map":
                viz_method = renderer.map
            elif viz_type.lower() == "bubble_map":
                viz_method = renderer.bubble_map
            elif viz_type.lower() == "map_heatmap":
                viz_method = renderer.map_heatmap
            
            # Execute the visualization method
            result = viz_method(data=data, **kwargs)
        else:
            # Handle standard visualization types
            if viz_type.lower() not in standard_viz_methods:
                logger.error(f"Visualization type '{viz_type}' not supported")
                raise ValueError(f"Visualization type '{viz_type}' not supported")
            
            result = standard_viz_methods[viz_type.lower()](data=data, **kwargs)
        
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Visualization created in {execution_time:.2f} seconds")
        
        return result
    
    def save_visualization(self, viz_result: Any, filename: str, 
                          format: str = "png", renderer_type: str = None, **kwargs) -> str:
        """Save a visualization to a file."""
        # Determine the renderer type from the viz_result if not specified
        if renderer_type is None:
            if isinstance(viz_result, tuple) and len(viz_result) == 2 and hasattr(viz_result[0], "figure"):
                renderer_type = "matplotlib"
            elif hasattr(viz_result, "update_layout"):
                renderer_type = "plotly"
            elif hasattr(viz_result, "to_dict") and hasattr(viz_result, "mark_type") or isinstance(viz_result, alt.Chart):
                renderer_type = "altair"
            elif hasattr(viz_result, "_repr_html_") and isinstance(viz_result, folium.Map):
                renderer_type = "folium"
            else:
                logger.warning("Could not determine renderer type from viz_result, using default")
        
        # Get the appropriate renderer
        renderer = self.get_renderer(renderer_type)
        
        # Generate a unique filename if not provided
        if not filename:
            filename = f"viz_{uuid.uuid4().hex[:8]}"
        
        # Adjust format based on renderer if not explicitly specified
        if kwargs.get('auto_format', True):
            if isinstance(renderer, MatplotlibRenderer) and format == "html":
                format = "png"
                logger.info("Adjusted format to 'png' for matplotlib renderer")
            elif isinstance(renderer, PlotlyRenderer) and format not in ["html", "json"]:
                format = "html"
                logger.info("Adjusted format to 'html' for plotly renderer")
            elif isinstance(renderer, AltairRenderer) and format not in ["html", "svg", "png"]:
                format = "html"
                logger.info("Adjusted format to 'html' for altair renderer")
            elif isinstance(renderer, FoliumRenderer):
                format = "html"
                logger.info("Using 'html' format for folium renderer (only supported format)")
        
        return renderer.save(viz_result, filename, format, **kwargs)


# Test code for the visualization factory
if __name__ == "__main__":
    # Set up logging
    logger.info("Running visualization factory test")
    
    # Create test data
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
    
    # Create visualization factory
    viz_factory = VisualizationFactory()
    
    try:
        # Test matplotlib renderer
        logger.info("Testing matplotlib renderer")
        line_plt = viz_factory.create_visualization(
            "line", 
            data,
            "matplotlib",
            x="date", 
            y=["value1", "value2"],
            title="Line Chart with Matplotlib"
        )
        viz_factory.save_visualization(line_plt, "test_matplotlib_line", "png")
        
        # Test plotly renderer
        logger.info("Testing plotly renderer")
        line_plotly = viz_factory.create_visualization(
            "line", 
            data,
            "plotly",
            x="date", 
            y=["value1", "value2"],
            title="Line Chart with Plotly"
        )
        viz_factory.save_visualization(line_plotly, "test_plotly_line", "html")
        
        # Test altair renderer
        logger.info("Testing altair renderer")
        scatter_altair = viz_factory.create_visualization(
            "scatter", 
            data,
            "altair",
            x="value1", 
            y="value2",
            color="category",
            size="size",
            title="Scatter Plot with Altair"
        )
        viz_factory.save_visualization(scatter_altair, "test_altair_scatter", "html")
        
        # Test folium map renderer
        logger.info("Testing folium map renderer")
        basic_map = viz_factory.create_visualization(
            "map", 
            data,
            "folium",
            lat="lat", 
            lon="lon",
            title="Basic Map with Folium",
            cluster=True
        )
        viz_factory.save_visualization(basic_map, "test_folium_map", "html")
        
        # Test folium bubble map
        logger.info("Testing folium bubble map")
        bubble_map = viz_factory.create_visualization(
            "bubble_map", 
            data,
            "folium",
            lat="lat", 
            lon="lon",
            size="size",
            color="category",
            title="Bubble Map with Folium"
        )
        viz_factory.save_visualization(bubble_map, "test_folium_bubble", "html")
        
        # Test folium heatmap
        logger.info("Testing folium heatmap")
        heat_map = viz_factory.create_visualization(
            "map_heatmap", 
            data,
            "folium",
            lat="lat", 
            lon="lon",
            weight="value1",
            title="Heat Map with Folium"
        )
        viz_factory.save_visualization(heat_map, "test_folium_heatmap", "html")
        
        logger.info("All visualization tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error in visualization factory test: {e}")
        import traceback
        traceback.print_exc()
    