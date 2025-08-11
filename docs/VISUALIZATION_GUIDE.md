# Ecological Monitoring Advanced Visualization Guide

This guide explains how to use the advanced visualization capabilities in the Ecological Monitoring system. The system now supports multiple visualization renderers including Matplotlib, Plotly, Altair, and Folium for map visualizations.

## Quick Start

The easiest way to create visualizations is to use natural language commands. Simply ask the agent for what you want to see:

```
create a line chart showing fish biomass over the years in Cabo Pulmo
```

```
create a bubble map showing fish biomass by location in Cabo Pulmo
```

## Available Visualization Types

### Basic Charts
- **Line charts**: Show trends over time
- **Bar charts**: Compare values across categories
- **Scatter plots**: Show relationships between two variables
- **Box plots**: Display distribution statistics
- **Heatmaps**: Visualize correlations between variables

### Map Visualizations
- **Basic maps**: Show sampling locations on a map
- **Bubble maps**: Display values with sized markers on a map
- **Heatmaps on maps**: Show density or hotspots on a map

### Interactive vs. Static Visualizations
- **Interactive (Plotly)**: Include hover information, zooming, and panning
- **Static (Matplotlib)**: Simple, publication-ready images

## Natural Language Interface

The system can interpret natural language requests and automatically determine:
- The visualization type
- Which renderer to use (Matplotlib, Plotly, Altair, or Folium)
- What data to display
- How to filter the data

### Example Requests

#### Basic Charts
```
create a line chart showing fish biomass over the years in Cabo Pulmo
```

```
show an interactive bar chart comparing fish biomass across reefs in La Paz
```

```
create a scatter plot of size vs biomass for fish in Loreto
```

#### Map Visualizations
```
create a map of sampling locations in Cabo Pulmo region
```

```
generate a bubble map showing fish biomass by location in Cabo Pulmo
```

```
make a heatmap of fish biomass density on a map of La Paz
```

### Tips for Effective Requests
1. **Be specific about the visualization type** you want (line chart, map, etc.)
2. **Mention the region** you're interested in (Cabo Pulmo, La Paz, Loreto)
3. **Specify organism type** (fish or invertebrates) when relevant
4. **Mention if you want interactive** visualizations
5. **Indicate time periods** if you want to filter by years

## Advanced Usage (JSON Interface)

For advanced users, you can use the technical interface with more control over parameters. This requires a JSON string with specific parameters:

```
create_advanced_visualization
```

with input:

```json
{
  "query": "SELECT Year, AVG(Biomass) as AvgBiomass FROM ltem_optimized_regions WHERE Region='Cabo Pulmo' AND Label='PEC' GROUP BY Year",
  "viz_type": "line",
  "params": {
    "title": "Fish Biomass in Cabo Pulmo",
    "x": "Year",
    "y": "AvgBiomass",
    "filename": "cabo_pulmo_biomass_trend",
    "renderer": "plotly"
  }
}
```

### Common Parameters for All Visualization Types

| Parameter | Description | Example |
|-----------|-------------|---------|
| title | Chart title | "Fish Biomass Over Time" |
| filename | Output filename (without extension) | "biomass_trend" |
| renderer | Specify renderer | "matplotlib", "plotly", "altair", "folium" |

### Parameters by Visualization Type

#### Line and Bar Charts
```json
{
  "x": "Year",           // X-axis column
  "y": "AvgBiomass",     // Y-axis column
  "color": "Region"      // Optional: group by color
}
```

#### Scatter Plots
```json
{
  "x": "Size",           // X-axis column
  "y": "Biomass",        // Y-axis column
  "color": "Species",    // Optional: color by category
  "size": "Quantity"     // Optional: size by value
}
```

#### Maps (Basic)
```json
{
  "lat": "Latitude",     // Latitude column
  "lon": "Longitude",    // Longitude column
  "tooltip": "Reef"      // Optional: hover information
}
```

#### Bubble Maps
```json
{
  "lat": "Latitude",     // Latitude column
  "lon": "Longitude",    // Longitude column
  "size": "AvgBiomass",  // Column for bubble size
  "color": "Region"      // Optional: color by category
}
```

## Output Formats

Visualizations are automatically saved in the appropriate format:
- Interactive visualizations (Plotly, Folium): HTML files
- Static visualizations (Matplotlib): PNG files
- Altair charts can be saved in various formats

## Common Visualization Patterns

### Time Series Analysis
```
create a line chart showing fish biomass over the years in Cabo Pulmo
```

### Regional Comparisons
```
create a bar chart comparing fish biomass across regions
```

### Spatial Patterns
```
create a bubble map showing fish biomass by location in Cabo Pulmo
```

### Correlation Analysis
```
create a scatter plot of size vs biomass for fish in Loreto
```

### Taxonomic Comparisons
```
create a bar chart comparing biomass of different fish species in La Paz
```

## Troubleshooting

If you encounter issues with visualizations:

1. **Check data availability**: Make sure data exists for your specific query
2. **Be specific in requests**: The more specific your request, the better the results
3. **Verify column names**: Use standard column names like Year, Region, Biomass, etc.
4. **Map issues**: For maps, ensure latitude and longitude data is available
5. **Missing dependencies**: If a renderer isn't working, check that its dependencies are installed

## Required Dependencies

- **Matplotlib & Seaborn**: For static visualizations
- **Plotly**: For interactive charts
- **Altair**: For grammar of graphics charts
- **Folium**: For map visualizations

To install all dependencies:
```bash
pip install matplotlib seaborn plotly altair folium
```
