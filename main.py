#!/usr/bin/env python3

import os
import time
import threading
import sys
from datetime import datetime
import re
from typing import Optional, Dict, List, Union, Tuple, Any

# Import our SQL formatter and data validator
from sql_formatter import format_sql_for_humans, translate_sql_to_english
from data_validator import DataValidator

# Import our new statistical analysis and PDF report modules
try:
    from statistical_analysis import (
        perform_ttest, perform_anova, perform_correlation_analysis,
        perform_regression_analysis, perform_nonparametric_tests,
        check_normality, format_statistical_results
    )
    from pdf_report_generator import (
        EcologicalReportGenerator, create_comprehensive_report,
        create_quick_summary_report
    )
    ADVANCED_STATS_AVAILABLE = True
    print("Advanced statistical analysis and PDF reporting modules loaded successfully.")
except ImportError as e:
    print(f"Warning: Advanced statistical modules not available: {e}")
    ADVANCED_STATS_AVAILABLE = False

# Basic dependencies that should be available
try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. Data analysis capabilities will be limited.")
    pd = None

# Visualization dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Visualization capabilities will be disabled.")
    VISUALIZATION_AVAILABLE = False

# Database dependencies
try:
    from sqlalchemy import create_engine
    DATABASE_AVAILABLE = True
except ImportError:
    print("Warning: sqlalchemy not installed. Database capabilities will be disabled.")
    DATABASE_AVAILABLE = False

# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    print("Loaded environment from .env file")
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")

# Track if LangChain dependencies are available
LANGCHAIN_AVAILABLE = False

# LangChain specific imports - make them all conditional
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
    from langchain_experimental.tools import PythonREPLTool
    from langchain.tools import Tool
    
    LANGCHAIN_AVAILABLE = True
    print("LangChain dependencies loaded successfully.")
except ImportError as e:
    print(f"WARNING: Missing LangChain dependencies: {e}")
    # Define dummy classes to avoid errors when not importing
    class Tool:
        def __init__(self, **kwargs):
            pass

# Environment variables should be loaded above in the try-except block

# --- Database schema description ---
SCHEMA_DESCRIPTION = """ 
This is a comprehensive dataset from a Long-Term Ecological Monitoring (LTEM) program for coral reef ecosystems. 

## `ltem_historical_database` Table 
This table is the foundational observational log. Each row represents a specific survey entry where organisms of the same size were recorded within a transect at a certain depth (Depth and Depth2). 
- **Purpose:** To log what species were seen, where, when, and in what quantity/size. 
- **Key Columns:** 
- `Label`: Has two factors, "INV" and "PEC", that represent Invertebrate and Fish surveys, respectively. 
- `Taxa1`, `Taxa2`, `Phylum`, `Species`, `Family`, `Genus`: Taxonomic information of the species recorded. 
- `Year`, `Month`, `Day`: Date of the observation. 
- `Region`, `Reef`: Character information of the Region and the specific Reef where the survey occurred. 
- `Habitat`: Type of substrate surveyed. 
- `Longitude`, `Latitude`: Precise location data in WGS84 degrees. 
- `Quantity`: The abundance, i.e., how many organisms of that size were counted within a transect. When abundance is called, you should divide this for the Area to get org/m2. 
- `Size`: The size class of the organisms. The unit is cm. 
- `Biomass`: Fish biomass (not available for invertebrates), calculated using size, quantity, and growth parameters. The unit is ton/ha. 
- `MPA`, `Protection_status`: Conservation status of the location. 
- `bleaching_coverage`: A key environmental indicator for coral health. 
- `TrophicLevelF`, `TrophicLevel`, `TrophicGroup`, `Functional_groups`: Functional traits from trophic levels in factor, number, groups, and functional groups. 
- `Area`: The area of the transect used. The unit is m2. 

--- 

## `ltem_spp_productivity` Table 
This is an advanced, analytical dataset focused on calculating the biological productivity of species populations. 
- **Purpose:** To model and estimate the rate at which new biomass is generated. 
- **Key Columns:** 
- `Linf`, `K`: Parameters from the von Bertalanffy growth model, used to describe organism growth over time. 
- `somatic_growth`, `mortality`, `Prod`: Direct components of population dynamics. `Prod` is the final calculated productivity, a key metric for ecosystem function. 
"""

def get_database_schema_description(query: str = None) -> str:
    """
    Returns a detailed description of the database schema.
    
    Args:
        query: Optional query parameter (not used but required by LangChain's tool calling framework)
        
    Returns:
        str: A formatted string containing the schema description.
    """
    # The query parameter is ignored as this function always returns the full schema
    return SCHEMA_DESCRIPTION

# --- Spinner animation class for better UX ---
class Spinner:
    def __init__(self, message="Thinking..."):
        self.spinner_cycle = ['-', '\\', '|', '/']
        self.message = message
        self.stop_running = False
        self.spin_thread = None
        # Flag to track if animation is running
        self.is_running = False

    def start(self):
        self.stop_running = False
        self.is_running = True
        self.spin_thread = threading.Thread(target=self.init_spin)
        self.spin_thread.start()

    def stop(self):
        self.stop_running = True
        self.is_running = False
        if self.spin_thread:
            self.spin_thread.join()
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()

    def init_spin(self):
        while not self.stop_running:
            for char in self.spinner_cycle:
                if self.stop_running:
                    break
                sys.stdout.write(f'\r{self.message} {char}')
                sys.stdout.flush()
                time.sleep(0.1)

# --- Optimized system prompt to prevent token overflow ---
SYSTEM_PROMPT = """
You are an ecological data analyst for Gulf of California coral reef monitoring data.

**Key Rules:**
1. Use proper scientific terminology and data-driven analysis
2. For density calculations: use `calculate_average_density` tool
3. For statistical tests: use `perform_ttest`, `perform_anova`, `perform_correlation_analysis`, `perform_regression_analysis`, `perform_nonparametric_test`
4. For reports: use `generate_pdf_report`
5. Report "Average Density" not "Total" values
6. Check statistical assumptions before selecting tests
"""

# Only try to create database connection if SQLAlchemy is available
db = None
engine = None
if DATABASE_AVAILABLE and LANGCHAIN_AVAILABLE:
    try:
        # Try to connect to the database using the environment variable
        DATABASE_URL = os.getenv("DATABASE_URL")
        if DATABASE_URL:
            engine = create_engine(DATABASE_URL)
            db = SQLDatabase(engine)
            DATABASE_AVAILABLE = True
            print(f"Connected to database at {DATABASE_URL}")
        else:
            # Fallback to SQLite in-memory for testing
            engine = create_engine("sqlite:///:memory:")
            db = SQLDatabase(engine)
            DATABASE_AVAILABLE = True
            print("No DATABASE_URL found. Connected to in-memory SQLite database.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        DATABASE_AVAILABLE = False
else:
    print("Database connection skipped due to missing dependencies.")
    DATABASE_AVAILABLE = False

# --- LLM Initialization ---
llm = None
prompt = None

if LANGCHAIN_AVAILABLE:
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # --- Create a prompt template that includes our system prompt ---
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            # Removed chat history placeholder to eliminate between-question memory
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        print("LLM and prompt initialized successfully.")
    except Exception as e:
        print(f"Error initializing LLM and prompt: {e}")
else:
    print("LLM initialization skipped due to missing dependencies.")

# --- Define plotting functions ---
# Versatile chart function
def create_chart(query: str, chart_type: str, title: str, filename: str = "chart.png") -> str:
    """
    Executes a SQL query, creates a specified type of chart from the results,
    and saves it to a file. Supported chart_types are 'bar', 'line', and 'scatter'.
    
    Args:
        query (str): The SQL query to execute to get the data.
        chart_type (str): Type of chart to create ('bar', 'line', or 'scatter').
        title (str): Title for the chart.
        filename (str): The name of the file to save the chart as.
    
    Returns:
        str: A confirmation message with the filename or an error.
    """
    if not VISUALIZATION_AVAILABLE:
        return "Error: Visualization capabilities are not available. Please install matplotlib and seaborn."
    
    if not DATABASE_AVAILABLE or pd is None:
        return "Error: Database or pandas capabilities are not available. Cannot execute SQL query."
    
    try:
        df = pd.read_sql(query, engine)
        if len(df.columns) != 2:
            return "Error: The SQL query for plotting must return exactly two columns."
            
        # Set plot style
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Extract the column names for labels
        x_column = df.columns[0]
        y_column = df.columns[1]
        
        # Create the appropriate chart type
        if chart_type.lower() == 'bar':
            ax = sns.barplot(x=x_column, y=y_column, data=df)
        elif chart_type.lower() == 'line':
            ax = sns.lineplot(x=x_column, y=y_column, data=df)
        elif chart_type.lower() == 'scatter':
            ax = sns.scatterplot(x=x_column, y=y_column, data=df)
        else:
            return f"Error: Chart type '{chart_type}' not supported. Use 'bar', 'line', or 'scatter'."
        
        # Improve the chart appearance
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return f"Chart saved as {filename}"
    except Exception as e:
        return f"Error creating chart: {e}"

# Legacy functions kept for reference
# This section is kept for reference but the create_chart function above should be used instead
# This section is kept for reference but the create_chart function above should be used instead

# --- Statistical summary function ---
def get_statistical_summary(query: str) -> str:
    """
    Executes a SQL query and returns a full statistical summary 
    (count, mean, std, min, max, quartiles) of the resulting numerical data.
    
    Args:
        query (str): The SQL query to execute to get the data.
    
    Returns:
        str: A formatted string containing the statistical summary.
    """
    if not DATABASE_AVAILABLE or pd is None:
        return "Error: Database or pandas capabilities are not available. Cannot execute SQL query."
        
    try:
        # Execute the SQL query
        df = pd.read_sql(query, engine)
        
        # Generate a detailed description of the data
        summary = df.describe(include='all').transpose()
        
        # Convert to a nicely formatted string
        result = "Statistical Summary:\n\n"
        result += summary.to_string()
        
        # Add additional info about the query result
        result += f"\n\nNumber of rows: {len(df)}"
        result += f"\nNumber of columns: {len(df.columns)}"
        result += f"\nColumn names: {', '.join(df.columns)}"
        
        return result
    except Exception as e:
        return f"Error generating statistical summary: {e}"


# --- Average Density Calculation function ---
def calculate_average_density(
    metric: str, 
    group_by: str, 
    where_clause: str = "1=1"
) -> str:
    """
    Calculates scientifically valid average density for a given metric, grouped by a specific column, with optional filters.
    This is the primary tool for any abundance or biomass calculation.
    
    Args:
        metric (str): The metric to calculate density for. Must be 'Biomass' or 'Quantity'.
        group_by (str): The column to group results by (e.g., 'Species', 'Year', 'Region')
        where_clause (str, optional): Additional filtering criteria. Defaults to "1=1" (no filtering).
        
    Returns:
        str: Formatted string with density results or error message
    """
    if engine is None:
        return "Error: Database connection is not available."
    
    # Validate the metric argument
    if metric not in ['Biomass', 'Quantity']:
        return f"Error: 'metric' must be 'Biomass' or 'Quantity', but got '{metric}'"
    
    # Escape '%' for logging to prevent format string errors with SQL LIKE patterns
    safe_where = where_clause.replace('%', '%%') if isinstance(where_clause, str) else where_clause
    log_progress(f"Calculating average {metric} density grouped by {group_by} with filter: {safe_where}", is_reasoning=True)
    
    # Construct SQL query for calculating average density
    # The query follows the scientific method of first summing by transect, then averaging
    transect_level_query = f"""
    SELECT 
        {group_by},
        SUM({metric}) as transect_total
    FROM 
        ltem_historical_database
    WHERE 
        {where_clause}
    GROUP BY 
        {group_by}, Year, Region, Reef, Depth2, Transect
    """
    
    try:
        # Execute the query and store results in a DataFrame
        df_transects = pd.read_sql(transect_level_query, engine)
        
        # Check if the DataFrame is empty
        if df_transects.empty:
            return "No data found for the specified criteria."
        
        # Log that we're calculating the final averages
        log_progress("Calculating final average densities across transects...", is_reasoning=True)
        
        # Use simpler, more robust aggregation approach
        grouped = df_transects.groupby(group_by)
        
        # Calculate statistics manually to avoid aggregation syntax issues
        mean_values = grouped['transect_total'].mean()
        std_values = grouped['transect_total'].std()
        count_values = grouped['transect_total'].count()
        
        # Create result DataFrame
        final_summary = pd.DataFrame({
            group_by: mean_values.index,
            'Mean': mean_values.values,
            'StdDev': std_values.values,
            'TransectCount': count_values.values
        })
        
        # Add units
        unit = "g/transect" if metric == "Biomass" else "individuals/transect"
        
        # Correcting the unit calculation for abundance
        if metric == 'Quantity':
            try:
                # Use default area value if query fails
                avg_area = 100  # Default value
                
                # Try to get actual area
                avg_area_query = "SELECT AVG(Area) as avg_area FROM ltem_historical_database WHERE Area > 0"
                avg_area_result = pd.read_sql(avg_area_query, engine)
                
                if not avg_area_result.empty and avg_area_result['avg_area'].iloc[0] > 0:
                    avg_area = avg_area_result['avg_area'].iloc[0]
                
                # Apply area correction
                final_summary['Mean'] = final_summary['Mean'] / avg_area
                final_summary['StdDev'] = final_summary['StdDev'] / avg_area
                unit = f"individuals/{avg_area}m²"
            except Exception as area_err:
                # If area calculation fails, just use the original values
                log_progress(f"Area calculation warning: {area_err}. Using uncorrected values.", is_reasoning=True)

        # Format the output
        result = f"Average {metric} Density by {group_by} (per {unit}):\n\n"
        result += final_summary.to_string(index=False)
        return result

    except Exception as e:
        error_msg = f"Error during density calculation: {e}"
        log_progress(error_msg, is_reasoning=True)
        return error_msg


def compare_taxa_density(
    metric: str, 
    taxa1: str, 
    taxa2: str, 
    year: Optional[int] = None, 
    region: Optional[str] = None
) -> str:
    """
    Compares the average density of two taxonomic groups for either 'Biomass' or 'Quantity'.
    Use this for any direct comparison question like "compare X and Y".
    The taxa names should be genus or species names.
    """
    if engine is None:
        return "Error: Database connection is not available."

    log_progress(f"Comparing {metric} density between {taxa1} and {taxa2}...", is_reasoning=True)

    # Validate required parameters
    if not metric or metric not in ['Biomass', 'Quantity']:
        return f"Error: metric parameter must be specified as either 'Biomass' or 'Quantity', but got '{metric}'"
    if not taxa1:
        return "Error: taxa1 parameter must be specified"
    if not taxa2:
        return "Error: taxa2 parameter must be specified"
        
    # Build WHERE clauses
    filters = []
    if year:
        filters.append(f"Year = {year}")
    if region:
        filters.append(f"Region = '{region}'")
    
    # More robust taxa search
    taxa1_filter = f"(Genus = '{taxa1}' OR Species = '{taxa1}' OR Taxa1 = '{taxa1}' OR Taxa2 = '{taxa1}')"
    taxa2_filter = f"(Genus = '{taxa2}' OR Species = '{taxa2}' OR Taxa1 = '{taxa2}' OR Taxa2 = '{taxa2}')"

    where1 = " AND ".join([taxa1_filter] + filters) if filters else taxa1_filter
    where2 = " AND ".join([taxa2_filter] + filters) if filters else taxa2_filter
    
    try:
        # Call the core density function for each taxon
        result1 = calculate_average_density(metric=metric, group_by='Genus', where_clause=where1)
        result2 = calculate_average_density(metric=metric, group_by='Genus', where_clause=where2)
        
        # Format the comparison
        filter_desc = []
        if year:
            filter_desc.append(f"Year: {year}")
        if region:
            filter_desc.append(f"Region: {region}")
        filter_str = ", ".join(filter_desc) if filter_desc else "all surveys"
        
        comparison = f"--- Comparison Result ---\n\n"
        comparison += f"Comparison of {taxa1} vs {taxa2} Average {metric} Density ({filter_str}):\n\n"
        comparison += f"Result for {taxa1}:\n{result1}\n\n"
        comparison += f"Result for {taxa2}:\n{result2}"
        
        return comparison
    
    except Exception as e:
        # Give the agent a more detailed error trace and a hint
        error_details = f"Error during taxa comparison: {e}. "
        error_details += f"This happened while comparing '{taxa1}' vs '{taxa2}'. "
        error_details += "Please check that the taxa names are correct and exist in the database."
        log_progress(error_details, is_reasoning=True)
        return error_details

# --- Advanced Statistical Analysis Functions ---
def perform_statistical_test(test_type: str, data_query: str, *args, **kwargs) -> str:
    """
    Unified function to perform various statistical tests on database query results.
    
    Args:
        test_type: Type of statistical test to perform
        data_query: SQL query to retrieve data
        *args, **kwargs: Additional arguments specific to each test type
        
    Returns:
        Formatted string with statistical test results
    """
    if not ADVANCED_STATS_AVAILABLE:
        return "Error: Advanced statistical analysis modules are not available. Please install required packages: scipy, statsmodels, scikit-learn."
    
    if engine is None:
        return "Error: Database connection is not available."
    
    try:
        # Execute the query to get data
        log_progress(f"Executing query for {test_type} analysis...", is_reasoning=True)
        data = pd.read_sql(data_query, engine)
        
        if data.empty:
            return f"Error: No data returned from query for {test_type} analysis."
        
        log_progress(f"Retrieved {len(data)} records for analysis", is_reasoning=True)
        
        # Perform the appropriate statistical test
        if test_type == "ttest":
            return _perform_ttest_analysis(data, *args, **kwargs)
        elif test_type == "anova":
            return _perform_anova_analysis(data, *args, **kwargs)
        elif test_type == "correlation":
            return _perform_correlation_analysis(data, *args, **kwargs)
        elif test_type == "regression":
            return _perform_regression_analysis(data, *args, **kwargs)
        elif test_type == "nonparametric":
            return _perform_nonparametric_analysis(data, *args, **kwargs)
        else:
            return f"Error: Unknown test type '{test_type}'"
            
    except Exception as e:
        error_msg = f"Error performing {test_type} analysis: {str(e)}"
        log_progress(error_msg, is_reasoning=True)
        return error_msg

def _perform_ttest_analysis(data: pd.DataFrame, group_column: str = None, 
                           test_type: str = "two_sample", alpha: float = 0.05) -> str:
    """Perform t-test analysis on the data."""
    try:
        if test_type == "one_sample":
            # For one-sample t-test, use the first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return "Error: No numeric columns found for t-test analysis."
            
            values = data[numeric_cols[0]]
            results = perform_ttest(values, test_type="one_sample")
            
        elif test_type in ["two_sample", "paired"]:
            if not group_column or group_column not in data.columns:
                return f"Error: Group column '{group_column}' not found in data."
            
            # Get unique groups
            groups = data[group_column].unique()
            if len(groups) != 2:
                return f"Error: Expected 2 groups for {test_type} t-test, found {len(groups)}"
            
            # Get the first numeric column for analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return "Error: No numeric columns found for t-test analysis."
            
            value_col = numeric_cols[0]
            group1_data = data[data[group_column] == groups[0]][value_col]
            group2_data = data[data[group_column] == groups[1]][value_col]
            
            results = perform_ttest(group1_data, group2_data, test_type=test_type, alpha=alpha)
            
        else:
            return f"Error: Unknown t-test type '{test_type}'"
        
        return format_statistical_results(results, f"{test_type.title()} T-Test")
        
    except Exception as e:
        return f"Error in t-test analysis: {str(e)}"

def _perform_anova_analysis(data: pd.DataFrame, dependent_var: str, 
                           independent_vars: list, alpha: float = 0.05) -> str:
    """Perform ANOVA analysis on the data."""
    try:
        if dependent_var not in data.columns:
            return f"Error: Dependent variable '{dependent_var}' not found in data."
        
        missing_vars = [var for var in independent_vars if var not in data.columns]
        if missing_vars:
            return f"Error: Independent variables not found: {missing_vars}"
        
        results = perform_anova(data, dependent_var, independent_vars, alpha=alpha)
        return format_statistical_results(results, "ANOVA")
        
    except Exception as e:
        return f"Error in ANOVA analysis: {str(e)}"

def _perform_correlation_analysis(data: pd.DataFrame, variables: list = None, 
                                 method: str = "pearson", alpha: float = 0.05) -> str:
    """Perform correlation analysis on the data."""
    try:
        if variables:
            missing_vars = [var for var in variables if var not in data.columns]
            if missing_vars:
                return f"Error: Variables not found: {missing_vars}"
        
        results = perform_correlation_analysis(data, variables, method, alpha)
        return format_statistical_results(results, f"{method.title()} Correlation Analysis")
        
    except Exception as e:
        return f"Error in correlation analysis: {str(e)}"

def _perform_regression_analysis(data: pd.DataFrame, dependent_var: str, 
                                independent_vars: list, alpha: float = 0.05) -> str:
    """Perform regression analysis on the data."""
    try:
        if dependent_var not in data.columns:
            return f"Error: Dependent variable '{dependent_var}' not found in data."
        
        missing_vars = [var for var in independent_vars if var not in data.columns]
        if missing_vars:
            return f"Error: Independent variables not found: {missing_vars}"
        
        results = perform_regression_analysis(data, dependent_var, independent_vars, alpha)
        return format_statistical_results(results, "Linear Regression Analysis")
        
    except Exception as e:
        return f"Error in regression analysis: {str(e)}"

def _perform_nonparametric_analysis(data: pd.DataFrame, group_column: str = None, 
                                   test_type: str = "mann_whitney", alpha: float = 0.05) -> str:
    """Perform non-parametric analysis on the data."""
    try:
        if test_type in ["mann_whitney", "wilcoxon"]:
            if not group_column or group_column not in data.columns:
                return f"Error: Group column '{group_column}' required for {test_type} test."
            
            groups = data[group_column].unique()
            if len(groups) != 2:
                return f"Error: Expected 2 groups for {test_type} test, found {len(groups)}"
            
            # Get the first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return "Error: No numeric columns found for analysis."
            
            value_col = numeric_cols[0]
            group1_data = data[data[group_column] == groups[0]][value_col]
            group2_data = data[data[group_column] == groups[1]][value_col]
            
            results = perform_nonparametric_tests(group1_data, group2_data, test_type=test_type)
            
        elif test_type == "kruskal_wallis":
            if not group_column or group_column not in data.columns:
                return f"Error: Group column '{group_column}' required for Kruskal-Wallis test."
            
            # Get the first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return "Error: No numeric columns found for analysis."
            
            value_col = numeric_cols[0]
            groups_df = data[[value_col, group_column]].rename(columns={value_col: 'value', group_column: 'group'})
            
            results = perform_nonparametric_tests(groups=groups_df, test_type=test_type)
            
        else:
            return f"Error: Unknown non-parametric test type '{test_type}'"
        
        return format_statistical_results(results, f"{test_type.replace('_', ' ').title()} Test")
        
    except Exception as e:
        return f"Error in non-parametric analysis: {str(e)}"

def generate_pdf_report(title: str = None, analysis_results: dict = None, 
                       include_plots: bool = True, filename: str = None) -> str:
    """
    Generate a comprehensive PDF report of statistical analysis results.
    
    Args:
        title: Report title
        analysis_results: Dictionary containing analysis results
        include_plots: Whether to include visualization plots
        filename: Output filename (optional)
        
    Returns:
        Status message with file path or error
    """
    if not ADVANCED_STATS_AVAILABLE:
        return "Error: PDF report generation is not available. Please install required packages: reportlab, weasyprint."
    
    try:
        log_progress("Generating comprehensive PDF report...", is_reasoning=True)
        
        # Extract and process parameters if they were passed as a single dictionary
        if analysis_results is None and isinstance(title, dict):
            # The function was likely called with a single dictionary parameter
            params = title
            title = params.get('title', 'Ecological Analysis Report')
            analysis_results = params.get('analysis_results', {})
            include_plots = params.get('include_plots', True)
            filename = params.get('filename', None)
        
        # Ensure we have a dictionary for analysis_results
        if analysis_results is None:
            analysis_results = {}
            
        # Create a simple analysis results dictionary if a string was passed
        if isinstance(analysis_results, str):
            analysis_results = {"analysis_summary": {"results": analysis_results}}
            
        # Add title to analysis_results if not already present
        if title and 'title' not in analysis_results:
            analysis_results['title'] = title
        
        log_progress(f"Generating report with parameters: title={title}, analysis_results keys={list(analysis_results.keys()) if isinstance(analysis_results, dict) else 'not a dict'}", is_reasoning=True)
        
        # Generate the report
        report_path = create_comprehensive_report(
            analysis_results=analysis_results,
            plots=None,  # Could be enhanced to include actual plot files
            data_tables=None,  # Could be enhanced to include data tables
            filename=filename
        )
        
        if report_path.startswith("Error:"):
            return report_path
        
        log_progress(f"PDF report generated successfully: {report_path}", is_reasoning=True)
        return f"✅ PDF report generated successfully!\n\nFile location: {report_path}\n\nThe report contains comprehensive statistical analysis results with professional formatting, including methodology descriptions, statistical test results, and interpretations."
        
    except Exception as e:
        error_msg = f"Error generating PDF report: {str(e)}"
        log_progress(error_msg, is_reasoning=True)
        return error_msg

# Legacy tool definitions are removed as they've been replaced by conditional creation below
# Tools are now created conditionally in the all_tools setup section

# --- Create a wrapper function to handle different input formats for plotting tools ---
def plotting_tool_wrapper(plotting_func):
    def wrapper(input_str):
        """
        Wrapper for the plotting functions to handle different input formats.
        
        Args:
            input_str: String or dictionary with query, title, and optional filename.
                
        Returns:
            Result from the plotting function
        """
        try:
            # Check if input is a dictionary or a string representation of a dict
            if isinstance(input_str, dict):
                params = input_str
            else:
                # Try to extract parameters from string input
                import re
                import json
                
                # First, try parsing as JSON
                try:
                    params = json.loads(input_str)
                except:
                    # If not valid JSON, try to extract parameters using regex
                    query_match = re.search(r'query["\s]*:["\s]*(.*?)["\s]*,', input_str)
                    chart_type_match = re.search(r'chart_type["\s]*:["\s]*(.*?)["\s]*,', input_str)
                    title_match = re.search(r'title["\s]*:["\s]*(.*?)["\s]*,', input_str) or re.search(r'title["\s]*:["\s]*(.*?)["\s]*\}', input_str)
                    filename_match = re.search(r'filename["\s]*:["\s]*(.*?)["\s]*\}', input_str)
                    
                    params = {}
                    if query_match:
                        params['query'] = query_match.group(1).strip('" ')
                    if chart_type_match:
                        params['chart_type'] = chart_type_match.group(1).strip('" ')
                    if title_match:
                        params['title'] = title_match.group(1).strip('" ')
                    if filename_match:
                        params['filename'] = filename_match.group(1).strip('" ')
            
            # Extract parameters
            query = params.get('query')
            chart_type = params.get('chart_type')
            title = params.get('title')
            filename = params.get('filename')
            
            # Validate required parameters
            if not query:
                return "Error: 'query' parameter is required"
            if not chart_type and plotting_func.__name__ == 'create_chart':
                return "Error: 'chart_type' parameter is required for create_chart"
            if not title:
                return "Error: 'title' parameter is required"
                
            # Call the appropriate function based on parameters
            if plotting_func.__name__ == 'create_chart':
                # For create_chart, we need to include the chart_type
                if filename:
                    return plotting_func(query, chart_type, title, filename)
                else:
                    return plotting_func(query, chart_type, title)
            else:
                # For legacy chart functions
                if filename:
                    return plotting_func(query, title, filename)
                else:
                    return plotting_func(query, title)
        except Exception as e:
            return f"Error parsing input for visualization: {e}. Please provide input in the format: {{\"query\": \"SELECT col1, col2 FROM table\", \"chart_type\": \"bar\", \"title\": \"Chart Title\", \"filename\": \"output.png\"}}"""
    return wrapper

# --- Create visualization tools ---
charting_tool = Tool(
    name="create_chart",
    func=plotting_tool_wrapper(create_chart),
    description="""
    Use this to create a visualization and save it as a file.
    It requires a SQL query, a chart_type, a title, and a filename.
    - Use chart_type 'bar' for comparing categories.
    - Use chart_type 'line' for showing trends over time.
    - Use chart_type 'scatter' for exploring relationships between two numeric variables.
    The SQL query MUST return exactly two columns corresponding to the x and y axes.
    
    You MUST provide input as a JSON object with these fields:
    - query: SQL query string that returns exactly 2 columns (required)  
    - chart_type: Type of chart to create ('bar', 'line', or 'scatter') (required)
    - title: Title for the chart (required)
    - filename: Name of the output file (optional, default: chart.png)
    
    Example: {"query": "SELECT Species, AVG(Biomass) FROM ltem_historical_database GROUP BY Species LIMIT 5", "chart_type": "bar", "title": "Average Biomass of Top 5 Species", "filename": "biomass_chart.png"}
    """
)

# --- Create the SQL Agent with our custom prompt ---
# --- Create summary tool ---
summary_tool = Tool(
    name="get_statistical_summary",
    func=get_statistical_summary,
    description="""
    Use this tool when the user asks for a 'full summary', 'statistics', 
    'breakdown', or 'statistical description' of numerical data.
    It takes a single SQL query as input and returns a complete statistical summary.
    """
)

# --- Create schema description tool ---
schema_tool = Tool(
    name="get_database_schema_description",
    func=get_database_schema_description,
    description="""
    Use this tool to get a detailed description of the database schema.
    This provides information about tables, their purposes, and the meaning of their columns.
    Call this when you need to understand the database structure or column meanings.
    No input is required.
    """
)
# Function to log progress to terminal with timestamps
def log_progress(message, elapsed_time=None, progress=None, is_sql=False, is_python=False, is_reasoning=False):
    """
    Log a progress message with timestamp, elapsed time, and progress bar if provided.
    
    Args:
        message (str): The progress message to display
        elapsed_time (float, optional): Elapsed time in seconds
        progress (float, optional): Progress value between 0.0 and 1.0
        is_sql (bool, optional): Whether this message is a SQL query
        is_python (bool, optional): Whether this message is Python code
        is_reasoning (bool, optional): Whether this message is agent reasoning
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Format elapsed time nicely
    time_str = ""
    if elapsed_time is not None:
        if elapsed_time < 1:
            time_str = f"(elapsed: {elapsed_time*1000:.2f}ms)"
        elif elapsed_time < 60:
            time_str = f"(elapsed: {elapsed_time:.2f}s)"
        else:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"(elapsed: {minutes}m {seconds:.2f}s)"
    
    # Create a progress bar if progress is provided
    progress_bar = ""
    if progress is not None:
        try:
            # Ensure progress is between 0 and 1
            progress = max(0, min(1, progress))
            
            # Create a 20-character progress bar
            bar_width = 20
            bar_fill = int(bar_width * progress)
            bar_empty = bar_width - bar_fill
            progress_bar = f"[{'█' * bar_fill}{' ' * bar_empty}] {progress*100:.1f}%"            
        except Exception:
            pass  # If there's an error creating the progress bar, just skip it
    
    # Format special message types with helpful prefixes and formatting
    prefix = ""
    if is_sql:
        prefix = "📊 SQL: "
        # We'll handle SQL formatting in the function call
    elif is_python:
        prefix = "🐍 PYTHON: "
    elif is_reasoning:
        prefix = "🧠 THINKING: "
    
    # Print the progress message with timestamp and progress bar if available
    if time_str and progress_bar:
        print(f"[{timestamp}] {prefix}{message} {progress_bar} {time_str}")
    elif time_str:
        print(f"[{timestamp}] {prefix}{message} {time_str}")
    elif progress_bar:
        print(f"[{timestamp}] {prefix}{message} {progress_bar}")
    else:
        print(f"[{timestamp}] {prefix}{message}")
    
    sys.stdout.flush()

# Intercept tool usage to provide progress messages with timing information
class ProgressCallback:
    def __init__(self):
        self.start_times = {}
        self.current_tool = None
        self.running_tools = {}
        self.progress_threads = {}
        self.estimated_durations = {
            # Tool name pattern: estimated seconds it typically takes
            "sql": 10,
            "python": 15,
        }
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Log when a tool starts running."""
        tool_name = serialized.get("name", "unknown_tool")
        tool_id = kwargs.get("run_id", str(time.time()))
        self.start_times[tool_id] = time.time()
        
        # Select message based on tool
        if "sql" in tool_name.lower():
            # Format and translate SQL for better understanding
            try:
                # Log the actual SQL for debugging
                log_progress(f"SQL Query: {input_str}", is_sql=True)
                
                # Translate to plain English
                translation = translate_sql_to_english(input_str)
                log_progress(translation, is_reasoning=True)
            except Exception as e:
                log_progress(f"Executing database query: {input_str[:50]}...")
            
            self._start_progress_thread(tool_id, "Running database query", "sql_query")
        elif "python" in tool_name.lower():
            # Format the Python code for better readability
            log_progress(f"Running Python analysis", is_python=True)
            if isinstance(input_str, str):
                # Try to display the Python code in a readable format
                log_progress(input_str.strip(), is_python=True)
            self._start_progress_thread(tool_id, "Analyzing data with Python", "python_repl")
        elif "chart" in tool_name.lower() or "visual" in tool_name.lower():
            # Extract chart details from input
            try:
                import json
                chart_details = json.loads(input_str)
                chart_type = chart_details.get('chart_type', 'unknown')
                chart_title = chart_details.get('title', 'Chart')
                log_progress(f"Creating {chart_type} chart: {chart_title}")
            except:
                log_progress(f"Creating visualization")
            self._start_progress_thread(tool_id, "Generating visualization", "visualization")
        else:
            message = f"Using {tool_name}"
            log_progress(message)
            self._start_progress_thread(tool_id, message)
    
    def on_tool_end(self, output, **kwargs):
        """Log when a tool finishes."""
        tool_id = kwargs.get("run_id", "unknown")
        if tool_id in self.start_times:
            elapsed_time = time.time() - self.start_times[tool_id]
            # Stop the progress thread if it's running
            self._stop_progress_thread(tool_id)
            log_progress("Operation completed", elapsed_time, 1.0)
            del self.start_times[tool_id]
    
    def on_tool_error(self, error, **kwargs):
        """Log when a tool errors."""
        tool_id = kwargs.get("run_id", "unknown")
        if tool_id in self.start_times:
            elapsed_time = time.time() - self.start_times[tool_id]
            # Stop the progress thread if it's running
            self._stop_progress_thread(tool_id)
            log_progress(f"Error during operation: {error}", elapsed_time)
            del self.start_times[tool_id]
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when an LLM starts."""
        llm_id = kwargs.get("run_id", str(time.time()))
        self.start_times[llm_id] = time.time()
        self._start_progress_thread(llm_id, "AI model processing", "llm")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Log when a chain starts - can capture agent reasoning."""
        # Try to extract reasoning/planning from agent chains
        try:
            # Check for agent reasoning in inputs
            if "input" in inputs and isinstance(inputs["input"], str) and \
               any(kw in inputs["input"].lower() for kw in ["step by step", "let's think", "i need to", "first", "plan"]):
                log_progress(f"Planning approach: {inputs['input'][:100]}...", is_reasoning=True)
        except Exception as e:
            pass  # Skip if we can't extract reasoning
    
    def on_llm_end(self, response, **kwargs):
        """Log when an LLM ends."""
        llm_id = kwargs.get("run_id", "unknown")
        if llm_id in self.start_times:
            elapsed_time = time.time() - self.start_times[llm_id]
            # Stop the progress thread if it's running
            self._stop_progress_thread(llm_id)
            log_progress("AI model processing completed", elapsed_time, 1.0)
            del self.start_times[llm_id]
    
            
    def _start_progress_thread(self, tool_id, message, tool_type=None):
        """Start a thread to periodically output progress messages for long-running operations."""
        # Create and start a daemon thread
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._progress_updater,
            args=(tool_id, message, stop_event, tool_type),
            daemon=True  # Daemon threads exit when the main program exits
        )
        thread.start()
        # Store the thread and stop event
        self.progress_threads[tool_id] = (thread, stop_event)
    
    def _stop_progress_thread(self, tool_id):
        """Stop the progress thread for the specified tool."""
        if tool_id in self.progress_threads:
            _, stop_event = self.progress_threads[tool_id]
            stop_event.set()  # Signal the thread to stop
            del self.progress_threads[tool_id]
            # No need to join, the thread will exit naturally
    
    def _progress_updater(self, tool_id, message, stop_event, tool_type=None):
        """Thread function that periodically outputs progress messages with a progress bar."""
        update_interval = 2  # Seconds between progress updates
        # Get the estimated duration for this tool type
        estimated_duration = self.estimated_durations.get(tool_type, 10) if tool_type else 10
        
        while tool_id in self.start_times and not stop_event.is_set():
            # Calculate elapsed time
            current_time = time.time()
            elapsed_time = current_time - self.start_times[tool_id]
            
            # Calculate progress as a fraction (0.0 to 1.0) based on estimated duration
            # Cap at 0.95 until completion to show it's still in progress
            progress = min(0.95, elapsed_time / estimated_duration)
            
            # Output a progress update with progress bar
            log_progress(f"{message}", elapsed_time, progress)
            
            # Wait for the next update or until stopped
            stop_event.wait(update_interval)

# --- Tool descriptions ---
SCHEMA_TOOL_DESCRIPTION = """Use this tool to get the schema of the database, including table and column names, types, and descriptions.
 This is useful when you need to know what tables and columns are available in the database."""

SQL_QUERY_TOOL_DESCRIPTION = """Use this tool to execute SQL queries against the database.
 The input should be a valid SQL query string."""

STATISTICAL_TOOL_DESCRIPTION = """Use this tool when the user asks for a 'full summary', 'statistics', 
'breakdown', or 'statistical description' of numerical data.
It takes a single SQL query as input and returns a complete statistical summary."""

VISUALIZATION_TOOL_DESCRIPTION = """Use this tool to create visualizations from SQL query results.
Args: query (str): SQL query to get data, chart_type (str): 'bar', 'line', or 'scatter', 
title (str): Chart title, filename (str, optional): Output filename."""

REPL_TOOL_DESCRIPTION = """Use this tool to execute Python code. It has access to pandas and numpy.
It's useful for complex data manipulation, statistics, and custom visualizations."""

DENSITY_TOOL_DESCRIPTION = """Calculate scientifically valid average density for a given metric, grouped by a specific column, with optional filters.
This is the primary tool for any abundance or biomass calculation. It properly sums by transect first, then averages the totals.
Args: metric (str): 'Quantity' or 'Biomass', group_by (str): column to group by, where_clause (str): optional SQL WHERE clause."""

COMPARE_TAXA_TOOL_DESCRIPTION = """Compare the average density of two taxonomic groups for either 'Biomass' or 'Quantity'.
Use this for any direct comparison question like 'compare X and Y'. The taxa names should be genus or species names.
Args: metric (str): 'Biomass' or 'Quantity', taxa1 (str): first taxon name, taxa2 (str): second taxon name, 
year (int, optional): specific year, region (str, optional): specific region."""

# --- Helper function to create tools ---
def create_tool(name, func, description, use_structured_tool=False):
    """Create a LangChain Tool with the given name, function, and description.
    
    Args:
        name (str): The name of the tool
        func (callable): The function to call when the tool is invoked
        description (str): A description of what the tool does
        use_structured_tool (bool): Whether to use StructuredTool for multi-argument functions
        
    Returns:
        Tool or StructuredTool: A LangChain tool object
    """
    from langchain.tools import Tool, StructuredTool
    
    if use_structured_tool:
        # Use StructuredTool for functions with multiple arguments
        return StructuredTool.from_function(
            func=func,
            name=name,
            description=description
        )
    else:
        # Use regular Tool for single-argument functions
        return Tool(
            name=name,
            func=func,
            description=description
        )

# --- Initialize tools collection ---
all_tools = []  # Start with an empty list

# Add schema tool - create it conditionally in case schema_description is referenced from LangChain
try:
    schema_tool = Tool(
        name="get_database_schema_description",
        func=get_database_schema_description,
        description="""Use this tool to get a detailed description of the database schema.
        This provides information about tables, their purposes, and the meaning of their columns.
        Call this when you need to understand the database structure or column meanings.
        No input is required."""
    )
    all_tools.append(schema_tool)
    print("Added schema description tool.")
except Exception as e:
    print(f"Could not create schema tool: {e}")

# Only try to create database tools if both LangChain and database are available
if DATABASE_AVAILABLE and pd is not None:
    try:
        # Create the database schema description tool
        database_schema_tool = create_tool(
            name="get_database_schema_description", 
            func=get_database_schema_description,
            description=SCHEMA_TOOL_DESCRIPTION
        )
        all_tools.append(database_schema_tool)
        print("Added schema description tool.")
        
        # Create the SQL query tool
        sql_query_tool = create_tool(
            name="run_sql_query", 
            func=lambda query: query, # This just returns the query itself, as the agent will handle it
            description=SQL_QUERY_TOOL_DESCRIPTION,
            use_structured_tool=True
        )
        all_tools.append(sql_query_tool)
        print("Added SQL query tool.")
        
        # Create the statistical summary tool
        stats_tool = create_tool(
            name="get_statistical_summary", 
            func=get_statistical_summary,
            description=STATISTICAL_TOOL_DESCRIPTION
        )
        all_tools.append(stats_tool)
        print("Added statistical summary tool.")
        
        # Add calculate_average_density tool if engine exists
        if 'engine' in globals() and engine is not None:
            from langchain.tools import StructuredTool
            
            # Register the calculate_average_density tool using StructuredTool
            density_tool = StructuredTool.from_function(
                func=calculate_average_density,
                name="calculate_average_density",
                description=DENSITY_TOOL_DESCRIPTION
            )
            all_tools.append(density_tool)
            
            # Register the compare_taxa_density tool using StructuredTool
            compare_taxa_tool = StructuredTool.from_function(
                func=compare_taxa_density,
                name="compare_taxa",
                description=COMPARE_TAXA_TOOL_DESCRIPTION
            )
            all_tools.append(compare_taxa_tool)
            print("Added density calculation and taxa comparison tools.")
        
        # Add advanced statistical analysis tools if available
        if ADVANCED_STATS_AVAILABLE:
            # T-test tool
            ttest_tool = StructuredTool.from_function(
                func=lambda data_query, group_column=None, test_type="two_sample", alpha=0.05: 
                    perform_statistical_test("ttest", data_query, group_column, test_type, alpha),
                name="perform_ttest",
                description="""Perform t-tests (one-sample, two-sample, or paired) on ecological data.
                Args:
                - data_query: SQL query to get the data
                - group_column: Column name to split data into groups (for two-sample tests)
                - test_type: 'one_sample', 'two_sample', or 'paired'
                - alpha: Significance level (default 0.05)
                Use this for comparing means between groups or against a hypothesized value."""
            )
            all_tools.append(ttest_tool)
            
            # ANOVA tool
            anova_tool = StructuredTool.from_function(
                func=lambda data_query, dependent_var, independent_vars, alpha=0.05:
                    perform_statistical_test("anova", data_query, dependent_var, independent_vars, alpha),
                name="perform_anova",
                description="""Perform one-way or two-way ANOVA on ecological data.
                Args:
                - data_query: SQL query to get the data
                - dependent_var: Name of the dependent variable column
                - independent_vars: List of independent variable column names (max 2)
                - alpha: Significance level (default 0.05)
                Use this for comparing means across multiple groups."""
            )
            all_tools.append(anova_tool)
            
            # Correlation analysis tool
            correlation_tool = StructuredTool.from_function(
                func=lambda data_query, variables=None, method="pearson", alpha=0.05:
                    perform_statistical_test("correlation", data_query, variables, method, alpha),
                name="perform_correlation_analysis",
                description="""Perform correlation analysis on ecological data.
                Args:
                - data_query: SQL query to get the data
                - variables: List of variable names (if None, use all numeric columns)
                - method: 'pearson', 'spearman', or 'kendall'
                - alpha: Significance level (default 0.05)
                Use this to identify relationships between variables."""
            )
            all_tools.append(correlation_tool)
            
            # Regression analysis tool
            regression_tool = StructuredTool.from_function(
                func=lambda data_query, dependent_var, independent_vars, alpha=0.05:
                    perform_statistical_test("regression", data_query, dependent_var, independent_vars, alpha),
                name="perform_regression_analysis",
                description="""Perform linear regression analysis on ecological data.
                Args:
                - data_query: SQL query to get the data
                - dependent_var: Name of the dependent variable
                - independent_vars: List of independent variable names
                - alpha: Significance level (default 0.05)
                Use this to model relationships and make predictions."""
            )
            all_tools.append(regression_tool)
            
            # Non-parametric tests tool
            nonparametric_tool = StructuredTool.from_function(
                func=lambda data_query, group_column=None, test_type="mann_whitney", alpha=0.05:
                    perform_statistical_test("nonparametric", data_query, group_column, test_type, alpha),
                name="perform_nonparametric_test",
                description="""Perform non-parametric statistical tests on ecological data.
                Args:
                - data_query: SQL query to get the data
                - group_column: Column name to split data into groups
                - test_type: 'mann_whitney', 'wilcoxon', or 'kruskal_wallis'
                - alpha: Significance level (default 0.05)
                Use this when data doesn't meet normality assumptions."""
            )
            all_tools.append(nonparametric_tool)
            
            # PDF report generation tool
            pdf_report_tool = StructuredTool.from_function(
                func=generate_pdf_report,
                name="generate_pdf_report",
                description="""Generate a comprehensive PDF report of statistical analysis results.
                Args:
                - title: Report title
                - analysis_results: Dictionary containing analysis results
                - include_plots: Whether to include visualization plots
                - filename: Output filename (optional)
                Use this to create professional reports of your analyses."""
            )
            all_tools.append(pdf_report_tool)
            
            print("Added advanced statistical analysis and PDF reporting tools.")
            
    except Exception as e:
        print(f"Could not create database tools: {e}")
else:
    print("Database tools are not available due to missing dependencies.")

# Add visualization tool if all required dependencies are available
if LANGCHAIN_AVAILABLE and VISUALIZATION_AVAILABLE and DATABASE_AVAILABLE:
    try:
        charting_tool = Tool(
            name="create_chart",
            func=plotting_tool_wrapper(create_chart),
            description="""Use this to create a visualization and save it as a file.
            It requires a SQL query, a chart_type, a title, and a filename."""
        )
        all_tools.append(charting_tool)
        print("Added visualization tool.")
    except Exception as e:
        print(f"Could not create visualization tool: {e}")
else:
    print("Visualization tool is not available due to missing dependencies.")

# Add Python REPL tool if LangChain is available
if LANGCHAIN_AVAILABLE:
    try:
        # Try to import from langchain_experimental as per error message
        try:
            from langchain_experimental.tools import PythonREPLTool
            python_repl_tool = PythonREPLTool()
            all_tools.append(python_repl_tool)
            print("Added Python REPL tool.")
        except ImportError:
            # Fall back to legacy import (though this will likely fail)
            from langchain.tools import PythonREPLTool
            python_repl_tool = PythonREPLTool()
            all_tools.append(python_repl_tool)
            print("Added Python REPL tool (legacy import).")
    except Exception as e:
        print(f"Could not create Python REPL tool: {e}")
else:
    print("Python REPL tool is not available due to missing LangChain dependencies.")

# Verify we have all needed components before attempting to create the agent
required_packages = {
    "LangChain": LANGCHAIN_AVAILABLE,
    "Database": DATABASE_AVAILABLE,
    "Visualization": VISUALIZATION_AVAILABLE
}

missing_packages = [pkg for pkg, available in required_packages.items() if not available]

# Print a summary of availability
print("\n--- System Status ---")
for pkg, available in required_packages.items():
    status = "✅ Available" if available else "❌ Missing"
    print(f"{pkg}: {status}")

# Create the agent and data validator only if we have the minimum required dependencies
agent_executor = None
data_validator = None
agent_available = False

# Check if we can create the agent (need LangChain and at least a database or REPL)
if LANGCHAIN_AVAILABLE and (DATABASE_AVAILABLE or pd is not None):
    # Initialize data validator if database is available
    if DATABASE_AVAILABLE and engine is not None:
        data_validator = DataValidator(engine)
        print("✅ Data Validator initialized successfully!")
    try:
        # --- Create the SQL Agent with our custom prompt ---
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,  # Pass the database directly
            agent_type="openai-tools",
            prompt=prompt, # Pass the custom prompt here
            verbose=False,  # Hide internal thoughts
            extra_tools=all_tools,  # Add our custom tools as extra_tools
            callbacks=[ProgressCallback()],
            max_iterations=40,  # Increased from 25 to 40 for complex comparison queries
            max_execution_time=600  # Increased timeout to 10 minutes for long analyses
        )
        
        agent_available = True
        print("\n✅ Agent created successfully with all available tools!")
    except Exception as e:
        print(f"\n❌ Could not create the agent: {e}")
else:
    print("\n❌ Cannot create agent due to missing core dependencies.")

# --- Chat History and Interactive Loop ---
print("\n--- Ecological Monitoring Agent ---")

# Exit if agent is not available
if not agent_available:
    print("\nWARNING: Agent is not available due to missing dependencies.")
    print("\nRequired packages:")
    print("  - langchain-openai, langchain-community, langchain-core, langchain-experimental")
    print("  - sqlalchemy, pandas")
    print("  - matplotlib, seaborn (for visualization)")
    print("\nInstallation command:")
    print("  pip install langchain-openai langchain-community langchain-core langchain-experimental sqlalchemy pandas matplotlib seaborn")
    sys.exit(1)

print("\nAgent is ready! Ask me questions about your database.")
print("Type 'exit', 'quit', or 'q' to end the session.")

# Chat history disabled to prevent memory retention between questions
# chat_history = []

# Main interaction loop
while True:
    user_question = input("\nYour Question: ")
    if user_question.lower() in ['exit', 'quit', 'q']:
        print("\nThank you for using the Ecological Monitoring Agent. Goodbye!")
        break
    
    spinner = Spinner("🐠 Analyzing data...")  # Create a spinner instance
    try:
        spinner.start()  # Start the animation
        
        # Add initial progress message with timestamp and progress bar
        start_time = time.time()
        log_progress("Starting analysis...")
        
        # Execute the agent with the user's question
        response = agent_executor.invoke({
            "input": user_question
            # "chat_history" removed to prevent memory retention between questions
        })
        
        # Sanitize the response to check for hallucinated regions/locations
        sanitized_output = response["output"]
        if data_validator is not None:
            log_progress("Validating regions and locations in response...")
            sanitized_output = data_validator.sanitize_response(response["output"])
        
        spinner.stop()  # Stop the animation
        
        # Calculate total elapsed time
        total_elapsed = time.time() - start_time
        
        # Add completion message with total time and 100% progress
        log_progress("Analysis complete", total_elapsed, 1.0)
        
        # Chat history updates removed to prevent memory retention between questions
        # chat_history.append(HumanMessage(content=user_question))
        # chat_history.append(AIMessage(content=sanitized_output))
        
        # Display the result
        print("\nAgent's Answer:")
        print(sanitized_output)
        
    except Exception as e:
        # Handle errors gracefully
        spinner.stop()  # Ensure spinner stops on error
        print(f"\nAn error occurred: {e}")
        
        # Display additional diagnostic information
        current_time = time.time()
        log_progress("Error details", current_time - start_time)
        
        # Helpful troubleshooting suggestions
        if "rate limit" in str(e).lower():
            print("This appears to be a rate limit error from OpenAI.")
            print("The agent is using gpt-4o-mini to avoid rate limiting where possible.")
            print("You may need to wait a few minutes before trying again.")
        else:
            print("If errors persist, check that all required dependencies are installed")
            print("and that your database connection information is correct.")
            
        # Ask if they want to continue
        print("\nDo you want to continue with another question? (yes/no)")
        continue_response = input().lower()
        if continue_response not in ["y", "yes"]:
            print("\nExiting due to error. Goodbye!")
            break

