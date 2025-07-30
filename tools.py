"""
tools.py - Tool definitions for the Deep Analysis Agent Framework

This module contains definitions for all tools used by the Data Analyst agent.
Each tool follows a consistent interface pattern and provides detailed logging.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import scipy.stats
from io import StringIO
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import traceback
import contextlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deep_analysis.log'
)
logger = logging.getLogger('deep_analysis_tools')

# Load environment variables
load_dotenv()

class DatabaseQueryTool:
    """
    A robust tool for executing SQL queries against a connected database.
    
    This tool handles connection management, query execution, error handling,
    and result formatting. It provides verbose logging for traceability.
    """
    
    def __init__(self, db_uri: Optional[str] = None):
        """
        Initialize the database query tool with connection parameters.
        
        Args:
            db_uri: Database URI string. If None, will be loaded from environment 
                   variables (DB_URI or DATABASE_URL).
        """
        self.db_uri = db_uri or os.getenv("DB_URI") or os.getenv("DATABASE_URL")
        if not self.db_uri:
            raise ValueError("Database URI not provided and not found in environment variables")
        
        # Create engine but defer connection until needed
        self.engine = create_engine(self.db_uri)
        logger.info(f"DatabaseQueryTool initialized with engine for {self.db_uri.split('@')[-1]}")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return the results as a list of dictionaries.
        
        Args:
            query: A string containing a valid SQL query.
            
        Returns:
            A list of dictionaries, where each dictionary represents a row with
            column names as keys and cell values as values.
            
        Raises:
            Exception: If the query execution fails.
        """
        start_time = datetime.now()
        logger.info(f"Executing query: {query}")
        
        try:
            # Execute the query
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                
                # Convert result to list of dictionaries
                column_names = result.keys()
                rows = result.fetchall()
                
                # Format the result
                formatted_result = [
                    {column: value for column, value in zip(column_names, row)}
                    for row in rows
                ]
                
                execution_time = (datetime.now() - start_time).total_seconds()
                row_count = len(formatted_result)
                logger.info(f"Query executed successfully in {execution_time:.2f}s. Returned {row_count} rows.")
                
                # Log first few rows for debugging
                if formatted_result:
                    sample = formatted_result[:3]
                    logger.debug(f"Sample results: {sample}")
                
                return formatted_result
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get the schema information for a specified table.
        
        Args:
            table_name: Name of the table to examine.
            
        Returns:
            A list of dictionaries containing column information.
        """
        logger.info(f"Retrieving schema for table: {table_name}")
        
        # This query works for most SQL databases (PostgreSQL, MySQL, SQLite)
        # For other databases, this might need to be adjusted
        query = f"""
        SELECT 
            column_name, 
            data_type,
            CASE WHEN character_maximum_length IS NOT NULL 
                 THEN character_maximum_length::text 
                 ELSE '' END as max_length
        FROM 
            information_schema.columns
        WHERE 
            table_name = '{table_name}'
        ORDER BY 
            ordinal_position
        """
        
        try:
            schema_info = self.execute_query(query)
            logger.info(f"Retrieved schema with {len(schema_info)} columns")
            return schema_info
        except Exception as e:
            # If the information_schema approach fails, try a database-agnostic approach
            logger.warning(f"Standard schema query failed: {str(e)}. Trying alternative approach.")
            
            try:
                # Get a sample row to infer schema
                sample_query = f"SELECT * FROM {table_name} LIMIT 1"
                sample = self.execute_query(sample_query)
                
                if sample:
                    # Extract column names and types from the sample
                    schema_info = [
                        {
                            "column_name": col,
                            "data_type": type(sample[0][col]).__name__,
                            "max_length": ""
                        }
                        for col in sample[0].keys()
                    ]
                    logger.info(f"Inferred schema with {len(schema_info)} columns")
                    return schema_info
                else:
                    logger.warning(f"Table {table_name} appears to be empty")
                    return []
            except Exception as inner_e:
                logger.error(f"Alternative schema retrieval failed: {str(inner_e)}")
                raise

class PythonInterpreterTool:
    """
    A secure Python code execution environment for data analysis and manipulation.
    
    This tool executes Python code within a sandboxed environment with access to
    pandas, numpy, and scipy libraries. It captures output and returns results.
    """
    
    def __init__(self):
        """Initialize the Python interpreter tool with required libraries"""
        # Libraries available in the execution environment
        self.available_libraries = {
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'scipy.stats': scipy.stats
        }
        logger.info("PythonInterpreterTool initialized with data science libraries")
    
    def execute_code(self, code: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code in a secure environment and return the results.
        
        Args:
            code: A string containing valid Python code to execute.
            input_data: Optional dictionary of input variables to be available in the code.
                        Typically contains DataFrames from database queries.
            
        Returns:
            A dictionary containing:
                'result': The return value of the code (if any)
                'output': Any printed output during execution
                'dataframes': Any pandas DataFrames created during execution
                'figures': Any matplotlib figures created during execution (as file paths)
                'error': Error message (if execution failed)
                
        Raises:
            No exceptions are raised; errors are captured and returned in the result.
        """
        start_time = datetime.now()
        logger.info(f"Executing Python code of length {len(code)} characters")
        logger.debug(f"Code to execute: {code}")
        
        # Create a secure namespace with allowed libraries
        namespace = self.available_libraries.copy()
        
        # Add input data to the namespace
        if input_data:
            namespace.update(input_data)
        
        # Capture stdout to get printed output
        stdout_capture = StringIO()
        
        result = {
            'result': None,
            'output': '',
            'dataframes': {},
            'figures': [],
            'error': None
        }
        
        # Execute the code with captured stdout
        try:
            with contextlib.redirect_stdout(stdout_capture):
                # Execute the code
                exec_result = exec(code, namespace)
                
                # Collect the return value if last statement is an expression
                last_line = code.strip().split('\n')[-1]
                if not (last_line.startswith('if ') or 
                        last_line.startswith('def ') or 
                        last_line.startswith('class ') or
                        last_line.startswith('for ') or
                        last_line.startswith('while ') or
                        last_line.startswith('#') or
                        '=' in last_line):
                    try:
                        result['result'] = eval(last_line, namespace)
                    except:
                        # If the last line isn't a valid expression, just ignore
                        pass
                
                # Collect any created pandas DataFrames
                for var_name, var_value in namespace.items():
                    if isinstance(var_value, pd.DataFrame) and var_name not in self.available_libraries:
                        # Only include dataframes that weren't input
                        if not (input_data and var_name in input_data):
                            result['dataframes'][var_name] = {
                                'shape': var_value.shape,
                                'columns': var_value.columns.tolist(),
                                'sample': var_value.head(5).to_dict('records')
                            }
                
                # Check for matplotlib figures
                if 'plt' in namespace or 'matplotlib.pyplot' in namespace:
                    plt = namespace.get('plt', namespace.get('matplotlib.pyplot'))
                    if plt and hasattr(plt, 'get_fignums') and plt.get_fignums():
                        for i, fig_num in enumerate(plt.get_fignums()):
                            fig = plt.figure(fig_num)
                            fig_path = f"figure_{start_time.strftime('%Y%m%d_%H%M%S')}_{i}.png"
                            fig.savefig(fig_path)
                            result['figures'].append(fig_path)
                
            # Get the captured stdout
            result['output'] = stdout_capture.getvalue()
            
            # Calculate and log execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Python code executed successfully in {execution_time:.2f}s")
            if result['dataframes']:
                logger.info(f"Created {len(result['dataframes'])} dataframes")
            if result['figures']:
                logger.info(f"Created {len(result['figures'])} figures")
            
        except Exception as e:
            # Capture the full stack trace
            error_tb = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_tb}"
            result['error'] = error_msg
            result['output'] = stdout_capture.getvalue()  # Include any output before the error
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Python code execution failed in {execution_time:.2f}s: {str(e)}")
        
        return result

class VisualizationTool:
    """
    A tool for creating data visualizations from pandas DataFrames.
    
    This tool generates charts using matplotlib and seaborn, saving them
    to image files that can be included in the final report.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the visualization tool with an output directory.
        
        Args:
            output_dir: Directory where visualization images will be saved.
        """
        self.output_dir = output_dir
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Import visualization libraries
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Store references to the libraries
        self.plt = plt
        self.sns = sns
        
        # Configure default styling
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })
        
        logger.info(f"VisualizationTool initialized with output directory: {output_dir}")
    
    def create_visualization(self, 
                          data: pd.DataFrame, 
                          chart_type: str, 
                          title: str,
                          x_axis: str, 
                          y_axis: str, 
                          hue: Optional[str] = None,
                          filename: Optional[str] = None,
                          dpi: int = 300,
                          figsize: tuple = (10, 6),
                          **kwargs) -> str:
        """
        Create a visualization from a pandas DataFrame.
        
        Args:
            data: The pandas DataFrame containing the data to visualize.
            chart_type: Type of chart to create ('bar', 'line', 'scatter', 'box', 'hist', etc.)
            title: Title for the chart.
            x_axis: Column name to use for the x-axis.
            y_axis: Column name to use for the y-axis.
            hue: Optional column name to use for color grouping.
            filename: Optional custom filename for the saved image. If None, a name will be generated.
            dpi: Resolution of the output image in dots per inch.
            figsize: Tuple of (width, height) for the figure size in inches.
            **kwargs: Additional keyword arguments to pass to the plotting function.
            
        Returns:
            File path to the saved visualization image.
        """
        start_time = datetime.now()
        logger.info(f"Creating {chart_type} chart with dimensions {figsize}, title: '{title}'")
        
        try:
            # Create a new figure with the specified size
            self.plt.figure(figsize=figsize)
            
            # Select the appropriate plotting function based on chart type
            if chart_type.lower() == 'bar':
                if hue:
                    ax = self.sns.barplot(x=x_axis, y=y_axis, hue=hue, data=data, **kwargs)
                else:
                    ax = self.sns.barplot(x=x_axis, y=y_axis, data=data, **kwargs)
                    
            elif chart_type.lower() == 'line':
                if hue:
                    ax = self.sns.lineplot(x=x_axis, y=y_axis, hue=hue, data=data, **kwargs)
                else:
                    ax = self.sns.lineplot(x=x_axis, y=y_axis, data=data, **kwargs)
                    
            elif chart_type.lower() == 'scatter':
                if hue:
                    ax = self.sns.scatterplot(x=x_axis, y=y_axis, hue=hue, data=data, **kwargs)
                else:
                    ax = self.sns.scatterplot(x=x_axis, y=y_axis, data=data, **kwargs)
                    
            elif chart_type.lower() == 'box':
                if hue:
                    ax = self.sns.boxplot(x=x_axis, y=y_axis, hue=hue, data=data, **kwargs)
                else:
                    ax = self.sns.boxplot(x=x_axis, y=y_axis, data=data, **kwargs)
                    
            elif chart_type.lower() == 'hist':
                ax = self.sns.histplot(data=data, x=x_axis, **kwargs)
                
            elif chart_type.lower() == 'heatmap':
                # For heatmap, data should typically be a pivot table or correlation matrix
                ax = self.sns.heatmap(data, annot=True, cmap="YlGnBu", **kwargs)
                
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            # Set title and labels
            self.plt.title(title)
            self.plt.xlabel(x_axis)
            self.plt.ylabel(y_axis)
            
            # Adjust layout to prevent labels from being cut off
            self.plt.tight_layout()
            
            # Generate a filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{chart_type}_{timestamp}.png"
            
            # Make sure the filename has a .png extension
            if not filename.endswith('.png'):
                filename += '.png'
            
            # Create the full file path
            file_path = os.path.join(self.output_dir, filename)
            
            # Save the figure
            self.plt.savefig(file_path, dpi=dpi)
            
            # Close the figure to free memory
            self.plt.close()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Chart created successfully in {execution_time:.2f}s: {file_path}")
            
            return file_path
            
        except Exception as e:
            # Log the error and return None
            logger.error(f"Failed to create chart: {str(e)}")
            self.plt.close()  # Make sure to close the figure even if there's an error
            raise

# Instantiate the tools for direct import
database_query_tool = DatabaseQueryTool().execute_query
python_interpreter_tool = PythonInterpreterTool().execute_code
visualization_tool = VisualizationTool().create_visualization
