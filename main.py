#!/usr/bin/env python3

import os
import time
import threading
import sys
from datetime import datetime

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
- `Quantity`: The abundance, i.e., how many organisms of that size were counted within a transect. 
- `Size`: The size class of the organisms. 
- `Biomass`: Fish biomass (not available for invertebrates), calculated using size, quantity, and growth parameters. 
- `MPA`, `Protection_status`: Conservation status of the location. 
- `bleaching_coverage`: A key environmental indicator for coral health. 
- `TrophicLevelF`, `TrophicLevel`, `TrophicGroup`, `Functional_groups`: Functional traits from trophic levels in factor, number, groups, and functional groups. 
- `Area`: The area of the transect used. 

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

# --- Define a powerful, custom prompt for our agent ---
SYSTEM_PROMPT = """
You are a highly intelligent ecological data analyst AI for the Gulf of California.
You have access to a SQL database and a suite of tools including a Python code interpreter and a chart generator.
Your primary goal is to answer user questions about this data, performing analysis and creating visualizations as needed.

**Key Instructions:**
1.  **Understand the Schema First:** If you are ever unsure about table names, column meanings, or how to calculate a metric, your first step **MUST** be to use the `get_database_schema_description` tool.
2.  **Maintain Context:** For follow-up questions, you **MUST** look at the previous conversation turn to maintain context (e.g., for `Species`, `Location`, or `Year`).
3.  **Think Step-by-Step:** For any complex request, formulate a plan first before executing it. Decide which tools you need and in what order.

---
### **CRITICAL ANALYSIS RULE: Data Aggregation**
To avoid skewed results from raw observational data, you **MUST** follow this rule for any analysis involving `Quantity` or `Biomass`:

1.  **Standardize First:** Before you compute any averages or perform other high-level analysis, you must first create a standardized dataset.
2.  **Aggregation Step:** Your first SQL query should sum the total `Quantity` and total `Biomass` for each unique combination of `Year`, `Region`, `Reef`, `Depth2`, and `Transect`. This creates a dataset where each row represents the total for one transect survey.
3.  **Analyze Second:** You can then use this pre-aggregated data for further analysis (e.g., calculating the average biomass per reef, or analyzing trends per year). This is often best done by passing the result of the first query to the `python_repl_tool`.

---
### **CRITICAL ANALYSIS RULE: Statistical Measures for Comparisons**
For any comparative analysis between regions, years, species, or other categories, you **MUST** use proper statistical measures:

1. **Always Use Summary Statistics:** When comparing groups, you must use averages, medians, quantiles, or other statistical summaries rather than individual data points.
2. **Include Variability Measures:** Always report measures of variability (standard deviation, interquartile range, confidence intervals) alongside central tendencies.
3. **Appropriate Statistical Tests:** When determining if differences are significant, use appropriate statistical tests (t-tests, ANOVA, etc.) and report p-values.
4. **Visual Comparison:** When creating visualizations for comparisons, include error bars or box plots to represent distributions, not just point estimates.

---
### TOOL USAGE RULES

1.  **For Simple Queries:** If the user asks a direct question that can be answered with a single SQL query (and does not violate the aggregation rule), use the `query_sql_database` tool.
2.  **For Advanced Analysis:** For complex calculations, use the two-step process: First, fetch the data with `query_sql_database` (respecting the aggregation rule), then analyze it with `python_repl_tool`.
3.  **For Visualizations:** Use the `create_chart` tool. Infer the best `chart_type` from the user's request.
4.  **For Statistical Summaries:** Use the `get_statistical_summary` tool for requests like "give me a full breakdown."
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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # --- Create a prompt template that includes our system prompt ---
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
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
def log_progress(message, elapsed_time=None, progress=None):
    """
    Log a progress message with timestamp, elapsed time, and progress bar if provided.
    
    Args:
        message (str): The progress message to display
        elapsed_time (float, optional): Elapsed time in seconds
        progress (float, optional): Progress value between 0.0 and 1.0
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
            filled_width = int(bar_width * progress)
            bar = '█' * filled_width + '░' * (bar_width - filled_width)
            
            # Add percentage
            percentage = progress * 100
            progress_bar = f" [{bar}] {percentage:.1f}%"
        except Exception:
            # In case of any errors with the progress bar, ignore it
            pass
    
    # Print the final message with all components
    if time_str and progress_bar:
        print(f"[{timestamp}] {message}{progress_bar} {time_str}")
    elif time_str:
        print(f"[{timestamp}] {message} {time_str}")
    elif progress_bar:
        print(f"[{timestamp}] {message}{progress_bar}")
    else:
        print(f"[{timestamp}] {message}")
    
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
            "chart": 8,
            "llm": 12
        }
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown tool")
        tool_id = kwargs.get("run_id", str(time.time()))
        
        # Record the start time for this tool execution
        self.start_times[tool_id] = time.time()
        self.current_tool = tool_id
        self.running_tools[tool_id] = tool_name
        
        # Output a progress message for long-running tools
        if "sql" in tool_name.lower():
            log_progress(f"Executing SQL query: {input_str[:80]}{'...' if len(input_str) > 80 else ''}")
            # Start a thread to periodically update progress for long-running SQL queries
            self._start_progress_thread(tool_id, "SQL query in progress", "sql")
            
        elif "python" in tool_name.lower():
            log_progress(f"Running Python analysis: {input_str[:80]}{'...' if len(input_str) > 80 else ''}")
            # Start a thread for Python operations
            self._start_progress_thread(tool_id, "Python analysis in progress", "python")
            
        elif "chart" in tool_name.lower() or "plot" in tool_name.lower():
            log_progress(f"Generating visualization")
            # Start a thread for chart creation
            self._start_progress_thread(tool_id, "Chart generation in progress", "chart")
            
        # You could add more specific handlers for other tool types here
        
    def on_tool_end(self, output, **kwargs):
        tool_id = kwargs.get("run_id", self.current_tool)
        tool_name = self.running_tools.get(tool_id, "unknown tool")
        
        if tool_id in self.start_times:
            # Calculate the elapsed time
            elapsed_time = time.time() - self.start_times[tool_id]
            
            # Stop the progress thread if it's running
            self._stop_progress_thread(tool_id)
            
            # Output completion message with elapsed time and 100% progress
            if "sql" in tool_name.lower():
                log_progress("SQL query completed", elapsed_time, 1.0)
            elif "python" in tool_name.lower():
                log_progress("Python analysis completed", elapsed_time, 1.0)
            elif "chart" in tool_name.lower() or "plot" in tool_name.lower():
                log_progress("Chart generation completed", elapsed_time, 1.0)
            
            # Clean up
            del self.start_times[tool_id]
            if tool_id in self.running_tools:
                del self.running_tools[tool_id]
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Record start time for LLM calls
        llm_id = kwargs.get("run_id", str(time.time()))
        self.start_times[llm_id] = time.time()
        log_progress("Processing with AI model...")
        # Start a thread for LLM processing
        self._start_progress_thread(llm_id, "AI processing in progress", "llm")
    
    def on_llm_end(self, response, **kwargs):
        llm_id = kwargs.get("run_id", None)
        if llm_id and llm_id in self.start_times:
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

# --- Create a set of tools based on available dependencies ---
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
if LANGCHAIN_AVAILABLE and DATABASE_AVAILABLE:
    try:
        # Create SQL query tool - we need to import QuerySQLDatabaseTool here to avoid reference errors
        from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
        sql_query_tool = QuerySQLDatabaseTool(db=db)
        all_tools.append(sql_query_tool)
        print("Added SQL query tool.")
        
        # Add statistical summary tool
        summary_tool = Tool(
            name="get_statistical_summary",
            func=get_statistical_summary,
            description="""Use this tool when the user asks for a 'full summary', 'statistics', 
            'breakdown', or 'statistical description' of numerical data.
            It takes a single SQL query as input and returns a complete statistical summary."""
        )
        all_tools.append(summary_tool)
        print("Added statistical summary tool.")
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

# Create the agent only if we have the minimum required dependencies
agent_executor = None
agent_available = False

# Check if we can create the agent (need LangChain and at least a database or REPL)
if LANGCHAIN_AVAILABLE and (DATABASE_AVAILABLE or pd is not None):
    try:
        # --- Create the SQL Agent with our custom prompt ---
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,  # Pass the database directly
            agent_type="openai-tools",
            prompt=prompt, # Pass the custom prompt here
            verbose=False,  # Hide internal thoughts
            extra_tools=all_tools,  # Add our custom tools as extra_tools
            callbacks=[ProgressCallback()]
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

# Initialize chat history
chat_history = []

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
            "input": user_question,
            "chat_history": chat_history
        })
        
        spinner.stop()  # Stop the animation
        
        # Calculate total elapsed time
        total_elapsed = time.time() - start_time
        
        # Add completion message with total time and 100% progress
        log_progress("Analysis complete", total_elapsed, 1.0)
        
        # Update chat history
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response["output"]))
        
        # Display the result
        print("\nAgent's Answer:")
        print(response["output"])
        
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

