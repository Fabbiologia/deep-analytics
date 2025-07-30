import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
import threading
import sys

# LangChain specific imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent  # Updated import path
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool  # Corrected tool name
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool  # Import Tool for creating custom tools

# Load environment variables from the .env file
load_dotenv()

# --- Spinner animation class for better UX ---
class Spinner:
    def __init__(self, message="Thinking..."):
        self.spinner_cycle = ['-', '\\', '|', '/']
        self.message = message
        self.stop_running = False
        self.spin_thread = None

    def start(self):
        self.spin_thread = threading.Thread(target=self.init_spin)
        self.spin_thread.start()

    def stop(self):
        self.stop_running = True
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
1.  The main data tables are `ltem_historical_database` and `ltem_spp_productivity`. Prioritize using these tables for observational data.
2.  When a user asks a follow-up question, you **MUST** look at the previous conversation turn to maintain context (e.g., for `Species`, `Location`, or `Year`).
3.  **Think step-by-step!** For any complex request, formulate a plan first before executing it. Decide which tools you need and in what order.

--- TOOL USAGE RULES ---

1.  **For Simple Queries:** If the user asks a direct question that can be answered with a single SQL query, use the `query_sql_database` tool and provide the answer.

2.  **For Advanced Analysis:** If a question requires complex calculations (like correlations, advanced statistics, or data transformations difficult for SQL), you **MUST** use a two-step process:
    * **Step 1:** Use the `query_sql_database` tool to fetch the necessary raw data.
    * **Step 2:** Use the `python_repl_tool` to perform the analysis on the data. You **must** `print()` the final result of your Python calculation.

3.  **For Visualizations:** Choose the most appropriate visualization tool based on the task:
    a. **create_bar_chart**: For comparing values across categories (e.g., "Show me average biomass by species")
    b. **create_scatter_plot**: For exploring relationships between two numerical variables (e.g., "Plot size vs. biomass for triggerfish")
    c. **create_box_plot**: For showing distributions across categories (e.g., "Show size distributions by location")
    
    Each tool requires a direct SQL query. Formulate the correct query to get the data needed for the plot.
    Do not try to describe the data in text. Your primary goal is to generate the image file.

**Example Complex Plan:**
*User Question:* "Can you show me a chart of the top 5 most observed species and also tell me the overall average size for all species in the database?"

*Your Internal Plan:*
1.  This is a multi-part request. I need to generate a chart and perform a separate calculation.
2.  First, I will use the `create_bar_chart` tool with a SQL query to find the top 5 species by observation count to generate the image.
3.  Second, I will use the `query_sql_database` tool with a separate SQL query (`SELECT AVG(Size) FROM ltem_historical_database`) to get the overall average size.
4.  Finally, I will combine these results into one comprehensive answer for the user.
"""

# --- Database Connection ---
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL environment variable not set.")

engine = create_engine(db_url)
db = SQLDatabase(engine=engine)

# --- LLM Initialization ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- NEW: Create a prompt template that includes our system prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Define plotting functions ---

# Bar Chart function
def create_bar_chart(query: str, title: str, filename: str = "bar_chart.png") -> str:
    """
    Executes a SQL query, creates a bar chart from the results,
    and saves it to a file. The query must return exactly two columns.
    The first column will be the X-axis and the second will be the Y-axis.

    Args:
        query (str): The SQL query to execute to get the data.
        title (str): The title for the chart.
        filename (str): The name of the file to save the chart as.

    Returns:
        str: A confirmation message with the filename.
    """
    try:
        # Use the existing SQLDatabase connection to run the query
        df = pd.read_sql(query, engine)

        if len(df.columns) != 2:
            return "Error: The SQL query for plotting must return exactly two columns."

        x_col, y_col = df.columns[0], df.columns[1]

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(filename)
        plt.close() # Close the plot to free up memory

        return f"Success! Bar chart saved as '{filename}'"
    except Exception as e:
        return f"An error occurred while creating the chart: {e}"

# Scatter Plot function
def create_scatter_plot(query: str, title: str, filename: str = "scatter_plot.png") -> str:
    """
    Executes a SQL query, creates a scatter plot from the results, and saves it to a file.
    The query must return exactly two numerical columns for the X and Y axes.
    Best used for exploring relationships between two continuous variables.

    Args:
        query (str): The SQL query to execute to get the data.
        title (str): The title for the chart.
        filename (str): The name of the file to save the chart as.

    Returns:
        str: A confirmation message with the filename.
    """
    try:
        # Use the existing SQLDatabase connection to run the query
        df = pd.read_sql(query, engine)

        if len(df.columns) != 2:
            return "Error: The SQL query for scatter plotting must return exactly two columns."

        x_col, y_col = df.columns[0], df.columns[1]

        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
            return "Error: Both columns for scatter plotting must contain numerical data."

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        
        # Add regression line if there are enough data points
        if len(df) > 2:
            sns.regplot(data=df, x=x_col, y=y_col, scatter=False, ci=None, line_kws={"color": "red"})

        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()

        plt.savefig(filename)
        plt.close() # Close the plot to free up memory

        return f"Success! Scatter plot saved as '{filename}'"
    except Exception as e:
        return f"An error occurred while creating the scatter plot: {e}"

# Box Plot function
def create_box_plot(query: str, title: str, filename: str = "box_plot.png") -> str:
    """
    Executes a SQL query, creates a box plot from the results, and saves it to a file.
    The query must return at least two columns: a categorical column for groups and 
    a numerical column for values.
    Best used for showing the distribution of values across different categories.

    Args:
        query (str): The SQL query to execute to get the data.
        title (str): The title for the chart.
        filename (str): The name of the file to save the chart as.

    Returns:
        str: A confirmation message with the filename.
    """
    try:
        # Use the existing SQLDatabase connection to run the query
        df = pd.read_sql(query, engine)

        if len(df.columns) < 2:
            return "Error: The SQL query for box plotting must return at least two columns."

        x_col, y_col = df.columns[0], df.columns[1]

        # Check if y-column is numeric
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return "Error: The second column (y-axis) must contain numerical data for box plotting."

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(filename)
        plt.close() # Close the plot to free up memory

        return f"Success! Box plot saved as '{filename}'"
    except Exception as e:
        return f"An error occurred while creating the box plot: {e}"

# --- Create tool for SQL queries that returns structured data ---
sql_query_tool = QuerySQLDatabaseTool(db=db)  # Corrected class name

# --- Create Python REPL tool for advanced analysis ---
python_repl_tool = PythonREPLTool()

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
                    title_match = re.search(r'title["\s]*:["\s]*(.*?)["\s]*,', input_str) or re.search(r'title["\s]*:["\s]*(.*?)["\s]*\}', input_str)
                    filename_match = re.search(r'filename["\s]*:["\s]*(.*?)["\s]*\}', input_str)
                    
                    params = {}
                    if query_match:
                        params['query'] = query_match.group(1).strip('" ')
                    if title_match:
                        params['title'] = title_match.group(1).strip('" ')
                    if filename_match:
                        params['filename'] = filename_match.group(1).strip('" ')
            
            # Extract parameters
            query = params.get('query')
            title = params.get('title')
            filename = params.get('filename')
            
            # Validate required parameters
            if not query:
                return "Error: 'query' parameter is required"
            if not title:
                return "Error: 'title' parameter is required"
                
            # Call the original function
            if filename:
                return plotting_func(query, title, filename)
            else:
                return plotting_func(query, title)
        except Exception as e:
            return f"Error parsing input for visualization: {e}. Please provide input in the format: {{\"query\": \"SELECT col1, col2 FROM table\", \"title\": \"Chart Title\", \"filename\": \"output.png\"}}"
    return wrapper

# --- Create visualization tools ---
bar_chart_tool = Tool(
    name="create_bar_chart",
    func=plotting_tool_wrapper(create_bar_chart),
    description="""
    Use this to create a BAR CHART and save it as a file. 
    It takes a SQL query that MUST return exactly two columns for the x and y axes, a title for the chart, and a filename.
    
    You MUST provide input as a JSON object with these fields:
    - query: SQL query string that returns exactly 2 columns (required)  
    - title: Title for the chart (required)
    - filename: Name of the output file (optional, default: bar_chart.png)
    
    Example: {"query": "SELECT Species, AVG(Biomass) FROM ltem_historical_database GROUP BY Species LIMIT 5", "title": "Average Biomass of Top 5 Species", "filename": "biomass_chart.png"}
    """
)

scatter_plot_tool = Tool(
    name="create_scatter_plot",
    func=plotting_tool_wrapper(create_scatter_plot),
    description="""
    Use this to create a SCATTER PLOT and save it as a file.
    It takes a SQL query that MUST return exactly two NUMERICAL columns for the x and y axes, a title for the chart, and a filename.
    A regression line will automatically be added if there are enough data points.
    
    You MUST provide input as a JSON object with these fields:
    - query: SQL query string that returns exactly 2 NUMERICAL columns (required)  
    - title: Title for the chart (required)
    - filename: Name of the output file (optional, default: scatter_plot.png)
    
    Example: {"query": "SELECT Size, Biomass FROM ltem_historical_database WHERE Species='Balistes polylepis' LIMIT 50", "title": "Size vs Biomass for Finescale Triggerfish", "filename": "scatter_plot.png"}
    """
)

box_plot_tool = Tool(
    name="create_box_plot",
    func=plotting_tool_wrapper(create_box_plot),
    description="""
    Use this to create a BOX PLOT and save it as a file.
    It takes a SQL query that MUST return at least two columns: a categorical column (x-axis) and a numerical column (y-axis).
    Box plots show the distribution of values across different categories, including median, quartiles, and outliers.
    
    You MUST provide input as a JSON object with these fields:
    - query: SQL query string that returns a categorical column and a numerical column (required)  
    - title: Title for the chart (required)
    - filename: Name of the output file (optional, default: box_plot.png)
    
    Example: {"query": "SELECT Location, Size FROM ltem_historical_database WHERE Species='Scarus ghobban' LIMIT 100", "title": "Size Distribution of Blue-barred Parrotfish by Location", "filename": "box_plot.png"}
    """
)

# --- Create the SQL Agent with our custom prompt ---
agent_executor = create_sql_agent(
    llm=llm,
    db=db,  # Pass the database directly
    agent_type="openai-tools",
    prompt=prompt, # Pass the custom prompt here
    verbose=False,  # Hide internal thoughts
    extra_tools=[python_repl_tool, bar_chart_tool, scatter_plot_tool, box_plot_tool]  # Add our custom tools as extra_tools
)

# --- Chat History and Interactive Loop ---
chat_history = []
print("--- 🐠 Ecological Monitoring Agent is ready! (With Custom Prompt) ---")
print("Ask me questions about your database. Type 'exit' to end.")

while True:
    user_question = input("\nYour Question: ")
    if user_question.lower() == 'exit':
        print("Goodbye!")
        break
    
    spinner = Spinner("🐠 Analyzing data...")  # Create a spinner instance
    try:
        spinner.start()  # Start the animation
        
        response = agent_executor.invoke({
            "input": user_question,
            "chat_history": chat_history
        })
        
        spinner.stop()  # Stop the animation
        
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response["output"]))
        
        print("\nAgent's Answer:")
        print(response["output"])
        
    except Exception as e:
        spinner.stop()  # Ensure spinner stops on error
        print(f"An error occurred: {e}")

