import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

# --- Define a powerful, custom prompt for our agent ---
SYSTEM_PROMPT = """
You are a highly intelligent ecological data analyst AI for the Gulf of California.
You have access to a SQL database with several tables. Your primary goal is to answer user questions about this data.

**Key Instructions:**
1.  The main data tables are `ltem_historical_database` and `ltem_spp_productivity`. These tables contain most of the observational data like `Species`, `Size`, and `Biomass`. Prioritize using these tables.
2.  When a user asks a follow-up question (e.g., "and its average biomass?"), you **MUST** look at the previous conversation turn. You must reuse any filters from the previous turn, such as the `Species` name, `Location`, or `Year`.
3.  **Example Memory Usage:** If the previous question was "What is the average size of species X?" and the new question is "What about its biomass?", you **MUST** generate a query like `SELECT AVG(Biomass) FROM your_table WHERE Species = 'X'`. Do NOT forget the `WHERE` clause.
4.  Think step-by-step. First, understand the user's question. Then, identify the correct table and columns. Formulate the SQL query. Finally, execute it and provide the answer.
5.  The `ltem_species_size` table is a general lookup table and is likely NOT what you need for specific species metrics. Query the historical or productivity tables directly.

**Advanced Analysis with Python:**
6.  You also have a Python code interpreter (python_repl_tool) for advanced analysis. If a question requires complex calculations (like correlations, trends, or advanced statistics) or data manipulation that is difficult in SQL, follow these steps:
   a. First, use the query_sql_database tool to fetch the necessary raw data.
   b. Then, use the python_repl_tool to perform the analysis on the retrieved data.
   c. When using Python, you must print() the final answer or result so it can be returned to the user.

**Creating Visualizations:**
7.  If the user asks for a visualization like a "bar chart," "graph," or "plot," you MUST use the create_bar_chart tool. Do not try to describe the data in text; generate the image file. Provide the tool with a complete SQL query to get the data, a descriptive title, and a filename.
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

# --- Define the custom plotting function ---
def create_bar_chart(query: str, title: str, filename: str = "chart.png") -> str:
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

        return f"Success! Chart saved as '{filename}'"
    except Exception as e:
        return f"An error occurred while creating the chart: {e}"

# --- Create tool for SQL queries that returns structured data ---
sql_query_tool = QuerySQLDatabaseTool(db=db)  # Corrected class name

# --- Create Python REPL tool for advanced analysis ---
python_repl_tool = PythonREPLTool()

# --- Create a wrapper function for the charting tool to handle different input formats ---
def create_bar_chart_wrapper(input_str):
    """
    Wrapper for the create_bar_chart function to handle different input formats.
    
    Args:
        input_str: String or dictionary with query, title, and optional filename.
            
    Returns:
        Result from create_bar_chart function
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
        filename = params.get('filename', 'chart.png')
        
        # Validate required parameters
        if not query:
            return "Error: 'query' parameter is required"
        if not title:
            return "Error: 'title' parameter is required"
            
        # Call the original function
        return create_bar_chart(query, title, filename)
    except Exception as e:
        return f"Error parsing input for chart creation: {e}. Please provide input in the format: {{\"query\": \"SELECT col1, col2 FROM table\", \"title\": \"Chart Title\", \"filename\": \"output.png\"}}"

# --- Create visualization tool ---
charting_tool = Tool(
    name="create_bar_chart",
    func=create_bar_chart_wrapper,
    description="""
    Use this to create a bar chart and save it as a file. 
    It takes a SQL query that MUST return exactly two columns for the x and y axes, a title for the chart, and a filename.
    
    You MUST provide input as a JSON object with these fields:
    - query: SQL query string that returns exactly 2 columns (required)  
    - title: Title for the chart (required)
    - filename: Name of the output file (optional, default: chart.png)
    
    Example: {"query": "SELECT Species, AVG(Biomass) FROM ltem_historical_database GROUP BY Species LIMIT 5", "title": "Average Biomass of Top 5 Species", "filename": "biomass_chart.png"}
    """
)

# --- Create the SQL Agent with our custom prompt ---
agent_executor = create_sql_agent(
    llm=llm,
    db=db,  # Pass the database directly
    agent_type="openai-tools",
    prompt=prompt, # Pass the custom prompt here
    verbose=True,
    extra_tools=[python_repl_tool, charting_tool]  # Add our custom tools as extra_tools
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
    
    try:
        response = agent_executor.invoke({
            "input": user_question,
            "chat_history": chat_history
        })
        
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response["output"]))
        
        print("\nAgent's Answer:")
        print(response["output"])
        
    except Exception as e:
        print(f"An error occurred: {e}")

