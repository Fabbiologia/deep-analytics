import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# LangChain specific imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # --- NEW: Import prompt tools ---

# Load environment variables from the .env file
load_dotenv()

# --- NEW: Define a powerful, custom prompt for our agent ---
SYSTEM_PROMPT = """
You are a highly intelligent ecological data analyst AI for the Gulf of California.
You have access to a SQL database with several tables. Your primary goal is to answer user questions about this data.

**Key Instructions:**
1.  The main data tables are `ltem_historical_database` and `ltem_spp_productivity`. These tables contain most of the observational data like `Species`, `Size`, and `Biomass`. Prioritize using these tables.
2.  When a user asks a follow-up question (e.g., "and its average biomass?"), you **MUST** look at the previous conversation turn. You must reuse any filters from the previous turn, such as the `Species` name, `Location`, or `Year`.
3.  **Example Memory Usage:** If the previous question was "What is the average size of species X?" and the new question is "What about its biomass?", you **MUST** generate a query like `SELECT AVG(Biomass) FROM your_table WHERE Species = 'X'`. Do NOT forget the `WHERE` clause.
4.  Think step-by-step. First, understand the user's question. Then, identify the correct table and columns. Formulate the SQL query. Finally, execute it and provide the answer.
5.  The `ltem_species_size` table is a general lookup table and is likely NOT what you need for specific species metrics. Query the historical or productivity tables directly.
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

# --- Create the SQL Agent with our custom prompt ---
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    prompt=prompt, # Pass the custom prompt here
    verbose=True
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

