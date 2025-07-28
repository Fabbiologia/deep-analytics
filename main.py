import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# LangChain specific imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from the .env file
load_dotenv()

# --- Database Connection ---
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL environment variable not set.")

engine = create_engine(db_url)
db = SQLDatabase(engine=engine)

# --- LLM Initialization ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Create the SQL Agent ---
# We remove the 'memory' argument here and will manage it manually.
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,
    handle_parsing_errors=True
)

# --- ADDED: Simple list for chat history ---
chat_history = []

# --- Interactive Chat Loop ---
print("--- 🐠 Ecological Monitoring Agent is ready! (Corrected Memory) ---")
print("Ask me questions about your database. Type 'exit' to end.")

while True:
    user_question = input("\nYour Question: ")
    if user_question.lower() == 'exit':
        print("Goodbye!")
        break
    
    try:
        # --- MODIFIED: Pass history directly into invoke ---
        # This forces the agent to consider the past conversation for every new question.
        response = agent_executor.invoke({
            "input": user_question,
            "chat_history": chat_history
        })
        
        # --- ADDED: Manually append the latest interaction to our history list ---
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response["output"]))
        
        print("\nAgent's Answer:")
        print(response["output"])
    except Exception as e:
        print(f"An error occurred: {e}")

