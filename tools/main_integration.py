#!/usr/bin/env python3
"""
Main Integration Module - Phase 1 Implementation
Shows how to integrate the Query Engine and Context-Aware Assistant with existing main.py
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional

# Import the newly created modules
try:
    from query_engine import QueryEngine
    from context_aware_assistant import ContextAwareAssistant
    PHASE1_COMPONENTS_AVAILABLE = True
    print("Phase 1 components loaded successfully.")
except ImportError as e:
    PHASE1_COMPONENTS_AVAILABLE = False
    print(f"Warning: Could not load Phase 1 components: {e}")

# Try importing required dependencies
try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. Data analysis capabilities will be limited.")
    pd = None

# Try to import dotenv for environment variables
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
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
    print("LangChain dependencies loaded successfully.")
except ImportError as e:
    print(f"WARNING: Missing LangChain dependencies: {e}")

# Try importing your existing modules
try:
    from sql_formatter import format_sql_for_humans, translate_sql_to_english
except ImportError:
    print("Warning: sql_formatter module not found.")
    def format_sql_for_humans(sql): return sql
    def translate_sql_to_english(sql): return f"SQL Query: {sql}"
    
try:
    from data_validator import DataValidator
except ImportError:
    print("Warning: data_validator module not found.")
    DataValidator = None

try:
    from statistical_analysis import (
        perform_ttest, perform_anova, perform_correlation_analysis,
        perform_regression_analysis, check_normality, format_statistical_results
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

# Database dependencies
try:
    from sqlalchemy import create_engine
    DATABASE_AVAILABLE = True
except ImportError:
    print("Warning: sqlalchemy not installed. Database capabilities will be disabled.")
    DATABASE_AVAILABLE = False

# --- Database schema description ---
# Use the same schema description from your main.py
SCHEMA_DESCRIPTION = """ 
This is a comprehensive dataset from a Long-Term Ecological Monitoring (LTEM) program for coral reef ecosystems. 

## `ltem_optimized_regions` Table 
This table is an optimized subset of the historical observational log containing only data from Loreto, La Paz, and Cabo Pulmo regions. Each row represents a specific survey entry where organisms of the same size were recorded within a transect at a certain depth (Depth and Depth2). 
- **Purpose:** To log what species were seen, where, when, and in what quantity/size. 
- **Regions Filter:** Only contains data from Loreto, La Paz, and Cabo Pulmo.
- **Optimizations Applied:** 
  - Removed columns: `bleaching_coverage`, `Functional_groups`, and `Protection_status` 
  - Reef names formatted: underscores replaced with spaces, words capitalized (e.g., "LOS_ISLOTES" → "Los Islotes")
- **Key Columns:** 
- `Label`: 'INV' for invertebrates, 'PEC' for fish.
- `Taxa1`, `Taxa2`, `Phylum`, `Species`: Taxonomic information.
- `Year`, `Month`, `Day`: Date of the observation. 
- `Region`, `Reef`: Character information of the Region and the specific Reef where the survey occurred. 
- `Habitat`: Type of substrate surveyed. 
- `Longitude`, `Latitude`: Precise location data in WGS84 degrees. 
- `Quantity`: The abundance, i.e., how many organisms of that size were counted within a transect. When abundance is called, you should divide this for the Area to get org/m2. 
- `Area`: Area surveyed, in square meters. Very important for density calculations!
- `Biomass`: Weight in grams. This is NULL/NA for invertebrates (INV).
- `MPA`: Whether the survey was inside a Marine Protected Area (Yes/No).
- `TrophicLevel`: Trophic level on a 1-5 scale with 1 = primary producers, 5 = apex predators. 
- `somatic_growth`, `mortality`, `Prod`: Direct components of population dynamics. `Prod` is the final calculated productivity, a key metric for ecosystem function. 
"""

# --- Helper Functions for Integration ---

def setup_phase1_components():
    """
    Set up the Phase 1 components (Query Engine and Context-Aware Assistant)
    
    Returns:
        Dictionary with components and their availability status
    """
    components = {
        'query_engine': None,
        'assistant': None,
        'memory': None,
        'available': False
    }
    
    # Check if required components are available
    if not PHASE1_COMPONENTS_AVAILABLE or not LANGCHAIN_AVAILABLE:
        print("Phase 1 components or LangChain not available. Cannot set up Phase 1.")
        return components
        
    try:
        # Set up database connection if available
        db_connection = None
        if DATABASE_AVAILABLE:
            # Try to connect to the database using the environment variable
            DATABASE_URL = os.getenv("DATABASE_URL")
            if DATABASE_URL:
                db_connection = create_engine(DATABASE_URL)
                print("Database connection established.")
            else:
                print("DATABASE_URL environment variable not set.")
                
        # Create LLM client (gpt-5-mini requires default temperature only)
        llm = ChatOpenAI(model="gpt-5-mini")
        
        # Create shared memory manager
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create analysis tools dictionary
        analysis_tools = {}
        if ADVANCED_STATS_AVAILABLE:
            analysis_tools = {
                "t_test": perform_ttest,
                "anova": perform_anova,
                "correlation": perform_correlation_analysis,
                "regression": perform_regression_analysis,
                "normality_check": check_normality
            }
            
        # Initialize QueryEngine
        query_engine = QueryEngine(
            db_connection=db_connection,
            schema_info=SCHEMA_DESCRIPTION,
            llm_client=llm
        )
        
        # Initialize ContextAwareAssistant
        assistant = ContextAwareAssistant(
            llm_client=llm,
            memory_manager=memory,
            analysis_tools=analysis_tools
        )
        
        # Update components dictionary
        components['query_engine'] = query_engine
        components['assistant'] = assistant
        components['memory'] = memory
        components['available'] = True
        
        print("Phase 1 components initialized successfully.")
        
        return components
        
    except Exception as e:
        print(f"Error setting up Phase 1 components: {e}")
        return components

def process_query_with_phase1(query: str, components: Dict[str, Any]):
    """
    Process a user query using the Phase 1 components
    
    Args:
        query: User's natural language query
        components: Dictionary with Phase 1 components
        
    Returns:
        Dictionary with query results and analysis suggestions
    """
    if not components['available']:
        return {
            'error': 'Phase 1 components not available',
            'raw_query': query
        }
        
    try:
        # Process the query with QueryEngine
        query_results = components['query_engine'].process_query(query)
        
        # Generate analysis suggestions based on query and results
        data = query_results.get('results', {}).get('dataframe')
        suggestions = components['assistant'].suggest_analysis(query, data)
        
        # Assess data quality if data is available
        data_quality = {}
        if data is not None:
            data_quality = components['assistant'].assess_data_quality(data)
            
        return {
            'query_results': query_results,
            'analysis_suggestions': suggestions,
            'data_quality': data_quality,
            'success': True
        }
        
    except Exception as e:
        print(f"Error processing query with Phase 1 components: {e}")
        return {
            'error': str(e),
            'raw_query': query,
            'success': False
        }


# --- Integration Demo ---

def run_integration_demo():
    """Run a simple demo of the Phase 1 integration"""
    print("\n=== Phase 1 Integration Demo ===")
    
    # Set up Phase 1 components
    print("\nSetting up Phase 1 components...")
    components = setup_phase1_components()
    
    if not components['available']:
        print("Cannot run demo - Phase 1 components not available.")
        return
    
    print("\nPhase 1 components ready!")
    print("- Query Engine: OK")
    print("- Context-Aware Assistant: OK")
    
    # Process sample queries
    sample_queries = [
        "What is the average biomass by region in 2018?",
        "Show me the species diversity in Cabo Pulmo over time",
        "Compare trophic levels between protected and non-protected areas"
    ]
    
    for i, query in enumerate(sample_queries):
        print(f"\n\n--- Sample Query {i+1} ---")
        print(f"Query: {query}")
        
        # Process the query
        print("\nProcessing query...")
        start_time = time.time()
        result = process_query_with_phase1(query, components)
        elapsed_time = time.time() - start_time
        
        # Display results
        print(f"\nQuery processed in {elapsed_time:.2f} seconds")
        
        if result.get('success', False):
            # Show query results
            query_results = result['query_results']
            if 'sql_query' in query_results and query_results['sql_query']:
                print("\nGenerated SQL:")
                print(query_results['sql_query'])
                
            # Show row count
            if 'results' in query_results and 'row_count' in query_results['results']:
                row_count = query_results['results']['row_count']
                print(f"\nFound {row_count} records")
                
                # Show first few records if available
                if row_count > 0 and 'records' in query_results['results']:
                    records = query_results['results']['records']
                    print("\nFirst record:")
                    print(records[0])
            
            # Show analysis suggestions
            suggestions = result.get('analysis_suggestions', {}).get('suggestions', [])
            if suggestions:
                print("\nAnalysis Suggestions:")
                for i, suggestion in enumerate(suggestions):
                    print(f"{i+1}. {suggestion.get('method_name')}")
                    print(f"   {suggestion.get('description')}")
                    print(f"   Rationale: {suggestion.get('rationale', 'N/A')}")
            
            # Show data quality assessment
            data_quality = result.get('data_quality', {})
            if data_quality:
                print("\nData Quality Assessment:")
                print(f"Score: {data_quality.get('quality_score', 'N/A')}/10")
                print(f"Issues: {', '.join(data_quality.get('major_issues', ['None']))}")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
    
    print("\n\n=== Demo Complete ===")
    print("To integrate Phase 1 components into your main application:")
    print("1. Copy the setup_phase1_components() function to main.py")
    print("2. Copy the process_query_with_phase1() function to main.py")
    print("3. Modify your agent_executor to use these functions for processing queries")
    print("4. See detailed instructions in the integration guide below")


# --- Integration Guide ---

INTEGRATION_GUIDE = """
# Phase 1 Integration Guide for main.py

Follow these steps to integrate the Query Engine and Context-Aware Assistant into your main application:

## Step 1: Add Import Statements
Add these imports at the top of your main.py:
```python
# Import Phase 1 components
try:
    from query_engine import QueryEngine
    from context_aware_assistant import ContextAwareAssistant
    PHASE1_COMPONENTS_AVAILABLE = True
    print("Phase 1 components loaded successfully.")
except ImportError as e:
    PHASE1_COMPONENTS_AVAILABLE = False
    print(f"Warning: Could not load Phase 1 components: {e}")
```

## Step 2: Add Component Setup Function
Add the setup_phase1_components() function to your main.py.

## Step 3: Add Query Processing Function
Add the process_query_with_phase1() function to your main.py.

## Step 4: Modify Your Agent Creation Code
Update your agent creation code to use the Phase 1 components:

```python
# Set up Phase 1 components
phase1_components = setup_phase1_components()

# Create tools list with new components
tools = []

# Add database tool
if db is not None:
    tools.append(
        Tool(
            name="query_database",
            description="Useful for querying the database directly with SQL",
            func=lambda q: str(db.run(q))
        )
    )
    
# Add Phase 1 query processing tool if available
if phase1_components['available']:
    tools.append(
        Tool(
            name="process_query",
            description="Process natural language queries about ecological data",
            func=lambda q: json.dumps(process_query_with_phase1(q, phase1_components), indent=2)
        )
    )
    
# Add your other existing tools...
```

## Step 5: Update Your Agent's System Prompt
Update your system prompt to include information about the new capabilities:

```python
SYSTEM_PROMPT = \"\"\"
You are an ecological data analyst for Gulf of California coral reef monitoring data.

**Key Rules:**
1. Use proper scientific terminology and data-driven analysis
2. First analyze the query to determine the best approach:
   - For natural language data queries, use `process_query`
   - For complex analyses or when you need context, use the context-aware assistant
   - For direct SQL, use `query_database`
3. For visualizations: use `create_chart` with appropriate parameters
4. For reports: use `generate_pdf_report`
5. Report "Average Density" not "Total" values
6. Check statistical assumptions before selecting tests
\"\"\"
```

## Step 6: Update Your Agent's Response Logic
In your main interaction loop, update how you handle agent responses:

```python
# Inside your main interaction loop:
try:
    spinner.start()  # Start the animation
    
    # Add initial progress message with timestamp and progress bar
    start_time = time.time()
    log_progress("Starting analysis...")
    
    # Process the query
    if PHASE1_COMPONENTS_AVAILABLE and phase1_components['available']:
        # Try to use Phase 1 components first
        phase1_result = process_query_with_phase1(user_question, phase1_components)
        
        if phase1_result.get('success', False):
            # Phase 1 processing succeeded
            response = {
                "output": format_phase1_response(phase1_result, user_question)
            }
        else:
            # Fall back to regular agent
            response = agent_executor.invoke({
                "input": user_question
            })
    else:
        # Use regular agent if Phase 1 not available
        response = agent_executor.invoke({
            "input": user_question
        })
    
    # Rest of your existing code...
```

## Step 7: Add Response Formatting Function
Add a function to format Phase 1 responses:

```python
def format_phase1_response(result, query):
    \"\"\"Format Phase 1 processing results into a user-friendly response\"\"\"
    response_parts = []
    
    # Add header
    response_parts.append(f"I've analyzed your question: \"{query}\"")
    
    # Add query results
    query_results = result['query_results']
    if 'results' in query_results and 'row_count' in query_results['results']:
        row_count = query_results['results']['row_count']
        response_parts.append(f"\n## Data Results\nFound {row_count} records matching your query.")
        
        # Add SQL explanation if available
        if 'sql_explanation' in query_results['results']:
            response_parts.append(f"\nQuery translated to: {query_results['results']['sql_explanation']}")
            
        # Add stats if available
        if 'stats' in query_results['results']:
            stats = query_results['results']['stats']
            if stats:
                response_parts.append("\n### Key Statistics")
                for col, stat in stats.items():
                    response_parts.append(f"- {col}: mean={stat['mean']:.2f}, min={stat['min']:.2f}, max={stat['max']:.2f}")
    
    # Add analysis suggestions
    suggestions = result.get('analysis_suggestions', {}).get('suggestions', [])
    if suggestions:
        response_parts.append("\n## Suggested Analyses")
        for i, suggestion in enumerate(suggestions[:3]):  # Limit to top 3
            response_parts.append(f"### {i+1}. {suggestion.get('method_name')}")
            response_parts.append(f"{suggestion.get('description')}")
            if 'rationale' in suggestion:
                response_parts.append(f"*Rationale*: {suggestion.get('rationale')}")
    
    # Add data quality notes
    data_quality = result.get('data_quality', {})
    if data_quality:
        response_parts.append(f"\n## Data Quality Assessment")
        response_parts.append(f"Quality Score: {data_quality.get('quality_score', 'N/A')}/10")
        
        if 'major_issues' in data_quality and data_quality['major_issues']:
            response_parts.append("Potential Issues:")
            for issue in data_quality['major_issues']:
                response_parts.append(f"- {issue}")
                
        if 'recommendations' in data_quality and data_quality['recommendations']:
            response_parts.append("Recommendations:")
            for rec in data_quality['recommendations']:
                response_parts.append(f"- {rec}")
    
    return "\\n".join(response_parts)
```

These changes will integrate the Phase 1 components into your existing application
while maintaining compatibility with your current functionality.
"""


# --- Main Execution ---

if __name__ == "__main__":
    # Run the integration demo
    run_integration_demo()
    
    # Print the integration guide
    print("\n\n" + INTEGRATION_GUIDE)
