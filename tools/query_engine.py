#!/usr/bin/env python3
"""
Query Engine Module for Ecological Data Analysis
Translates natural language queries to structured database operations.
"""

import os
import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union

# Import conditionally to handle missing dependencies gracefully
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. QueryEngine will have limited functionality.")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: LangChain dependencies not available: {e}")

# Import local modules if available
try:
    from sql_formatter import format_sql_for_humans, translate_sql_to_english
except ImportError:
    print("Warning: sql_formatter module not found.")
    def format_sql_for_humans(sql): return sql
    def translate_sql_to_english(sql): return f"SQL Query: {sql}"


class QueryEngine:
    """
    Natural language query engine that translates ecological research questions
    into structured database operations and returns results.
    """
    
    def __init__(self, db_connection=None, schema_info=None, llm_client=None):
        """
        Initialize the QueryEngine with database connection and LLM client.
        
        Args:
            db_connection: SQLAlchemy engine or connection string
            schema_info: String describing the database schema
            llm_client: LangChain LLM client (optional)
        """
        self.db_connection = db_connection
        self.schema_info = schema_info
        self.llm_client = llm_client
        self.query_history = []
        
        # Create database engine if connection is a string
        if isinstance(db_connection, str):
            try:
                # Add pool health checks and conservative pool sizing
                self.db_connection = create_engine(
                    db_connection,
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    pool_timeout=30,
                    pool_size=5,
                    max_overflow=10,
                )
            except Exception as e:
                print(f"Error creating database engine: {e}")
                self.db_connection = None
                
        # Create default LLM client if not provided
        if llm_client is None and LANGCHAIN_AVAILABLE:
            try:
                self.llm_client = ChatOpenAI(
                    model="gpt-5-mini"
                )
            except Exception as e:
                print(f"Error creating LLM client: {e}")
                self.llm_client = None
                
        # Create JSON parser
        if LANGCHAIN_AVAILABLE:
            self.parser = JsonOutputParser()
                
        # Set up prompt template for query parsing
        self._setup_prompt_templates()
        
    def _setup_prompt_templates(self):
        """Set up the prompt templates for various query operations"""
        if not LANGCHAIN_AVAILABLE:
            return
            
        # Template for parsing natural language to query structure
        self.parse_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a specialized ecological data analysis assistant. Your task is to parse 
            natural language queries about ecological data into structured query components.
            
            Parse the user query into a JSON structure with these fields:
            - query_type: 'sql' (requires database) or 'filter' (can work on dataframes)
            - target_variables: List of variables/metrics the user wants to analyze
            - group_by: List of variables to group by (if any)
            - filters: Dictionary of filters to apply (field: value)
            - sort_by: How to sort results (field and direction)
            - limit: Number of results to return (default: 100)
            - time_range: Time period for analysis (if specified)
            - locations: Specific geographical locations mentioned
            - aggregations: List of aggregation functions to apply
            - ecological_context: What ecological question is being asked
            
            SCHEMA INFORMATION:
            {schema_info}
            """),
            ("user", "{query}")
        ])
        
        # Template for SQL generation
        self.sql_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an SQL expert specializing in ecological data analysis. 
            Generate a precise SQL query for the requested analysis.
            
            The query should be executable on a database with this schema:
            {schema_info}
            
            Use the parsed query components to build the SQL.
            Only include the raw SQL query without any explanations or markdown formatting.
            For aggregations, use proper SQL functions like AVG, SUM, COUNT, etc.
            For density calculations, be sure to divide by Area when appropriate.
            
            IMPORTANT RULES:
            1. NEVER return non-SQL text
            2. ALWAYS include filter conditions in the WHERE clause
            3. Format numbers with proper casting to avoid division by zero
            4. Use HAVING clause only for aggregated fields
            5. Table name is 'ltem_optimized_regions'
            """),
            ("user", "Generate SQL for: {parsed_query}")
        ])
        
    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Process a natural language query and return structured results
        
        Args:
            natural_language_query: The user's ecological data question
            
        Returns:
            Dictionary with query results and metadata
        """
        # Step 1: Parse and validate the query
        parsed_query = self._parse_query(natural_language_query)
        
        # Step 2: Generate SQL or filter operations
        if parsed_query.get('query_type') == 'sql' and self.db_connection is not None:
            sql_query = self._generate_sql(parsed_query)
            results = self._execute_sql(sql_query)
        else:
            # Fall back to filter operations if no database or not SQL type
            filtered_data = self._apply_filters(parsed_query)
            results = filtered_data
            
        # Step 3: Save query to history
        self._add_to_history(natural_language_query, parsed_query, results)
        
        return {
            'results': results,
            'parsed_query': parsed_query,
            'sql_query': sql_query if 'sql_query' in locals() else None,
            'query_type': parsed_query.get('query_type', 'unknown')
        }
        
    def _parse_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into structured components
        
        Args:
            natural_language_query: The user's query string
            
        Returns:
            Dictionary with parsed query components
        """
        if not LANGCHAIN_AVAILABLE or self.llm_client is None:
            # Simple fallback parsing if LangChain not available
            return {
                'query_type': 'filter',
                'raw_query': natural_language_query,
                'target_variables': ['*'],
                'filters': {}
            }
            
        try:
            # Format the query parsing prompt
            formatted_prompt = self.parse_template.format(
                schema_info=self.schema_info,
                query=natural_language_query
            )
            
            # Get the LLM response
            response = self.llm_client.invoke(formatted_prompt)
            
            # Extract JSON from the response
            parsed_content = response.content
            
            # Clean up the response content
            # Remove markdown code blocks if present
            parsed_content = re.sub(r'```json\s*|\s*```', '', parsed_content)
            parsed_content = re.sub(r'```\s*|\s*```', '', parsed_content)
            
            # Parse the JSON
            parsed_query = json.loads(parsed_content)
            
            # Add the original query
            parsed_query['raw_query'] = natural_language_query
            
            return parsed_query
            
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Fallback for error cases
            return {
                'query_type': 'filter',
                'raw_query': natural_language_query,
                'target_variables': ['*'],
                'filters': {},
                'error': str(e)
            }
            
    def _generate_sql(self, parsed_query: Dict[str, Any]) -> str:
        """
        Generate SQL from parsed query components
        
        Args:
            parsed_query: Dictionary with parsed query components
            
        Returns:
            SQL query string
        """
        if not LANGCHAIN_AVAILABLE or self.llm_client is None:
            # Simple fallback SQL generation
            return "SELECT * FROM ltem_optimized_regions LIMIT 100"
            
        try:
            # Format the SQL generation prompt
            formatted_prompt = self.sql_template.format(
                schema_info=self.schema_info,
                parsed_query=json.dumps(parsed_query, indent=2)
            )
            
            # Get the LLM response
            response = self.llm_client.invoke(formatted_prompt)
            
            # Extract SQL from response
            sql_query = response.content.strip()
            
            # Remove markdown code blocks if present
            sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query)
            sql_query = re.sub(r'```\s*|\s*```', '', sql_query)
            
            # Format the SQL for readability
            formatted_sql = format_sql_for_humans(sql_query)
            
            return formatted_sql
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            # Fallback for error cases
            return "SELECT * FROM ltem_optimized_regions LIMIT 100"
            
    def _execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute the SQL query and return results
        
        Args:
            sql_query: SQL query string to execute
            
        Returns:
            Dictionary with query results and metadata
        """
        if self.db_connection is None:
            return {'error': 'No database connection available'}
            
        # Retry loop for transient DB errors (e.g., MySQL 2013/2006)
        attempts = 0
        max_attempts = 3
        backoff = 1.0
        last_err = None
        while attempts < max_attempts:
            try:
                # Execute the query
                if PANDAS_AVAILABLE:
                    # Use pandas to execute and get a DataFrame
                    result_df = pd.read_sql(sql_query, self.db_connection)
                    
                    # Convert to records for consistent return format
                    records = result_df.to_dict(orient='records')
                    
                    # Get basic stats for numerical columns
                    stats = {}
                    for col in result_df.select_dtypes(include=['number']).columns:
                        stats[col] = {
                            'mean': result_df[col].mean(),
                            'min': result_df[col].min(),
                            'max': result_df[col].max(),
                            'median': result_df[col].median()
                        }
                    
                    return {
                        'records': records,
                        'dataframe': result_df,
                        'row_count': len(result_df),
                        'column_count': len(result_df.columns),
                        'columns': list(result_df.columns),
                        'stats': stats,
                        'sql_query': sql_query,
                        'sql_explanation': translate_sql_to_english(sql_query)
                    }
                else:
                    # Direct execution without pandas
                    with self.db_connection.connect() as conn:
                        result = conn.execute(text(sql_query))
                        records = [dict(zip(result.keys(), row)) for row in result.fetchall()]
                        
                    return {
                        'records': records,
                        'row_count': len(records),
                        'column_count': len(records[0]) if records else 0,
                        'columns': list(records[0].keys()) if records else [],
                        'sql_query': sql_query,
                        'sql_explanation': translate_sql_to_english(sql_query)
                    }
            except OperationalError as e:
                # Check MySQL disconnect errors: 2013 (lost connection), 2006 (server gone away)
                err_code = getattr(e.orig, 'args', [None])[0] if hasattr(e, 'orig') else None
                if err_code in (2006, 2013):
                    attempts += 1
                    last_err = e
                    try:
                        # Dispose of pool to refresh stale connections
                        if hasattr(self.db_connection, 'dispose'):
                            self.db_connection.dispose()
                    except Exception:
                        pass
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    print(f"OperationalError executing SQL: {e}")
                    return {'error': str(e), 'sql_query': sql_query}
            except Exception as e:
                print(f"Error executing SQL: {e}")
                return {'error': str(e), 'sql_query': sql_query}
        # If we exhausted retries
        print(f"Error executing SQL after retries: {last_err}")
        return {'error': str(last_err) if last_err else 'Unknown error', 'sql_query': sql_query}
            
    def _apply_filters(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply filters to in-memory data when SQL execution is not available
        
        Args:
            parsed_query: Dictionary with parsed query components
            
        Returns:
            Dictionary with filtered results
        """
        # This is a placeholder for the in-memory filtering logic
        # In a real implementation, this would use pandas operations on an in-memory DataFrame
        return {
            'message': 'In-memory filtering not implemented yet',
            'parsed_query': parsed_query
        }
            
    def _add_to_history(self, natural_language_query: str, parsed_query: Dict[str, Any], 
                       results: Dict[str, Any]) -> None:
        """
        Add the query and results to the history
        
        Args:
            natural_language_query: Original query string
            parsed_query: Parsed query components
            results: Query results
        """
        # Get timestamp for the query
        timestamp = pd.Timestamp.now() if PANDAS_AVAILABLE else None
        
        # Create history entry with limited result data to avoid memory issues
        history_entry = {
            'timestamp': timestamp,
            'natural_language_query': natural_language_query,
            'parsed_query': parsed_query,
            'result_summary': {
                'row_count': results.get('row_count'),
                'column_count': results.get('column_count'),
                'columns': results.get('columns'),
                'error': results.get('error')
            }
        }
        
        # Add to history
        self.query_history.append(history_entry)
        
        # Limit history size to avoid memory issues
        if len(self.query_history) > 10:
            self.query_history = self.query_history[-10:]
            
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the query history
        
        Returns:
            List of query history entries
        """
        return self.query_history
        
    def get_last_query(self) -> Optional[Dict[str, Any]]:
        """
        Get the last query from history
        
        Returns:
            The last query history entry or None if history is empty
        """
        if not self.query_history:
            return None
        return self.query_history[-1]


# Standalone testing code
if __name__ == "__main__":
    # Example usage
    # Load environment variables if dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    
    # Define a simple schema for testing
    test_schema = """
    Table: ltem_optimized_regions
    Columns: Species, Year, Region, Reef, Quantity, Biomass, Area, Trophic_level
    """
    
    # Initialize the query engine
    query_engine = QueryEngine(database_url, test_schema)
    
    # Test with a simple query
    test_query = "What is the average biomass by region in 2018?"
    print(f"Testing query: {test_query}")
    
    results = query_engine.process_query(test_query)
    
    # Display results
    if 'error' in results:
        print(f"Error: {results['error']}")
    elif 'records' in results:
        print(f"Found {results.get('row_count', 0)} records")
        if results.get('row_count', 0) > 0:
            print("First record:", results['records'][0])
    else:
        print("Results:", results)
