#!/usr/bin/env python3

import os
import time
import threading
import sys
from datetime import datetime
import re
import difflib
import unicodedata
from functools import lru_cache
from typing import Optional, Dict, List, Union, Tuple, Any

# Import our SQL formatter and data validator from tools folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
from sql_formatter import format_sql_for_humans, translate_sql_to_english
from data_validator import DataValidator

# Import statistical analysis, PDF reports, and Phase 3 modules from tools folder
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

# Import Phase 3 components
try:
    from insights_engine import InsightsEngine
    from report_generator import ReportGenerator, generate_quick_report, generate_insights_report
    PHASE3_AVAILABLE = True
    print("Phase 3 Insights Engine and Report Generator loaded successfully.")
except ImportError as e:
    print(f"Warning: Phase 3 modules not available: {e}")
    PHASE3_AVAILABLE = False

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
    
    # Try importing advanced visualization components from tools folder
    try:
        from visualization_agent import VisualizationAgent
        from visualization_factory import VisualizationFactory
        from visualization_integration import (
            integrate_visualization_system, 
            initialize_visualization_system
        )
        from natural_viz_tool import (
            create_natural_viz_tool,
            get_natural_viz_tool_description
        )
        ADVANCED_VIZ_AVAILABLE = True
        NATURAL_VIZ_AVAILABLE = True
        print("Advanced visualization components loaded successfully.")
    except ImportError as e:
        ADVANCED_VIZ_AVAILABLE = False
        NATURAL_VIZ_AVAILABLE = False
        print(f"Warning: Advanced visualization not available: {e}")
except ImportError:
    print("Warning: matplotlib/seaborn not installed. Visualization capabilities will be disabled.")
    VISUALIZATION_AVAILABLE = False

# Database dependencies
try:
    from sqlalchemy import create_engine, text
    DATABASE_AVAILABLE = True
except ImportError:
    print("Warning: sqlalchemy not installed. Database capabilities will be disabled.")
    DATABASE_AVAILABLE = False

# Optional, better fuzzy matching
try:
    from rapidfuzz import fuzz, process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

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

# Optional UI dependency for in-app visualization controls
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

# Environment variables should be loaded above in the try-except block

# --- Database schema description ---
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
- `Label`: Has two factors, "INV" and "PEC", that represent Invertebrate and Fish surveys, respectively. 
- `Taxa1`, `Taxa2`, `Phylum`, `Species`, `Family`, `Genus`: Taxonomic information of the species recorded. 
- `Year`, `Month`, `Day`: Date of the observation. 
- `Region`, `Reef`: Character information of the Region and the specific Reef where the survey occurred. 
- `Habitat`: Type of substrate surveyed. 
- `Longitude`, `Latitude`: Precise location data in WGS84 degrees. 
- `Quantity`: The abundance, i.e., how many organisms of that size were counted within a transect. When abundance is called, you should divide this for the Area to get org/m2. 
- `Size`: The size class of the organisms. The unit is cm. 
- `Biomass`: Fish biomass (not available for invertebrates), calculated using size, quantity, and growth parameters. The unit is ton/ha. 
- `MPA`: Conservation status of the location. 

- `TrophicLevelF`, `TrophicLevel`, `TrophicGroup`: Functional traits from trophic levels in factor, number, and groups. 
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

# --- Concise system prompt to prevent token overflow ---
SYSTEM_PROMPT = """
You are an AI ecological data analyst for Gulf of California coral reef monitoring.

**Core Capabilities:**
- SQL database queries and statistical analysis
- Automated insights discovery (trends, anomalies, correlations)
- Advanced visualizations and comprehensive reports
- Publication-ready PDF and HTML report generation

**Analysis Guidelines:**
1. Use scientific terminology and validate statistical significance
2. Create clear visualizations with proper labels
3. Consider temporal/spatial patterns
4. Generate automated insights when analyzing large datasets
5. Provide comprehensive reports for complex analyses
6. CRITICAL: Never divide Biomass by Area. Biomass values are already area-standardized in the dataset. Do not compute biomass density.
7. For abundance (Quantity) metrics, area-normalization (e.g., per m^2) may be applied when appropriate. This exception does not apply to Biomass.
"""

# Only try to create database connection if SQLAlchemy is available
db = None
engine = None
if DATABASE_AVAILABLE and LANGCHAIN_AVAILABLE:
    try:
        # Try to connect to the database using the environment variable
        DATABASE_URL = os.getenv("DATABASE_URL")
        if DATABASE_URL:
            # Use connection pool health checks and sane defaults for MySQL
            # - pool_pre_ping avoids stale connections
            # - pool_recycle proactively refreshes connections (seconds)
            # - pool_timeout limits waiting for a connection
            # - pool_size/max_overflow control concurrency
            engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=1800,
                pool_timeout=30,
                pool_size=5,
                max_overflow=10,
            )
            
            # Initialize advanced visualization system if available
            viz_components = None
            advanced_viz_tool_func = None
            if 'ADVANCED_VIZ_AVAILABLE' in globals() and ADVANCED_VIZ_AVAILABLE:
                try:
                    # Pass the database engine to the visualization system
                    advanced_viz_tool_func, advanced_viz_description, viz_components = integrate_visualization_system(engine)
                    print("Advanced visualization system initialized successfully.")
                except Exception as e:
                    print(f"Warning: Could not initialize advanced visualization system: {e}")
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

# --- Reference data and fuzzy matching helpers ---

def _strip_accents(s: str) -> str:
    """Remove accents/diacritics for robust matching."""
    if s is None:
        return ""
    try:
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    except Exception:
        return s

@lru_cache(maxsize=1)
def get_reference_values() -> Dict[str, List[str]]:
    """Load distinct reef and species terms from the database (cached)."""
    refs = {"reef_names": [], "species_terms": []}
    try:
        if not DATABASE_AVAILABLE or engine is None:
            return refs
        with engine.connect() as conn:
            # Reefs
            try:
                reef_rows = conn.execute(
                    text("SELECT DISTINCT Reef FROM ltem_optimized_regions WHERE Reef IS NOT NULL AND TRIM(Reef) <> ''")
                ).fetchall()
                refs["reef_names"] = sorted({str(r[0]).strip() for r in reef_rows if r and r[0]})
            except Exception as e:
                print(f"Warning: could not load reef names: {e}")
            # Species terms: robust to schema differences
            try:
                # Inspect available columns
                cols = []
                try:
                    col_rows = conn.execute(text("SHOW COLUMNS FROM ltem_monitoring_species")).fetchall()
                    cols = [str(r[0]).strip() for r in col_rows]
                except Exception:
                    # Fallback for engines that don't support SHOW COLUMNS
                    # Try a lightweight select limit 1 to infer keys from result metadata
                    try:
                        probe = conn.execute(text("SELECT * FROM ltem_monitoring_species LIMIT 1"))
                        cols = list(probe.keys())
                    except Exception:
                        cols = []

                cols_lower = {c.lower(): c for c in cols}
                terms = set()

                # Helper to add non-empty trimmed strings
                def _add_safe(values):
                    for v in values:
                        if v is not None:
                            s = str(v).strip()
                            if s:
                                terms.add(s)

                # Prefer explicit Genus/Species if present (any casing)
                if 'genus' in cols_lower and 'species' in cols_lower:
                    q = text(f"SELECT DISTINCT `{cols_lower['genus']}`, `{cols_lower['species']}` FROM ltem_monitoring_species")
                    for g, s in conn.execute(q).fetchall():
                        _add_safe([g, s])
                else:
                    # Try common alternates
                    candidates = [
                        'scientific_name', 'binomial', 'species_name', 'taxon', 'taxon_name', 'species', 'genus'
                    ]
                    present = [cols_lower[k] for k in candidates if k in cols_lower]
                    for col in present:
                        q = text(f"SELECT DISTINCT `{col}` FROM ltem_monitoring_species WHERE `{col}` IS NOT NULL AND TRIM(`{col}`) <> ''")
                        _add_safe([r[0] for r in conn.execute(q).fetchall()])

                refs["species_terms"] = sorted(terms)
            except Exception as e:
                print(f"Warning: could not load species terms: {e}")
    except Exception as e:
        print(f"Warning: reference load failed: {e}")
    return refs

def _normalize(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace, and simplify punctuation for matching."""
    s = _strip_accents(s or "")
    s = s.lower()
    # Keep letters, digits, spaces, hyphens and apostrophes; replace others with space
    s = re.sub(r"[^a-z0-9\s'\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_location_cues(text: str) -> List[str]:
    """Extract simple locality cues like 'in La Paz', 'at Cabo Pulmo'. Normalized."""
    cues = set()
    if not text:
        return []
    for m in re.finditer(r"\b(?:in|at|near|around|off)\s+([A-Za-zÀ-ÿ'-]+(?:\s+[A-Za-zÀ-ÿ'-]+)?)", text, flags=re.IGNORECASE):
        cues.add(_normalize(m.group(1)))
    # Also add common 2-gram windows to catch cases without prepositions
    words = [w for w in re.findall(r"[A-Za-zÀ-ÿ'-]+", text) if len(w) >= 3]
    for i in range(len(words) - 1):
        cues.add(_normalize(words[i] + " " + words[i+1]))
    return [c for c in cues if c]

def _find_exact_substring_matches(text: str, candidates: List[str]) -> List[str]:
    """Find normalized substring and token-contained matches."""
    t = _normalize(text)
    hits = []
    for c in candidates:
        cn = _normalize(c)
        if not cn:
            continue
        if cn in t:
            hits.append(c)
            continue
        # token containment heuristic: all tokens of candidate appear in text
        toks = [tok for tok in cn.split() if len(tok) >= 3]
        if toks and all(tok in t for tok in toks):
            hits.append(c)
    return hits

def _suggest_from_question(text: str, candidates: List[str], topn: int = 5) -> List[str]:
    """Suggest closest candidates using RapidFuzz if available, else difflib.
    Applies light locality filtering to narrow search space when possible.
    """
    if not text or not candidates:
        return []
    qn = _normalize(text)

    # Optional: filter candidates by locality cues present in the question
    cues = _extract_location_cues(text)
    cand_pool = candidates
    if cues:
        subset = []
        for c in candidates:
            cn = _normalize(c)
            if any(cue and cue in cn for cue in cues):
                subset.append(c)
        if 0 < len(subset) <= max(50, len(candidates)//2):  # only narrow if meaningfully reduces
            cand_pool = subset

    if 'RAPIDFUZZ_AVAILABLE' in globals() and RAPIDFUZZ_AVAILABLE:
        # RapidFuzz token_set_ratio is robust to word order and duplicates
        # Build a list of (candidate, normalized) to avoid recomputing in scorer
        norm_map = {c: _normalize(c) for c in cand_pool}
        def scorer(cand: str) -> int:
            return fuzz.token_set_ratio(qn, norm_map[cand])
        results = sorted(((c, scorer(c)) for c in cand_pool), key=lambda x: x[1], reverse=True)
        # Keep suggestions with reasonable confidence
        suggestions = [c for c, s in results if s >= 65][:topn]
        return suggestions
    else:
        # Fallback to difflib using grams as in previous version
        words = [w for w in re.findall(r"[A-Za-zÀ-ÿ'-]+", text) if len(w) >= 3]
        grams = set()
        for i in range(len(words)):
            for L in (1, 2, 3):
                if i + L <= len(words):
                    grams.add(" ".join(words[i:i+L]))
        scored = {}
        for cand in cand_pool:
            best = 0.0
            cn = _normalize(cand)
            for g in grams:
                score = difflib.SequenceMatcher(None, _normalize(g), cn).ratio()
                if score > best:
                    best = score
            scored[cand] = best
        suggestions = [c for c, s in sorted(scored.items(), key=lambda x: x[1], reverse=True) if s >= 0.6]
        return suggestions[:topn]

def maybe_request_clarification(user_question: str) -> Optional[str]:
    """If question mentions reefs or species and no exact matches are found, suggest alternatives."""
    q = user_question or ""
    ql = q.lower()
    intents_reef = any(k in ql for k in ["reef", "reefs", "site", "sites", "location", "locations"]) 
    intents_species = any(k in ql for k in ["species", "fish", "fishes", "taxa", "genus"]) 
    if not (intents_reef or intents_species):
        return None
    refs = get_reference_values()

    # Reefs
    if intents_reef and refs.get("reef_names"):
        reef_names = refs["reef_names"]
        # Try exact/contained first
        hits = _find_exact_substring_matches(q, reef_names)
        if not hits:
            # Better suggestions with locality-aware filtering
            sugg = _suggest_from_question(q, reef_names, topn=5)
            if sugg:
                return (
                    "I couldn't find an exact reef match. Did you mean one of these?\n- " + "\n- ".join(sugg) +
                    "\nPlease confirm the correct reef name."
                )

    # Species
    if intents_species and refs.get("species_terms"):
        terms = refs["species_terms"]
        hits = _find_exact_substring_matches(q, terms)
        if not hits:
            sugg = _suggest_from_question(q, terms, topn=5)
            if sugg:
                return (
                    "I couldn't find an exact species match. Possible alternatives:\n- " + "\n- ".join(sugg) +
                    "\nPlease confirm the correct species or genus."
                )
    return None

 

# --- LLM Initialization ---
llm = None
prompt = None

# Note: Using optimized database table (ltem_optimized_regions) with only Loreto, La Paz, and Cabo Pulmo regions
# This helps prevent token limit issues by reducing the dataset size and columns
# Optimizations: removed bleaching_coverage, Functional_groups, Protection_status columns and formatted Reef names

if LANGCHAIN_AVAILABLE:
    try:
        llm = ChatOpenAI(model="gpt-5-mini")
        
        # --- Create a prompt template that includes our system prompt ---
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            # Include chat history so the agent can remember context between turns
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
    # Escape '%' for DB-API execution as well to avoid old-style percent formatting interpretation
    db_where = where_clause.replace('%', '%%') if isinstance(where_clause, str) else where_clause
    log_progress(f"Calculating average {metric} density grouped by {group_by} with filter: {safe_where}", is_reasoning=True)
    
    # Construct SQL query for calculating average density
    # The query follows the scientific method of first summing by transect, then averaging
    transect_level_query = f"""
    SELECT 
        {group_by},
        SUM({metric}) as transect_total
    FROM 
        ltem_optimized_regions
    WHERE 
        {db_where}
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
                avg_area_query = "SELECT AVG(Area) as avg_area FROM ltem_optimized_regions WHERE Area > 0"
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
        # Normalize independent_vars input (can be str, list, or mis-specified dict)
        if isinstance(independent_vars, str):
            independent_vars = [independent_vars]
        elif isinstance(independent_vars, dict):
            # Common agent mistake: passing a dict with a key
            # Try typical keys, otherwise fail fast with a helpful error
            for key in ("independent_vars", "x", "features"):
                if key in independent_vars and isinstance(independent_vars[key], (list, str)):
                    val = independent_vars[key]
                    independent_vars = [val] if isinstance(val, str) else list(val)
                    break
            else:
                return "Error: 'independent_vars' should be a list or string. Received a dict without a usable key."

        if dependent_var not in data.columns:
            return f"Error: Dependent variable '{dependent_var}' not found in data."
        
        missing_vars = [var for var in independent_vars if var not in data.columns]
        if missing_vars:
            return f"Error: Independent variables not found: {missing_vars}"
        
        # Attempt to coerce numeric types when possible (e.g., Year)
        df = data.copy()
        try:
            if df[dependent_var].dtype.kind not in ("i", "u", "f"):
                df[dependent_var] = pd.to_numeric(df[dependent_var], errors="coerce")
        except Exception:
            pass
        for v in independent_vars:
            try:
                if df[v].dtype.kind not in ("i", "u", "f"):
                    df[v] = pd.to_numeric(df[v], errors="coerce")
            except Exception:
                pass
        df = df[[dependent_var] + independent_vars].dropna()
        if len(df) < len(independent_vars) + 2:
            return "Error: Insufficient numeric data for regression analysis after cleaning."

        results = perform_regression_analysis(df, dependent_var, independent_vars, alpha)
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
    
    Example: {"query": "SELECT Species, AVG(Biomass) FROM ltem_optimized_regions GROUP BY Species LIMIT 5", "chart_type": "bar", "title": "Average Biomass of Top 5 Species", "filename": "biomass_chart.png"}
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
if LANGCHAIN_AVAILABLE and VISUALIZATION_AVAILABLE:
    # Ensure charting tool is added once and uses the wrapper to parse inputs
    try:
        already_added = any(getattr(t, 'name', '') == 'create_chart' for t in all_tools)
    except Exception:
        already_added = False
    if not already_added:
        charting_tool = Tool(
            name="create_chart",
            func=plotting_tool_wrapper(create_chart),
            description="Creates a chart from SQL query results. Parameters: query (SQL to execute), chart_type (bar, line, scatter), title, filename"
        )
        all_tools.append(charting_tool)
    
    # Add advanced visualization tool if available
    if 'ADVANCED_VIZ_AVAILABLE' in globals() and ADVANCED_VIZ_AVAILABLE and advanced_viz_tool_func:
        try:
            advanced_viz_tool = Tool(
                name="create_advanced_visualization",
                func=advanced_viz_tool_func,
                description=advanced_viz_description
            )
            all_tools.append(advanced_viz_tool)
            print("Added advanced visualization tool with support for maps and interactive charts.")
            
            # Add natural language visualization tool
            if 'NATURAL_VIZ_AVAILABLE' in globals() and NATURAL_VIZ_AVAILABLE:
                natural_viz_func = create_natural_viz_tool(advanced_viz_tool_func)
                natural_viz_tool = Tool(
                    name="create_visualization",
                    func=natural_viz_func,
                    description=get_natural_viz_tool_description()
                )
                all_tools.append(natural_viz_tool)
                print("Added natural language visualization tool for plain English requests.")
        except Exception as e:
            print(f"Could not add advanced visualization tool: {e}")
    print("Added visualization tool.")
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

# --- Visualization Controls UI (Gradio Blocks) ---
# Provides toggles and parameters for advanced line charts (grouping, points, lines, error bars, smoothing)
# Uses the already initialized `advanced_viz_tool_func` and database `engine`.
def launch_visualization_controls():
    """
    Launch a Gradio Blocks UI exposing advanced visualization controls.
    - Consolidates grouped lines, error bars, points, smoothing (GLM/LOESS/GAM) into one UI.
    - Calls the integrated advanced visualization tool in this module.
    Returns the Gradio Blocks app if Gradio is available; otherwise returns a string message.
    """
    if not ('GRADIO_AVAILABLE' in globals() and GRADIO_AVAILABLE):
        return "Gradio is not installed. Please `pip install gradio` to use the visualization UI."
    if 'advanced_viz_tool_func' not in globals() or advanced_viz_tool_func is None:
        return "Advanced visualization tool is not available. Ensure visualization components initialized."
    if 'engine' not in globals() or engine is None:
        return "Database engine is not available. Set DATABASE_URL or ensure DB initialization succeeded."

    import json

    def _submit(
        sql_query: str,
        viz_type: str,
        renderer: str,
        x: str,
        y: str,
        group: str,
        show_points: bool,
        show_line: bool,
        compute_error: bool,
        error_type: str,
        error_lower: str,
        error_upper: str,
        smoother: str,
        loess_frac: float,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str
    ):
        # Handle y as list if comma-separated
        y_clean = (y or "").strip()
        if "," in y_clean:
            y_value = [part.strip() for part in y_clean.split(",") if part.strip()]
        else:
            y_value = y_clean or None

        # Normalize smoother
        smoother_norm = (smoother or "").strip().lower()
        smoother_value = None if smoother_norm == "none" else (smoother_norm or None)

        params = {
            "renderer": renderer or None,
            "x": (x or "").strip() or None,
            "y": y_value,
            "group": (group or "").strip() or None,
            "color": (group or "").strip() or None,
            "show_points": bool(show_points),
            "show_line": bool(show_line),
            "compute_error": bool(compute_error),
            "error_type": (error_type or "se").lower(),
            "error_lower": (error_lower or "").strip() or None,
            "error_upper": (error_upper or "").strip() or None,
            "smoother": smoother_value,
            "smoother_params": {"frac": float(loess_frac) if loess_frac else 0.5},
            "title": (title or "").strip() or None,
            "xlabel": (xlabel or "").strip() or None,
            "ylabel": (ylabel or "").strip() or None,
            "filename": (filename or "").strip() or None,
        }
        params = {k: v for k, v in params.items() if v is not None}

        payload = {
            "query": sql_query,
            "viz_type": (viz_type or "line").lower(),
            "params": params,
        }
        return advanced_viz_tool_func(json.dumps(payload))

    with gr.Blocks(title="Advanced Visualization Controls") as demo:
        gr.Markdown("## Advanced Visualization Controls\nConfigure grouped lines, error bars, points, and smoothing. Uses VisualizationFactory under the hood.")

        with gr.Row():
            sql_query = gr.Textbox(label="SQL Query", lines=5, placeholder="SELECT Year, AVG(Biomass) AS AvgBiomass, Region FROM ltem_optimized_regions WHERE Label='PEC' GROUP BY Year, Region")

        with gr.Row():
            viz_type = gr.Dropdown(choices=["line", "bar", "scatter", "histogram", "heatmap", "box_plot", "map", "bubble_map", "map_heatmap"], value="line", label="Visualization Type")
            renderer = gr.Dropdown(choices=["plotly", "matplotlib", "altair", "folium"], value="plotly", label="Renderer")

        with gr.Row():
            x = gr.Textbox(label="X Column", placeholder="Year")
            y = gr.Textbox(label="Y Column or comma-separated list", placeholder="AvgBiomass")
            group = gr.Textbox(label="Group/Color Column (optional)", placeholder="Region")

        with gr.Row():
            show_points = gr.Checkbox(label="Show Points", value=True)
            show_line = gr.Checkbox(label="Show Line", value=True)
            compute_error = gr.Checkbox(label="Compute Error (SE/CI)", value=False)
            error_type = gr.Dropdown(choices=["se", "ci"], value="se", label="Error Type")

        with gr.Row():
            error_lower = gr.Textbox(label="Error Lower Col (optional)")
            error_upper = gr.Textbox(label="Error Upper Col (optional)")

        with gr.Row():
            smoother = gr.Dropdown(choices=["none", "glm", "loess", "gam"], value="none", label="Smoother")
            loess_frac = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="LOESS frac")

        with gr.Row():
            title = gr.Textbox(label="Title", placeholder="Fish Biomass Density by Year")
            xlabel = gr.Textbox(label="X Label", placeholder="Year")
            ylabel = gr.Textbox(label="Y Label", placeholder="Biomass")
            filename = gr.Textbox(label="Filename (no extension)", placeholder="biomass_trend")

        out = gr.Textbox(label="Result", lines=3)

        submit = gr.Button("Create Visualization")
        submit.click(
            _submit,
            inputs=[sql_query, viz_type, renderer, x, y, group, show_points, show_line, compute_error, error_type, error_lower, error_upper, smoother, loess_frac, title, xlabel, ylabel, filename],
            outputs=[out]
        )

    return demo

# Add Phase 3 tools if available
if PHASE3_AVAILABLE and LANGCHAIN_AVAILABLE:
    try:
        from langchain.tools import StructuredTool
        
        # Initialize Phase 3 components
        insights_engine = InsightsEngine()
        report_generator = ReportGenerator()
        
        # Automated Insights Discovery Tool
        def discover_insights_tool(data_query: str, target_columns: str = None, max_insights: int = 10):
            """Discover automated insights in ecological data"""
            try:
                # Execute query to get data
                if engine is not None:
                    df = pd.read_sql(data_query, engine)
                else:
                    return "Error: Database not available for insights discovery"
                
                # Parse target columns if provided
                target_cols = None
                if target_columns:
                    target_cols = [col.strip() for col in target_columns.split(',')]
                
                # Discover insights
                insights = insights_engine.discover_insights(
                    df, 
                    target_columns=target_cols,
                    context={'analysis_type': 'agent_query', 'ecosystem': 'gulf_of_california'}
                )
                
                # Format results for agent
                if not insights:
                    return "No significant patterns or insights were automatically detected in the data."
                
                results = []
                for i, insight in enumerate(insights[:max_insights], 1):
                    insight_text = f"{i}. {insight.get('type', 'Unknown').replace('_', ' ').title()}: {insight.get('description', 'No description')}"
                    if insight.get('narrative'):
                        insight_text += f"\n   Analysis: {insight.get('narrative')}"
                    results.append(insight_text)
                
                summary = f"Discovered {len(insights)} insights from {len(df)} records:\n\n" + "\n\n".join(results)
                return summary
                
            except Exception as e:
                return f"Error discovering insights: {str(e)}"
        
        insights_tool = StructuredTool.from_function(
            func=discover_insights_tool,
            name="discover_insights",
            description="""Automatically discover patterns, trends, anomalies, and ecological insights in data.
            Args:
            - data_query: SQL query to get the data for analysis
            - target_columns: Comma-separated list of specific columns to focus on (optional)
            - max_insights: Maximum number of insights to return (default: 10)
            Use this when you want to find hidden patterns or get automated analysis of complex datasets."""
        )
        all_tools.append(insights_tool)
        
        # Comprehensive Report Generation Tool
        def generate_comprehensive_report_tool(data_query: str, title: str = "Ecological Analysis Report", formats: str = "html"):
            """Generate comprehensive reports with data analysis and insights"""
            try:
                # Execute query to get data
                if engine is not None:
                    df = pd.read_sql(data_query, engine)
                else:
                    return "Error: Database not available for report generation"
                
                # Parse formats
                format_list = [fmt.strip().lower() for fmt in formats.split(',')]
                
                # Generate report
                config = {
                    'title': title,
                    'author': 'Gulf of California LTEM Analysis System',
                    'formats': format_list,
                    'output_dir': 'outputs',
                    'include_insights': True,
                    'max_insights': 15
                }
                
                results = report_generator.generate_report(df, config)
                
                # Format response
                response_parts = [f"Generated comprehensive report: '{title}'"]
                response_parts.append(f"Data analyzed: {len(df)} records with {len(df.columns)} variables")
                
                if results.get('metadata', {}).get('insights_count', 0) > 0:
                    insights_count = results['metadata']['insights_count']
                    response_parts.append(f"Automated insights discovered: {insights_count}")
                
                # Add file paths
                for fmt in format_list:
                    if fmt in results:
                        response_parts.append(f"{fmt.upper()} report: {results[fmt]}")
                
                return "\n".join(response_parts)
                
            except Exception as e:
                return f"Error generating report: {str(e)}"
        
        report_tool = StructuredTool.from_function(
            func=generate_comprehensive_report_tool,
            name="generate_comprehensive_report",
            description="""Generate comprehensive PDF and/or HTML reports with automated insights and professional formatting.
            Args:
            - data_query: SQL query to get the data for the report
            - title: Title for the report (optional)
            - formats: Comma-separated list of formats: 'pdf', 'html' (default: 'html')
            Use this to create publication-ready reports with automated insights and professional layouts."""
        )
        all_tools.append(report_tool)
        
        print("✅ Added Phase 3 automated insights and comprehensive reporting tools.")
        
    except Exception as e:
        print(f"Could not create Phase 3 tools: {e}")
else:
    if not PHASE3_AVAILABLE:
        print("Phase 3 tools not available - automated insights and reporting will be limited.")

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

# --- Agent Analysis Function ---
def run_analysis(user_question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Runs the full analysis pipeline for a given user question.
    This function is designed to be called from a GUI.
    """
    if not agent_available or not agent_executor:
        return "Error: Agent is not available due to missing dependencies or initialization failure."

    # The spinner is a command-line feature, so we'll just log progress for the GUI.
    print(f"Starting analysis for: {user_question}")
    start_time = time.time()

    # If the question mentions reefs/species but no direct matches are found,
    # ask for clarification with suggestions before running expensive analysis
    try:
        clarification = maybe_request_clarification(user_question)
        if clarification:
            return clarification
    except Exception:
        pass

    try:
        # Prepare chat history messages (token-safe: keep last 6 turns max)
        lc_history = []
        try:
            if chat_history and isinstance(chat_history, list):
                # Keep only the last 6 messages (3 user/assistant pairs)
                trimmed = chat_history[-6:]
                for m in trimmed:
                    role = (m or {}).get("role")
                    content = (m or {}).get("content", "")
                    if not isinstance(content, str):
                        content = str(content)
                    if role == "user":
                        lc_history.append(HumanMessage(content=content))
                    elif role == "assistant":
                        lc_history.append(AIMessage(content=content))
        except Exception:
            lc_history = []

        # Execute the agent with the user's question and optional history
        payload = {"input": user_question}
        if lc_history:
            payload["chat_history"] = lc_history
        response = agent_executor.invoke(payload)

        # Sanitize the response
        sanitized_output = response["output"]
        if data_validator is not None:
            print("Validating response...")
            sanitized_output = data_validator.sanitize_response(response["output"]) 

        # Redact internal details such as raw SQL echoes and label jargon
        def _redact(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Remove lines that disclose internal queries
            lines = []
            for ln in text.splitlines():
                if re.search(r"^\s*(Query|SQL) used\s*:\s*", ln, flags=re.IGNORECASE):
                    continue
                lines.append(ln)
            redacted = "\n".join(lines)
            # Remove parenthetical Label specifications like (Label = 'PEC')
            redacted = re.sub(r"\(\s*Label\s*=\s*['\"]?PEC['\"]?\s*\)", "", redacted, flags=re.IGNORECASE)
            # Tidy multiple spaces left by removals
            redacted = re.sub(r"\s{2,}", " ", redacted)
            return redacted.strip()

        sanitized_output = _redact(sanitized_output)
        
        total_elapsed = time.time() - start_time
        print(f"Analysis complete in {total_elapsed:.2f} seconds.")
        
        return sanitized_output

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        # Provide specific advice for common errors
        if "rate limit" in str(e).lower():
            return (f"{error_message}\n\nThis appears to be a rate limit error from OpenAI. "
                    "Please wait a few minutes before trying again.")
        return error_message

# --- Main Interactive Loop (for command-line use) ---
if __name__ == "__main__":
    print("\n--- Ecological Monitoring Agent (Command-Line Interface) ---")

    # Exit if agent is not available
    if not agent_available:
        print("\nWARNING: Agent is not available due to missing dependencies.")
        print("Please check your installation and environment variables.")
        sys.exit(1)

    print("\nAgent is ready! Ask me questions about your database.")
    print("Type 'exit', 'quit', or 'q' to end the session.")

    # Main interaction loop
    while True:
        user_question = input("\nYour Question: ")
        if user_question.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Ecological Monitoring Agent. Goodbye!")
            break
        
        # Use the new run_analysis function
        spinner = Spinner("🐠 Analyzing data...")
        spinner.start()
        
        answer = run_analysis(user_question)
        
        spinner.stop()
        
        print("\nAgent's Answer:")
        print(answer)

