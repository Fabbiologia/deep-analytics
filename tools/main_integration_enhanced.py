#!/usr/bin/env python3
"""
Enhanced Main Integration Module - Phases 1-3 Implementation
Integrates Query Engine, Context-Aware Assistant, Insights Engine, and Report Generation
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import Phase 1 components
try:
    from query_engine import QueryEngine
    from context_aware_assistant import ContextAwareAssistant
    PHASE1_COMPONENTS_AVAILABLE = True
    print("Phase 1 components loaded successfully.")
except ImportError as e:
    PHASE1_COMPONENTS_AVAILABLE = False
    print(f"Warning: Could not load Phase 1 components: {e}")

# Import Phase 3 components
try:
    from insights_engine import InsightsEngine
    from report_generator import ReportGenerator, generate_quick_report, generate_insights_report
    PHASE3_COMPONENTS_AVAILABLE = True
    print("Phase 3 components loaded successfully.")
except ImportError as e:
    PHASE3_COMPONENTS_AVAILABLE = False
    print(f"Warning: Could not load Phase 3 components: {e}")

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
SCHEMA_DESCRIPTION = """ 
This is a comprehensive dataset from a Long-Term Ecological Monitoring (LTEM) program for coral reef ecosystems. 

## `ltem_optimized_regions` Table 
This table is an optimized subset containing data from Loreto, La Paz, and Cabo Pulmo regions. 
Key columns include:
- `Label`: 'INV' for invertebrates, 'PEC' for fish
- `Taxa1`, `Taxa2`, `Phylum`, `Species`: Taxonomic information
- `Region`: Survey region (Loreto, La Paz, Cabo Pulmo)
- `Year`: Survey year
- `TrophicLevel`: Trophic level (1-5 scale)
- `Biomass`: Weight in grams (NULL for invertebrates)
- `Area`: Area surveyed in square meters
- `MPA`: Marine Protected Area status (Yes/No)
"""


class EnhancedAnalysisSystem:
    """
    Enhanced Analysis System integrating Phases 1-3 capabilities:
    - Phase 1: Query Engine + Context-Aware Assistant
    - Phase 3: Insights Engine + Report Generation
    """
    
    def __init__(self):
        """Initialize the enhanced analysis system."""
        self.components = self._setup_all_components()
        self.analysis_history = []
        
    def _setup_all_components(self) -> Dict[str, Any]:
        """Set up all available components from Phases 1-3."""
        components = {
            'phase1_available': False,
            'phase3_available': False,
            'query_engine': None,
            'assistant': None,
            'insights_engine': None,
            'report_generator': None
        }
        
        # Set up Phase 1 components
        if PHASE1_COMPONENTS_AVAILABLE and DATABASE_AVAILABLE:
            try:
                # Database connection (you might need to adjust this)
                db_url = os.getenv('DATABASE_URL', 'sqlite:///your_database.db')
                
                components['query_engine'] = QueryEngine(db_url, SCHEMA_DESCRIPTION)
                components['assistant'] = ContextAwareAssistant()
                components['phase1_available'] = True
                print("✅ Phase 1 components initialized successfully")
                
            except Exception as e:
                print(f"❌ Error setting up Phase 1 components: {e}")
        
        # Set up Phase 3 components
        if PHASE3_COMPONENTS_AVAILABLE:
            try:
                components['insights_engine'] = InsightsEngine()
                components['report_generator'] = ReportGenerator()
                components['phase3_available'] = True
                print("✅ Phase 3 components initialized successfully")
                
            except Exception as e:
                print(f"❌ Error setting up Phase 3 components: {e}")
        
        return components
    
    def process_enhanced_query(self, query: str, config: Dict = None) -> Dict[str, Any]:
        """
        Process a user query using all available components.
        
        Args:
            query: User's natural language query
            config: Configuration for analysis and reporting
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if config is None:
            config = {}
            
        start_time = time.time()
        results = {
            'query': query,
            'timestamp': start_time,
            'phase1_results': None,
            'phase3_results': None,
            'reports_generated': None,
            'success': False,
            'errors': []
        }
        
        print(f"\n🔍 Processing query: {query}")
        
        # Phase 1: Query processing and context analysis
        if self.components['phase1_available']:
            try:
                print("📊 Running Phase 1 analysis...")
                phase1_results = self._run_phase1_analysis(query)
                results['phase1_results'] = phase1_results
                print(f"✅ Phase 1 completed in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Phase 1 error: {str(e)}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        # Phase 3: Insights discovery and report generation
        if self.components['phase3_available']:
            try:
                print("🔬 Running Phase 3 analysis...")
                
                # Get data from Phase 1 if available
                data = None
                if results['phase1_results'] and results['phase1_results'].get('success'):
                    data = results['phase1_results']['query_results'].get('results', {}).get('dataframe')
                
                # If no data from Phase 1, could load from file or generate sample
                if data is None:
                    print("ℹ️ No data from Phase 1, using sample data for demonstration")
                    data = self._generate_sample_data()
                
                phase3_results = self._run_phase3_analysis(data, query, config)
                results['phase3_results'] = phase3_results
                
                # Generate reports if requested
                if config.get('generate_reports', True) and data is not None:
                    reports = self._generate_reports(data, phase3_results.get('insights', []), config)
                    results['reports_generated'] = reports
                
                print(f"✅ Phase 3 completed in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Phase 3 error: {str(e)}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        # Overall success determination
        results['success'] = (
            (results['phase1_results'] and results['phase1_results'].get('success')) or
            (results['phase3_results'] and results['phase3_results'].get('success'))
        )
        
        results['processing_time'] = time.time() - start_time
        
        # Add to analysis history
        self.analysis_history.append({
            'query': query,
            'timestamp': start_time,
            'success': results['success'],
            'processing_time': results['processing_time']
        })
        
        return results
    
    def _run_phase1_analysis(self, query: str) -> Dict[str, Any]:
        """Run Phase 1 query processing and context analysis."""
        try:
            # Process the query with QueryEngine
            query_results = self.components['query_engine'].process_query(query)
            
            # Generate analysis suggestions based on query and results
            data = query_results.get('results', {}).get('dataframe')
            suggestions = self.components['assistant'].suggest_analysis(query, data)
            
            # Assess data quality if data is available
            data_quality = {}
            if data is not None:
                data_quality = self.components['assistant'].assess_data_quality(data)
                
            return {
                'query_results': query_results,
                'analysis_suggestions': suggestions,
                'data_quality': data_quality,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def _run_phase3_analysis(self, data: pd.DataFrame, query: str, config: Dict) -> Dict[str, Any]:
        """Run Phase 3 insights discovery and analysis."""
        try:
            # Discover insights
            insights = self.components['insights_engine'].discover_insights(
                data,
                target_columns=config.get('target_columns'),
                context={
                    'query': query,
                    'analysis_type': 'enhanced_query_processing',
                    'ecosystem': 'gulf_of_california'
                }
            )
            
            # Analyze insights patterns
            insights_summary = self._analyze_insights_patterns(insights)
            
            return {
                'insights': insights,
                'insights_summary': insights_summary,
                'data_shape': data.shape if data is not None else None,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def _generate_reports(self, data: pd.DataFrame, insights: List[Dict], config: Dict) -> Dict[str, Any]:
        """Generate comprehensive reports."""
        try:
            # Prepare report configuration
            report_config = {
                'title': config.get('report_title', 'Enhanced Gulf of California LTEM Analysis'),
                'author': 'Gulf of California LTEM Analysis System - Enhanced Version',
                'formats': config.get('report_formats', ['html']),  # Default to HTML for speed
                'template': config.get('report_template', 'comprehensive'),
                'output_dir': config.get('output_dir', 'outputs'),
                'include_insights': True,
                'max_insights': config.get('max_insights', 10)
            }
            
            # Generate reports
            report_results = self.components['report_generator'].generate_report(data, report_config)
            
            return report_results
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def _analyze_insights_patterns(self, insights: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in discovered insights."""
        if not insights:
            return {'total': 0, 'types': {}, 'top_insights': []}
        
        # Count insights by type
        type_counts = {}
        for insight in insights:
            insight_type = insight.get('type', 'unknown')
            type_counts[insight_type] = type_counts.get(insight_type, 0) + 1
        
        # Get top insights by score
        top_insights = sorted(insights, key=lambda x: x.get('score', 0), reverse=True)[:5]
        
        return {
            'total': len(insights),
            'types': type_counts,
            'top_insights': [
                {
                    'type': insight.get('type', 'unknown'),
                    'description': insight.get('description', 'N/A'),
                    'score': insight.get('score', 0)
                }
                for insight in top_insights
            ]
        }
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demonstration when no real data is available."""
        import numpy as np
        
        np.random.seed(42)  # For reproducible results
        
        # Create sample ecological data
        years = list(range(2015, 2021))
        regions = ['Loreto', 'La Paz', 'Cabo Pulmo']
        species = ['Species_A', 'Species_B', 'Species_C', 'Species_D']
        trophic_levels = [2.0, 2.5, 3.0, 3.5, 4.0]
        
        data = []
        
        for year in years:
            for region in regions:
                for species_name in species:
                    base_biomass = np.random.exponential(30)
                    
                    # Add temporal trend
                    trend_factor = 1 + (year - 2015) * 0.02 * np.random.normal(0, 0.1)
                    
                    # Add regional differences
                    region_factor = {'Loreto': 1.2, 'La Paz': 1.0, 'Cabo Pulmo': 0.8}[region]
                    
                    biomass = base_biomass * trend_factor * region_factor
                    
                    # Ensure non-negative values
                    biomass = max(0, biomass)
                    
                    data.append({
                        'Year': year,
                        'Region': region,
                        'Species': species_name,
                        'TrophicLevel': np.random.choice(trophic_levels),
                        'Biomass': biomass,
                        'CPUE': biomass * np.random.lognormal(0, 0.2),
                        'Label': 'PEC',
                        'MPA': np.random.choice(['Yes', 'No'])
                    })
        
        return pd.DataFrame(data)
    
    def get_capabilities_summary(self) -> Dict[str, Any]:
        """Get a summary of available system capabilities."""
        return {
            'phase1_available': self.components['phase1_available'],
            'phase3_available': self.components['phase3_available'],
            'capabilities': {
                'natural_language_queries': self.components['phase1_available'],
                'automated_insights': self.components['phase3_available'],
                'report_generation': self.components['phase3_available'],
                'data_quality_assessment': self.components['phase1_available'],
                'analysis_suggestions': self.components['phase1_available'],
                'statistical_analysis': ADVANCED_STATS_AVAILABLE
            },
            'analysis_history_count': len(self.analysis_history)
        }
    
    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run a comprehensive demonstration of all system capabilities."""
        print("\n" + "="*60)
        print("ENHANCED GULF OF CALIFORNIA LTEM ANALYSIS SYSTEM DEMO")
        print("="*60)
        
        # Display system capabilities
        capabilities = self.get_capabilities_summary()
        print(f"\n📋 System Capabilities:")
        for capability, available in capabilities['capabilities'].items():
            status = "✅" if available else "❌"
            print(f"  {status} {capability.replace('_', ' ').title()}")
        
        # Run sample queries
        sample_queries = [
            "What is the average biomass by region over time?",
            "Show me trophic level distribution patterns",
            "Analyze biodiversity trends in marine protected areas"
        ]
        
        results = []
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n\n🔍 Sample Query {i}: {query}")
            print("-" * 50)
            
            # Configure analysis
            config = {
                'generate_reports': True,
                'report_formats': ['html'],
                'max_insights': 5,
                'target_columns': ['Biomass', 'TrophicLevel', 'CPUE']
            }
            
            # Process query
            result = self.process_enhanced_query(query, config)
            results.append(result)
            
            # Display summary
            if result['success']:
                print(f"✅ Analysis completed in {result['processing_time']:.2f}s")
                
                # Show Phase 1 results
                if result['phase1_results']:
                    print("📊 Phase 1 Results: Query processed successfully")
                
                # Show Phase 3 results
                if result['phase3_results']:
                    insights_summary = result['phase3_results'].get('insights_summary', {})
                    print(f"🔬 Phase 3 Results: {insights_summary.get('total', 0)} insights discovered")
                    
                    # Show top insights
                    for insight in insights_summary.get('top_insights', [])[:2]:
                        print(f"  • {insight['type']}: {insight['description']}")
                
                # Show report generation
                if result['reports_generated']:
                    reports = result['reports_generated']
                    if 'html' in reports:
                        print(f"📄 Report generated: {reports['html']}")
            else:
                print("❌ Analysis failed")
                for error in result.get('errors', []):
                    print(f"  Error: {error}")
        
        return {
            'demo_results': results,
            'system_capabilities': capabilities,
            'total_queries_processed': len(results)
        }


# --- Integration Functions ---

def setup_enhanced_system() -> EnhancedAnalysisSystem:
    """Set up the enhanced analysis system with all available components."""
    print("\n🚀 Initializing Enhanced Gulf of California LTEM Analysis System...")
    system = EnhancedAnalysisSystem()
    print("✅ System initialization completed")
    return system


def process_user_query(system: EnhancedAnalysisSystem, query: str, 
                      generate_reports: bool = True) -> Dict[str, Any]:
    """
    Process a user query with the enhanced system.
    
    Args:
        system: Enhanced analysis system instance
        query: User's natural language query
        generate_reports: Whether to generate reports
        
    Returns:
        Comprehensive analysis results
    """
    config = {
        'generate_reports': generate_reports,
        'report_formats': ['html', 'pdf'] if generate_reports else [],
        'max_insights': 10,
        'report_title': f'Analysis: {query[:50]}...' if len(query) > 50 else f'Analysis: {query}'
    }
    
    return system.process_enhanced_query(query, config)


# --- Main Execution ---

def main():
    """Main function demonstrating the enhanced system."""
    try:
        # Initialize system
        system = setup_enhanced_system()
        
        # Run comprehensive demo
        demo_results = system.run_comprehensive_demo()
        
        print(f"\n\n🎉 Demo completed successfully!")
        print(f"Processed {demo_results['total_queries_processed']} queries")
        
        # Display final system status
        capabilities = demo_results['system_capabilities']
        print(f"\n📊 Final System Status:")
        print(f"  Phase 1 Available: {capabilities['phase1_available']}")
        print(f"  Phase 3 Available: {capabilities['phase3_available']}")
        
        return demo_results
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()
