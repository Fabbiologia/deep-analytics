#!/usr/bin/env python3
"""
Context-Aware Analysis Assistant for Ecological Data Analysis
Guides users through appropriate ecological analysis methods and maintains context.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

# Import conditionally to handle missing dependencies gracefully
try:
    import pandas as pd
    import numpy as np
    NUMPY_PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_PANDAS_AVAILABLE = False
    print("Warning: pandas/numpy not installed. ContextAwareAssistant will have limited functionality.")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: LangChain dependencies not available: {e}")

# Try to import optional dependencies
try:
    from statistical_analysis import (
        perform_ttest, perform_anova, perform_correlation_analysis,
        perform_regression_analysis, check_normality
    )
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("Warning: statistical_analysis module not found. Statistical capabilities will be limited.")


class ContextAwareAssistant:
    """
    An assistant that maintains context about the user's analysis session
    and provides guidance on ecological data analysis methods.
    """
    
    def __init__(self, llm_client=None, memory_manager=None, analysis_tools=None):
        """
        Initialize the ContextAwareAssistant
        
        Args:
            llm_client: LangChain LLM client (optional)
            memory_manager: Memory system for conversation history
            analysis_tools: Dictionary of analysis tool functions
        """
        self.llm_client = llm_client
        self.memory = memory_manager if memory_manager else self._create_default_memory()
        self.tools = analysis_tools if analysis_tools else self._setup_default_tools()
        
        # Context tracking
        self.context = {
            "current_analysis": None,
            "data_summary": None,
            "recent_queries": [],
            "ecological_focus": [],
            "data_quality_notes": [],
            "suggested_analyses": [],
            "time_periods": [],
            "regions": [],
            "species": []
        }
        
        # Create default LLM client if not provided
        if llm_client is None and LANGCHAIN_AVAILABLE:
            try:
                self.llm_client = ChatOpenAI(
                    model="gpt-5-mini"
                )
            except Exception as e:
                print(f"Error creating LLM client: {e}")
                self.llm_client = None
                
        # Set up prompt templates
        self._setup_prompt_templates()
    
    def _create_default_memory(self):
        """Create a default memory system"""
        if LANGCHAIN_AVAILABLE:
            return ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return {"messages": []}
        
    def _setup_default_tools(self):
        """Set up default analysis tools"""
        tools = {}
        
        # Add statistical analysis tools if available
        if STATS_AVAILABLE:
            tools.update({
                "t_test": perform_ttest,
                "anova": perform_anova,
                "correlation": perform_correlation_analysis,
                "regression": perform_regression_analysis,
                "normality_check": check_normality
            })
            
        return tools
        
    def _setup_prompt_templates(self):
        """Set up prompt templates for various assistant functions"""
        if not LANGCHAIN_AVAILABLE:
            return
            
        # Template for suggesting analyses
        self.analysis_suggestion_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an ecological data analysis expert specializing in marine ecosystems,
            particularly coral reef monitoring data. Your task is to suggest appropriate 
            scientific analyses based on the user's request and available data.
            
            AVAILABLE DATA CONTEXT:
            {data_context}
            
            USER'S ANALYSIS HISTORY:
            {analysis_history}
            
            Suggest 1-3 specific analytical approaches that would be appropriate.
            For each suggestion:
            1. Name the analytical method
            2. Explain why it's appropriate for the ecological question
            3. List required assumptions and prerequisites
            4. Describe expected ecological insights
            5. Note any statistical considerations or limitations
            
            Format your response as a JSON array of analysis objects with these fields:
            - method_name: The name of the analytical method
            - description: Brief description of the method
            - rationale: Why this method is appropriate
            - prerequisites: List of prerequisites/assumptions
            - expected_insights: What ecological insights it might provide
            - required_columns: Data columns required for this analysis
            - implementation_notes: Statistical considerations and implementation tips
            """),
            ("user", "{request}")
        ])
        
        # Template for data quality assessment
        self.data_quality_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a data quality specialist for ecological datasets. Assess the quality
            and suitability of the provided data summary for ecological analysis.
            
            Look for these potential issues:
            1. Missing values or incomplete records
            2. Outliers that might skew analysis
            3. Temporal gaps in monitoring data
            4. Spatial biases in sampling
            5. Taxonomic biases or identification issues
            6. Measurement consistency issues
            
            Provide a concise assessment formatted as a JSON object with these fields:
            - quality_score: Overall score from 1-10
            - major_issues: List of major data quality concerns
            - recommendations: Preprocessing steps to address issues
            - suitable_analyses: Analyses that would be robust despite quality issues
            - cautions: Analyses to avoid given the data quality
            """),
            ("user", "Data summary: {data_summary}")
        ])
        
    def suggest_analysis(self, user_request: str, current_data=None) -> Dict[str, Any]:
        """
        Suggest appropriate analysis based on user request and context
        
        Args:
            user_request: The user's analysis request
            current_data: Current dataframe or data summary (optional)
            
        Returns:
            Dictionary with analysis suggestions and context
        """
        # Update context with current data if provided
        if current_data is not None:
            self._update_context_with_data(current_data)
            
        # If LangChain not available, return basic suggestions
        if not LANGCHAIN_AVAILABLE or self.llm_client is None:
            return {
                'suggestions': [
                    {
                        'method_name': 'Basic Statistical Summary',
                        'description': 'Calculate mean, median, min, max values',
                        'implementation_notes': 'Use pandas describe() function'
                    }
                ],
                'context_updated': False
            }
            
        # Prepare context information
        data_context = self._format_data_context()
        analysis_history = self._format_analysis_history()
        
        try:
            # Format the analysis suggestion prompt
            formatted_prompt = self.analysis_suggestion_template.format(
                data_context=data_context,
                analysis_history=analysis_history,
                request=user_request
            )
            
            # Get suggestions from LLM
            response = self.llm_client.invoke(formatted_prompt)
            
            # Extract JSON from the response
            content = response.content
            
            # Remove markdown code blocks if present
            content = re.sub(r'```json\s*|\s*```', '', content)
            content = re.sub(r'```\s*|\s*```', '', content)
            
            # Parse suggestions
            suggestions = json.loads(content)
            
            # Update context with the request and suggestions
            self._update_context_with_request(user_request, suggestions)
            
            return {
                'suggestions': suggestions,
                'context_updated': True
            }
            
        except Exception as e:
            print(f"Error generating analysis suggestions: {e}")
            # Fallback for error cases
            return {
                'suggestions': [
                    {
                        'method_name': 'Basic Statistical Summary',
                        'description': 'Calculate mean, median, min, max values',
                        'implementation_notes': 'Use pandas describe() function'
                    }
                ],
                'error': str(e),
                'context_updated': False
            }
            
    def assess_data_quality(self, data) -> Dict[str, Any]:
        """
        Assess the quality of the provided data
        
        Args:
            data: DataFrame or data summary to assess
            
        Returns:
            Dictionary with data quality assessment
        """
        # Generate data summary if a DataFrame is provided
        data_summary = self._generate_data_summary(data) if NUMPY_PANDAS_AVAILABLE and isinstance(data, pd.DataFrame) else data
        
        # If LangChain not available, return basic assessment
        if not LANGCHAIN_AVAILABLE or self.llm_client is None:
            return {
                'quality_score': 5,
                'major_issues': ['Unable to perform detailed assessment without LLM'],
                'recommendations': ['Check for missing values and outliers manually']
            }
            
        try:
            # Format the data quality prompt
            formatted_prompt = self.data_quality_template.format(
                data_summary=json.dumps(data_summary)
            )
            
            # Get assessment from LLM
            response = self.llm_client.invoke(formatted_prompt)
            
            # Extract JSON from the response
            content = response.content
            
            # Remove markdown code blocks if present
            content = re.sub(r'```json\s*|\s*```', '', content)
            content = re.sub(r'```\s*|\s*```', '', content)
            
            # Parse assessment
            assessment = json.loads(content)
            
            # Update context with the assessment
            self.context['data_quality_notes'].append(assessment)
            
            return assessment
            
        except Exception as e:
            print(f"Error assessing data quality: {e}")
            # Fallback for error cases
            return {
                'quality_score': 5,
                'major_issues': [f'Error assessing data quality: {e}'],
                'recommendations': ['Check for missing values and outliers manually']
            }
            
    def execute_analysis(self, analysis_type: str, data, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a specific analysis on the provided data
        
        Args:
            analysis_type: Type of analysis to perform
            data: DataFrame to analyze
            parameters: Additional parameters for the analysis
            
        Returns:
            Dictionary with analysis results
        """
        parameters = parameters or {}
        
        # Check if the requested analysis is available in our tools
        if analysis_type in self.tools:
            try:
                # Execute the analysis tool
                result = self.tools[analysis_type](data, **parameters)
                
                # Update context with the analysis
                self._update_context_with_analysis(analysis_type, parameters, result)
                
                return {
                    'analysis_type': analysis_type,
                    'parameters': parameters,
                    'result': result,
                    'success': True
                }
            except Exception as e:
                print(f"Error executing {analysis_type} analysis: {e}")
                return {
                    'analysis_type': analysis_type,
                    'parameters': parameters,
                    'error': str(e),
                    'success': False
                }
        else:
            return {
                'analysis_type': analysis_type,
                'error': f"Analysis type '{analysis_type}' not available",
                'available_analyses': list(self.tools.keys()),
                'success': False
            }
            
    def _update_context_with_data(self, data) -> None:
        """
        Update the assistant's context with the provided data
        
        Args:
            data: DataFrame or data summary
        """
        # Generate data summary if a DataFrame is provided
        if NUMPY_PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            summary = self._generate_data_summary(data)
            self.context['data_summary'] = summary
            
            # Extract key context elements from the data
            self._extract_context_from_data(data)
        else:
            # If data is already a summary, just store it
            self.context['data_summary'] = data
            
    def _generate_data_summary(self, df) -> Dict[str, Any]:
        """
        Generate a summary of the DataFrame
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data summary
        """
        if not NUMPY_PANDAS_AVAILABLE or not isinstance(df, pd.DataFrame):
            return {'error': 'DataFrame not available or invalid'}
            
        try:
            summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': {col: int(df[col].isna().sum()) for col in df.columns},
                'numeric_summary': {}
            }
            
            # Add numeric column summaries
            for col in df.select_dtypes(include=['number']).columns:
                summary['numeric_summary'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'std': float(df[col].std())
                }
                
            # Add categorical column summaries
            categorical_summary = {}
            for col in df.select_dtypes(include=['object', 'category']).columns:
                value_counts = df[col].value_counts()
                if len(value_counts) <= 20:  # Only include if not too many unique values
                    categorical_summary[col] = {
                        'unique_count': len(value_counts),
                        'top_values': {str(k): int(v) for k, v in value_counts.head(5).items()}
                    }
                else:
                    categorical_summary[col] = {
                        'unique_count': len(value_counts),
                        'too_many_values': True
                    }
            summary['categorical_summary'] = categorical_summary
            
            return summary
            
        except Exception as e:
            print(f"Error generating data summary: {e}")
            return {'error': str(e)}
            
    def _extract_context_from_data(self, df) -> None:
        """
        Extract key context elements from the data
        
        Args:
            df: DataFrame to extract context from
        """
        if not NUMPY_PANDAS_AVAILABLE or not isinstance(df, pd.DataFrame):
            return
            
        try:
            # Extract time periods if 'Year' column exists
            if 'Year' in df.columns:
                years = sorted(df['Year'].unique())
                self.context['time_periods'] = years
                
            # Extract regions if 'Region' column exists
            if 'Region' in df.columns:
                regions = sorted(df['Region'].unique())
                self.context['regions'] = regions
                
            # Extract species if 'Species' column exists
            if 'Species' in df.columns:
                # Limit to top species to avoid context overload
                species_counts = df['Species'].value_counts()
                top_species = species_counts.head(20).index.tolist()
                self.context['species'] = top_species
                
        except Exception as e:
            print(f"Error extracting context from data: {e}")
            
    def _update_context_with_request(self, user_request: str, suggestions: List[Dict[str, Any]]) -> None:
        """
        Update the context with the user's request and suggestions
        
        Args:
            user_request: The user's analysis request
            suggestions: The analysis suggestions provided
        """
        # Add request to recent queries
        self.context['recent_queries'].append({
            'timestamp': datetime.now().isoformat(),
            'request': user_request
        })
        
        # Limit recent queries to last 5
        self.context['recent_queries'] = self.context['recent_queries'][-5:]
        
        # Add suggestions to context
        self.context['suggested_analyses'].extend(suggestions)
        
        # Limit suggested analyses to last 10
        self.context['suggested_analyses'] = self.context['suggested_analyses'][-10:]
        
        # Update memory if using LangChain memory
        if LANGCHAIN_AVAILABLE and hasattr(self.memory, 'save_context'):
            self.memory.save_context(
                {"input": user_request},
                {"output": f"Suggested {len(suggestions)} analyses"}
            )
            
    def _update_context_with_analysis(self, analysis_type: str, parameters: Dict[str, Any], result: Any) -> None:
        """
        Update the context with the executed analysis
        
        Args:
            analysis_type: The type of analysis performed
            parameters: The parameters used for the analysis
            result: The analysis result
        """
        # Set current analysis
        self.context['current_analysis'] = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'parameters': parameters,
            'result_summary': str(result)[:100] + '...' if isinstance(result, str) and len(str(result)) > 100 else result
        }
        
    def _format_data_context(self) -> str:
        """
        Format the data context for use in prompts
        
        Returns:
            String representation of the data context
        """
        context_parts = []
        
        # Add data summary if available
        if self.context['data_summary']:
            context_parts.append(f"DATA SUMMARY: {json.dumps(self.context['data_summary'])[:500]}...")
            
        # Add time periods if available
        if self.context['time_periods']:
            context_parts.append(f"TIME PERIODS: {self.context['time_periods']}")
            
        # Add regions if available
        if self.context['regions']:
            context_parts.append(f"REGIONS: {self.context['regions']}")
            
        # Add species if available
        if self.context['species']:
            context_parts.append(f"TOP SPECIES: {self.context['species']}")
            
        # Add data quality notes if available
        if self.context['data_quality_notes']:
            latest_note = self.context['data_quality_notes'][-1]
            context_parts.append(f"DATA QUALITY: Score {latest_note.get('quality_score', 'N/A')}/10, Issues: {latest_note.get('major_issues', ['None'])}")
            
        return "\n".join(context_parts)
        
    def _format_analysis_history(self) -> str:
        """
        Format the analysis history for use in prompts
        
        Returns:
            String representation of the analysis history
        """
        history_parts = []
        
        # Add recent queries
        for query in self.context['recent_queries']:
            history_parts.append(f"QUERY: {query['request']}")
            
        # Add current analysis if available
        if self.context['current_analysis']:
            analysis = self.context['current_analysis']
            history_parts.append(f"LAST ANALYSIS: {analysis['analysis_type']} with parameters {analysis['parameters']}")
            
        # Add suggested analyses if available
        if self.context['suggested_analyses']:
            suggestions = [s['method_name'] for s in self.context['suggested_analyses'][-3:]]
            history_parts.append(f"RECENTLY SUGGESTED: {', '.join(suggestions)}")
            
        return "\n".join(history_parts)
        
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context
        
        Returns:
            The assistant's context dictionary
        """
        return self.context


# Standalone testing code
if __name__ == "__main__":
    # Example usage
    # Create assistant
    assistant = ContextAwareAssistant()
    
    # Test with sample data if pandas is available
    if NUMPY_PANDAS_AVAILABLE:
        # Create sample data
        data = pd.DataFrame({
            'Year': [2018, 2018, 2019, 2019, 2020, 2020],
            'Region': ['Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo', 'Loreto', 'Cabo Pulmo'],
            'Species': ['Scarus ghobban', 'Scarus ghobban', 'Scarus ghobban', 'Lutjanus argentiventris', 'Lutjanus argentiventris', 'Balistes polylepis'],
            'Quantity': [5, 3, 7, 2, 4, 6],
            'Biomass': [1250.5, 800.2, 1750.1, 950.3, 1100.8, 1400.6],
            'TrophicLevel': [4.2, 4.2, 4.2, 4.5, 4.5, 3.8]
        })
        
        # Update context with data
        assistant._update_context_with_data(data)
        
        # Test suggestion
        test_request = "I want to analyze how biomass varies by region over time"
        print(f"Testing request: {test_request}")
        
        suggestions = assistant.suggest_analysis(test_request)
        
        # Display results
        print("\nSuggested Analyses:")
        for suggestion in suggestions.get('suggestions', []):
            print(f"- {suggestion.get('method_name')}: {suggestion.get('description')}")
            print(f"  Rationale: {suggestion.get('rationale', 'N/A')}")
            print()
        
        # Test data quality assessment
        quality = assistant.assess_data_quality(data)
        print("\nData Quality Assessment:")
        print(f"Score: {quality.get('quality_score')}/10")
        print(f"Issues: {quality.get('major_issues', ['None'])}")
        
    else:
        print("Cannot run test - pandas not available")
