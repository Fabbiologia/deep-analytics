#!/usr/bin/env python3
"""
Comprehensive Report Generation System for Ecological Data Analysis
Integrates insights, visualizations, and statistical results into publication-ready reports.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from pathlib import Path

# Import existing PDF generator
try:
    from pdf_report_generator import EcologicalReportGenerator, create_comprehensive_report
    PDF_AVAILABLE = True
except ImportError:
    logger.warning("PDF report generator not available")
    PDF_AVAILABLE = False

# Import insights engine
try:
    from insights_engine import InsightsEngine
    INSIGHTS_AVAILABLE = True
except ImportError:
    logger.warning("Insights engine not available")
    INSIGHTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Main report generation system that creates comprehensive reports
    from data analysis, insights, and visualizations.
    """
    
    def __init__(self):
        """Initialize the report generator with all components."""
        self.pdf_generator = PDFReportGenerator() if PDF_AVAILABLE else None
        self.html_generator = HTMLReportGenerator()
        self.template_manager = ReportTemplateManager()
        self.insights_engine = InsightsEngine() if INSIGHTS_AVAILABLE else None
        
        logger.info("ReportGenerator initialized")
        
    def generate_report(self, data: pd.DataFrame, config: Dict = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report with data, insights, and visualizations.
        
        Args:
            data: Input DataFrame for analysis
            config: Configuration dictionary with report settings
            
        Returns:
            Dictionary containing report paths and metadata
        """
        if config is None:
            config = {}
            
        logger.info(f"Starting report generation for {len(data)} rows of data")
        
        # Set default configuration
        config = self._set_default_config(config)
        
        # Discover insights if insights engine is available
        insights = []
        if self.insights_engine and config.get('include_insights', True):
            try:
                logger.info("Discovering automated insights...")
                insights = self.insights_engine.discover_insights(
                    data, 
                    target_columns=config.get('target_columns'),
                    context=config.get('context', {})
                )
                logger.info(f"Found {len(insights)} insights")
            except Exception as e:
                logger.error(f"Error discovering insights: {str(e)}")
        
        # Prepare report content
        content = self._prepare_content(data, insights, config)
        
        # Select template
        template = self.template_manager.get_template(config.get('template', 'comprehensive'))
        
        # Generate reports in requested formats
        report_results = {}
        
        formats = config.get('formats', ['pdf'])
        if isinstance(formats, str):
            formats = [formats]
            
        for fmt in formats:
            try:
                if fmt == 'pdf' and self.pdf_generator:
                    report_path = self.pdf_generator.generate(content, template, config)
                    report_results['pdf'] = report_path
                elif fmt == 'html':
                    report_path = self.html_generator.generate(content, template, config)
                    report_results['html'] = report_path
                else:
                    logger.warning(f"Unsupported format: {fmt}")
            except Exception as e:
                logger.error(f"Error generating {fmt} report: {str(e)}")
                
        # Add metadata
        report_results['metadata'] = {
            'generation_time': datetime.now().isoformat(),
            'data_shape': data.shape,
            'insights_count': len(insights),
            'config': config
        }
        
        logger.info(f"Report generation completed. Generated formats: {list(report_results.keys())}")
        return report_results
    
    def _set_default_config(self, config: Dict) -> Dict:
        """Set default configuration values."""
        defaults = {
            'title': 'Ecological Data Analysis Report',
            'author': 'Gulf of California LTEM Analysis System',
            'template': 'comprehensive',
            'formats': ['pdf'],
            'include_insights': True,
            'include_visualizations': True,
            'include_statistical_tests': True,
            'max_insights': 10,
            'output_dir': 'outputs'
        }
        
        # Merge with provided config
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
                
        return config
    
    def _prepare_content(self, data: pd.DataFrame, insights: List[Dict], config: Dict) -> Dict:
        """Prepare structured content for report generation."""
        content = {
            'title': config.get('title', 'Ecological Data Analysis Report'),
            'author': config.get('author', 'Analysis System'),
            'generation_date': datetime.now().strftime("%Y-%m-%d"),
            'data_summary': self._create_data_summary(data),
            'insights': insights[:config.get('max_insights', 10)],
            'statistical_results': {},
            'visualizations': [],
            'conclusions': self._generate_conclusions(insights)
        }
        
        return content
    
    def _create_data_summary(self, data: pd.DataFrame) -> Dict:
        """Create a summary of the input data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        summary = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'missing_values': data.isnull().sum().sum(),
            'date_range': self._get_date_range(data),
            'column_info': {}
        }
        
        # Add basic stats for numeric columns
        for col in numeric_cols:
            if not data[col].empty:
                summary['column_info'][col] = {
                    'type': 'numeric',
                    'mean': float(data[col].mean()) if not data[col].isna().all() else None,
                    'std': float(data[col].std()) if not data[col].isna().all() else None,
                    'min': float(data[col].min()) if not data[col].isna().all() else None,
                    'max': float(data[col].max()) if not data[col].isna().all() else None,
                    'missing': int(data[col].isnull().sum())
                }
        
        # Add info for categorical columns
        for col in categorical_cols:
            if not data[col].empty:
                summary['column_info'][col] = {
                    'type': 'categorical',
                    'unique_values': int(data[col].nunique()),
                    'most_common': data[col].mode().iloc[0] if not data[col].empty else None,
                    'missing': int(data[col].isnull().sum())
                }
        
        return summary
    
    def _get_date_range(self, data: pd.DataFrame) -> Optional[Dict]:
        """Extract date range from data if temporal columns exist."""
        time_columns = ['Year', 'Date', 'TIME', 'year', 'date']
        
        for col in time_columns:
            if col in data.columns and not data[col].empty:
                try:
                    min_val = data[col].min()
                    max_val = data[col].max()
                    return {
                        'column': col,
                        'start': str(min_val),
                        'end': str(max_val)
                    }
                except Exception:
                    continue
                    
        return None
    
    def _generate_conclusions(self, insights: List[Dict]) -> List[str]:
        """Generate conclusions based on discovered insights."""
        conclusions = []
        
        if not insights:
            conclusions.append("No significant patterns were automatically detected in the current dataset.")
            return conclusions
        
        # Count insights by type
        insight_counts = {}
        for insight in insights:
            insight_type = insight.get('type', 'unknown')
            insight_counts[insight_type] = insight_counts.get(insight_type, 0) + 1
        
        # Generate type-specific conclusions
        if 'temporal_trend' in insight_counts:
            conclusions.append(f"Analysis identified {insight_counts['temporal_trend']} significant temporal trends in the dataset.")
        
        if 'trophic_structure_shift' in insight_counts:
            conclusions.append(f"Detected {insight_counts['trophic_structure_shift']} shifts in trophic structure, indicating ecosystem changes.")
        
        if 'significant_difference' in insight_counts:
            conclusions.append(f"Statistical testing revealed {insight_counts['significant_difference']} significant differences between groups.")
        
        if 'correlation' in insight_counts:
            conclusions.append(f"Found {insight_counts['correlation']} significant correlations between variables.")
        
        if 'anomaly' in insight_counts:
            conclusions.append(f"Identified {insight_counts['anomaly']} data anomalies that may require further investigation.")
        
        # Add general conclusion
        total_insights = len(insights)
        if total_insights > 5:
            conclusions.append(f"The {total_insights} automated insights provide strong evidence of ecological patterns and changes.")
        elif total_insights > 0:
            conclusions.append("The automated analysis successfully identified key patterns in the ecological data.")
        
        conclusions.append("These findings should be validated with domain expertise and additional analysis.")
        
        return conclusions


class PDFReportGenerator:
    """PDF report generation using existing EcologicalReportGenerator."""
    
    def generate(self, content: Dict, template: Dict, config: Dict) -> str:
        """Generate PDF report using existing infrastructure."""
        if not PDF_AVAILABLE:
            raise RuntimeError("PDF generation not available")
        
        # Create output directory
        output_dir = Path(config.get('output_dir', 'outputs'))
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"ecological_analysis_report_{timestamp}.pdf"
        
        # Prepare analysis results for existing PDF generator
        analysis_results = {
            'summary': {
                'total_insights': len(content.get('insights', [])),
                'data_rows': content.get('data_summary', {}).get('total_rows', 0),
                'analysis_date': content.get('generation_date', ''),
                'key_findings': [insight.get('description', '') for insight in content.get('insights', [])[:5]]
            }
        }
        
        # Add insights as analysis results
        for i, insight in enumerate(content.get('insights', [])):
            analysis_results[f'insight_{i+1}'] = {
                'type': insight.get('type', 'unknown'),
                'description': insight.get('description', ''),
                'narrative': insight.get('narrative', ''),
                'score': insight.get('score', 0),
                'statistical_test': insight.get('statistical_test', 'N/A'),
                'p_value': insight.get('p_value')
            }
        
        # Use existing comprehensive report generator
        try:
            report_path = create_comprehensive_report(
                analysis_results=analysis_results,
                plots=content.get('visualizations', []),
                filename=str(filename)
            )
            logger.info(f"PDF report generated: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise


class HTMLReportGenerator:
    """HTML report generation with modern styling."""
    
    def generate(self, content: Dict, template: Dict, config: Dict) -> str:
        """Generate HTML report."""
        # Create output directory
        output_dir = Path(config.get('output_dir', 'outputs'))
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"ecological_analysis_report_{timestamp}.html"
        
        # Generate HTML content
        html_content = self._generate_html(content, template)
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filename}")
        return str(filename)
    
    def _generate_html(self, content: Dict, template: Dict) -> str:
        """Generate HTML content from template and content."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{content.get('title', 'Ecological Analysis Report')}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{content.get('title', 'Ecological Analysis Report')}</h1>
            <p class="meta">Generated by {content.get('author', 'Analysis System')} on {content.get('generation_date', '')}</p>
        </header>
        
        <section class="data-summary">
            <h2>Data Summary</h2>
            {self._generate_data_summary_html(content.get('data_summary', {}))}
        </section>
        
        <section class="insights">
            <h2>Automated Insights</h2>
            {self._generate_insights_html(content.get('insights', []))}
        </section>
        
        <section class="conclusions">
            <h2>Conclusions</h2>
            {self._generate_conclusions_html(content.get('conclusions', []))}
        </section>
        
        <footer class="report-footer">
            <p>Report generated automatically by the Gulf of California LTEM Analysis System</p>
        </footer>
    </div>
</body>
</html>
"""
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the HTML report."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .report-header {
            border-bottom: 3px solid #2c5282;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            color: #2c5282;
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }
        
        .meta {
            color: #666;
            font-style: italic;
            margin: 0;
        }
        
        h2 {
            color: #2d3748;
            border-left: 4px solid #38a169;
            padding-left: 15px;
            margin-top: 30px;
        }
        
        .insight-card {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #4299e1;
        }
        
        .insight-type {
            background: #4299e1;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 10px;
        }
        
        .insight-description {
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        .insight-narrative {
            color: #4a5568;
            line-height: 1.5;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-box {
            background: #edf2f7;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #2c5282;
            display: block;
        }
        
        .stat-label {
            color: #4a5568;
            font-size: 0.9em;
        }
        
        .conclusions ul {
            padding-left: 20px;
        }
        
        .conclusions li {
            margin-bottom: 8px;
        }
        
        .report-footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        """
    
    def _generate_data_summary_html(self, data_summary: Dict) -> str:
        """Generate HTML for data summary section."""
        if not data_summary:
            return "<p>No data summary available.</p>"
        
        html = f"""
        <div class="stats-grid">
            <div class="stat-box">
                <span class="stat-number">{data_summary.get('total_rows', 0):,}</span>
                <span class="stat-label">Total Records</span>
            </div>
            <div class="stat-box">
                <span class="stat-number">{data_summary.get('total_columns', 0)}</span>
                <span class="stat-label">Variables</span>
            </div>
            <div class="stat-box">
                <span class="stat-number">{data_summary.get('numeric_columns', 0)}</span>
                <span class="stat-label">Numeric Variables</span>
            </div>
            <div class="stat-box">
                <span class="stat-number">{data_summary.get('categorical_columns', 0)}</span>
                <span class="stat-label">Categorical Variables</span>
            </div>
        </div>
        """
        
        # Add date range if available
        date_range = data_summary.get('date_range')
        if date_range:
            html += f"""
            <p><strong>Temporal Coverage:</strong> {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')} 
               (Column: {date_range.get('column', 'N/A')})</p>
            """
        
        return html
    
    def _generate_insights_html(self, insights: List[Dict]) -> str:
        """Generate HTML for insights section."""
        if not insights:
            return "<p>No automated insights were discovered in this dataset.</p>"
        
        html = f"<p>Discovered <strong>{len(insights)}</strong> automated insights:</p>"
        
        for i, insight in enumerate(insights, 1):
            insight_type = insight.get('type', 'unknown').replace('_', ' ').title()
            description = insight.get('description', 'No description available')
            narrative = insight.get('narrative', insight.get('description', ''))
            
            html += f"""
            <div class="insight-card">
                <span class="insight-type">{insight_type}</span>
                <div class="insight-description">{description}</div>
                <div class="insight-narrative">{narrative}</div>
            </div>
            """
        
        return html
    
    def _generate_conclusions_html(self, conclusions: List[str]) -> str:
        """Generate HTML for conclusions section."""
        if not conclusions:
            return "<p>No conclusions generated.</p>"
        
        html = "<ul>"
        for conclusion in conclusions:
            html += f"<li>{conclusion}</li>"
        html += "</ul>"
        
        return html


class ReportTemplateManager:
    """Manages report templates and configurations."""
    
    def __init__(self):
        """Initialize template manager with default templates."""
        self.templates = {
            'comprehensive': self._get_comprehensive_template(),
            'summary': self._get_summary_template(),
            'insights_only': self._get_insights_template()
        }
    
    def get_template(self, template_name: str) -> Dict:
        """Get template configuration by name."""
        return self.templates.get(template_name, self.templates['comprehensive'])
    
    def _get_comprehensive_template(self) -> Dict:
        """Get comprehensive report template."""
        return {
            'name': 'comprehensive',
            'sections': [
                'title_page',
                'executive_summary', 
                'data_summary',
                'automated_insights',
                'statistical_results',
                'visualizations',
                'conclusions',
                'methods'
            ],
            'styling': 'professional',
            'include_metadata': True
        }
    
    def _get_summary_template(self) -> Dict:
        """Get summary report template."""
        return {
            'name': 'summary',
            'sections': [
                'title_page',
                'data_summary',
                'key_insights',
                'conclusions'
            ],
            'styling': 'clean',
            'include_metadata': False
        }
    
    def _get_insights_template(self) -> Dict:
        """Get insights-only report template."""
        return {
            'name': 'insights_only',
            'sections': [
                'title_page',
                'automated_insights',
                'conclusions'
            ],
            'styling': 'minimal',
            'include_metadata': True
        }
    
    def add_custom_template(self, name: str, template: Dict):
        """Add a custom template."""
        self.templates[name] = template
        logger.info(f"Added custom template: {name}")


# Convenience functions for easy report generation
def generate_quick_report(data: pd.DataFrame, title: str = None, output_dir: str = 'outputs') -> Dict[str, Any]:
    """
    Generate a quick comprehensive report from data.
    
    Args:
        data: Input DataFrame
        title: Report title
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with report paths and metadata
    """
    generator = ReportGenerator()
    
    config = {
        'title': title or 'Quick Ecological Data Analysis',
        'output_dir': output_dir,
        'formats': ['pdf', 'html'],
        'template': 'comprehensive'
    }
    
    return generator.generate_report(data, config)


def generate_insights_report(data: pd.DataFrame, target_columns: List[str] = None, 
                           output_dir: str = 'outputs') -> Dict[str, Any]:
    """
    Generate a report focused on automated insights.
    
    Args:
        data: Input DataFrame
        target_columns: Specific columns to analyze
        output_dir: Output directory for reports
        
    Returns:
        Dictionary with report paths and metadata
    """
    generator = ReportGenerator()
    
    config = {
        'title': 'Automated Insights Report',
        'output_dir': output_dir,
        'formats': ['html'],
        'template': 'insights_only',
        'target_columns': target_columns,
        'max_insights': 15
    }
    
    return generator.generate_report(data, config)
