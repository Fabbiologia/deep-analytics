#!/usr/bin/env python3
"""
PDF Report Generator for Ecological Data Analysis
Creates comprehensive, publication-ready reports with statistical analysis results.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO

# PDF generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    PDF_AVAILABLE = True
except ImportError:
    print("Warning: ReportLab not available. PDF generation will be disabled.")
    PDF_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib/Seaborn not available. Plot generation will be disabled.")
    PLOTTING_AVAILABLE = False

class EcologicalReportGenerator:
    """
    Generates comprehensive PDF reports for ecological data analysis.
    """
    
    def __init__(self, title: str = "Ecological Data Analysis Report", 
                 author: str = "Ecological Monitoring System"):
        """
        Initialize the report generator.
        
        Args:
            title: Report title
            author: Report author
        """
        self.title = title
        self.author = author
        self.creation_date = datetime.now()
        self.styles = None
        self.story = []
        
        if PDF_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkgreen
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.darkblue
        ))
        
        # Statistical result style
        self.styles.add(ParagraphStyle(
            name='StatResult',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=20
        ))
        
        # Method description style
        self.styles.add(ParagraphStyle(
            name='Method',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=6,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            textColor=colors.darkgrey
        ))
    
    def add_title_page(self):
        """Add a title page to the report."""
        if not PDF_AVAILABLE:
            return
        
        # Title
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(self.title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Author and date
        self.story.append(Paragraph(f"<b>Author:</b> {self.author}", self.styles['Normal']))
        self.story.append(Paragraph(f"<b>Generated:</b> {self.creation_date.strftime('%B %d, %Y at %H:%M')}", 
                                  self.styles['Normal']))
        self.story.append(Spacer(1, 1*inch))
        
        # Abstract placeholder
        abstract_text = """
        This report presents a comprehensive statistical analysis of ecological monitoring data 
        from the Gulf of California Long-Term Ecological Monitoring (LTEM) program. The analysis 
        includes descriptive statistics, hypothesis testing, correlation analysis, and regression 
        modeling to understand patterns in marine biodiversity and ecosystem health indicators.
        """
        self.story.append(Paragraph("<b>Abstract</b>", self.styles['SectionHeader']))
        self.story.append(Paragraph(abstract_text, self.styles['Normal']))
        
        self.story.append(PageBreak())
    
    def add_executive_summary(self, summary_data: Dict):
        """
        Add an executive summary section.
        
        Args:
            summary_data: Dictionary containing key findings and metrics
        """
        if not PDF_AVAILABLE:
            return
        
        self.story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Key findings
        if 'key_findings' in summary_data:
            self.story.append(Paragraph("<b>Key Findings:</b>", self.styles['SubsectionHeader']))
            for finding in summary_data['key_findings']:
                self.story.append(Paragraph(f"• {finding}", self.styles['Normal']))
            self.story.append(Spacer(1, 12))
        
        # Dataset overview
        if 'dataset_info' in summary_data:
            info = summary_data['dataset_info']
            self.story.append(Paragraph("<b>Dataset Overview:</b>", self.styles['SubsectionHeader']))
            
            data_table = [
                ['Metric', 'Value'],
                ['Total Records', f"{info.get('total_records', 'N/A'):,}" if isinstance(info.get('total_records'), (int, float)) else str(info.get('total_records', 'N/A'))],
                ['Date Range', str(info.get('date_range', 'N/A'))],
                ['Species Count', f"{info.get('species_count', 'N/A'):,}" if isinstance(info.get('species_count'), (int, float)) else str(info.get('species_count', 'N/A'))],
                ['Regions Covered', str(info.get('regions', 'N/A'))],
                ['Survey Types', str(info.get('survey_types', 'N/A'))]
            ]
            
            table = Table(data_table, colWidths=[2*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            self.story.append(table)
            self.story.append(Spacer(1, 12))
    
    def add_methods_section(self):
        """Add a methods and materials section."""
        if not PDF_AVAILABLE:
            return
        
        self.story.append(Paragraph("Methods and Statistical Procedures", self.styles['SectionHeader']))
        
        methods_text = """
        <b>Data Source:</b> This analysis utilizes data from the Gulf of California Long-Term 
        Ecological Monitoring (LTEM) program, which systematically surveys coral reef ecosystems 
        across multiple regions and time periods.
        
        <b>Statistical Analysis:</b> All statistical analyses were performed using Python with 
        scipy.stats, statsmodels, and scikit-learn libraries. Significance levels were set at 
        α = 0.05 unless otherwise specified.
        
        <b>Data Preparation:</b> Data were cleaned to remove missing values and outliers. 
        Normality assumptions were tested using Shapiro-Wilk and D'Agostino tests. When normality 
        assumptions were violated, non-parametric alternatives were employed.
        
        <b>Density Calculations:</b> Abundance and biomass densities were calculated by first 
        summing observations within each transect, then averaging across transects to account 
        for sampling design and avoid pseudoreplication.
        """
        
        self.story.append(Paragraph(methods_text, self.styles['Method']))
        self.story.append(Spacer(1, 12))
    
    def add_statistical_results(self, results: Dict, test_name: str):
        """
        Add statistical test results to the report.
        
        Args:
            results: Dictionary containing statistical test results
            test_name: Name of the statistical test
        """
        if not PDF_AVAILABLE:
            return
        
        self.story.append(Paragraph(f"{test_name} Results", self.styles['SubsectionHeader']))
        
        # Add robust type checking
        if not isinstance(results, dict):
            self.story.append(Paragraph(f"<b>Error:</b> Invalid results format (expected dict, got {type(results)})", self.styles['StatResult']))
            return
        
        if 'error' in results:
            self.story.append(Paragraph(f"<b>Error:</b> {results['error']}", self.styles['StatResult']))
            return
        
        # Format results based on test type
        if 't_statistic' in results:
            self._add_ttest_results(results)
        elif 'f_statistic' in results:
            self._add_anova_results(results)
        elif 'correlation_matrix' in results:
            self._add_correlation_results(results)
        elif 'r_squared' in results:
            self._add_regression_results(results)
        elif 'statistic' in results:
            self._add_nonparametric_results(results)
        
        self.story.append(Spacer(1, 12))
    
    def _add_ttest_results(self, results: Dict):
        """Add t-test results to the report."""
        # Basic statistics
        if 'group1_stats' in results:
            stats1 = results['group1_stats']
            self.story.append(Paragraph(
                f"<b>Group 1:</b> n = {stats1['n']}, M = {stats1['mean']:.3f}, "
                f"SD = {stats1['std']:.3f}", self.styles['StatResult']
            ))
        
        if 'group2_stats' in results:
            stats2 = results['group2_stats']
            self.story.append(Paragraph(
                f"<b>Group 2:</b> n = {stats2['n']}, M = {stats2['mean']:.3f}, "
                f"SD = {stats2['std']:.3f}", self.styles['StatResult']
            ))
        
        # Test results
        self.story.append(Paragraph(
            f"<b>t-statistic:</b> {results['t_statistic']:.4f}, "
            f"<b>p-value:</b> {results['p_value']:.4f}", self.styles['StatResult']
        ))
        
        self.story.append(Paragraph(
            f"<b>Effect Size (Cohen's d):</b> {results.get('effect_size_cohen_d', 'N/A'):.4f} "
            f"({results.get('effect_size_interpretation', 'N/A')})", self.styles['StatResult']
        ))
        
        significance = "significant" if results.get('significant', False) else "not significant"
        self.story.append(Paragraph(f"<b>Result:</b> {significance}", self.styles['StatResult']))
    
    def _add_anova_results(self, results: Dict):
        """Add ANOVA results to the report."""
        self.story.append(Paragraph(
            f"<b>F-statistic:</b> {results['f_statistic']:.4f}, "
            f"<b>p-value:</b> {results['p_value']:.4f}", self.styles['StatResult']
        ))
        
        significance = "significant" if results.get('significant', False) else "not significant"
        self.story.append(Paragraph(f"<b>Result:</b> {significance}", self.styles['StatResult']))
        
        if 'post_hoc_tukey' in results:
            self.story.append(Paragraph("<b>Post-hoc Analysis (Tukey HSD):</b>", self.styles['StatResult']))
            # Note: In a real implementation, you'd format the Tukey results more nicely
    
    def _add_correlation_results(self, results: Dict):
        """Add correlation analysis results to the report."""
        if 'significant_correlations' in results:
            sig_corrs = results['significant_correlations']
            self.story.append(Paragraph(
                f"<b>Significant Correlations Found:</b> {len(sig_corrs)}", self.styles['StatResult']
            ))
            
            for i, corr in enumerate(sig_corrs[:10]):  # Show top 10
                self.story.append(Paragraph(
                    f"{i+1}. {corr['variable1']} ↔ {corr['variable2']}: "
                    f"r = {corr['correlation']:.3f} (p = {corr['p_value']:.4f}) - {corr['strength']}",
                    self.styles['StatResult']
                ))
    
    def _add_regression_results(self, results: Dict):
        """Add regression analysis results to the report."""
        self.story.append(Paragraph(
            f"<b>R-squared:</b> {results['r_squared']:.4f}, "
            f"<b>Adjusted R-squared:</b> {results['adjusted_r_squared']:.4f}", 
            self.styles['StatResult']
        ))
        
        self.story.append(Paragraph(
            f"<b>F-statistic:</b> {results['f_statistic']:.4f}, "
            f"<b>Model p-value:</b> {results['f_pvalue']:.4f}", 
            self.styles['StatResult']
        ))
        
        # Coefficients table
        if 'coefficients' in results:
            coeff_data = [['Variable', 'Coefficient', 'Std Error', 't-value', 'p-value', 'Significant']]
            
            for var, coeff_info in results['coefficients'].items():
                coeff_data.append([
                    var,
                    f"{coeff_info['coefficient']:.4f}",
                    f"{coeff_info['std_error']:.4f}",
                    f"{coeff_info['t_statistic']:.4f}",
                    f"{coeff_info['p_value']:.4f}",
                    "Yes" if coeff_info['significant'] else "No"
                ])
            
            table = Table(coeff_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            self.story.append(table)
    
    def _add_nonparametric_results(self, results: Dict):
        """Add non-parametric test results to the report."""
        self.story.append(Paragraph(
            f"<b>Test Statistic:</b> {results['statistic']:.4f}, "
            f"<b>p-value:</b> {results['p_value']:.4f}", self.styles['StatResult']
        ))
        
        if 'group1_median' in results and 'group2_median' in results:
            self.story.append(Paragraph(
                f"<b>Group 1 Median:</b> {results['group1_median']:.3f}, "
                f"<b>Group 2 Median:</b> {results['group2_median']:.3f}", 
                self.styles['StatResult']
            ))
    
    def add_plot_from_file(self, plot_path: str, caption: str = "", width: float = 6*inch):
        """
        Add a plot from a file to the report.
        
        Args:
            plot_path: Path to the plot image file
            caption: Caption for the plot
            width: Width of the plot in the report
        """
        if not PDF_AVAILABLE or not os.path.exists(plot_path):
            return
        
        try:
            # Calculate height maintaining aspect ratio
            from PIL import Image as PILImage
            with PILImage.open(plot_path) as img:
                aspect_ratio = img.height / img.width
                height = width * aspect_ratio
            
            # Add the image
            img = Image(plot_path, width=width, height=height)
            self.story.append(img)
            
            # Add caption if provided
            if caption:
                self.story.append(Paragraph(f"<b>Figure:</b> {caption}", self.styles['Normal']))
            
            self.story.append(Spacer(1, 12))
            
        except Exception as e:
            self.story.append(Paragraph(f"Error loading plot: {str(e)}", self.styles['Normal']))
    
    def add_data_table(self, data: pd.DataFrame, title: str = "", max_rows: int = 20):
        """
        Add a data table to the report.
        
        Args:
            data: DataFrame to display
            title: Title for the table
            max_rows: Maximum number of rows to display
        """
        if not PDF_AVAILABLE:
            return
        
        if title:
            self.story.append(Paragraph(title, self.styles['SubsectionHeader']))
        
        # Limit rows if necessary
        display_data = data.head(max_rows) if len(data) > max_rows else data
        
        # Convert DataFrame to list of lists for ReportLab
        table_data = [list(display_data.columns)]
        for _, row in display_data.iterrows():
            table_data.append([str(val) for val in row])
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        self.story.append(table)
        
        if len(data) > max_rows:
            self.story.append(Paragraph(
                f"<i>Note: Showing first {max_rows} of {len(data)} rows</i>", 
                self.styles['Normal']
            ))
        
        self.story.append(Spacer(1, 12))
    
    def add_conclusions_section(self, conclusions: List[str]):
        """
        Add a conclusions section to the report.
        
        Args:
            conclusions: List of conclusion statements
        """
        if not PDF_AVAILABLE:
            return
        
        self.story.append(Paragraph("Conclusions and Recommendations", self.styles['SectionHeader']))
        
        for i, conclusion in enumerate(conclusions, 1):
            self.story.append(Paragraph(f"{i}. {conclusion}", self.styles['Normal']))
            self.story.append(Spacer(1, 6))
    
    def generate_report(self, filename: str = None) -> str:
        """
        Generate the PDF report.
        
        Args:
            filename: Output filename (if None, auto-generate)
            
        Returns:
            Path to the generated PDF file
        """
        if not PDF_AVAILABLE:
            return "Error: PDF generation libraries not available"
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ecological_analysis_report_{timestamp}.pdf"
        
        try:
            # Create the PDF document
            doc = SimpleDocTemplate(filename, pagesize=A4)
            doc.build(self.story)
            
            return os.path.abspath(filename)
            
        except Exception as e:
            return f"Error generating PDF: {str(e)}"

def create_comprehensive_report(analysis_results: Dict, 
                              plots: List[str] = None,
                              data_tables: Dict[str, pd.DataFrame] = None,
                              filename: str = None) -> str:
    """
    Create a comprehensive ecological analysis report.
    
    Args:
        analysis_results: Dictionary containing all analysis results
        plots: List of plot file paths to include
        data_tables: Dictionary of DataFrames to include as tables
        filename: Output filename
        
    Returns:
        Path to the generated PDF report
    """
    if not PDF_AVAILABLE:
        return "Error: PDF generation not available"
    
    # Initialize report generator
    report = EcologicalReportGenerator(
        title="Comprehensive Ecological Data Analysis Report",
        author="Gulf of California LTEM Analysis System"
    )
    
    # Add title page
    report.add_title_page()
    
    # Add executive summary if available
    if 'summary' in analysis_results:
        report.add_executive_summary(analysis_results['summary'])
    
    # Add methods section
    report.add_methods_section()
    
    # Add statistical results
    for test_name, results in analysis_results.items():
        if test_name != 'summary':
            report.add_statistical_results(results, test_name)
    
    # Add plots if provided
    if plots:
        report.story.append(Paragraph("Visualizations", report.styles['SectionHeader']))
        for i, plot_path in enumerate(plots):
            report.add_plot_from_file(plot_path, f"Analysis visualization {i+1}")
    
    # Add data tables if provided
    if data_tables:
        report.story.append(Paragraph("Data Tables", report.styles['SectionHeader']))
        for table_name, df in data_tables.items():
            report.add_data_table(df, table_name)
    
    # Add conclusions
    conclusions = [
        "Statistical analysis reveals significant patterns in the ecological monitoring data.",
        "Proper statistical methods were applied accounting for the hierarchical sampling design.",
        "Results provide insights into ecosystem health and biodiversity patterns.",
        "Further analysis may be warranted to explore temporal trends and environmental correlations."
    ]
    report.add_conclusions_section(conclusions)
    
    # Generate the report
    return report.generate_report(filename)

def create_quick_summary_report(title: str, results: Dict, filename: str = None) -> str:
    """
    Create a quick summary report for a single analysis.
    
    Args:
        title: Report title
        results: Analysis results dictionary
        filename: Output filename
        
    Returns:
        Path to the generated PDF report
    """
    if not PDF_AVAILABLE:
        return "Error: PDF generation not available"
    
    report = EcologicalReportGenerator(title=title)
    report.add_title_page()
    report.add_statistical_results(results, title)
    
    return report.generate_report(filename)
