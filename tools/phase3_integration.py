#!/usr/bin/env python3
"""
Phase 3 Integration Script: Automated Insights & Reporting
Demonstrates the complete Phase 3 functionality with insights discovery and report generation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.getcwd())

from insights_engine import InsightsEngine
from report_generator import ReportGenerator, generate_quick_report, generate_insights_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase3Demo:
    """
    Demonstration class for Phase 3 automated insights and reporting functionality.
    """
    
    def __init__(self):
        """Initialize the Phase 3 demo system."""
        self.insights_engine = InsightsEngine()
        self.report_generator = ReportGenerator()
        logger.info("Phase 3 Demo System initialized")
    
    def run_comprehensive_demo(self, data_file: str = None) -> dict:
        """
        Run a comprehensive demonstration of Phase 3 capabilities.
        
        Args:
            data_file: Optional path to data file (CSV format)
            
        Returns:
            Dictionary containing demo results and report paths
        """
        logger.info("Starting Phase 3 comprehensive demonstration")
        
        # Load or generate sample data
        if data_file and os.path.exists(data_file):
            logger.info(f"Loading data from {data_file}")
            data = pd.read_csv(data_file)
        else:
            logger.info("Generating sample ecological data for demonstration")
            data = self._generate_sample_data()
        
        logger.info(f"Working with dataset: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Step 1: Discover insights
        logger.info("Step 1: Discovering automated insights...")
        insights = self.insights_engine.discover_insights(
            data,
            target_columns=['Biomass', 'TrophicLevel', 'CPUE'] if 'Biomass' in data.columns else None,
            context={'analysis_type': 'ecological_monitoring', 'ecosystem': 'gulf_of_california'}
        )
        
        logger.info(f"Discovered {len(insights)} insights")
        for i, insight in enumerate(insights[:3], 1):  # Show top 3
            logger.info(f"  {i}. {insight.get('type', 'unknown')}: {insight.get('description', 'N/A')}")
        
        # Step 2: Generate comprehensive report
        logger.info("Step 2: Generating comprehensive reports...")
        
        # Create outputs directory
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        # Generate reports using different methods
        results = {}
        
        # Method 1: Full report generator
        try:
            config = {
                'title': 'Phase 3 Demo: Automated Ecological Insights',
                'author': 'Gulf of California LTEM Analysis System - Phase 3',
                'formats': ['pdf', 'html'],
                'template': 'comprehensive',
                'output_dir': str(output_dir),
                'include_insights': True,
                'max_insights': 10
            }
            
            full_report_results = self.report_generator.generate_report(data, config)
            results['comprehensive_reports'] = full_report_results
            logger.info("Comprehensive reports generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating comprehensive reports: {str(e)}")
            results['comprehensive_reports'] = {'error': str(e)}
        
        # Method 2: Quick report function
        try:
            quick_report_results = generate_quick_report(
                data, 
                title="Quick Phase 3 Analysis Demo",
                output_dir=str(output_dir)
            )
            results['quick_reports'] = quick_report_results
            logger.info("Quick reports generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating quick reports: {str(e)}")
            results['quick_reports'] = {'error': str(e)}
        
        # Method 3: Insights-focused report
        try:
            insights_report_results = generate_insights_report(
                data,
                target_columns=['Biomass', 'TrophicLevel'] if 'Biomass' in data.columns else None,
                output_dir=str(output_dir)
            )
            results['insights_reports'] = insights_report_results
            logger.info("Insights-focused reports generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating insights reports: {str(e)}")
            results['insights_reports'] = {'error': str(e)}
        
        # Step 3: Summary and results
        logger.info("Phase 3 demonstration completed!")
        
        # Create summary
        summary = {
            'data_summary': {
                'shape': data.shape,
                'columns': list(data.columns),
                'has_temporal_data': any(col in data.columns for col in ['Year', 'Date', 'TIME']),
                'has_trophic_data': 'TrophicLevel' in data.columns,
                'has_biomass_data': 'Biomass' in data.columns
            },
            'insights_summary': {
                'total_insights': len(insights),
                'insight_types': list(set(insight.get('type', 'unknown') for insight in insights)),
                'top_insights': [
                    {
                        'type': insight.get('type', 'unknown'),
                        'description': insight.get('description', 'N/A'),
                        'score': insight.get('score', 0)
                    }
                    for insight in insights[:5]
                ]
            },
            'reports_generated': results
        }
        
        # Log summary
        self._log_demo_summary(summary)
        
        return summary
    
    def test_insights_engine_only(self, data_file: str = None) -> list:
        """
        Test only the insights engine component.
        
        Args:
            data_file: Optional path to data file
            
        Returns:
            List of discovered insights
        """
        logger.info("Testing insights engine independently...")
        
        # Load or generate data
        if data_file and os.path.exists(data_file):
            data = pd.read_csv(data_file)
        else:
            data = self._generate_sample_data()
        
        # Test each analyzer independently
        insights_by_analyzer = {}
        
        for analyzer_name, analyzer in self.insights_engine.analyzers.items():
            try:
                logger.info(f"Testing {analyzer_name} analyzer...")
                analyzer_insights = analyzer.analyze(data)
                insights_by_analyzer[analyzer_name] = analyzer_insights
                logger.info(f"  {analyzer_name}: {len(analyzer_insights)} insights")
            except Exception as e:
                logger.error(f"  {analyzer_name}: Error - {str(e)}")
                insights_by_analyzer[analyzer_name] = []
        
        # Test full insights discovery
        all_insights = self.insights_engine.discover_insights(data)
        
        logger.info(f"Total insights from full engine: {len(all_insights)}")
        
        return all_insights
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample ecological data for demonstration."""
        np.random.seed(42)  # For reproducible results
        
        # Create sample data structure
        years = list(range(2000, 2021))
        regions = ['North', 'Central', 'South']
        species = ['Species_A', 'Species_B', 'Species_C', 'Species_D']
        trophic_levels = [2.0, 2.5, 3.0, 3.5, 4.0]
        
        data = []
        
        for year in years:
            for region in regions:
                for species in species:
                    # Simulate some temporal trends and regional differences
                    base_biomass = np.random.exponential(50)
                    
                    # Add temporal trend
                    trend_factor = 1 + (year - 2000) * 0.02 * np.random.normal(0, 0.1)
                    
                    # Add regional differences
                    if region == 'North':
                        region_factor = 1.2
                    elif region == 'Central':
                        region_factor = 1.0
                    else:
                        region_factor = 0.8
                    
                    biomass = base_biomass * trend_factor * region_factor
                    
                    # Add some outliers occasionally
                    if np.random.random() < 0.05:  # 5% chance of outlier
                        biomass *= np.random.choice([0.1, 5.0])  # Very low or very high
                    
                    trophic_level = np.random.choice(trophic_levels)
                    cpue = biomass * np.random.lognormal(0, 0.3)  # CPUE correlated with biomass
                    
                    data.append({
                        'Year': year,
                        'Region': region,
                        'Species': species,
                        'TrophicLevel': trophic_level,
                        'Biomass': max(0, biomass),  # Ensure non-negative
                        'CPUE': max(0, cpue),
                        'Label': 'PEC',  # Fish label for trophic analysis
                        'Temperature': 20 + np.random.normal(0, 3),  # Additional variable
                        'Depth': np.random.uniform(10, 200)  # Additional variable
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated sample data with {len(df)} records")
        return df
    
    def _log_demo_summary(self, summary: dict):
        """Log a comprehensive summary of the demo results."""
        logger.info("="*60)
        logger.info("PHASE 3 DEMONSTRATION SUMMARY")
        logger.info("="*60)
        
        # Data summary
        data_info = summary['data_summary']
        logger.info(f"Data Shape: {data_info['shape']}")
        logger.info(f"Temporal Data: {data_info['has_temporal_data']}")
        logger.info(f"Trophic Data: {data_info['has_trophic_data']}")
        logger.info(f"Biomass Data: {data_info['has_biomass_data']}")
        
        # Insights summary
        insights_info = summary['insights_summary']
        logger.info(f"Total Insights: {insights_info['total_insights']}")
        logger.info(f"Insight Types: {', '.join(insights_info['insight_types'])}")
        
        # Top insights
        logger.info("Top Insights:")
        for i, insight in enumerate(insights_info['top_insights'], 1):
            logger.info(f"  {i}. [{insight['type']}] {insight['description']} (Score: {insight['score']:.3f})")
        
        # Reports generated
        reports_info = summary['reports_generated']
        logger.info("Reports Generated:")
        for report_type, report_data in reports_info.items():
            if 'error' in report_data:
                logger.error(f"  {report_type}: Error - {report_data['error']}")
            else:
                logger.info(f"  {report_type}: Success")
                if 'pdf' in report_data:
                    logger.info(f"    PDF: {report_data['pdf']}")
                if 'html' in report_data:
                    logger.info(f"    HTML: {report_data['html']}")
        
        logger.info("="*60)


def main():
    """Main function to run Phase 3 demonstration."""
    print("Phase 3: Automated Insights & Reporting Demo")
    print("=" * 50)
    
    # Initialize demo system
    demo = Phase3Demo()
    
    # Check if data file is provided as argument
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run comprehensive demo
    try:
        results = demo.run_comprehensive_demo(data_file)
        print("\nDemo completed successfully!")
        print(f"Generated {len(results['insights_summary']['top_insights'])} insights")
        
        # Print report locations
        print("\nGenerated Reports:")
        for report_type, report_data in results['reports_generated'].items():
            if 'error' not in report_data:
                print(f"  {report_type.replace('_', ' ').title()}:")
                if 'pdf' in report_data:
                    print(f"    PDF: {report_data['pdf']}")
                if 'html' in report_data:
                    print(f"    HTML: {report_data['html']}")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"Demo failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
