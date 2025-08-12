#!/usr/bin/env python3
"""
Quick test script to verify ANOVA functionality and get coral coverage analysis results.
"""

import sys
import os
import pandas as pd
from sqlalchemy import create_engine

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our statistical analysis functions
try:
    from statistical_analysis import perform_anova, format_statistical_results
    print("✅ Statistical analysis modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing statistical modules: {e}")
    sys.exit(1)

# Database connection (you may need to adjust this based on your .env file)
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get database connection details from environment
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER') 
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME')
    
    if not all([db_host, db_user, db_password, db_name]):
        print("❌ Missing database connection details in .env file")
        sys.exit(1)
    
    # Create database connection
    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(connection_string)
    print("✅ Database connection established")
    
except Exception as e:
    print(f"❌ Database connection error: {e}")
    sys.exit(1)

def test_coral_coverage_anova():
    """Test ANOVA analysis for coral coverage across regions."""
    print("\n🔬 Testing ANOVA: Coral coverage differences across regions")
    print("=" * 60)
    
    # Query to get coral coverage data by region
    query = """
    SELECT 
        Region,
        bleaching_coverage as coral_coverage
    FROM ltem_historical_database 
    WHERE bleaching_coverage IS NOT NULL 
        AND bleaching_coverage >= 0
        AND Region IS NOT NULL
    LIMIT 10000
    """
    
    try:
        # Get the data
        print("📊 Executing query...")
        data = pd.read_sql(query, engine)
        print(f"✅ Retrieved {len(data)} records")
        
        if data.empty:
            print("❌ No data retrieved")
            return
        
        # Show data summary
        print(f"\n📈 Data Summary:")
        print(f"Regions: {data['Region'].nunique()} unique regions")
        print(f"Region counts: {data['Region'].value_counts().head()}")
        print(f"Coverage range: {data['coral_coverage'].min():.2f} - {data['coral_coverage'].max():.2f}")
        
        # Perform ANOVA
        print("\n🧮 Performing ANOVA analysis...")
        results = perform_anova(
            data=data, 
            dependent_var='coral_coverage', 
            independent_vars=['Region'], 
            alpha=0.05
        )
        
        # Format and display results
        print("\n📋 ANOVA Results:")
        print("=" * 60)
        formatted_results = format_statistical_results(results, "One-Way ANOVA")
        print(formatted_results)
        
        # Also print raw results for debugging
        print("\n🔍 Raw Results (for debugging):")
        print(f"Result type: {type(results)}")
        print(f"Result keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        if isinstance(results, dict):
            for key, value in results.items():
                print(f"  {key}: {type(value)} = {value}")
        
    except Exception as e:
        print(f"❌ Error in ANOVA test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coral_coverage_anova()
