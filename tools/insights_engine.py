#!/usr/bin/env python3
"""
Automated Insights Engine for Ecological Data Analysis
Discovers patterns, trends, and anomalies in ecological monitoring data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightsEngine:
    """Main engine for discovering automated insights in ecological data."""
    
    def __init__(self):
        """Initialize the insights engine with all analyzer components."""
        self.analyzers = {
            "trend": TrendAnalyzer(),
            "anomaly": AnomalyAnalyzer(), 
            "correlation": CorrelationAnalyzer(),
            "trophic_shift": TrophicStructureShiftAnalyzer(),
            "significant_difference": SignificanceTesterAnalyzer()
        }
        self.narrative_generator = NarrativeGenerator()
        self.insight_prioritizer = InsightPrioritizer()
        
    def discover_insights(self, df: pd.DataFrame, target_columns: List[str] = None, 
                         context: Dict = None) -> List[Dict]:
        """
        Discover insights in the provided dataframe.
        
        Args:
            df: Input DataFrame
            target_columns: Specific columns to focus on (optional)
            context: Additional context about the analysis
            
        Returns:
            List of insight objects sorted by relevance/significance
        """
        logger.info(f"Starting insights discovery on DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        if context is None:
            context = {}
            
        all_insights = []
        
        # Apply each analyzer to discover insights
        for analyzer_name, analyzer in self.analyzers.items():
            try:
                logger.info(f"Running {analyzer_name} analyzer...")
                insights = analyzer.analyze(df, target_columns, context)
                all_insights.extend(insights)
                logger.info(f"{analyzer_name} analyzer found {len(insights)} insights")
            except Exception as e:
                logger.error(f"Error in {analyzer_name} analyzer: {str(e)}")
                
        logger.info(f"Total insights discovered: {len(all_insights)}")
        
        # Generate narratives for each insight
        for insight in all_insights:
            try:
                insight["narrative"] = self.narrative_generator.generate_narrative(insight)
            except Exception as e:
                logger.error(f"Error generating narrative for insight: {str(e)}")
                insight["narrative"] = f"Analysis found: {insight.get('description', 'Pattern detected')}"
                
        # Prioritize insights
        prioritized_insights = self.insight_prioritizer.prioritize(all_insights)
        
        logger.info(f"Returning {len(prioritized_insights)} prioritized insights")
        return prioritized_insights


class TrendAnalyzer:
    """Analyzes temporal trends in the data."""
    
    def analyze(self, df: pd.DataFrame, target_columns: List[str] = None, 
                context: Dict = None) -> List[Dict]:
        """Analyze temporal trends in the data."""
        insights = []
        
        # Check if we have temporal columns
        time_columns = ['Year', 'Date', 'TIME', 'year']
        time_col = None
        for col in time_columns:
            if col in df.columns:
                time_col = col
                break
                
        if time_col is None:
            return insights
            
        # Focus on numeric columns for trend analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_columns:
            numeric_cols = [col for col in numeric_cols if col in target_columns]
            
        # Remove time column from analysis
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
            
        # Analyze trends for each numeric column
        for col in numeric_cols:
            try:
                trend_insight = self._analyze_column_trend(df, col, time_col)
                if trend_insight:
                    insights.append(trend_insight)
            except Exception as e:
                logger.error(f"Error analyzing trend for {col}: {str(e)}")
                
        return insights
        
    def _analyze_column_trend(self, df: pd.DataFrame, col: str, time_col: str) -> Optional[Dict]:
        """Analyze trend for a specific column over time."""
        # Group by time and calculate mean values
        grouped = df.groupby(time_col)[col].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 3]  # Need at least 3 data points
        
        if len(grouped) < 3:
            return None
            
        # Perform linear regression to detect trend
        x = grouped[time_col].values
        y = grouped['mean'].values
        
        # Check for sufficient variance
        if np.std(y) < 0.001:
            return None
            
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Only report significant trends
            if p_value <= 0.05 and abs(r_value) >= 0.5:
                trend_direction = "increasing" if slope > 0 else "decreasing"
                
                return {
                    "type": "temporal_trend",
                    "variable": col,
                    "trend_direction": trend_direction,
                    "slope": slope,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "time_range": [int(x.min()), int(x.max())],
                    "description": f"{col} shows a {trend_direction} trend over time (R² = {r_value**2:.3f}, p = {p_value:.3f})",
                    "score": abs(r_value) * (1 - p_value),
                    "statistical_test": "linear_regression"
                }
        except Exception:
            return None


class AnomalyAnalyzer:
    """Detects anomalous values in the data."""
    
    def analyze(self, df: pd.DataFrame, target_columns: List[str] = None, 
                context: Dict = None) -> List[Dict]:
        """Detect anomalies in the data using statistical methods."""
        insights = []
        
        # Focus on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_columns:
            numeric_cols = [col for col in numeric_cols if col in target_columns]
            
        for col in numeric_cols:
            try:
                anomaly_insights = self._detect_column_anomalies(df, col)
                insights.extend(anomaly_insights)
            except Exception as e:
                logger.error(f"Error detecting anomalies for {col}: {str(e)}")
                
        return insights
        
    def _detect_column_anomalies(self, df: pd.DataFrame, col: str) -> List[Dict]:
        """Detect anomalies in a specific column using IQR method."""
        insights = []
        values = df[col].dropna()
        
        if len(values) < 10:
            return insights
            
        # IQR-based outlier detection
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            outlier_percentage = len(outliers) / len(values) * 100
            
            if len(outliers) > 0 and outlier_percentage < 20:
                extreme_values = outliers.sort_values()
                
                insight = {
                    "type": "anomaly",
                    "method": "IQR",
                    "variable": col,
                    "outlier_count": len(outliers),
                    "outlier_percentage": outlier_percentage,
                    "extreme_values": [float(extreme_values.iloc[0]), float(extreme_values.iloc[-1])],
                    "normal_range": [float(lower_bound), float(upper_bound)],
                    "description": f"{col} contains {len(outliers)} outliers ({outlier_percentage:.1f}% of values)",
                    "score": min(outlier_percentage / 5, 1.0),
                    "statistical_test": "IQR_outlier_detection"
                }
                insights.append(insight)
                
        return insights


class CorrelationAnalyzer:
    """Analyzes correlations between variables."""
    
    def analyze(self, df: pd.DataFrame, target_columns: List[str] = None, 
                context: Dict = None) -> List[Dict]:
        """Analyze correlations between numeric variables."""
        insights = []
        
        # Focus on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_columns:
            numeric_cols = [col for col in numeric_cols if col in target_columns]
            
        # Need at least 2 columns for correlation
        if len(numeric_cols) < 2:
            return insights
            
        # Calculate correlations for all pairs
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                try:
                    correlation_insight = self._analyze_correlation(df, col1, col2)
                    if correlation_insight:
                        insights.append(correlation_insight)
                except Exception as e:
                    logger.error(f"Error analyzing correlation between {col1} and {col2}: {str(e)}")
                    
        return insights
        
    def _analyze_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> Optional[Dict]:
        """Analyze correlation between two columns."""
        # Get clean data for both columns
        clean_data = df[[col1, col2]].dropna()
        
        if len(clean_data) < 10:
            return None
            
        x, y = clean_data[col1], clean_data[col2]
        
        # Check for sufficient variance
        if np.std(x) < 0.001 or np.std(y) < 0.001:
            return None
            
        try:
            # Calculate Pearson correlation
            corr_coef, p_value = pearsonr(x, y)
            
            # Only report significant correlations
            if p_value <= 0.05 and abs(corr_coef) >= 0.3:
                strength = "strong" if abs(corr_coef) >= 0.7 else "moderate"
                direction = "positive" if corr_coef > 0 else "negative"
                
                return {
                    "type": "correlation",
                    "variables": [col1, col2],
                    "correlation_coefficient": corr_coef,
                    "p_value": p_value,
                    "strength": strength,
                    "direction": direction,
                    "sample_size": len(clean_data),
                    "description": f"{strength.capitalize()} {direction} correlation between {col1} and {col2} (r = {corr_coef:.3f}, p = {p_value:.3f})",
                    "score": abs(corr_coef) * (1 - p_value),
                    "statistical_test": "pearson_correlation"
                }
        except Exception:
            return None


class TrophicStructureShiftAnalyzer:
    """Analyzes shifts in trophic structure over time or between regions."""
    
    def analyze(self, df: pd.DataFrame, target_columns: List[str] = None, 
                context: Dict = None) -> List[Dict]:
        """Analyze shifts in trophic structure."""
        insights = []
        
        # Check for necessary columns
        required_columns = ["TrophicLevel", "Biomass", "Year"]
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) < 3:
            return insights
            
        # Work with fish data if Label column exists
        working_df = df.copy()
        if "Label" in working_df.columns:
            working_df = working_df[working_df["Label"] == "PEC"].copy()
            
        if len(working_df) < 20:
            return insights
            
        # Analyze temporal shifts
        time_shifts = self._detect_time_shifts(working_df)
        insights.extend(time_shifts)
            
        return insights
        
    def _detect_time_shifts(self, df: pd.DataFrame) -> List[Dict]:
        """Detect shifts in trophic structure over time."""
        insights = []
        
        try:
            # Pivot data to get Year x TrophicLevel matrix
            pivot_df = pd.pivot_table(
                df, values="Biomass", index="Year", columns="TrophicLevel", aggfunc="sum"
            ).fillna(0)
        except Exception:
            return insights
            
        if len(pivot_df) < 2:
            return insights
            
        # Calculate relative contribution
        row_sums = pivot_df.sum(axis=1)
        row_sums = row_sums.replace(0, np.nan)
        relative_df = pivot_df.div(row_sums, axis=0) * 100
        relative_df = relative_df.fillna(0)
        
        # Find consecutive years with significant changes
        years = sorted(relative_df.index.unique())
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            if curr_year - prev_year > 3:
                continue
                
            try:
                changes = relative_df.loc[curr_year] - relative_df.loc[prev_year]
                sig_changes = changes[abs(changes) >= 5]
                
                if not sig_changes.empty:
                    for tl, change in sig_changes.items():
                        insights.append({
                            "type": "trophic_structure_shift",
                            "subtype": "temporal",
                            "description": f"Trophic level {tl} showed a {abs(change):.1f}% " + 
                                          f"{'increase' if change > 0 else 'decrease'} from {prev_year} to {curr_year}",
                            "from_year": prev_year,
                            "to_year": curr_year,
                            "trophic_level": float(tl),
                            "change_percent": float(change),
                            "score": abs(change) / 10,
                            "statistical_test": "percent_change"
                        })
            except Exception:
                continue
                
        return insights


class SignificanceTesterAnalyzer:
    """Performs statistical significance testing between groups."""
    
    def analyze(self, df: pd.DataFrame, target_columns: List[str] = None, 
                context: Dict = None) -> List[Dict]:
        """Perform statistical significance tests between groups."""
        insights = []
        
        # Look for categorical grouping variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_columns:
            numeric_cols = [col for col in numeric_cols if col in target_columns]
            
        # Test differences between groups
        for cat_col in categorical_cols:
            if cat_col in ['Label', 'Region', 'Site', 'Species']:
                for num_col in numeric_cols:
                    try:
                        significance_insight = self._test_group_differences(df, cat_col, num_col)
                        if significance_insight:
                            insights.append(significance_insight)
                    except Exception as e:
                        logger.error(f"Error testing significance between {cat_col} groups for {num_col}: {str(e)}")
                        
        return insights
        
    def _test_group_differences(self, df: pd.DataFrame, group_col: str, value_col: str) -> Optional[Dict]:
        """Test for significant differences between groups."""
        # Clean data
        clean_data = df[[group_col, value_col]].dropna()
        
        if len(clean_data) < 10:
            return None
            
        # Get groups
        groups = clean_data[group_col].unique()
        if len(groups) < 2:
            return None
            
        # Prepare data for testing
        group_data = [clean_data[clean_data[group_col] == group][value_col].values 
                     for group in groups]
        
        # Filter out groups with too few observations
        group_data = [group for group in group_data if len(group) >= 5]
        if len(group_data) < 2:
            return None
            
        try:
            if len(group_data) == 2:
                # Two groups: use Mann-Whitney U test
                stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                test_name = "Mann-Whitney U"
            else:
                # Multiple groups: use Kruskal-Wallis test
                stat, p_value = kruskal(*group_data)
                test_name = "Kruskal-Wallis"
                
            # Only report significant results
            if p_value <= 0.05:
                # Calculate group statistics
                group_stats = []
                for i, group in enumerate(groups[:len(group_data)]):
                    data = group_data[i]
                    group_stats.append({
                        'group': group,
                        'median': float(np.median(data)),
                        'mean': float(np.mean(data)),
                        'n': len(data)
                    })
                    
                return {
                    "type": "significant_difference",
                    "grouping_variable": group_col,
                    "response_variable": value_col,
                    "test_statistic": float(stat),
                    "p_value": float(p_value),
                    "test_name": test_name,
                    "group_statistics": group_stats,
                    "description": f"Significant difference in {value_col} between {group_col} groups (p = {p_value:.3f})",
                    "score": (1 - p_value),
                    "statistical_test": test_name.lower().replace('-', '_').replace(' ', '_')
                }
        except Exception:
            return None


class NarrativeGenerator:
    """Generates human-readable narratives for insights."""
    
    def generate_narrative(self, insight: Dict) -> str:
        """Generate a narrative description for an insight."""
        try:
            insight_type = insight.get("type", "unknown")
            
            if insight_type == "temporal_trend":
                return self._generate_trend_narrative(insight)
            elif insight_type == "correlation":
                return self._generate_correlation_narrative(insight)
            elif insight_type == "anomaly":
                return self._generate_anomaly_narrative(insight)
            elif insight_type == "trophic_structure_shift":
                return self._generate_trophic_narrative(insight)
            elif insight_type == "significant_difference":
                return self._generate_significance_narrative(insight)
            else:
                return insight.get("description", "Pattern detected in the data.")
        except Exception as e:
            logger.error(f"Error generating narrative: {str(e)}")
            return insight.get("description", "Analysis found interesting patterns in the data.")
            
    def _generate_trend_narrative(self, insight: Dict) -> str:
        """Generate narrative for trend insights."""
        variable = insight.get("variable", "Variable")
        direction = insight.get("trend_direction", "changing")
        r_squared = insight.get("r_squared", 0)
        time_range = insight.get("time_range", [])
        
        strength = "strong" if r_squared >= 0.7 else "moderate" if r_squared >= 0.3 else "weak"
        
        narrative = f"Analysis reveals a {strength} {direction} trend in {variable}"
        if time_range and len(time_range) == 2:
            narrative += f" from {time_range[0]} to {time_range[1]}"
        narrative += f". This trend explains {r_squared*100:.1f}% of the variation over time."
            
        return narrative
        
    def _generate_correlation_narrative(self, insight: Dict) -> str:
        """Generate narrative for correlation insights."""
        variables = insight.get("variables", ["Variable 1", "Variable 2"])
        strength = insight.get("strength", "moderate")
        direction = insight.get("direction", "positive")
        corr_coef = insight.get("correlation_coefficient", 0)
        
        narrative = f"A {strength} {direction} correlation exists between {variables[0]} and {variables[1]} "
        narrative += f"(correlation coefficient = {corr_coef:.3f}). "
        
        if direction == "positive":
            narrative += f"As {variables[0]} increases, {variables[1]} tends to increase as well."
        else:
            narrative += f"As {variables[0]} increases, {variables[1]} tends to decrease."
            
        return narrative
        
    def _generate_anomaly_narrative(self, insight: Dict) -> str:
        """Generate narrative for anomaly insights."""
        variable = insight.get("variable", "Variable")
        count = insight.get("outlier_count", 0)
        percentage = insight.get("outlier_percentage", 0)
        
        narrative = f"Unusual values detected in {variable}: {count} observations ({percentage:.1f}% of the data) "
        narrative += "fall outside the expected range. These may represent measurement errors or genuine ecological extremes."
            
        return narrative
        
    def _generate_trophic_narrative(self, insight: Dict) -> str:
        """Generate narrative for trophic structure insights."""
        return insight.get("description", "Changes in trophic structure detected.")
        
    def _generate_significance_narrative(self, insight: Dict) -> str:
        """Generate narrative for significance test insights."""
        group_var = insight.get("grouping_variable", "groups")
        response_var = insight.get("response_variable", "variable")
        p_value = insight.get("p_value", 0)
        
        narrative = f"Statistical analysis reveals significant differences in {response_var} between {group_var} categories "
        narrative += f"(p = {p_value:.3f}). This suggests that {group_var} is an important factor influencing {response_var}."
        
        return narrative


class InsightPrioritizer:
    """Prioritizes insights based on statistical significance and ecological relevance."""
    
    def prioritize(self, insights: List[Dict]) -> List[Dict]:
        """
        Prioritize insights based on their scores and types.
        
        Args:
            insights: List of insight dictionaries
            
        Returns:
            Sorted list of insights by priority
        """
        if not insights:
            return insights
            
        # Define type priorities (higher = more important)
        type_priorities = {
            "trophic_structure_shift": 1.0,
            "temporal_trend": 0.9,
            "significant_difference": 0.8,
            "correlation": 0.7,
            "anomaly": 0.6
        }
        
        # Calculate final priority scores
        for insight in insights:
            base_score = insight.get("score", 0.5)
            type_priority = type_priorities.get(insight.get("type", "unknown"), 0.5)
            
            # Boost score for statistically significant results
            p_value = insight.get("p_value")
            if p_value is not None and p_value <= 0.001:
                significance_boost = 0.3
            elif p_value is not None and p_value <= 0.01:
                significance_boost = 0.2
            elif p_value is not None and p_value <= 0.05:
                significance_boost = 0.1
            else:
                significance_boost = 0.0
                
            insight["priority_score"] = base_score * type_priority + significance_boost
            
        # Sort by priority score (descending)
        return sorted(insights, key=lambda x: x.get("priority_score", 0), reverse=True)
