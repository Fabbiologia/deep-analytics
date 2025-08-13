#!/usr/bin/env python3
"""
Advanced Statistical Analysis Module for Ecological Data
Provides comprehensive statistical testing and analysis capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
try:
    from scipy import stats
    from scipy.stats import normaltest, levene, mannwhitneyu, kruskal
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.power import ttest_power
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    STATS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Statistical libraries not available: {e}")
    STATS_AVAILABLE = False

def check_normality(data: pd.Series, alpha: float = 0.05) -> Dict:
    """
    Check normality of data using multiple tests.
    
    Args:
        data: Pandas Series of numerical data
        alpha: Significance level for tests
        
    Returns:
        Dictionary with normality test results
    """
    if not STATS_AVAILABLE:
        return {"error": "Statistical libraries not available"}
    
    # Remove NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 3:
        return {"error": "Insufficient data for normality testing"}
    
    results = {
        "sample_size": len(clean_data),
        "mean": clean_data.mean(),
        "std": clean_data.std(),
        "skewness": stats.skew(clean_data),
        "kurtosis": stats.kurtosis(clean_data)
    }
    
    # Shapiro-Wilk test (best for small samples)
    if len(clean_data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(clean_data)
        results["shapiro_wilk"] = {
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "is_normal": shapiro_p > alpha
        }
    
    # D'Agostino's normality test
    if len(clean_data) >= 8:
        dagostino_stat, dagostino_p = normaltest(clean_data)
        results["dagostino"] = {
            "statistic": dagostino_stat,
            "p_value": dagostino_p,
            "is_normal": dagostino_p > alpha
        }
    
    return results

def perform_ttest(group1: pd.Series, group2: pd.Series = None, 
                  test_type: str = "two_sample", mu: float = 0, 
                  alpha: float = 0.05) -> Dict:
    """
    Perform various t-tests.
    
    Args:
        group1: First group data
        group2: Second group data (for two-sample tests)
        test_type: "one_sample", "two_sample", or "paired"
        mu: Hypothesized mean for one-sample test
        alpha: Significance level
        
    Returns:
        Dictionary with t-test results
    """
    if not STATS_AVAILABLE:
        return {"error": "Statistical libraries not available"}
    
    # Clean data
    group1_clean = group1.dropna()
    
    if len(group1_clean) < 2:
        return {"error": "Insufficient data in group 1"}
    
    results = {
        "test_type": test_type,
        "alpha": alpha,
        "group1_stats": {
            "n": len(group1_clean),
            "mean": group1_clean.mean(),
            "std": group1_clean.std(),
            "sem": group1_clean.sem()
        }
    }
    
    if test_type == "one_sample":
        t_stat, p_value = stats.ttest_1samp(group1_clean, mu)
        results.update({
            "hypothesized_mean": mu,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "effect_size_cohen_d": (group1_clean.mean() - mu) / group1_clean.std()
        })
        
    elif test_type in ["two_sample", "paired"]:
        if group2 is None:
            return {"error": "Group 2 data required for two-sample/paired tests"}
        
        group2_clean = group2.dropna()
        if len(group2_clean) < 2:
            return {"error": "Insufficient data in group 2"}
        
        results["group2_stats"] = {
            "n": len(group2_clean),
            "mean": group2_clean.mean(),
            "std": group2_clean.std(),
            "sem": group2_clean.sem()
        }
        
        if test_type == "two_sample":
            # Check equal variances
            levene_stat, levene_p = levene(group1_clean, group2_clean)
            equal_var = levene_p > alpha
            
            t_stat, p_value = stats.ttest_ind(group1_clean, group2_clean, equal_var=equal_var)
            
            # Cohen's d for independent samples
            pooled_std = np.sqrt(((len(group1_clean) - 1) * group1_clean.var() + 
                                 (len(group2_clean) - 1) * group2_clean.var()) / 
                                (len(group1_clean) + len(group2_clean) - 2))
            cohens_d = (group1_clean.mean() - group2_clean.mean()) / pooled_std
            
            results.update({
                "equal_variances": equal_var,
                "levene_test": {"statistic": levene_stat, "p_value": levene_p}
            })
            
        else:  # paired
            if len(group1_clean) != len(group2_clean):
                return {"error": "Paired test requires equal sample sizes"}
            
            t_stat, p_value = stats.ttest_rel(group1_clean, group2_clean)
            
            # Cohen's d for paired samples
            diff = group1_clean - group2_clean
            cohens_d = diff.mean() / diff.std()
        
        results.update({
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "effect_size_cohen_d": cohens_d
        })
    
    # Interpret effect size
    abs_d = abs(results.get("effect_size_cohen_d", 0))
    if abs_d < 0.2:
        effect_interpretation = "negligible"
    elif abs_d < 0.5:
        effect_interpretation = "small"
    elif abs_d < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    results["effect_size_interpretation"] = effect_interpretation
    
    return results

def perform_anova(data: pd.DataFrame, dependent_var: str, 
                  independent_vars: List[str], alpha: float = 0.05) -> Dict:
    """
    Perform one-way or two-way ANOVA.
    
    Args:
        data: DataFrame containing the data
        dependent_var: Name of dependent variable column
        independent_vars: List of independent variable column names
        alpha: Significance level
        
    Returns:
        Dictionary with ANOVA results
    """
    if not STATS_AVAILABLE:
        return {"error": "Statistical libraries not available"}
    
    # Clean data
    clean_data = data[[dependent_var] + independent_vars].dropna()
    
    if len(clean_data) < 3:
        return {"error": "Insufficient data for ANOVA"}
    
    results = {
        "dependent_variable": dependent_var,
        "independent_variables": independent_vars,
        "sample_size": len(clean_data),
        "alpha": alpha
    }
    
    try:
        if len(independent_vars) == 1:
            # One-way ANOVA
            groups = [group[dependent_var].values for name, group in clean_data.groupby(independent_vars[0])]
            f_stat, p_value = stats.f_oneway(*groups)
            
            results.update({
                "test_type": "one_way_anova",
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < alpha
            })
            
            # Post-hoc Tukey HSD if significant
            if p_value < alpha:
                tukey_results = pairwise_tukeyhsd(
                    clean_data[dependent_var], 
                    clean_data[independent_vars[0]]
                )
                results["post_hoc_tukey"] = str(tukey_results)
                
        elif len(independent_vars) == 2:
            # Two-way ANOVA using statsmodels
            formula = f"{dependent_var} ~ C({independent_vars[0]}) + C({independent_vars[1]}) + C({independent_vars[0]}):C({independent_vars[1]})"
            model = sm.formula.ols(formula, data=clean_data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            results.update({
                "test_type": "two_way_anova",
                "anova_table": anova_table.to_dict(),
                "model_summary": str(model.summary())
            })
        
        else:
            return {"error": "ANOVA supports maximum 2 independent variables"}
            
    except Exception as e:
        results["error"] = f"ANOVA calculation failed: {str(e)}"
    
    return results

def perform_correlation_analysis(data: pd.DataFrame, 
                                variables: List[str] = None,
                                method: str = "pearson",
                                alpha: float = 0.05) -> Dict:
    """
    Perform correlation analysis between variables.
    
    Args:
        data: DataFrame containing the data
        variables: List of variable names (if None, use all numeric columns)
        method: "pearson", "spearman", or "kendall"
        alpha: Significance level
        
    Returns:
        Dictionary with correlation results
    """
    if not STATS_AVAILABLE:
        return {"error": "Statistical libraries not available"}
    
    # Select numeric columns if variables not specified
    if variables is None:
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        numeric_data = data[variables].select_dtypes(include=[np.number])
    
    if numeric_data.shape[1] < 2:
        return {"error": "Need at least 2 numeric variables for correlation"}
    
    # Remove rows with any NaN values
    clean_data = numeric_data.dropna()
    
    if len(clean_data) < 3:
        return {"error": "Insufficient data for correlation analysis"}
    
    results = {
        "method": method,
        "sample_size": len(clean_data),
        "variables": list(clean_data.columns),
        "alpha": alpha
    }
    
    # Calculate correlation matrix
    if method == "pearson":
        corr_matrix = clean_data.corr(method='pearson')
    elif method == "spearman":
        corr_matrix = clean_data.corr(method='spearman')
    elif method == "kendall":
        corr_matrix = clean_data.corr(method='kendall')
    else:
        return {"error": "Method must be 'pearson', 'spearman', or 'kendall'"}
    
    results["correlation_matrix"] = corr_matrix.to_dict()
    
    # Calculate p-values for correlations
    n_vars = len(clean_data.columns)
    p_values = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                if method == "pearson":
                    _, p_val = stats.pearsonr(clean_data.iloc[:, i], clean_data.iloc[:, j])
                elif method == "spearman":
                    _, p_val = stats.spearmanr(clean_data.iloc[:, i], clean_data.iloc[:, j])
                elif method == "kendall":
                    _, p_val = stats.kendalltau(clean_data.iloc[:, i], clean_data.iloc[:, j])
                p_values[i, j] = p_val
    
    p_value_df = pd.DataFrame(p_values, 
                             index=clean_data.columns, 
                             columns=clean_data.columns)
    results["p_values"] = p_value_df.to_dict()
    
    # Identify significant correlations
    significant_pairs = []
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_values[i, j]
            if p_val < alpha:
                significant_pairs.append({
                    "variable1": clean_data.columns[i],
                    "variable2": clean_data.columns[j],
                    "correlation": corr_val,
                    "p_value": p_val,
                    "strength": interpret_correlation_strength(abs(corr_val))
                })
    
    results["significant_correlations"] = significant_pairs
    
    return results

def interpret_correlation_strength(r: float) -> str:
    """Interpret correlation coefficient strength."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "weak"
    elif abs_r < 0.5:
        return "moderate"
    elif abs_r < 0.7:
        return "strong"
    else:
        return "very strong"

def perform_regression_analysis(data: pd.DataFrame, 
                               dependent_var: str,
                               independent_vars: List[str],
                               alpha: float = 0.05) -> Dict:
    """
    Perform linear regression analysis.
    
    Args:
        data: DataFrame containing the data
        dependent_var: Name of dependent variable
        independent_vars: List of independent variable names
        alpha: Significance level
        
    Returns:
        Dictionary with regression results
    """
    if not STATS_AVAILABLE:
        return {"error": "Statistical libraries not available"}
    
    # Prepare data
    all_vars = [dependent_var] + independent_vars
    clean_data = data[all_vars].dropna()
    
    if len(clean_data) < len(independent_vars) + 2:
        return {"error": "Insufficient data for regression analysis"}
    
    X = clean_data[independent_vars]
    y = clean_data[dependent_var]
    
    results = {
        "dependent_variable": dependent_var,
        "independent_variables": independent_vars,
        "sample_size": len(clean_data),
        "alpha": alpha
    }
    
    try:
        # Fit regression model using statsmodels for detailed statistics
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        results.update({
            "r_squared": model.rsquared,
            "adjusted_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "aic": model.aic,
            "bic": model.bic,
            "coefficients": {},
            "residual_analysis": {}
        })
        
        # Extract coefficient information
        for i, var in enumerate(['const'] + independent_vars):
            results["coefficients"][var] = {
                "coefficient": model.params[i],
                "std_error": model.bse[i],
                "t_statistic": model.tvalues[i],
                "p_value": model.pvalues[i],
                "significant": model.pvalues[i] < alpha,
                "conf_int_lower": model.conf_int().iloc[i, 0],
                "conf_int_upper": model.conf_int().iloc[i, 1]
            }
        
        # Residual analysis
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        results["residual_analysis"] = {
            "mean_residual": residuals.mean(),
            "residual_std": residuals.std(),
            "durbin_watson": sm.stats.durbin_watson(residuals),
            "jarque_bera_test": {
                "statistic": sm.stats.jarque_bera(residuals)[0],
                "p_value": sm.stats.jarque_bera(residuals)[1]
            }
        }
        
        # Model interpretation
        if model.f_pvalue < alpha:
            results["model_significance"] = "significant"
        else:
            results["model_significance"] = "not_significant"
            
        results["model_summary"] = str(model.summary())
        
    except Exception as e:
        results["error"] = f"Regression analysis failed: {str(e)}"
    
    return results

def perform_nonparametric_tests(group1: pd.Series, group2: pd.Series = None,
                               groups: pd.DataFrame = None, 
                               test_type: str = "mann_whitney") -> Dict:
    """
    Perform non-parametric statistical tests.
    
    Args:
        group1: First group data
        group2: Second group data (for two-group tests)
        groups: DataFrame with 'value' and 'group' columns (for multi-group tests)
        test_type: "mann_whitney", "wilcoxon", or "kruskal_wallis"
        
    Returns:
        Dictionary with test results
    """
    if not STATS_AVAILABLE:
        return {"error": "Statistical libraries not available"}
    
    results = {"test_type": test_type}
    
    try:
        if test_type == "mann_whitney":
            if group2 is None:
                return {"error": "Mann-Whitney U test requires two groups"}
            
            group1_clean = group1.dropna()
            group2_clean = group2.dropna()
            
            if len(group1_clean) < 1 or len(group2_clean) < 1:
                return {"error": "Insufficient data for Mann-Whitney U test"}
            
            statistic, p_value = mannwhitneyu(group1_clean, group2_clean, 
                                            alternative='two-sided')
            
            results.update({
                "statistic": statistic,
                "p_value": p_value,
                "group1_median": group1_clean.median(),
                "group2_median": group2_clean.median(),
                "group1_n": len(group1_clean),
                "group2_n": len(group2_clean)
            })
            
        elif test_type == "wilcoxon":
            if group2 is None:
                return {"error": "Wilcoxon signed-rank test requires two groups"}
            
            group1_clean = group1.dropna()
            group2_clean = group2.dropna()
            
            if len(group1_clean) != len(group2_clean):
                return {"error": "Wilcoxon test requires paired data of equal length"}
            
            statistic, p_value = stats.wilcoxon(group1_clean, group2_clean)
            
            results.update({
                "statistic": statistic,
                "p_value": p_value,
                "median_difference": (group1_clean - group2_clean).median(),
                "n_pairs": len(group1_clean)
            })
            
        elif test_type == "kruskal_wallis":
            if groups is None:
                return {"error": "Kruskal-Wallis test requires groups DataFrame"}
            
            if 'value' not in groups.columns or 'group' not in groups.columns:
                return {"error": "Groups DataFrame must have 'value' and 'group' columns"}
            
            clean_groups = groups.dropna()
            group_data = [group['value'].values for name, group in clean_groups.groupby('group')]
            
            if len(group_data) < 2:
                return {"error": "Need at least 2 groups for Kruskal-Wallis test"}
            
            statistic, p_value = kruskal(*group_data)
            
            # Calculate group medians
            group_medians = clean_groups.groupby('group')['value'].median().to_dict()
            
            results.update({
                "statistic": statistic,
                "p_value": p_value,
                "group_medians": group_medians,
                "total_n": len(clean_groups)
            })
            
    except Exception as e:
        results["error"] = f"Non-parametric test failed: {str(e)}"
    
    return results

def format_statistical_results(results: Dict, test_name: str) -> str:
    """
    Format statistical results for display.
    
    Args:
        results: Dictionary with statistical test results
        test_name: Name of the statistical test
        
    Returns:
        Formatted string with results
    """
    if "error" in results:
        return f"❌ {test_name} Error: {results['error']}"
    
    output = [f"📊 {test_name} Results", "=" * 50]
    
    # Add debug information if results seem empty
    if not results or len(results) == 0:
        return f"⚠️ {test_name}: No results to display (empty results dictionary)"
    
    # Add basic info about what's in the results
    output.append(f"Result keys: {list(results.keys())}")
    
    # Safe float formatter
    def _fmt(val, nd=4):
        try:
            return f"{float(val):.{nd}f}"
        except Exception:
            return str(val)

    # Sample size information
    if "sample_size" in results:
        output.append(f"Sample Size: {results['sample_size']}")
    
    # Test-specific formatting
    if "t_statistic" in results:
        output.extend([
            f"t-statistic: {_fmt(results.get('t_statistic'))}",
            f"p-value: {_fmt(results.get('p_value'))}",
            f"Significant: {'Yes' if results.get('significant', False) else 'No'}",
            f"Effect Size (Cohen's d): {_fmt(results.get('effect_size_cohen_d', 'N/A'))}",
            f"Effect Size Interpretation: {results.get('effect_size_interpretation', 'N/A')}"
        ])
    
    elif "f_statistic" in results and "anova_table" not in results:
        # This is for simple ANOVA results (not the complex statsmodels output)
        output.extend([
            f"F-statistic: {_fmt(results.get('f_statistic'))}",
            f"p-value: {_fmt(results.get('p_value'))}",
            f"Significant: {'Yes' if results.get('significant', False) else 'No'}"
        ])
        
        if "post_hoc_tukey" in results:
            output.append("\nPost-hoc Analysis (Tukey HSD):")
            output.append(str(results['post_hoc_tukey']))
    
    elif "anova_table" in results:
        # This handles complex two-way ANOVA results from statsmodels
        output.append("ANOVA Results:")
        if "test_type" in results:
            output.append(f"Test Type: {results['test_type']}")
        
        # Try to format the ANOVA table if it exists
        try:
            anova_table = results['anova_table']
            output.append("\nANOVA Table:")
            output.append("Source\t\tF-value\tp-value\tSignificant")
            output.append("-" * 50)
            
            for source, values in anova_table.items():
                if isinstance(values, dict) and 'F' in values and 'PR(>F)' in values:
                    f_val = values['F'][0] if isinstance(values['F'], list) else values['F']
                    p_val = values['PR(>F)'][0] if isinstance(values['PR(>F)'], list) else values['PR(>F)']
                    significant = "Yes" if p_val < 0.05 else "No"
                    output.append(f"{source}\t\t{_fmt(f_val)}\t{_fmt(p_val)}\t{significant}")
        except Exception as e:
            output.append(f"Note: ANOVA table formatting error: {e}")
            
        if "model_summary" in results:
            output.append("\nDetailed Model Summary Available")
    
    elif "correlation_matrix" in results:
        output.append("Correlation Matrix:")
        # Format correlation matrix (simplified)
        if "significant_correlations" in results:
            output.append(f"\nSignificant Correlations ({len(results['significant_correlations'])} found):")
            for corr in results['significant_correlations'][:5]:  # Show top 5
                output.append(f"  {corr['variable1']} ↔ {corr['variable2']}: "
                            f"r = {corr['correlation']:.3f} (p = {corr['p_value']:.4f}) - {corr['strength']}")
    
    elif "r_squared" in results:
        output.extend([
            f"R-squared: {_fmt(results.get('r_squared'))}",
            f"Adjusted R-squared: {_fmt(results.get('adjusted_r_squared'))}",
            f"F-statistic: {_fmt(results.get('f_statistic'))}",
            f"Model p-value: {_fmt(results.get('f_pvalue'))}",
            f"Model Significance: {results.get('model_significance', 'N/A')}"
        ])
    
    elif "statistic" in results:  # Non-parametric tests
        output.extend([
            f"Test Statistic: {_fmt(results.get('statistic'))}",
            f"p-value: {_fmt(results.get('p_value'))}"
        ])
    
    return "\n".join(output)
