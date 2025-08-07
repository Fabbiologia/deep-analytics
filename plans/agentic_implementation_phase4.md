# Agentic Implementation Guide - Phase 4: Predictive Modeling

## Phase 4: Predictive Modeling

### 4.1 Predictive Modeling System - Agentic Instructions

1. **Predictive Modeling Architecture**:

```python
class PredictiveModelingSystem:
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.feature_engineer = FeatureEngineer()
        self.model_selector = ModelSelector()
        self.explainer = ModelExplainer()
        
    def build_predictive_model(self, data, target_variable, model_type=None, features=None, parameters=None):
        """
        Build a predictive model for the given target variable
        
        Args:
            data: Input DataFrame
            target_variable: Target variable to predict
            model_type: Optional specific model type ('time_series', 'regression', 'gam', 'classification')
            features: Optional specific features to use
            parameters: Optional model parameters
            
        Returns:
            Model object with predictions, performance metrics, and explanations
        """
        # Step 1: Prepare features
        if features is None:
            # Auto-select relevant features
            features, feature_importances = self.feature_engineer.select_features(
                data, target_variable
            )
            
        X, y, feature_transformers = self.feature_engineer.prepare_features(
            data, target_variable, features
        )
        
        # Step 2: Select appropriate model type if not specified
        if model_type is None:
            model_type = self.model_selector.recommend_model(X, y, target_variable)
            
        # Step 3: Build and train model
        model = self.model_registry.get_model(
            model_type, parameters
        )
        
        # Step 4: Cross-validate and evaluate
        cv_results = self._cross_validate_model(model, X, y)
        
        # Step 5: Train final model on all data
        model.fit(X, y)
        
        # Step 6: Generate predictions
        predictions = model.predict(X)
        
        # Step 7: Create explanations
        explanations = self.explainer.explain_model(model, X, y, feature_names=features)
        
        # Step 8: Package everything into result object
        result = {
            "model": model,
            "model_type": model_type,
            "features": features,
            "feature_transformers": feature_transformers,
            "cv_results": cv_results,
            "predictions": predictions,
            "explanations": explanations,
            "metrics": self._calculate_metrics(y, predictions, model_type)
        }
        
        return result
```

2. **Time Series Forecasting Implementation**:

```python
class TimeSeriesModelBuilder:
    def build_forecasting_model(self, df, target_column, date_column='Year', 
                               groupby_columns=None, forecast_periods=5, seasonality_mode='additive'):
        """
        Build a time series forecasting model using Prophet
        
        Args:
            df: Input DataFrame
            target_column: Column to forecast
            date_column: Column containing dates/years
            groupby_columns: Columns to group by before forecasting
            forecast_periods: Number of periods to forecast
            seasonality_mode: 'additive' or 'multiplicative'
            
        Returns:
            Dictionary with models, forecasts, and performance metrics
        """
        import pandas as pd
        from prophet import Prophet
        import numpy as np
        
        results = {}
        
        # Handle both grouped and ungrouped forecasting
        if groupby_columns:
            # For each group, build a separate forecast
            groups = df.groupby(groupby_columns)
            
            for group_name, group_df in groups:
                # Prepare data for Prophet
                prophet_df = self._prepare_prophet_data(
                    group_df, target_column, date_column
                )
                
                # Build model and forecast
                model, forecast, metrics = self._fit_prophet_model(
                    prophet_df, forecast_periods, seasonality_mode
                )
                
                # Store results for this group
                if isinstance(group_name, tuple):
                    group_key = '_'.join([str(x) for x in group_name])
                else:
                    group_key = str(group_name)
                    
                results[group_key] = {
                    'model': model,
                    'forecast': forecast,
                    'metrics': metrics,
                    'group_values': group_name
                }
        else:
            # Single forecast for the entire dataset
            prophet_df = self._prepare_prophet_data(
                df, target_column, date_column
            )
            
            model, forecast, metrics = self._fit_prophet_model(
                prophet_df, forecast_periods, seasonality_mode
            )
            
            results['overall'] = {
                'model': model,
                'forecast': forecast,
                'metrics': metrics,
                'group_values': None
            }
            
        return results
        
    def _prepare_prophet_data(self, df, target_column, date_column):
        """Convert data to Prophet format (ds, y)"""
        # Create a copy to avoid modifying the original
        prophet_df = df[[date_column, target_column]].copy()
        
        # Rename columns to Prophet's required format
        prophet_df.columns = ['ds', 'y']
        
        # Convert ds to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
            # If it's just a year, convert to datetime at the middle of the year
            if prophet_df['ds'].dtype == int or prophet_df['ds'].dtype == float:
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
            else:
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                
        # Sort by date
        prophet_df = prophet_df.sort_values('ds')
        
        return prophet_df
        
    def _fit_prophet_model(self, prophet_df, forecast_periods, seasonality_mode):
        """Fit Prophet model and generate forecast"""
        from prophet import Prophet
        
        # Initialize and fit the model
        model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(prophet_df)
        
        # Create future dataframe for forecasting
        future = model.make_future_dataframe(
            periods=forecast_periods,
            freq='Y'  # Yearly forecasting
        )
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Calculate metrics using cross-validation
        from prophet.diagnostics import cross_validation, performance_metrics
        
        initial = str(int(len(prophet_df) * 0.5)) + ' days'
        period = str(int(len(prophet_df) * 0.1)) + ' days'
        horizon = str(int(forecast_periods)) + ' days'
        
        try:
            # Try cross-validation, but it might fail for small datasets
            cv_results = cross_validation(
                model, 
                initial=initial, 
                period=period, 
                horizon=horizon,
                parallel="processes"
            )
            metrics = performance_metrics(cv_results)
        except Exception as e:
            # Fallback to simpler validation
            metrics = {
                'note': 'Cross-validation failed, using simple validation',
                'error': str(e),
                'rmse': self._calculate_simple_error(prophet_df['y'], 
                                                  forecast[forecast['ds'].isin(prophet_df['ds'])]['yhat'])
            }
        
        return model, forecast, metrics
        
    def _calculate_simple_error(self, actual, predicted):
        """Calculate simple RMSE for cases where cross-validation fails"""
        import numpy as np
        
        # Match lengths if necessary
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        return float(rmse)
```

3. **Generalized Additive Models for Ecological Analysis**:

```python
class GAMModelBuilder:
    def build_ecological_gam(self, df, target_variable, predictor_variables=None, 
                            spline_terms=None, family='gaussian'):
        """
        Build a Generalized Additive Model for ecological data
        
        Args:
            df: Input DataFrame
            target_variable: Response variable
            predictor_variables: List of predictor variables
            spline_terms: Dict mapping variable names to spline parameters
            family: Distribution family ('gaussian', 'poisson', 'binomial', etc.)
            
        Returns:
            Dictionary with model, predictions, and diagnostics
        """
        import pandas as pd
        import numpy as np
        from pygam import LinearGAM, s, f, l
        
        # Automatically select predictors if not provided
        if predictor_variables is None:
            # Remove non-numeric columns and other inappropriate predictors
            exclude_cols = [target_variable, 'Year', 'Month', 'Day', 'ID']
            numeric_cols = df.select_dtypes(include=np.number).columns
            predictor_variables = [col for col in numeric_cols if col not in exclude_cols]
        
        # Prepare data
        X = df[predictor_variables].copy()
        y = df[target_variable].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Define spline terms if not provided
        if spline_terms is None:
            spline_terms = {var: s(i) for i, var in enumerate(predictor_variables)}
        
        # Build formula
        gam_terms = []
        for var, term in spline_terms.items():
            if term == 'linear':
                gam_terms.append(l(predictor_variables.index(var)))
            else:
                gam_terms.append(s(predictor_variables.index(var)))
        
        # Create and fit the model
        gam = LinearGAM(gam_terms, distribution=family)
        gam.fit(X, y)
        
        # Generate predictions
        y_pred = gam.predict(X)
        
        # Calculate performance metrics
        r_squared = gam.statistics_['pseudo_r2']['explained_deviance']
        AIC = gam.statistics_['AIC']
        
        # Create partial dependence plots
        pdp_data = {}
        for i, var in enumerate(predictor_variables):
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            pdp_data[var] = {
                'x': XX[:, i],
                'y': pdep,
                'lower': confi[:, 0],
                'upper': confi[:, 1]
            }
        
        # Create residual plots
        residuals = y - y_pred
        residual_data = {
            'fitted': y_pred,
            'residuals': residuals,
            'standardized_residuals': residuals / np.std(residuals)
        }
        
        # Package results
        results = {
            'model': gam,
            'predictor_variables': predictor_variables,
            'target_variable': target_variable,
            'predictions': y_pred,
            'metrics': {
                'r_squared': r_squared,
                'AIC': AIC,
                'summary': str(gam.summary())
            },
            'partial_dependence': pdp_data,
            'residuals': residual_data
        }
        
        return results
```

4. **Model Explainability Framework**:

```python
class ModelExplainer:
    def explain_model(self, model, X, y, feature_names=None, method="shap"):
        """
        Generate explanations for a trained model
        
        Args:
            model: Trained model object
            X: Feature data
            y: Target data
            feature_names: List of feature names
            method: Explainability method ('shap', 'lime', or 'both')
            
        Returns:
            Dictionary with explanation objects
        """
        result = {}
        
        # Ensure feature_names are available
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Convert X to numpy array if it's a pandas DataFrame
        X_array = X.values if hasattr(X, 'values') else X
        
        # Generate SHAP explanations
        if method in ["shap", "both"]:
            result["shap"] = self._generate_shap_explanations(model, X_array, feature_names)
        
        # Generate LIME explanations
        if method in ["lime", "both"]:
            result["lime"] = self._generate_lime_explanations(model, X_array, y, feature_names)
            
        return result
    
    def _generate_shap_explanations(self, model, X, feature_names):
        """Generate SHAP explanations"""
        import shap
        
        # Try to determine model type to use appropriate explainer
        try:
            # Check if model has predict_proba (for classification)
            has_predict_proba = hasattr(model, 'predict_proba')
            
            # Select appropriate explainer
            if hasattr(model, 'predict_log_proba'):
                # For scikit-learn classifiers
                explainer = shap.explainers.Tree(model, X) if hasattr(model, 'estimators_') else shap.explainers.Linear(model, X)
            elif hasattr(model, 'layers'):
                # For Keras models
                explainer = shap.explainers.Deep(model, X)
            else:
                # General case
                explainer = shap.explainers.Kernel(model.predict, X)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # For multi-class classification, use the first class's explanations
            if isinstance(shap_values, list) and not isinstance(shap_values[0], float):
                shap_values = shap_values[0]
                
            # Create a base value (mean prediction)
            if hasattr(explainer, 'expected_value'):
                base_value = explainer.expected_value
                if isinstance(base_value, list):
                    base_value = base_value[0]
            else:
                base_value = model.predict(X).mean()
                
            # Format results
            result = {
                "shap_values": shap_values,
                "base_value": base_value,
                "feature_names": feature_names
            }
            
            return result
        except Exception as e:
            # Fallback if SHAP fails
            return {
                "error": str(e),
                "note": "SHAP explanation generation failed"
            }
    
    def _generate_lime_explanations(self, model, X, y, feature_names):
        """Generate LIME explanations for a few representative samples"""
        import numpy as np
        from lime import lime_tabular
        
        try:
            # Determine if classification or regression
            is_classifier = hasattr(model, 'predict_proba')
            
            # Create explainer
            mode = 'classification' if is_classifier else 'regression'
            explainer = lime_tabular.LimeTabularExplainer(
                X,
                feature_names=feature_names,
                mode=mode
            )
            
            # Select representative samples for explanation (3 samples)
            if len(X) <= 3:
                indices = range(len(X))
            else:
                # Try to select diverse samples
                if is_classifier:
                    # For classification, get samples from different classes
                    predictions = model.predict(X)
                    unique_classes = np.unique(predictions)
                    indices = []
                    for cls in unique_classes[:3]:  # Up to 3 classes
                        class_indices = np.where(predictions == cls)[0]
                        if len(class_indices) > 0:
                            indices.append(class_indices[0])
                    # Fill remaining slots if needed
                    while len(indices) < 3 and len(indices) < len(X):
                        remaining = np.setdiff1d(range(len(X)), indices)
                        if len(remaining) == 0:
                            break
                        indices.append(remaining[0])
                else:
                    # For regression, select min, max and median predictions
                    predictions = model.predict(X)
                    indices = [
                        np.argmin(predictions),
                        np.argmax(predictions),
                        np.argsort(predictions)[len(predictions)//2]
                    ]
            
            # Generate explanations
            explanations = []
            for idx in indices:
                if is_classifier:
                    exp = explainer.explain_instance(
                        X[idx], model.predict_proba, num_features=min(10, len(feature_names))
                    )
                else:
                    exp = explainer.explain_instance(
                        X[idx], model.predict, num_features=min(10, len(feature_names))
                    )
                
                # Extract explanation data
                exp_data = {
                    'instance_index': int(idx),
                    'instance_values': X[idx].tolist(),
                    'prediction': float(model.predict(X[idx].reshape(1, -1))[0]),
                    'explanation': dict(exp.as_list())
                }
                explanations.append(exp_data)
                
            return {
                "explanations": explanations,
                "mode": mode
            }
        except Exception as e:
            # Fallback if LIME fails
            return {
                "error": str(e),
                "note": "LIME explanation generation failed"
            }
```

5. **Interactive What-If Analysis**:

```python
class WhatIfAnalyzer:
    def __init__(self, model_system):
        self.model_system = model_system
        
    def create_scenario_analysis(self, baseline_model, X, feature_ranges, target_variable, 
                               scenarios=None, num_points=20):
        """
        Create what-if scenarios by varying feature values
        
        Args:
            baseline_model: Trained model from model_system
            X: Original feature dataset
            feature_ranges: Dict of features and their min/max ranges
            target_variable: Target variable name
            scenarios: Predefined scenarios to evaluate
            num_points: Number of points to evaluate in each range
            
        Returns:
            Dictionary with scenario analysis results
        """
        import numpy as np
        import pandas as pd
        
        # Get model and features from baseline model
        model = baseline_model['model']
        features = baseline_model['features']
        
        # Initialize result container
        results = {
            'single_feature_effects': {},
            'scenarios': {}
        }
        
        # For each feature, analyze its isolated effect
        for feature, range_values in feature_ranges.items():
            if feature not in features:
                continue
                
            feature_idx = features.index(feature)
            
            # Create a range of values to test
            min_val, max_val = range_values
            test_values = np.linspace(min_val, max_val, num_points)
            
            # Create test data by varying only this feature
            X_test = np.tile(X.mean(axis=0), (num_points, 1))
            for i, val in enumerate(test_values):
                X_test[i, feature_idx] = val
                
            # Get predictions for each value
            predictions = model.predict(X_test)
            
            # Store results
            results['single_feature_effects'][feature] = {
                'feature_values': test_values.tolist(),
                'predictions': predictions.tolist(),
                'baseline': float(X[:, feature_idx].mean()),
                'baseline_prediction': float(model.predict(X.mean(axis=0).reshape(1, -1))[0])
            }
            
        # Evaluate predefined scenarios if provided
        if scenarios:
            for scenario_name, scenario_values in scenarios.items():
                # Create scenario data
                X_scenario = X.mean(axis=0).reshape(1, -1).copy()
                
                # Apply scenario values
                for feature, value in scenario_values.items():
                    if feature in features:
                        feature_idx = features.index(feature)
                        X_scenario[0, feature_idx] = value
                
                # Get prediction for scenario
                prediction = model.predict(X_scenario)[0]
                
                # Store scenario results
                results['scenarios'][scenario_name] = {
                    'values': scenario_values,
                    'prediction': float(prediction),
                    'change_from_baseline': float(prediction - model.predict(X.mean(axis=0).reshape(1, -1))[0])
                }
                
        # Add metadata
        results['metadata'] = {
            'target_variable': target_variable,
            'features': features,
            'baseline_prediction': float(model.predict(X.mean(axis=0).reshape(1, -1))[0])
        }
                
        return results
        
    def generate_interactive_widget(self, model_result, feature_ranges):
        """
        Generate code for an interactive Streamlit widget for what-if analysis
        
        Args:
            model_result: Model result from model_system
            feature_ranges: Dict of features and their min/max ranges
            
        Returns:
            String with Python code for the interactive widget
        """
        # Extract model and features
        model = model_result['model']
        features = model_result['features']
        
        # Generate widget code
        code = []
        code.append("import streamlit as st")
        code.append("import numpy as np")
        code.append("import pandas as pd")
        code.append("import plotly.express as px")
        code.append("")
        code.append("st.title('What-If Analysis')")
        code.append("st.write('Adjust the sliders to see how changes in feature values affect predictions.')")
        code.append("")
        code.append("# Create sliders for features")
        
        # Add feature sliders
        for feature in features:
            if feature in feature_ranges:
                min_val, max_val = feature_ranges[feature]
                default = (min_val + max_val) / 2
                step = (max_val - min_val) / 100
                code.append(f"{feature}_val = st.slider('{feature}', {min_val}, {max_val}, {default}, {step})")
        
        code.append("")
        code.append("# Create input array for prediction")
        code.append("input_data = np.array([")
        
        # Build input array
        feature_values = []
        for feature in features:
            if feature in feature_ranges:
                feature_values.append(f"{feature}_val")
            else:
                feature_values.append("0.0  # Default value")
                
        code.append("    " + ", ".join(feature_values))
        code.append("])")
        
        code.append("")
        code.append("# Make prediction with the model")
        code.append("prediction = model.predict(input_data.reshape(1, -1))[0]")
        code.append("")
        code.append("# Display prediction")
        code.append("st.subheader('Prediction')")
        code.append("st.write(f'Predicted value: {prediction:.4f}')")
        
        return "\n".join(code)
```
