# Agentic Implementation Guide - Phase 3: Automated Insights & Reporting

## Phase 3: Automated Insights & Reporting

### 3.1 Automated Insights Engine - Agentic Instructions

1. **Insights Discovery Architecture**:

```python
class InsightsEngine:
    def __init__(self):
        self.analyzers = {
            "trend": TrendAnalyzer(),
            "anomaly": AnomalyAnalyzer(),
            "correlation": CorrelationAnalyzer(),
            "outlier": OutlierAnalyzer(),
            "trophic_shift": TrophicStructureShiftAnalyzer(),
            "significant_difference": SignificanceTesterAnalyzer()
        }
        self.narrative_generator = NarrativeGenerator()
        self.insight_prioritizer = InsightPrioritizer()
        
    def discover_insights(self, df, target_columns=None, context=None):
        """
        Discover insights in the provided dataframe
        
        Args:
            df: Input DataFrame
            target_columns: Specific columns to focus on (optional)
            context: Additional context about the analysis
            
        Returns:
            List of insight objects sorted by relevance/significance
        """
        all_insights = []
        
        # Apply each analyzer to discover insights
        for analyzer_name, analyzer in self.analyzers.items():
            try:
                insights = analyzer.analyze(df, target_columns, context)
                all_insights.extend(insights)
            except Exception as e:
                logging.error(f"Error in {analyzer_name}: {str(e)}")
                
        # Generate narratives for each insight
        for insight in all_insights:
            insight["narrative"] = self.narrative_generator.generate_narrative(insight)
            
        # Prioritize insights
        prioritized_insights = self.insight_prioritizer.prioritize(all_insights)
        
        return prioritized_insights
```

2. **Specific Analyzer Implementations**:

```python
class TrophicStructureShiftAnalyzer:
    def analyze(self, df, target_columns=None, context=None):
        """
        Analyze shifts in trophic structure over time or between regions
        
        Args:
            df: Input DataFrame with trophic and biomass data
            target_columns: Specific columns to analyze
            context: Additional context
            
        Returns:
            List of trophic structure shift insights
        """
        insights = []
        
        # Check if we have necessary columns
        required_columns = ["TrophicLevel", "Biomass", "Year", "Region"]
        if not all(col in df.columns for col in required_columns):
            return insights
            
        # Ensure we're working with fish data (PEC)
        if "Label" in df.columns:
            df = df[df["Label"] == "PEC"].copy()
            
        # Group by Year and TrophicLevel to detect shifts over time
        time_shifts = self._detect_time_shifts(df)
        insights.extend(time_shifts)
        
        # Group by Region and TrophicLevel to detect differences between regions
        region_differences = self._detect_region_differences(df)
        insights.extend(region_differences)
        
        return insights
        
    def _detect_time_shifts(self, df):
        """
        Detect shifts in trophic structure over time
        """
        insights = []
        
        # Pivot data to get Year x TrophicLevel matrix with Biomass values
        pivot_df = pd.pivot_table(
            df,
            values="Biomass",
            index="Year",
            columns="TrophicLevel",
            aggfunc="sum"
        ).fillna(0)
        
        # Calculate relative contribution of each trophic level
        relative_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        
        # Find years with at least 5% change in any trophic level
        years = sorted(relative_df.index.unique())
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            # Skip if too many years apart
            if curr_year - prev_year > 3:
                continue
                
            # Calculate changes
            changes = relative_df.loc[curr_year] - relative_df.loc[prev_year]
            
            # Find significant changes
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
                        "trophic_level": tl,
                        "change_percent": change,
                        "score": abs(change) / 10,  # Score based on magnitude of change
                        "statistical_test": "percent_change",
                        "p_value": None
                    })
        
        return insights
```

3. **Statistical Testing Framework**:

```python
class SignificanceTesterAnalyzer:
    def analyze(self, df, target_columns=None, context=None):
        """
        Run statistical tests to find significant differences between groups
        
        Args:
            df: Input DataFrame
            target_columns: Numeric columns to test (if None, use all numeric)
            context: Additional context
            
        Returns:
            List of significant difference insights
        """
        insights = []
        
        if target_columns is None:
            # Find numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            # Exclude ID columns or other non-meaningful numerics
            exclude_patterns = ['id', 'ID', 'code']
            target_columns = [col for col in numeric_cols if not any(pat in col for pat in exclude_patterns)]
        
        # Find categorical columns for grouping
        categorical_cols = ['Region', 'MPA', 'Habitat', 'Year']
        available_cat_cols = [col for col in categorical_cols if col in df.columns]
        
        for cat_col in available_cat_cols:
            # Skip if too many unique values
            if df[cat_col].nunique() > 10:
                continue
                
            # Skip if fewer than 2 groups
            if df[cat_col].nunique() < 2:
                continue
                
            for target_col in target_columns:
                # Skip testing categorical against itself
                if cat_col == target_col:
                    continue
                
                # Run appropriate test based on number of groups and normality
                if df[cat_col].nunique() == 2:
                    # Two groups: t-test or Mann-Whitney
                    insight = self._run_two_group_test(df, cat_col, target_col)
                    if insight:
                        insights.append(insight)
                else:
                    # Multiple groups: ANOVA or Kruskal-Wallis
                    insight = self._run_multi_group_test(df, cat_col, target_col)
                    if insight:
                        insights.append(insight)
        
        return insights
        
    def _check_normality(self, data):
        """Check if data is normally distributed using Shapiro-Wilk test"""
        # Skip test if too few samples
        if len(data) < 3:
            return True
            
        # Skip test if too many samples (Shapiro-Wilk limited to 5000)
        if len(data) > 5000:
            return False
            
        from scipy import stats
        try:
            stat, p = stats.shapiro(data)
            return p > 0.05
        except Exception:
            return False
            
    def _run_two_group_test(self, df, group_col, target_col):
        """Run t-test or Mann-Whitney U test for two groups"""
        from scipy import stats
        
        # Get the two groups
        groups = df[group_col].unique()
        
        if len(groups) != 2:
            return None
            
        data1 = df[df[group_col] == groups[0]][target_col].dropna()
        data2 = df[df[group_col] == groups[1]][target_col].dropna()
        
        # Skip if either group has too little data
        if len(data1) < 3 or len(data2) < 3:
            return None
            
        # Check normality to determine test
        if self._check_normality(data1) and self._check_normality(data2):
            # Use t-test
            test_name = "t-test"
            stat, p = stats.ttest_ind(data1, data2, equal_var=False)
        else:
            # Use Mann-Whitney U test
            test_name = "Mann-Whitney U"
            stat, p = stats.mannwhitneyu(data1, data2)
            
        # Only return insight if significant
        if p <= 0.05:
            mean1 = data1.mean()
            mean2 = data2.mean()
            diff = mean2 - mean1
            
            return {
                "type": "significant_difference",
                "subtype": "two_group",
                "description": f"Significant difference in {target_col} between {groups[0]} and {groups[1]}",
                "narrative": f"{test_name} shows a significant difference in {target_col} between {group_col} values " + 
                             f"{groups[0]} (mean: {mean1:.2f}) and {groups[1]} (mean: {mean2:.2f}). " + 
                             f"The {groups[1]} group is {abs(diff):.2f} " + 
                             f"{'higher' if diff > 0 else 'lower'} (p-value: {p:.4f}).",
                "group_column": group_col,
                "target_column": target_col,
                "group1": groups[0],
                "group2": groups[1],
                "mean1": float(mean1),
                "mean2": float(mean2),
                "difference": float(diff),
                "statistical_test": test_name,
                "statistic": float(stat),
                "p_value": float(p),
                "score": (0.05 - p) / 0.05  # Higher score for more significant results
            }
            
        return None
```

4. **Narrative Generation System**:

```python
class NarrativeGenerator:
    def __init__(self):
        self.templates = {
            "trend": self._trend_template,
            "anomaly": self._anomaly_template,
            "correlation": self._correlation_template,
            "outlier": self._outlier_template,
            "trophic_structure_shift": self._trophic_shift_template,
            "significant_difference": self._significant_difference_template
        }
        
    def generate_narrative(self, insight):
        """Generate a narrative description for an insight"""
        if insight["type"] in self.templates:
            return self.templates[insight["type"]](insight)
        else:
            return f"Analysis shows {insight['description']}."
            
    def _correlation_template(self, insight):
        direction = "positive" if insight["score"] > 0 else "negative"
        strength = "strong" if abs(insight["score"]) > 0.7 else "moderate" if abs(insight["score"]) > 0.4 else "weak"
        
        narrative = f"Analysis reveals a {strength} {direction} correlation "
        narrative += f"({insight['statistical_test']}: r = {insight['score']:.2f}) "
        narrative += f"between {insight['var1']} and {insight['var2']}. "
        
        # Add p-value information if available
        if "p_value" in insight and insight["p_value"] is not None:
            sig_level = "highly significant" if insight["p_value"] < 0.01 else "significant" if insight["p_value"] < 0.05 else "not statistically significant"
            narrative += f"This correlation is {sig_level} (p = {insight['p_value']:.4f}). "
            
        # Add ecological interpretation
        if direction == "positive":
            narrative += f"As {insight['var1']} increases, {insight['var2']} tends to increase as well. "
        else:
            narrative += f"As {insight['var1']} increases, {insight['var2']} tends to decrease. "
            
        return narrative
        
    def _trophic_shift_template(self, insight):
        if insight["subtype"] == "temporal":
            narrative = f"The analysis identified a significant change in trophic structure between "
            narrative += f"{insight['from_year']} and {insight['to_year']}. "
            narrative += f"Trophic level {insight['trophic_level']} showed a {abs(insight['change_percent']):.1f}% "
            narrative += f"{'increase' if insight['change_percent'] > 0 else 'decrease'}. "
            
            # Add ecological context
            if insight['change_percent'] > 0 and insight['trophic_level'] >= 4:
                narrative += "This increase in higher trophic levels may indicate ecosystem recovery "
                narrative += "or improved conditions for top predators. "
            elif insight['change_percent'] < 0 and insight['trophic_level'] >= 4:
                narrative += "This decrease in higher trophic levels may indicate fishing pressure "
                narrative += "or other disturbances affecting top predators. "
            elif insight['change_percent'] > 0 and insight['trophic_level'] <= 2:
                narrative += "This increase in lower trophic levels may indicate changes in primary "
                narrative += "productivity or reduced predation pressure. "
                
            return narrative
        else:
            # For other subtypes
            return insight["description"]
```

### 3.2 PDF & HTML Report Generation - Agentic Instructions

1. **Report System Architecture**:

```python
class ReportingSystem:
    def __init__(self):
        self.pdf_generator = PDFReportGenerator()
        self.html_generator = HTMLReportGenerator()
        self.template_manager = ReportTemplateManager()
        
    def generate_report(self, format_type, template_name, data, metadata=None):
        """
        Generate a report in the specified format
        
        Args:
            format_type: 'pdf' or 'html'
            template_name: Template to use (e.g., 'executive', 'scientific', 'monitoring')
            data: Dictionary with data to include in the report
            metadata: Report metadata (title, author, date, etc.)
            
        Returns:
            Path to the generated report
        """
        # Load the appropriate template
        template = self.template_manager.get_template(template_name)
        
        # Generate report in requested format
        if format_type.lower() == 'pdf':
            return self.pdf_generator.generate(template, data, metadata)
        elif format_type.lower() == 'html':
            return self.html_generator.generate(template, data, metadata)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
```

2. **Template Customization System**:

```python
class ReportTemplateManager:
    def __init__(self):
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("templates"),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Register custom filters
        self.jinja_env.filters['format_pvalue'] = self._format_pvalue
        
        # Template configurations
        self.templates = {
            "executive": {
                "pdf": "executive_summary_pdf.xml",
                "html": "executive_summary.html",
                "config": {
                    "sections": ["summary", "highlights", "recommendations"],
                    "max_plots": 5,
                    "technical_details": False
                }
            },
            "scientific": {
                "pdf": "scientific_report_pdf.xml",
                "html": "scientific_report.html",
                "config": {
                    "sections": ["abstract", "introduction", "methods", "results", "discussion", "references"],
                    "max_plots": 10,
                    "technical_details": True,
                    "citation_style": "apa"
                }
            },
            "annual_reef": {
                "pdf": "annual_reef_health_pdf.xml",
                "html": "annual_reef_health.html",
                "config": {
                    "sections": ["overview", "indicator_trends", "biodiversity", "trophic_structure", "recommendations"],
                    "max_plots": 15,
                    "technical_details": True,
                    "include_raw_data": False
                }
            }
        }
        
    def get_template(self, template_name):
        """Get a template configuration by name"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
            
        return self.templates[template_name]
        
    def render_template(self, template_path, data):
        """Render a Jinja2 template with the provided data"""
        template = self.jinja_env.get_template(template_path)
        return template.render(**data)
        
    def _format_pvalue(self, pvalue):
        """Format p-values according to scientific conventions"""
        if pvalue < 0.001:
            return "p < 0.001"
        else:
            return f"p = {pvalue:.3f}"
```

3. **PDF Report Generation Implementation**:

```python
class PDFReportGenerator:
    def generate(self, template, data, metadata=None):
        """
        Generate a PDF report using ReportLab
        
        Args:
            template: Template configuration
            data: Report data
            metadata: Report metadata
            
        Returns:
            Path to the generated PDF
        """
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        
        # Initialize metadata
        if metadata is None:
            metadata = {}
            
        title = metadata.get('title', 'Ecological Data Analysis Report')
        author = metadata.get('author', 'AI Analysis System')
        date = metadata.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # Create output filename
        filename = f"{title.replace(' ', '_').lower()}_{date}.pdf"
        filepath = os.path.join('reports', filename)
        os.makedirs('reports', exist_ok=True)
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create document elements list
        elements = []
        
        # Add title page
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {date}", normal_style))
        elements.append(Paragraph(f"Author: {author}", normal_style))
        elements.append(Spacer(1, 36))
        
        # Add sections based on template configuration
        for section in template['config']['sections']:
            if section in data:
                # Add section title
                elements.append(Paragraph(section.replace('_', ' ').title(), heading1_style))
                elements.append(Spacer(1, 12))
                
                section_data = data[section]
                
                # Add text content
                if 'text' in section_data:
                    elements.append(Paragraph(section_data['text'], normal_style))
                    elements.append(Spacer(1, 12))
                
                # Add subsections
                for subsection_name, subsection in section_data.items():
                    if subsection_name != 'text' and isinstance(subsection, dict):
                        elements.append(Paragraph(subsection_name.replace('_', ' ').title(), heading2_style))
                        elements.append(Spacer(1, 6))
                        
                        # Add subsection content
                        if 'text' in subsection:
                            elements.append(Paragraph(subsection['text'], normal_style))
                            elements.append(Spacer(1, 6))
                            
                        # Add figures
                        if 'figures' in subsection and isinstance(subsection['figures'], list):
                            for i, figure in enumerate(subsection['figures']):
                                if i >= template['config']['max_plots']:
                                    break
                                    
                                if 'path' in figure:
                                    img = Image(figure['path'], width=450, height=300)
                                    elements.append(img)
                                    if 'caption' in figure:
                                        elements.append(Paragraph(f"<i>Figure: {figure['caption']}</i>", normal_style))
                                    elements.append(Spacer(1, 12))
                                    
                        # Add tables
                        if 'tables' in subsection and isinstance(subsection['tables'], list):
                            for table_data in subsection['tables']:
                                if 'data' in table_data and isinstance(table_data['data'], list):
                                    table = Table(table_data['data'], repeatRows=1)
                                    table.setStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                    ])
                                    elements.append(table)
                                    if 'caption' in table_data:
                                        elements.append(Paragraph(f"<i>Table: {table_data['caption']}</i>", normal_style))
                                    elements.append(Spacer(1, 12))
        
        # Build the PDF document
        doc.build(elements)
        return filepath
```
