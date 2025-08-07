# Masterplan Sonnet: AI-Enhanced Ecological Data Exploration (Integrated Version)

This document presents a unified development plan for the `chatMPA_deep_research` application. It integrates the original "Masterplan Sonnet" with expert-level enhancements tailored to the `ltem_optimized_regions` table, ensuring the final tool has deep scientific rigor and intuitive ecological awareness.

---

## Objective

To evolve the current application into a state-of-the-art data exploration system for a specific long-term ecological monitoring (LTEM) dataset. The aim is to enable accurate, intuitive, and insightful analysis via natural language interaction, interactive dashboards, and automated reporting. This tool is **not** multimodal; its strength lies in its deep, domain-specific capabilities.

---

## Implementation Strategy

Each prompt includes: **Role**, **Goal**, **Input/Output**, **Constraints**, and **Implementation Guidance**. Enhanced guidance blocks follow to assist AI agents.

---

## Phase 1: Core AI Enhancement & Natural Language Interface

### Prompt 1.1: Natural Language Query Engine Integration

- **Role**: NLP & Data Engineer
- **Goal**: Enable natural language querying of the dataset
- **Input/Output**:
  - Input: User query (string)
  - Output: SQL or pandas query result (`DataFrame`)
- **Constraints**:
  - Use schema awareness
  - Handle errors gracefully
  - Optionally explain queries
- **Implementation Guidance**:
  - Use `pandas_nql` or `LangChain SQLChain`
  - Add `nl_query_engine.py` with function:

    ```python
    def process_natural_language_query(query: str) -> pd.DataFrame
    ```

  - Include schema:

    ```
    - date: datetime
    - region: str
    - species: str
    - abundance: int
    - temperature: float
    - depth: float
    ```

  - Test with 5 domain-relevant queries
  - Update `requirements.txt`

#### Enhanced Guidance for AI Agent

1. **Specific Enriched Schema Context**

    ```python
    SCHEMA_CONTEXT = """
    The data is in a pandas DataFrame with the following columns:
    - Label: str ('INV' or 'PEC')
    - Taxa1, Taxa2, Phylum, Species, Family, Genus: str
    - Year, Month, Day: int
    - Region, Reef: str
    - Habitat: str
    - Longitude, Latitude: float
    - Quantity: int
    - Size: float
    - Biomass: float (NaN for invertebrates)
    - MPA: str
    - TrophicLevelF, TrophicLevel, TrophicGroup: str/float
    - Area: float
    """
    ```

2. **Calculation Constraints**

    - **Density**: Use `Quantity.sum() / Area.sum()` and inform the user that the metric is density (org/m²).
    - **Filtering**: Automatically filter `Label == 'PEC'` when handling biomass or trophic queries and clarify this to the user.

3. **Metadata Awareness**

    - Normalize reef names (e.g., 'los_islotes' → 'Los Islotes').

---

### Prompt 1.2: Context-Aware AI Analysis Assistant

- **Role**: Conversational Assistant
- **Goal**: Maintain context and guide analyses
- **Input/Output**:
  - Input: Sequential user queries
  - Output: DataFrames, explanations, suggestions
- **Constraints**:
  - Keep memory of prior queries
  - Use predefined flows
- **Implementation Guidance**:
  - Use `ConversationBufferMemory`
  - Implement in `ai_analysis_assistant.py`
  - Add flows:
    1. Species diversity by region
    2. Trends in target species
    3. Temperature-abundance correlations
  - Export conversation history (JSON/Markdown)
  - Suggest follow-ups via `SentenceTransformers`

#### Enhanced Guidance for AI Agent

1. **Biodiversity Analysis**: Compare biodiversity indices across regions or MPA status.

2. **Trophic Structure Analysis**: Visualize biomass across trophic groups and reefs.

3. **Population Structure**: Plot size-frequency histograms for selected species.

---

## Phase 2: Visualization & Interactive Dashboards

### Prompt 2.1: Streamlit Dashboard Framework

- **Role**: UI/UX Developer
- **Goal**: Build dashboard with tabs, plots, and inputs
- **Input/Output**:
  - Input: Session state, query, parameters
  - Output: Streamlit interface
- **Constraints**:
  - Use `st.experimental_memo`
  - Add map visualization
- **Implementation Guidance**:
  - Use Streamlit v2+, Plotly, `streamlit-folium`
  - Add `dashboard_app.py`
  - Maintain session history
  - Include tooltips and examples

#### Enhanced Guidance for AI Agent

Tabs:

- **Overview**: Key metrics (e.g., biomass, richness)
- **Community Explorer**: Composition and ordination plots
- **Trophic Analysis**: Biomass pyramids and levels
- **Geospatial Viewer**: Reef-based maps
- **Data Dictionary**: Schema documentation

---

### Prompt 2.2: Visualization System API

- **Role**: Visualization Specialist
- **Goal**: Reusable API for ecological plots
- **Input/Output**:
  - Input: DataFrame, plot type, parameters
  - Output: Plot (static or interactive)
- **Constraints**:
  - Support matplotlib/plotly
  - Export to PNG/PDF/HTML
- **Implementation Guidance**:
  - Create `visualizations.py`
  - Functions:
    - `plot_trend()`
    - `plot_map()`
    - `plot_composition()`

#### Enhanced Guidance for AI Agent

Add:

- `plot_ordination()`
- `plot_biomass_pyramid()`
- `plot_size_histogram()`
- `plot_map_bubbles()`

---

## Phase 3: Automated Insights & Reporting

### Prompt 3.1: Automated Insights Engine

- **Role**: Statistical Analysis Assistant
- **Goal**: Discover patterns in LTEM data
- **Input/Output**:
  - Input: Cleaned `DataFrame`
  - Output: JSON list of insights
- **Constraints**:
  - Insight types: trend, anomaly, correlation, outlier
  - Provide narrative explanation
- **Implementation Guidance**:
  - Use `scipy.stats`, `prophet`, `statsmodels`, `sklearn`
  - Output format:

    ```json
    {
      "type": "trend",
      "description": "...",
      "score": 0.93,
      "supporting_visual": "<fig>"
    }
    ```

#### Enhanced Guidance for AI Agent

1. **Add statistical rigor**:

    ```json
    {
      "type": "correlation",
      "description": "Negative correlation found...",
      "narrative": "...",
      "score": -0.78,
      "statistical_test": "Pearson",
      "p_value": 0.005
    }
    ```

2. **New insight types**:

- **Trophic Structure Shift**
- **Significant Difference (ANOVA/Kruskal-Wallis)**

---

### Prompt 3.2: PDF & HTML Report Generation

- **Role**: Reporting System Architect
- **Goal**: Generate static + interactive reports
- **Input/Output**:
  - Input: Analysis session
  - Output: PDF/HTML reports
- **Constraints**:
  - Include summaries, plots, narrative
  - Interactive HTML filters
- **Implementation Guidance**:
  - Use `ReportLab`, `jinja2`, `plotly`, `MathJax`
  - Add templates: executive, scientific, monitoring

#### Enhanced Guidance for AI Agent

Templates:

- **Executive Summary**: For policymakers
- **Scientific Report**: Research-paper style
- **Annual Reef Health**: Yearly reef assessments

---

## Phase 4: Predictive Modeling

### Prompt 4.1: Predictive Modeling System

- **Role**: Ecological Forecaster
- **Goal**: Forecast trends and test hypotheses
- **Input/Output**:
  - Input: Historical data + params
  - Output: Forecast + plot + explanation
- **Constraints**:
  - Allow species-specific forecasting
  - Include model performance
- **Implementation Guidance**:
  - Use `Prophet`, `sklearn`, `SHAP`, `LIME`, `pyGAM`
  - Add sliders for what-if analysis

#### Enhanced Guidance for AI Agent

Models:

- **Time Series**: Biomass forecast by region
- **GAMs**: Non-linear ecological responses
- **Classification**: Habitat type prediction

Constraint:

- Always include performance metrics (e.g., R², AIC) and residual plots.

---

## Phase 5: User Experience & Deployment

### Prompt 5.1: UX and Help System

- **Role**: Front-End Developer
- **Goal**: Improve accessibility and help features
- **Input/Output**:
  - Input: Session
  - Output: Guided UI + help
- **Constraints**:
  - Support non-technical users
  - Include workflows and themes
- **Implementation Guidance**:
  - Use `streamlit-extras`, `streamlit-option-menu`

#### Enhanced Guidance for AI Agent

- Add **Data Dictionary Tab**
- Implement **Guided Workflows** (e.g., "Compare Two Reefs")

---

### Prompt 5.2: Deployment System

- **Role**: DevOps Engineer
- **Goal**: Containerize and deploy the app
- **Output**: Docker image, install script, setup guide
- **Constraints**:
  - Cloud/local support
  - Backup/export/logging
- **Implementation Guidance**:
  - Add `deployment/` folder:
    - `Dockerfile`, `docker-compose.yml`, `install.sh`, `cloud_config.yaml`
    - Backup/export scripts
    - Documentation (`README.md`)

#### Enhanced Guidance for AI Agent

- **Data Versioning**: Display active data version
- **Performance Planning**: Mention scalability to Polars if needed

---

## Implementation Roadmap

1. **Phase 1**: Natural language + AI logic  
2. **Phase 2**: Visual interface + ecological plots  
3. **Phase 3**: Automated insight + reporting  
4. **Phase 4**: Forecasting + ecological hypothesis testing  
5. **Phase 5**: UX improvement + deployment

---

## Success Metrics

- Time-to-insight reduction
- NLP query accuracy and relevance
- Validity of automated insights
- User adoption and satisfaction
- Quality of generated reports
- Deployment performance and stability
- Novel ecological hypotheses derived from insights 