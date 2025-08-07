# Agentic Implementation Guide for Ecological Data Analysis

This document provides detailed agentic instructions for implementing the capabilities outlined in the Masterplan Sonnet. Each section includes specific guidance for AI agents on how to build and integrate the various components.

## Core Agentic Architecture Design

Before implementing individual components, establish a unified agentic architecture:

```python
# Architecture Overview
class AgentCoordinator:
    def __init__(self):
        self.agents = {
            "query": NLQueryAgent(),
            "analysis": AnalysisAgent(),
            "visualization": VisualizationAgent(),
            "insights": InsightsAgent(),
            "reporting": ReportingAgent(),
            "prediction": PredictionAgent()
        }
        self.memory = ConversationMemory()
        self.workflow_manager = WorkflowManager()
    
    def process_request(self, user_input):
        # Determine intent and delegate to appropriate agent
        pass
```

## Phase 1: Core AI Enhancement & Natural Language Interface

### 1.1 Natural Language Query Engine - Agentic Instructions

1. **Initialization Setup**:

```python
def setup_nlquery_agent():
    # Define environmental variables
    os.environ['OPENAI_API_KEY'] = config.get('OPENAI_API_KEY')
    
    # Initialize the database schema parser
    schema_parser = SchemaParser(SCHEMA_CONTEXT)
    
    # Create the NL Query Agent with components
    return NLQueryAgent(
        schema_parser=schema_parser,
        query_converter=QueryConverter(),
        result_formatter=ResultFormatter(),
        error_handler=QueryErrorHandler()
    )
```

2. **Query Processing Pipeline**:

```python
class NLQueryAgent:
    def process_query(self, query):
        # Step 1: Parse and understand query intent
        intent = self.parse_intent(query)
        
        # Step 2: Convert to appropriate query language
        if intent.requires_sql:
            executable_query = self.sql_converter.convert(query, self.schema)
        else:
            executable_query = self.pandas_converter.convert(query, self.schema)
            
        # Step 3: Execute with error handling
        try:
            result = self.execute_query(executable_query)
            return self.format_result(result, query)
        except Exception as e:
            return self.error_handler.handle(e, query, executable_query)
```

3. **Ecological Domain Awareness Implementation**:

```python
class EcologicalQueryEnhancer:
    def enhance_query(self, query, converted_query):
        # Check for ecological concepts that need special handling
        if "biomass" in query.lower() or "trophic" in query.lower():
            # Add Label filter for PEC
            if "WHERE" in converted_query:
                return converted_query.replace("WHERE", "WHERE Label = 'PEC' AND")
            else:
                # Add WHERE clause if none exists
                return converted_query + " WHERE Label = 'PEC'"
                
        # Handle density calculations
        if "density" in query.lower():
            # Ensure proper density calculation using area
            if "GROUP BY" in converted_query:
                # Complex case with grouping
                return self._add_density_calculation_grouped(converted_query)
            else:
                # Simple case
                return self._add_density_calculation_simple(converted_query)
        
        return converted_query
```

### 1.2 Context-Aware AI Analysis Assistant - Agentic Instructions

1. **Memory Management Implementation**:

```python
class AnalysisMemoryManager:
    def __init__(self):
        self.conversation_memory = ConversationBufferMemory(memory_key="chat_history")
        self.analysis_history = []
        self.entity_memory = {}  # For tracking specific entities of interest
    
    def add_interaction(self, query, response):
        # Add to conversation memory
        self.conversation_memory.chat_memory.add_user_message(query)
        self.conversation_memory.chat_memory.add_ai_message(str(response))
        
        # Extract and store entities
        entities = self._extract_entities(query)
        for entity in entities:
            if entity not in self.entity_memory:
                self.entity_memory[entity] = {"mentions": 1, "last_mentioned": datetime.now()}
            else:
                self.entity_memory[entity]["mentions"] += 1
                self.entity_memory[entity]["last_mentioned"] = datetime.now()
    
    def get_relevant_context(self, query):
        # Retrieve relevant history based on semantic similarity
        return self.conversation_memory.load_memory_variables({})["chat_history"]
    
    def get_analysis_flow(self, query):
        # Determine if query fits into a predefined analysis flow
        flows = {
            "biodiversity": self._check_biodiversity_intent,
            "trends": self._check_trends_intent,
            "correlation": self._check_correlation_intent,
            # Add more flows
        }
        
        for flow_name, check_func in flows.items():
            if check_func(query):
                return flow_name, self._get_flow_steps(flow_name)
                
        return None, None
```

2. **Analysis Flow Management**:

```python
class AnalysisFlowManager:
    def __init__(self):
        self.active_flows = {}
        self.flow_templates = {
            "biodiversity": [
                {"step": "data_selection", "description": "Select regions/sites to compare"},
                {"step": "index_calculation", "description": "Calculate biodiversity indices"},
                {"step": "visualization", "description": "Visualize biodiversity comparison"},
                {"step": "statistical_test", "description": "Run statistical tests"},
                {"step": "interpretation", "description": "Interpret results"}
            ],
            "trophic_structure": [
                {"step": "data_filtering", "description": "Filter by regions and trophic groups"},
                {"step": "pyramid_calculation", "description": "Calculate biomass at each trophic level"},
                {"step": "pyramid_visualization", "description": "Create trophic pyramid visualization"},
                {"step": "comparison", "description": "Compare across regions/reefs"}
            ],
            # Add more flows
        }
    
    def start_flow(self, flow_name, context=None):
        flow_id = str(uuid.uuid4())
        steps = deepcopy(self.flow_templates.get(flow_name, []))
        
        self.active_flows[flow_id] = {
            "name": flow_name,
            "steps": steps,
            "current_step": 0,
            "context": context or {},
            "results": {},
            "started_at": datetime.now()
        }
        
        return flow_id, steps[0]
    
    def advance_flow(self, flow_id, step_result):
        if flow_id not in self.active_flows:
            return None
            
        flow = self.active_flows[flow_id]
        flow["results"][flow["current_step"]] = step_result
        flow["current_step"] += 1
        
        if flow["current_step"] >= len(flow["steps"]):
            # Flow completed
            return self._compile_flow_results(flow_id)
            
        # Return next step
        return flow["steps"][flow["current_step"]]
```

3. **Analysis Assistant Interface**:

```python
class AnalysisAssistant:
    def __init__(self):
        self.memory_manager = AnalysisMemoryManager()
        self.flow_manager = AnalysisFlowManager()
        self.nlp_engine = SentenceTransformer('all-MiniLM-L6-v2')
        
    def process_query(self, query):
        # Get context from previous interactions
        context = self.memory_manager.get_relevant_context(query)
        
        # Check if query fits into an analysis flow
        flow_name, flow_steps = self.memory_manager.get_analysis_flow(query)
        
        if flow_name:
            # User is starting a known analysis flow
            flow_id, first_step = self.flow_manager.start_flow(flow_name)
            response = self._generate_flow_step_response(flow_id, first_step, query)
        else:
            # Check if continuing an active flow
            flow_id = self._check_if_continuing_flow(query)
            if flow_id:
                # Process the current step of the flow
                response = self._process_flow_step(flow_id, query)
            else:
                # Regular query processing
                response = self._process_regular_query(query, context)
        
        # Update memory
        self.memory_manager.add_interaction(query, response)
        return response
```

## Phase 2: Visualization & Interactive Dashboards

### 2.1 Streamlit Dashboard Framework - Agentic Instructions

1. **Dashboard Component Management**:

```python
class DashboardComponentManager:
    def __init__(self):
        self.components = {
            "overview": OverviewTab(),
            "community_explorer": CommunityExplorerTab(),
            "trophic_analysis": TrophicAnalysisTab(),
            "geospatial_viewer": GeospatialViewerTab(),
            "data_dictionary": DataDictionaryTab()
        }
        self.session_manager = SessionManager()
    
    def render_dashboard(self):
        # Set page config
        st.set_page_config(
            page_title="Ecological Data Explorer",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Create sidebar navigation
        selected_tab = st.sidebar.radio(
            "Navigation", 
            list(self.components.keys()),
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Render the selected tab
        self.components[selected_tab].render()
        
        # Store the session state
        self.session_manager.update_state()
```

2. **Interactive Components Implementation**:

```python
class InteractiveComponentFactory:
    @staticmethod
    def create_filter_widgets(dataframe, session_state):
        # Create region filter
        regions = sorted(dataframe['Region'].unique().tolist())
        selected_regions = st.multiselect(
            "Select Regions", 
            regions, 
            default=session_state.get('selected_regions', regions[:2])
        )
        session_state.selected_regions = selected_regions
        
        # Create year range slider
        years = sorted(dataframe['Year'].unique().tolist())
        year_range = st.slider(
            "Year Range",
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years)),
            step=1,
            key="year_slider"
        )
        session_state.year_range = year_range
        
        # Create species selector with search
        species_list = sorted(dataframe['Species'].unique().tolist())
        species_search = st.text_input("Search Species", key="species_search")
        filtered_species = [sp for sp in species_list if species_search.lower() in sp.lower()]
        selected_species = st.multiselect(
            "Select Species",
            filtered_species,
            default=session_state.get('selected_species', []) if species_search == "" else []
        )
        session_state.selected_species = selected_species
        
        # Return the combined filter state
        return {
            "regions": selected_regions,
            "year_range": year_range,
            "species": selected_species
        }
```

### 2.2 Visualization System API - Agentic Instructions

1. **Visualization Factory Implementation**:

```python
class EcologicalVisualizationFactory:
    def __init__(self):
        self.renderers = {
            "matplotlib": MatplotlibRenderer(),
            "plotly": PlotlyRenderer(),
            "folium": FoliumMapRenderer()
        }
        
    def create_visualization(self, data, viz_type, params, renderer="plotly"):
        # Select the appropriate visualization method
        if viz_type == "trend":
            visualization = self._create_trend_plot(data, params)
        elif viz_type == "composition":
            visualization = self._create_composition_plot(data, params)
        elif viz_type == "map":
            visualization = self._create_map_plot(data, params)
        elif viz_type == "ordination":
            visualization = self._create_ordination_plot(data, params)
        elif viz_type == "biomass_pyramid":
            visualization = self._create_biomass_pyramid(data, params)
        elif viz_type == "size_histogram":
            visualization = self._create_size_histogram(data, params)
        elif viz_type == "map_bubbles":
            visualization = self._create_map_bubbles(data, params)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
            
        # Render using the selected renderer
        return self.renderers[renderer].render(visualization)
```

2. **Ecological Plot Implementations**:

```python
class EcologicalPlotGenerator:
    def generate_ordination_plot(self, df, ordination_type="nmds", distance_metric="bray"):
        """
        Generate an ordination plot (NMDS, PCA, etc.) for community data
        
        Args:
            df: DataFrame with species abundance/biomass data
            ordination_type: Type of ordination ('nmds', 'pca', 'rda', 'ca')
            distance_metric: Distance metric for ordination
            
        Returns:
            Figure object with the ordination plot
        """
        # Prepare data matrix (species as columns, sites as rows)
        pivot_df = pd.pivot_table(
            df, 
            values='Quantity', 
            index=['Region', 'Reef', 'Year'], 
            columns='Species', 
            aggfunc='sum',
            fill_value=0
        )
        
        # Get metadata for plotting
        metadata = pivot_df.index.to_frame()
        
        if ordination_type == "nmds":
            # Generate NMDS ordination
            import skbio.stats.ordination as ord
            
            # Calculate distance matrix
            dist_matrix = skbio.diversity.beta_diversity(
                distance_metric, 
                pivot_df.values, 
                ids=pivot_df.index
            )
            
            # Perform NMDS
            nmds_result = ord.nmds(dist_matrix, number_of_dimensions=2)
            
            # Create ordination plot
            fig = px.scatter(
                x=nmds_result.samples.iloc[:, 0],
                y=nmds_result.samples.iloc[:, 1],
                color=metadata['Region'],
                symbol=metadata['Year'],
                hover_data={
                    'Reef': metadata['Reef'],
                    'Year': metadata['Year']
                },
                labels={
                    'x': 'NMDS1',
                    'y': 'NMDS2'
                },
                title='NMDS Ordination of Communities'
            )
            
            # Add stress value
            fig.add_annotation(
                text=f"Stress: {nmds_result.stress:.3f}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False
            )
            
            return fig
```

3. **Visualization Export System**:

```python
class VisualizationExporter:
    def export_visualization(self, figure, format_type, filename=None, **kwargs):
        """
        Export a visualization to various formats
        
        Args:
            figure: The figure object to export
            format_type: One of 'png', 'pdf', 'svg', 'html', 'json'
            filename: Output filename (optional)
            **kwargs: Additional export parameters
            
        Returns:
            Path to exported file or the raw content
        """
        if filename is None:
            filename = f"visualization_{int(time.time())}"
            
        # Add appropriate extension if not present
        if not filename.endswith(f'.{format_type}'):
            filename = f"{filename}.{format_type}"
            
        # Create output directory if needed
        os.makedirs('exports', exist_ok=True)
        filepath = os.path.join('exports', filename)
            
        # Export based on format type
        if hasattr(figure, 'write_image') and format_type in ['png', 'jpg', 'pdf', 'svg']:
            # Handle Plotly figure
            figure.write_image(
                filepath,
                width=kwargs.get('width', 1200),
                height=kwargs.get('height', 800),
                scale=kwargs.get('scale', 2)
            )
        elif hasattr(figure, 'write_html') and format_type == 'html':
            # Export interactive HTML
            figure.write_html(
                filepath,
                include_plotlyjs=kwargs.get('include_plotlyjs', 'cdn'),
                full_html=kwargs.get('full_html', True)
            )
        elif hasattr(figure, 'savefig'):
            # Handle Matplotlib figure
            figure.savefig(
                filepath,
                dpi=kwargs.get('dpi', 300),
                bbox_inches=kwargs.get('bbox_inches', 'tight')
            )
        else:
            raise ValueError(f"Cannot export figure of type {type(figure)} to {format_type}")
            
        return filepath
```
