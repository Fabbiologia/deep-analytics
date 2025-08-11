# Agentic Implementation Guide - Phase 5: User Experience & Deployment

## Phase 5: User Experience & Deployment

### 5.1 UX and Help System - Agentic Instructions

1. **User Experience Manager Architecture**:

```python
class UXManager:
    def __init__(self):
        self.help_system = HelpSystem()
        self.workflow_manager = WorkflowManager()
        self.theme_manager = ThemeManager()
        self.onboarding_system = OnboardingSystem()
        
    def initialize_interface(self, config=None):
        """Initialize the UI with the specified configuration"""
        # Set up theme
        self.theme_manager.apply_theme(config.get('theme', 'light'))
        
        # Set up navigation
        self._setup_navigation()
        
        # Set up help system
        self.help_system.initialize()
        
    def _setup_navigation(self):
        """Create the main navigation structure"""
        import streamlit as st
        from streamlit_option_menu import option_menu
        
        # Create horizontal menu
        selected = option_menu(
            "Navigation",
            ["Dashboard", "Analysis", "Reports", "Data Dictionary", "Help"],
            icons=['house', 'graph-up', 'file-text', 'book', 'question-circle'],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal"
        )
        
        st.session_state.current_page = selected
```

2. **Guided Workflow Implementation**:

```python
class WorkflowManager:
    def __init__(self):
        self.workflows = {
            "compare_reefs": CompareReefsWorkflow(),
            "temporal_analysis": TemporalAnalysisWorkflow(),
            "biodiversity_assessment": BiodiversityAssessmentWorkflow(),
            "trophic_structure": TrophicStructureWorkflow(),
            "custom_query": CustomQueryWorkflow()
        }
        
    def start_workflow(self, workflow_name):
        """Initialize and start a guided workflow"""
        if workflow_name not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_name]
        return workflow.initialize()
        
    def render_current_workflow(self):
        """Render the current active workflow"""
        import streamlit as st
        
        if 'active_workflow' not in st.session_state:
            return False
            
        workflow_name = st.session_state.active_workflow
        if workflow_name not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_name]
        return workflow.render_current_step()
```

3. **Help System Implementation**:

```python
class HelpSystem:
    def __init__(self):
        self.help_content = {
            "dashboard": {
                "title": "Dashboard Help",
                "content": "The dashboard provides an overview of key metrics...",
                "sections": {
                    "filters": "Use these filters to narrow down the data...",
                    "charts": "These charts show the main trends in your data...",
                    "metrics": "These key metrics summarize the state of the ecosystem..."
                }
            },
            # More help sections here
        }
        
    def show_contextual_help(self, context):
        """Show help relevant to the current context"""
        import streamlit as st
        
        if context in self.help_content:
            help_data = self.help_content[context]
            
            with st.expander("Help"):
                st.subheader(help_data["title"])
                st.write(help_data["content"])
                
                for section, content in help_data.get("sections", {}).items():
                    st.subheader(section.capitalize())
                    st.write(content)
                    
    def show_data_dictionary(self):
        """Show the data dictionary with column explanations"""
        import streamlit as st
        
        st.header("Data Dictionary")
        st.write("This section explains the columns in the dataset.")
        
        # Create a DataFrame with column explanations
        import pandas as pd
        
        data_dict = {
            "Column": [
                "Label", "Taxa1", "Taxa2", "Phylum", "Species", 
                "Year", "Region", "Reef", "Quantity", "Biomass", 
                "MPA", "TrophicLevel", "Area"
            ],
            "Description": [
                "Label for data type ('INV' for invertebrates, 'PEC' for fish)",
                "Primary taxonomic classification",
                "Secondary taxonomic classification",
                "Phylum of the organism",
                "Species name",
                "Year of observation",
                "Geographic region",
                "Specific reef name",
                "Count of organisms",
                "Biomass of organisms (NaN for invertebrates)",
                "Marine Protected Area status",
                "Trophic level (1-5 scale)",
                "Area surveyed in square meters"
            ],
            "Type": [
                "str", "str", "str", "str", "str", 
                "int", "str", "str", "int", "float",
                "str", "float", "float"
            ],
            "Example": [
                "PEC", "Actinopterygii", "Perciformes", "Chordata", "Scarus ghobban",
                "2018", "Cabo Pulmo", "Los Islotes", "5", "1250.5",
                "Yes", "4.2", "100"
            ]
        }
        
        df_dict = pd.DataFrame(data_dict)
        st.table(df_dict)
```

### 5.2 Deployment System - Agentic Instructions

1. **Docker Deployment Architecture**:

```python
class DeploymentManager:
    def __init__(self):
        self.docker_manager = DockerManager()
        self.config_manager = ConfigManager()
        self.backup_manager = BackupManager()
        
    def prepare_deployment(self, deployment_type="docker"):
        """Prepare the application for deployment"""
        if deployment_type == "docker":
            return self._prepare_docker_deployment()
        elif deployment_type == "local":
            return self._prepare_local_deployment()
        else:
            raise ValueError(f"Unsupported deployment type: {deployment_type}")
            
    def _prepare_docker_deployment(self):
        """Create Docker deployment files"""
        # Create Dockerfile
        dockerfile_content = self._generate_dockerfile()
        
        # Create docker-compose.yml
        docker_compose_content = self._generate_docker_compose()
        
        # Create install script
        install_script_content = self._generate_install_script()
        
        # Create cloud configuration
        cloud_config_content = self._generate_cloud_config()
        
        # Return all generated files
        return {
            "Dockerfile": dockerfile_content,
            "docker-compose.yml": docker_compose_content,
            "install.sh": install_script_content,
            "cloud_config.yaml": cloud_config_content
        }
```

2. **Backup and Data Management**:

```python
class BackupManager:
    def setup_backup_system(self, config):
        """Set up automated backups based on configuration"""
        import os
        
        # Create backup directory
        os.makedirs("backups", exist_ok=True)
        
        # Generate backup script
        backup_script = self._generate_backup_script(config)
        
        # Generate restore script
        restore_script = self._generate_restore_script(config)
        
        return {
            "backup.py": backup_script,
            "restore.py": restore_script
        }
        
    def _generate_backup_script(self, config):
        """Generate a Python script for backing up data"""
        script = """
import os
import shutil
import datetime
import sqlite3
import argparse
import json

def backup_database(config):
    """Backup the SQLite database"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join('backups', timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Copy the database file
    shutil.copy(config['database_path'], os.path.join(backup_dir, 'database.sqlite'))
    
    # Export as SQL if specified
    if config.get('export_sql', False):
        conn = sqlite3.connect(config['database_path'])
        with open(os.path.join(backup_dir, 'database_dump.sql'), 'w') as f:
            for line in conn.iterdump():
                f.write(f'{line}\\n')
        conn.close()
    
    # Save configuration
    with open(os.path.join(backup_dir, 'backup_info.json'), 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'database': config['database_path'],
            'version': config.get('version', '1.0.0')
        }, f, indent=2)
        
    print(f"Backup completed to {backup_dir}")
    return backup_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backup the application database')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    backup_database(config)
"""
        return script
```

3. **Documentation Generation**:

```python
class DocumentationGenerator:
    def generate_readme(self, project_info):
        """Generate a README.md file for the project"""
        readme = f"""# {project_info['title']}

## Overview
{project_info['description']}

## Features
{self._format_list(project_info.get('features', []))}

## Installation

### Prerequisites
{self._format_list(project_info.get('prerequisites', []))}

### Docker Installation
1. Make sure you have Docker and Docker Compose installed
2. Clone this repository
3. Run `./install.sh` to set up the environment
4. Start the application with `docker-compose up`

### Local Installation
1. Create a virtual environment: `python -m venv env`
2. Activate it: `source env/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`

## Usage
{project_info.get('usage', 'Instructions on how to use the application.')}

## Configuration
The application can be configured using the `config.json` file:

```json
{
  "database_path": "path/to/database.sqlite",
  "api_key": "your_openai_api_key",
  "theme": "light",
  "log_level": "info"
}
```

## Backup and Restore
- To create a backup: `python backup.py`
- To restore from backup: `python restore.py --backup backup_folder`

"""
        return readme
```

4. **Multi-Agent Supervisor Architecture for Coordinator Class**:

```python
class AgentSupervisor:
    def __init__(self):
        # Create agent registry
        self.agents = {
            "query": None,  # Will be instantiated as needed
            "analysis": None,
            "visualization": None,
            "insights": None, 
            "reporting": None,
            "prediction": None
        }
        
        # Configuration
        self.config = {
            "max_concurrent_agents": 5,
            "max_iterations": 3,
            "allow_clarification": True
        }
        
        # Memory and state management
        self.conversation_memory = None
        self.research_state = None
        
    def initialize(self):
        """Initialize the supervisor and its components"""
        from langchain.memory import ConversationBufferMemory
        
        # Initialize memory
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize research state
        self.research_state = {
            "current_tasks": {},
            "completed_tasks": [],
            "insights": [],
            "clarification_needed": False
        }
        
    def process_request(self, user_query):
        """Process a user request by delegating to appropriate agents"""
        # Step 1: Determine intent and required agents
        required_agents = self._determine_required_agents(user_query)
        
        # Step 2: Load or instantiate required agents
        self._load_agents(required_agents)
        
        # Step 3: Create execution plan
        execution_plan = self._create_execution_plan(user_query, required_agents)
        
        # Step 4: Execute plan with agents
        results = self._execute_plan(execution_plan)
        
        # Step 5: Synthesize results
        final_response = self._synthesize_results(results, user_query)
        
        # Step 6: Update memory and state
        self._update_memory(user_query, final_response)
        
        return final_response
        
    def _determine_required_agents(self, query):
        """Determine which agents are needed based on the query"""
        required_agents = []
        
        # Always include query agent for natural language processing
        required_agents.append("query")
        
        # Check for analysis needs
        if any(kw in query.lower() for kw in ["analyze", "compare", "trend", "correlation"]):
            required_agents.append("analysis")
            
        # Check for visualization needs
        if any(kw in query.lower() for kw in ["plot", "chart", "graph", "visualize", "map"]):
            required_agents.append("visualization")
            
        # Check for insights needs
        if any(kw in query.lower() for kw in ["insight", "discover", "find pattern", "anomaly"]):
            required_agents.append("insights")
            
        # Check for reporting needs
        if any(kw in query.lower() for kw in ["report", "summary", "pdf", "document", "export"]):
            required_agents.append("reporting")
            
        # Check for prediction needs
        if any(kw in query.lower() for kw in ["predict", "forecast", "future", "estimate"]):
            required_agents.append("prediction")
            
        return required_agents
```

## Integration with Multi-Agent Architecture

Integrating the above components with the multi-agent supervisor architecture from the memory:

```python
def integrate_with_multi_agent_system():
    """
    Integrate the ecological analysis components with a multi-agent system
    based on the LangChain's deep_research project architecture.
    """
    # Create the agent supervisor
    supervisor = AgentSupervisor()
    
    # Register specialized agents
    supervisor.register_agent("method_selector", MethodSelectorAgent())
    supervisor.register_agent("data_analysis", DataAnalysisAgent())
    supervisor.register_agent("report_synthesis", ReportSynthesisAgent())
    supervisor.register_agent("critique", CritiqueAgent())
    
    # Set up configuration
    supervisor.configure({
        "max_concurrent_research_units": 5,
        "max_researcher_iterations": 3,
        "allow_clarification_questions": True,
        "specialized_models": {
            "summarization": "gpt-4-turbo",
            "research": "gpt-4-turbo",
            "compression": "gpt-3.5-turbo-16k",
            "final_report": "gpt-4-turbo"
        }
    })
    
    # Extend the existing AnalysisState
    class EnhancedAnalysisState(AnalysisState):
        def __init__(self):
            super().__init__()
            self.agent_outputs = {}
            self.current_agent = None
            self.completed_agents = []
            self.pending_agents = []
    
    # Return the integrated system
    return {
        "supervisor": supervisor,
        "state": EnhancedAnalysisState(),
        "tools": {
            "database_query_tool": database_query_tool,
            "python_interpreter_tool": python_interpreter_tool,
            "visualization_tool": visualization_tool
        }
    }
```
