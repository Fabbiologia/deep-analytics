# Ecological Data Analysis Agent

An advanced, AI-powered data analysis agent for ecological research with SQL database connectivity, Python analysis capabilities, and comprehensive visualization tools.

## 📋 Overview

This tool provides an intuitive way to analyze ecological data by:

- Creating **multi-step analysis plans** that orchestrate complex workflows
- Connecting to **SQL databases** and performing structured data retrieval
- Running **Python code** for advanced statistical analysis and data manipulation
- Generating **data visualizations** including bar charts, scatter plots, and box plots
- Providing a **clean, user-friendly interface** with elegant loading animations

Designed specifically for marine ecological research, this tool helps researchers and stakeholders understand trends in fish biomass, abundance, diversity, and other ecological metrics across different regions.

## 🌟 Key Features

- **🧠 Master Orchestrator Architecture**: Formulates multi-step plans to tackle complex analytical questions
- **🔍 Structured SQL Data Retrieval**: Executes SQL queries against ecological databases
- **🐍 Python Execution Engine**: Runs Python code for advanced statistical analysis and data transformation
- **📊 Visualization Suite**: Creates bar charts, scatter plots, and box plots based on SQL query results
- **💡 Self-Planning Workflow**: Determines the best sequence of tools to answer multi-part questions
- **🖥️ User-Friendly Interface**: Clean terminal UI with loading animations and concise responses
- **🎯 Region-Specific Analysis**: Supports filtering by specific marine regions (La Paz, Cabo Pulmo, etc.)
- **📈 Multiple Ecological Metrics**: Analyzes biomass, abundance, density, and diversity metrics

## 🛠️ Project Structure

```bash
chatMPA_deep_research/
├── main.py              # Main agent with SQL, Python, and visualization capabilities
├── tools.py             # Database and analysis utilities
├── requirements.txt     # Python dependencies
├── .env                 # Database configuration (not in repo)
├── .gitignore           # Git ignore file
├── *.png                # Generated visualization images
└── README.md            # This documentation
```

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- Access to an ecological database (PostgreSQL, MySQL, or SQLite)
- OpenAI API key for LLM capabilities
- Python libraries: langchain, pandas, seaborn, matplotlib

### Setup Instructions

1. **Clone this repository:**

   ```bash
   git clone <repository-url>
   cd chatMPA_deep_research
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   Create a `.env` file in the project root:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=your_database_connection_string_here
   ```

   **Database URL Examples:**
   - PostgreSQL: `postgresql://username:password@host:port/database`
   - MySQL: `mysql://username:password@host:port/database`
   - SQLite: `sqlite:///path/to/database.db`

## 🚀 Usage

### Running the Analysis Tool

```bash
python main.py
```

The tool will start with a friendly interface showing examples of questions you can ask.

### Example Analysis Workflow

1. **Start the Agent:**

   ```bash
   python main.py
   ```


2. **Ask a Question:**

   ```text
   🤔 Your Question: What's the relationship between size and biomass for triggerfish?
   ```

3. **Agent Plans and Executes:**
   The agent creates a multi-step plan:
   - Formulates SQL queries to retrieve relevant data
   - Determines which visualization type is most appropriate
   - Sequences tools in the optimal order

4. **Watch the Analysis:**
   A spinner animation shows progress while the agent:
   - 🔌 Connects to your database
   - 🔍 Executes SQL queries
   - 🐍 Runs Python code as needed
   - 📊 Generates appropriate visualizations

5. **Get Comprehensive Results:**
   Receive detailed answers including:
   - Key findings from the data
   - Generated visualization images
   - Statistical insights and interpretations

### Example Questions

#### SQL Queries

- "How many unique species are in the database?"
- "What are the top 5 most observed species?"
- "What's the average biomass by location?"

#### Python Analysis

- "Calculate the correlation between size and biomass for parrotfish"
- "What's the statistical significance of biomass differences between locations?"
- "Generate summary statistics for all species in Cabo Pulmo"

#### Visualizations

- "Create a bar chart showing average biomass by species"
- "Make a scatter plot of size vs. biomass for triggerfish"
- "Show me a box plot of fish sizes by location"

#### Complex Multi-Step Analysis

- "Show me a scatter plot of size vs. biomass for the most observed species and calculate the correlation coefficient"
- "What are the three largest fish species by average size, and create box plots showing their size distributions by location"

## 🔧 Technical Details

### Database Requirements

The tool automatically detects and adapts to your database structure:

- **Supported databases:** PostgreSQL, MySQL, SQLite
- **Table discovery:** Automatically finds tables with ecological data
- **Column mapping:** Intelligently matches columns for metrics (biomass, abundance, etc.) and regions
- **Temporal analysis:** Detects date/time columns for trend analysis

### Analysis Capabilities

- **SQL Analysis:** Complex queries, aggregations, filters, joins, and subqueries
- **Python Analysis:** Advanced statistical analysis with pandas, numpy, scipy
- **Visualization Types:**
  - **Bar Charts:** For comparing values across categories
  - **Scatter Plots:** For exploring relationships between two numerical variables (with regression lines)
  - **Box Plots:** For showing distributions across categories (with quartiles and outliers)
- **Multi-Step Reasoning:** Planning and executing complex analytical workflows
- **Tool Orchestration:** Selecting and sequencing appropriate tools based on the task

### AI Integration

- **Large Language Model:** Uses OpenAI models for planning and analysis
- **LangChain Framework:** Integrates various tools and capabilities
- **Agent Architecture:**
  - **Master Orchestrator:** Plans and coordinates analytical workflows
  - **SQL Tool:** Structured data retrieval from databases
  - **Python REPL Tool:** Executes Python code for advanced analysis
  - **Visualization Tools:** Creates appropriate charts based on data characteristics
- **Self-Improving:** Learns from interactions to provide better analysis

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys or secrets to your repository. Always use environment variables (.env file).

2. **Database Access**: Consider using read-only database credentials for the analysis tool to prevent accidental data modification.

3. **Data Privacy**: The tool only reads data for analysis - it does not modify or delete database records.

## 🛠️ Troubleshooting

### Common Issues

#### No database connection found

- Check that your `.env` file exists and contains valid `DATABASE_URL`
- Verify database credentials and network connectivity

#### Table not found or not accessible

- Ensure the table name exists in your database
- Check database permissions for the user account

#### No column found for metric

- The tool looks for columns containing keywords like 'biomass', 'abundance', etc.
- You may need to rename columns or specify them more clearly

#### OpenAI API errors

- Verify your `OPENAI_API_KEY` is valid and has sufficient credits
- Check your internet connection

## 📚 References

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain SQL Tools](https://python.langchain.com/docs/modules/agents/toolkits/sql_database)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## 📝 License

This project is designed for ecological research and follows best practices for scientific reproducibility.

---

### Note

*This agent follows best practices for code organization, documentation, and reproducibility as defined in the project development rules. Updated: July 30, 2025*
