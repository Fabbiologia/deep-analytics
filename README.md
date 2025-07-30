# Ecological Data Analysis Tool

A streamlined, narrative-driven tool for analyzing ecological and marine data with real database connectivity and human-readable explanations.

## 📋 Overview

This tool provides an intuitive way to analyze ecological data by:

- Creating **narrative, human-readable analysis plans** that anyone can understand and approve
- Connecting to **real databases** and performing genuine statistical analysis (no mock data!)
- Providing **verbose, step-by-step terminal output** showing exactly what's happening during analysis
- Generating meaningful insights from actual ecological monitoring data

Designed specifically for marine ecological research, this tool helps researchers and stakeholders understand trends in fish biomass, abundance, diversity, and other ecological metrics across different regions.

## 🌟 Key Features

- **📋 Narrative Plans**: Analysis plans written in plain English that non-technical stakeholders can understand and approve
- **🔍 Verbose Terminal Output**: Real-time step-by-step progress with clear status messages during analysis
- **🗄️ Real Database Analysis**: Connects to actual ecological databases and performs genuine statistical analysis
- **📊 Trend Analysis**: Calculates real temporal trends, percentage changes, and statistical significance
- **🎯 Region-Specific**: Supports analysis of specific marine regions (La Paz, Cabo Pulmo, Gulf of California, etc.)
- **📈 Multiple Metrics**: Analyzes biomass, abundance, density, and diversity metrics

## 🛠️ Project Structure

```
chatMPA_deep_research/
├── main.py              # Main analysis tool with narrative interface
├── tools.py             # Database and statistical analysis utilities
├── requirements.txt     # Python dependencies
├── .env                 # Database configuration (not in repo)
├── .gitignore          # Git ignore file
└── README.md           # This documentation
```

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- Access to an ecological database (PostgreSQL, MySQL, or SQLite)
- OpenAI API key for natural language processing

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

1. **Ask a Question:**
   ```
   🤔 Your Question: How has fish biomass changed in La Paz?
   ```

2. **Review the Narrative Plan:**
   The tool creates a human-readable plan explaining:
   - What you're really asking
   - What steps will be taken
   - Why this analysis matters
   - What you can expect to learn

3. **Approve and Execute:**
   ```
   🚀 Execute this analysis plan? (yes/no): yes
   ```

4. **Watch Real-Time Analysis:**
   See verbose output as the tool:
   - 🔌 Connects to your database
   - 🔍 Explores table structure
   - 📊 Queries real data
   - 📈 Calculates trends and statistics

5. **Get Meaningful Results:**
   Receive a comprehensive report with:
   - Key findings from real data
   - Statistical analysis and trends
   - Discussion of ecological implications

### Example Questions

- "How has fish biomass changed in La Paz?"
- "What are the abundance trends in Cabo Pulmo?"
- "Show me diversity patterns in the Gulf of California"
- "Has fish density increased in Loreto over time?"
- "Compare biomass trends between La Paz and Cabo Pulmo"

## 🔧 Technical Details

### Database Requirements

The tool automatically detects and adapts to your database structure:

- **Supported databases:** PostgreSQL, MySQL, SQLite
- **Table discovery:** Automatically finds tables with ecological data
- **Column mapping:** Intelligently matches columns for metrics (biomass, abundance, etc.) and regions
- **Temporal analysis:** Detects date/time columns for trend analysis

### Analysis Capabilities

- **Statistical Analysis:** Mean, standard deviation, min/max, trend calculations
- **Temporal Trends:** Linear regression, slope calculation, percentage change
- **Regional Filtering:** Flexible region matching (exact and partial matches)
- **Data Validation:** Automatic null value handling and data type conversion

### AI Integration

- **Natural Language Processing:** Uses OpenAI GPT-4 for plan generation and interpretation
- **Narrative Generation:** Creates human-readable explanations of technical processes
- **Adaptive Querying:** Dynamically constructs SQL queries based on database structure

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys or secrets to your repository. Always use environment variables (.env file).

2. **Database Access**: Consider using read-only database credentials for the analysis tool to prevent accidental data modification.

3. **Data Privacy**: The tool only reads data for analysis - it does not modify or delete database records.

## 🛠️ Troubleshooting

### Common Issues

**"No database connection found"**
- Check that your `.env` file exists and contains valid `DATABASE_URL`
- Verify database credentials and network connectivity

**"Table not found or not accessible"**
- Ensure the table name exists in your database
- Check database permissions for the user account

**"No column found for metric"**
- The tool looks for columns containing keywords like 'biomass', 'abundance', etc.
- You may need to rename columns or specify them more clearly

**OpenAI API errors**
- Verify your `OPENAI_API_KEY` is valid and has sufficient credits
- Check your internet connection

## 📚 References

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

## 📝 License

This project is designed for ecological research and follows best practices for scientific reproducibility.

---

*This tool follows best practices for code organization, documentation, and reproducibility as defined in the project development rules.*
