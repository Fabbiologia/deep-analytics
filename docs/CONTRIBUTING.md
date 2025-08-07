# Ecological Data Analysis Agent - Project Rules & Guidelines

This document outlines the coding standards, development practices, and guidelines for contributing to the Ecological Data Analysis Agent project. Following these rules ensures consistency, maintainability, and reproducibility.

## Table of Contents

1. [Python Environment & Dependency Management](#python-environment--dependency-management)
2. [Code Organization & Structure](#code-organization--structure)
3. [Documentation Standards](#documentation-standards)
4. [Data Visualization Guidelines](#data-visualization-guidelines)
5. [Testing & Quality Assurance](#testing--quality-assurance)
6. [Terminal Output & Logging](#terminal-output--logging)
7. [Git & Version Control](#git--version-control)
8. [AI Integration Guidelines](#ai-integration-guidelines)

---

## Python Environment & Dependency Management

### Virtual Environments

- **Always use Python virtual environments**
  ```bash
  python -m venv env && source env/bin/activate
  ```
  Always activate the environment before installing packages or running scripts.

- **Update pip before installing packages**
  ```bash
  pip install --upgrade pip
  ```

- **Install dependencies via requirements files**
  ```bash
  pip install -r requirements.txt
  pip freeze > requirements.txt  # After adding new packages
  ```

- **Use conda only for system dependencies**
  Use `virtualenv` and `pip` for most cases; use `conda` only for compiled libraries with complex dependencies.

### Package Versions

- **Pin specific versions in requirements.txt**
  ```
  langchain==0.1.0
  pandas==2.0.0
  ```

- **Document dependency decisions**
  Add comments in requirements.txt for non-obvious packages or version constraints.

---

## Code Organization & Structure

### File Structure

- **Maintain modular organization**
  ```
  /main.py                # Entry point, agent instantiation
  /tools.py               # Custom tools and utilities
  /config.py              # Configuration handling
  /agents/                # Agent-specific modules
  /tests/                 # Test files
  /docs/                  # Documentation
  ```

### Naming Conventions

- **Use descriptive names**
  - Clear, descriptive names for variables, functions, and classes
  - Avoid generic names like `data`, `df`, `x`, `y`
  - Follow Python conventions: snake_case for variables/functions, PascalCase for classes

### Code Style

- **Follow PEP 8 standards**
  - 4-space indentation
  - Maximum line length of 88 characters (compatible with Black formatter)
  - Proper spacing around operators

- **Use type hints**
  ```python
  def analyze_data(df: pd.DataFrame, metric: str) -> Dict[str, float]:
      ...
  ```

- **Apply consistent formatting**
  Use Black for Python code formatting:
  ```bash
  black .
  ```

---

## Documentation Standards

### Inline Documentation

- **Add docstrings to all functions, classes, and modules**
  ```python
  def create_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str) -> str:
      """
      Creates a bar chart visualization and saves it to a file.
      
      Args:
          data: Pandas DataFrame containing the data to plot
          x_col: Column to use for x-axis categories
          y_col: Column to use for y-axis values
          title: Title of the chart
          filename: Output filename (will be saved as PNG)
          
      Returns:
          Path to the saved image file
      """
  ```

- **Include inline comments for complex logic**
  Explain the "why" rather than the "what" for non-obvious code sections.

### Project Documentation

- **Write all documentation in Markdown (.md)**
  - Use headings, bullet points, code blocks (triple backticks), and links

- **Keep README.md updated**
  Ensure README accurately reflects current functionality, dependencies, and usage

- **Document LLM prompts and reasoning**
  Explain the design of system prompts and their intended behavior

---

## Data Visualization Guidelines

### Standard Practices

- **Always use theme_bw() or theme_minimal() for ggplot2 figures**
  For consistent, clean visualization style

- **Use seaborn for Python visualizations with consistent styling**
  ```python
  import seaborn as sns
  sns.set_theme(style="whitegrid")
  ```

- **Label axes and legends with meaningful names and units**
  Never use raw variable names as labels

- **Use color palettes accessible to colorblind users**
  ```python
  # Python
  import matplotlib.pyplot as plt
  from matplotlib.colors import LinearSegmentedColormap
  plt.cm.viridis  # Or other colorblind-friendly palettes
  
  # R
  library(RColorBrewer)
  brewer.pal(8, "Set2")
  ```

### Saving Visualizations

- **Save plots programmatically with consistent parameters**
  ```python
  # Python - matplotlib
  plt.savefig(filename, dpi=300, bbox_inches='tight')
  
  # R - ggplot2
  ggsave(filename, plot=p, dpi=300, width=8, height=6)
  ```

- **Include creation date and data source in filenames or metadata**
  ```
  biomass_by_location_20250730.png
  ```

---

## Testing & Quality Assurance

### Test Coverage

- **Write unit tests for all custom functions**
  Use pytest for Python projects

- **Create integration tests for end-to-end workflows**
  Test complete analytical pipelines

- **Add validation checks to analytical functions**
  Verify inputs, check for edge cases, validate outputs

### Reproducibility

- **Always seed random number generators**
  ```python
  # Python
  import numpy as np
  np.random.seed(123)
  
  # R
  set.seed(123)
  ```

- **Document computational environment**
  Record Python/R version, key package versions, OS, etc.

- **Make scripts executable from top to bottom**
  Avoid hardcoded paths or parameters; use config files or CLI arguments

---

## Terminal Output & Logging

### Verbosity

- **All code outputs must be verbose and traceable**
  Use print or logging statements for key actions
  Include file paths, data shapes, and parameters

- **Use progress indicators for long operations**
  ```python
  # Python
  from tqdm import tqdm
  for i in tqdm(range(total_steps)):
      # operation
  
  # R
  library(progress)
  pb <- progress_bar$new(total = total_steps)
  for (i in 1:total_steps) {
      pb$tick()
      # operation
  }
  ```

### Logging

- **Log outputs to file for long scripts**
  ```python
  # Python
  import logging
  logging.basicConfig(
      filename='analysis.log',
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s'
  )
  logging.info("Starting analysis with parameters: %s", params)
  
  # R
  sink("analysis.log", append=TRUE)
  cat(paste0(Sys.time(), " - INFO - Starting analysis with parameters: ", params, "\n"))
  ```

- **Provide time estimates for operations > 10 seconds**
  Log start times, progress updates, and completion times

---

## Git & Version Control

### Commit Practices

- **Make small, focused commits**
  Each commit should represent a single logical change

- **Use descriptive commit messages**
  Follow the format:
  ```
  <type>: <summary>
  
  <body>
  ```
  
  Types include: feat, fix, docs, style, refactor, test, chore

- **Create feature branches for major changes**
  Work in branches and merge via pull requests

### Version Tagging

- **Use semantic versioning**
  `MAJOR.MINOR.PATCH` format (e.g., 1.0.0)

- **Tag significant versions**
  ```bash
  git tag -a v1.0.0 -m "Version 1.0.0"
  git push origin v1.0.0
  ```

---

## AI Integration Guidelines

### LLM Usage

- **Document all system prompts**
  Keep prompts versioned and documented in the codebase

- **Maintain agent behavior consistency**
  Document expected agent behaviors and capabilities

- **Verify AI-generated code**
  Review, test, and validate any code suggested by AI assistants

### Agent Configuration

- **Use consistent prompt templates**
  Standardize how instructions are provided to the agent

- **Document agent limitations**
  Clearly describe what the agent can and cannot do

- **Keep LLM service tokens secure**
  Never commit API keys to the repository; always use environment variables

---

## Compliance

All contributors and maintainers are expected to follow these guidelines. Regular code reviews will ensure adherence to these standards.

**Last Updated:** July 30, 2025
