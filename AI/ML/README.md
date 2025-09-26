# Gene Pair ML Analysis System

An advanced machine learning system for identifying biologically significant gene pairs through meta-analysis of gene expression studies, with a specific focus on sepsis and septic shock research.

## 🧬 Overview

The Gene Pair ML Analysis System combines unsupervised machine learning with configurable rules-based ranking to provide interpretable recommendations for bioinformatics research. The system is designed to analyze meta-analysis statistics from gene expression studies and identify gene pairs with strong correlations and potential biological significance.

## ✨ Key Features

### Machine Learning Capabilities
- **Ensemble Methods**: Isolation Forest, DBSCAN, PCA, and Gaussian Mixture Models
- **Unsupervised Pattern Discovery**: Identify patterns without labeled training data
- **Feature Engineering**: Create derived features and dimensionality reduction
- **Anomaly Detection**: Identify exceptional gene pairs
- **Clustering Analysis**: Group similar gene pairs using multiple algorithms

### Rules-Based Ranking
- **Configurable Rule Engine**: Add, modify, and weight ranking rules
- **Default Rule Set**: Statistical significance, effect size, z-score strength, FDR correction, consistency
- **Custom Rules**: Simple conditional logic with comparison and logical operators
- **Visual Rule Management**: Web interface for rule configuration

### Interactive Visualizations
- **Box Plots**: Distribution of effect sizes by condition
- **Scatter Plots**: Gene pair correlations with significance highlighting
- **Clustering Visualizations**: Ensemble clustering results
- **Ranking Charts**: Top recommendations with scores
- **Interactive Dashboard**: Comprehensive results display

### Data Integration
- **File Upload**: Excel, CSV, JSON formats supported
- **Database Connectivity**: SQL Server, MySQL, PostgreSQL support
- **Batch Processing**: Efficient handling of large datasets
- **Data Validation**: Automatic structure and quality checks

## 🏗️ Architecture

### System Components

```
gene_pair_agent/          # Core ML analysis engine
├── gene_pair_analyzer.py     # Main analysis engine
├── rules_engine.py          # Configurable ranking system
├── feature_engineering.py   # Derived features and PCA
├── meta_analysis_processor.py # File data processing
└── database_connector.py    # Database connectivity

web_interface/            # Flask web application
├── app.py                   # Flask application
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── analyze.html
│   ├── results.html
│   ├── database.html
│   ├── rules.html
│   └── about.html
└── static/                 # CSS, JavaScript, images

visualization/            # Interactive charts and plots
├── chart_generator.py     # Plotly chart creation
├── interactive_plots.py   # HTML components
└── results_dashboard.py   # Dashboard functionality

data_processing/          # Data handling utilities
├── file_handler.py        # File I/O operations
├── database_handler.py    # Database operations
├── data_validator.py      # Data validation
└── export_handler.py      # Results export

config/                   # Configuration management
├── settings.py           # System configuration
└── logging_config.py     # Logging setup
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd gene-pair-ml-analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the web interface:**
```bash
cd web_interface
python app.py
```

4. **Or use the command-line interface:**
```bash
python main.py --file sample_data.xlsx --output results.json
```

### Command-Line Usage

```bash
# Run analysis on file (JSON export by default)
python main.py --file data.xlsx --output results.json

# Export recommendations as a flat CSV table
python main.py --file data.xlsx --output results.csv

# Generate an Excel workbook with multiple sheets
python main.py --file data.xlsx --output results.xlsx

# Create sample data
python main.py --sample --output sample_data.csv --n-pairs 100

# Run with configuration
python main.py --file data.xlsx --config config.json --output results.json

# Run database analysis
python main.py --database --config db_config.json --output results.json
```

> ℹ️ The CLI now infers the export format from the `--output` extension. JSON, CSV/TSV, and Excel workbooks are supported without additional flags.

### Web Interface

1. **Upload Data**: Drag and drop or browse to upload meta-analysis files
2. **Configure Rules**: Customize ranking rules through the visual interface
3. **Run Analysis**: Start the ensemble ML analysis with configurable parameters
4. **View Results**: Interactive visualizations and ranked recommendations
5. **Export**: Download results in Excel, JSON, or chart formats

## 📊 Data Format

### Required Columns
- `pair_id`: Unique identifier for gene pair
- `GeneAName`, `GeneBName`: Gene symbols
- `dz_ss_mean`, `dz_soth_mean`: Effect sizes (Cohen's d)
- `p_ss`, `p_soth`: P-values
- `q_ss`, `q_soth`: FDR-adjusted p-values
- `abs_dz_ss`, `abs_dz_soth`: Absolute effect sizes

### Optional Columns
- `study_key`: Study identifier
- `illness_label`: Condition type (control, sepsis, septic shock)
- `n_studies_ss`, `n_studies_soth`: Number of studies
- `dz_ss_I2`, `dz_soth_I2`: Heterogeneity measures
- `dz_ss_z`, `dz_soth_z`: Z-scores

## 🔧 Configuration

### Configuration File (JSON)

```json
{
  "analysis": {
    "n_features": 5,
    "contamination": 0.1,
    "random_state": 42
  },
  "rules": [
    {
      "name": "Statistical Significance",
      "condition": "(p_ss < 0.1) AND (p_soth < 0.01)",
      "weight": 0.25,
      "description": "Strong statistical significance"
    }
  ],
  "database": {
    "server": "localhost",
    "database": "BioinformaticsDB",
    "username": "sa",
    "password": "password"
  }
}
```

### Database Connection

```python
from gene_pair_agent import DatabaseConnector

config = {
    'driver': 'ODBC Driver 17 for SQL Server',
    'server': 'localhost',
    'database': 'BioinformaticsDB',
    'username': 'sa',
    'password': 'password'
}

connector = DatabaseConnector(config)
data = connector.get_gene_pair_data()
```

## 🎯 Analysis Pipeline

1. **Data Input**: Upload meta-analysis file or connect to database
2. **Data Validation**: Check structure, quality, and completeness
3. **Feature Engineering**: Create derived features and reduce dimensionality
4. **Ensemble Analysis**: Run multiple ML algorithms simultaneously
5. **Rules Application**: Apply configurable ranking rules
6. **Result Integration**: Combine ML and rules-based scores
7. **Visualization**: Generate interactive charts and summaries
8. **Export**: Download results in multiple formats

## 📈 Statistical Measures

### Effect Sizes
- **Cohen's d**: Standardized effect size for both conditions
- **Confidence Intervals**: 95% confidence bounds
- **Interpretation**: Effect magnitude and direction

### Statistical Significance
- **P-values**: Traditional significance testing
- **FDR Correction**: False discovery rate adjusted q-values
- **Multiple Testing**: Handles large numbers of comparisons

### Heterogeneity
- **I² Statistics**: Measure of study consistency
- **Q Statistics**: Heterogeneity test statistics
- **Interpretation**: Consistency across studies

### Quality Metrics
- **Sample Sizes**: Number of studies and participants
- **Kappa Statistics**: Agreement measures
- **Z-scores**: Standardized effect sizes

## 🎨 Visualizations

### Box Plots
- Distribution of effect sizes by condition
- Outlier identification
- Statistical significance highlighting

### Scatter Plots
- Gene pair correlations
- Quadrant analysis
- Significance-based color coding

### Clustering Visualizations
- DBSCAN clustering results
- Anomaly detection outcomes
- Consensus matrices

### Ranking Charts
- Top recommendations
- Score distributions
- Confidence intervals

## 🔬 Research Applications

### Sepsis Research
- **Biomarker Discovery**: Identify novel gene pair biomarkers
- **Diagnostic Signatures**: Develop multi-gene diagnostic panels
- **Therapeutic Targets**: Find potential intervention points
- **Mechanism Investigation**: Understand disease pathways

### Quality Control
- **Positive Control**: MS4A4A-CD86 validation
- **Cross-validation**: Multiple validation approaches
- **Statistical Rigor**: Comprehensive significance testing
- **Reproducibility**: Fixed random seeds and logging

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM recommended
- **Storage**: 2GB disk space
- **Browser**: Modern web browser for interface

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
plotly>=5.0.0
flask>=2.0.0
sqlalchemy>=1.4.0
openpyxl>=3.0.0
```

## 🛠️ Development

### Project Structure
```
gene-pair-ml-analysis/
├── gene_pair_agent/        # Core ML engine
├── web_interface/         # Flask web app
├── visualization/         # Charts and plots
├── data_processing/       # Data utilities
├── config/               # Configuration
├── main.py               # CLI entry point
├── requirements.txt      # Dependencies
└── README.md            # This file
```

### Testing
```bash
# Run sample analysis
python main.py --sample --output test_results.json

# Test web interface
cd web_interface && python app.py

# Validate installation
python -c "from gene_pair_agent import GenePairAnalyzer; print('Installation successful')"
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## 📞 Support

For technical support or questions:
- Create an issue in the repository
- Check the documentation
- Review example configurations

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
  - Ensemble ML analysis
  - Configurable rules engine
  - Interactive web interface
  - Database connectivity
  - Comprehensive visualizations

## 🙏 Acknowledgments

- **Research Community**: For providing the biological context and requirements
- **Open Source**: Built on excellent open-source libraries
- **Contributors**: All contributors who helped shape the system

---

**Note**: This system is designed for research purposes and should be used in accordance with appropriate ethical and scientific standards.