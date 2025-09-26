# Gene Pair ML Analysis System - Project Outline

## Architecture Overview
A comprehensive bioinformatics web application for gene pair correlation analysis using unsupervised machine learning and rules-based ranking.

## System Components

### 1. Core AI Agent (`gene_pair_agent/`)
- **gene_pair_analyzer.py**: Main analysis engine with ensemble ML methods
- **rules_engine.py**: Configurable rules-based ranking system
- **feature_engineering.py**: Derived feature creation and dimensionality reduction
- **meta_analysis_processor.py**: File-based meta-analysis data processing
- **database_connector.py**: Database connectivity for source data
- **models/**: Trained model persistence and versioning

### 2. Web Interface (`web_interface/`)
- **app.py**: Flask/FastAPI application with batch processing endpoints
- **templates/**: HTML templates for data upload and results
- **static/**: CSS, JavaScript, and visualization assets
- **config.py**: Web application configuration

### 3. Data Processing (`data_processing/`)
- **file_handler.py**: Excel/CSV file processing for meta-analysis data
- **database_handler.py**: Database query and data extraction
- **data_validator.py**: Input data validation and cleaning
- **export_handler.py**: Results export in multiple formats

### 4. Visualization (`visualization/`)
- **chart_generator.py**: Box plots and scatter plot generation
- **interactive_plots.py**: Plotly-based interactive visualizations
- **results_dashboard.py**: Comprehensive results display

### 5. Configuration & Utilities (`config/`)
- **settings.py**: System-wide configuration management
- **logging_config.py**: Logging setup and management
- **exceptions.py**: Custom exception handling

## Key Features Implementation

### Best Practice Architecture
- **Hybrid Data Access**: File-based meta-analysis + database source data
- **Modular Design**: Separation of concerns with clear interfaces
- **Configurable System**: JSON-based configuration for rules and parameters
- **Batch Processing**: Asynchronous processing for large datasets
- **Comprehensive Visualization**: Interactive charts with Plotly.js

### ML Implementation
- **Ensemble Methods**: Isolation Forest, DBSCAN, PCA, Gaussian Mixture Models
- **Feature Engineering**: Derived statistical features and composite measures
- **Rules Engine**: Configurable ranking with visual rule management
- **Positive Control**: MS4A4A-CD86 pair as configurable baseline

### User Interface
- **Data Upload**: Drag-and-drop file upload with validation
- **Rule Configuration**: Visual interface for adding/modifying ranking rules
- **Results Dashboard**: Interactive visualization of recommendations
- **Export Options**: CSV, Excel, and PDF report generation

## File Structure
```
/mnt/okcomputer/output/
├── gene_pair_agent/
│   ├── __init__.py
│   ├── gene_pair_analyzer.py
│   ├── rules_engine.py
│   ├── feature_engineering.py
│   ├── meta_analysis_processor.py
│   ├── database_connector.py
│   └── models/
├── web_interface/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── config.py
├── data_processing/
│   ├── __init__.py
│   ├── file_handler.py
│   ├── database_handler.py
│   ├── data_validator.py
│   └── export_handler.py
├── visualization/
│   ├── __init__.py
│   ├── chart_generator.py
│   ├── interactive_plots.py
│   └── results_dashboard.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── logging_config.py
├── main.py
├── requirements.txt
└── README.md
```

## Implementation Phases
1. **Core Agent Development**: ML algorithms and rules engine
2. **Data Processing Layer**: File and database handlers
3. **Web Interface**: Upload, configuration, and results display
4. **Visualization**: Interactive charts and dashboards
5. **Integration & Testing**: End-to-end system validation

## Technology Stack
- **Backend**: Python with scikit-learn, pandas, numpy
- **Web Framework**: Flask/FastAPI with Jinja2 templates
- **Database**: SQLAlchemy for database connectivity
- **Visualization**: Plotly.js for interactive charts
- **Frontend**: HTML5, CSS3, JavaScript with responsive design