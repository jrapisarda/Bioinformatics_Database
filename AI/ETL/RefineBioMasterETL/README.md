# Production-Grade Bioinformatics ETL Pipeline

A robust, scalable ETL (Extract, Transform, Load) pipeline designed for processing refine.bio RNA-seq gene expression data into a dimensional data warehouse. This solution implements a BioStar schema approach with comprehensive logging, data validation, and MERGE-based deduplication.

## 🚀 Features

### Core Capabilities
- **Scalable Study Code Support**: Handles various study code patterns (SRP*, EX-AUR-1004, etc.)
- **Performance Optimized**: Batch processing, parallel execution, memory-efficient operations
- **Data Validation**: Multi-level validation with quality scoring
- **Deduplication**: MERGE statements prevent duplicate dimensions and facts
- **Comprehensive Logging**: Detailed audit trails and performance monitoring
- **Error Handling**: Robust error recovery and retry mechanisms

### Technical Features
- **Matrix to Long Format**: Transforms gene expression matrices to warehouse-ready format
- **Dynamic File Discovery**: Automatically discovers study codes from directory structure
- **Staging-First Approach**: Loads to staging tables before production migration
- **Performance Monitoring**: Real-time metrics and throughput analysis
- **Quality Assurance**: Automated data quality checks and reporting

## 📁 Project Structure

```
├── robust_main_etl.py              # Main pipeline orchestrator
├── robust_etl_config.py            # Configuration management
├── robust_data_extractor.py        # Data extraction module
├── robust_data_validator.py        # Data validation module
├── robust_data_transformer.py      # Data transformation module
├── robust_data_loader.py           # Data loading module
├── robust_database_setup.sql       # Database schema and procedures
├── test_etl_pipeline.py            # Comprehensive test suite
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- SQL Server 2017+ or compatible database
- 8GB+ RAM recommended for large datasets

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd etl-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up database**
   ```bash
   # Run the database setup script
   sqlcmd -S your_server -d BioinformaticsWarehouse -i robust_database_setup.sql
   ```

4. **Configure environment variables**
   ```bash
   # Set required environment variables
   export ETL_BASE_PATH="/path/to/your/data"
   export DB_CONNECTION_STRING="Server=localhost;Database=BioinformaticsWarehouse;Trusted_Connection=true;"
   export ETL_LOG_LEVEL="INFO"
   ```

## 📊 Data Format Requirements

### Input Files

1. **JSON Metadata** (`aggregated_metadata.json`)
   - Study and sample metadata
   - Experiment descriptions and annotations
   - Platform and processing information

2. **TSV Metadata** (`metadata_<study_code>.tsv`)
   - Sample-level annotations
   - Clinical and experimental metadata
   - Sample classifications and groupings

3. **Expression Data** (`<study_code>.tsv`)
   - Gene expression matrix (genes × samples)
   - TPM or similar normalized values
   - Gene symbols as row indices

### Expected Directory Structure

```
base_path/
├── aggregated_metadata.json
├── SRP049820/
│   ├── metadata_SRP049820.tsv
│   └── SRP049820.tsv
├── EX-AUR-1004/
│   ├── metadata_EX-AUR-1004.tsv
│   └── EX-AUR-1004.tsv
└── ...
```

## 🚦 Usage

### Basic Usage

```python
from robust_main_etl import ETLPipelineOrchestrator
from robust_etl_config import ETLConfig

# Load configuration
config = ETLConfig.from_environment()

# Create and run pipeline
pipeline = ETLPipelineOrchestrator(config)
results = pipeline.execute_full_pipeline()

print(f"Processed {results['pipeline_stats']['studies_processed']} studies")
```

### Command Line Interface

```bash
# Process all discovered studies
python robust_main_etl.py

# Process specific studies
python robust_main_etl.py --studies SRP049820 EX-AUR-1004

# Validate only (no database loading)
python robust_main_etl.py --validate-only

# Dry run (no permanent changes)
python robust_main_etl.py --dry-run

# Custom logging level
python robust_main_etl.py --log-level DEBUG
```

### Running Tests

```bash
# Run comprehensive test suite
python test_etl_pipeline.py

# Run with coverage
pytest test_etl_pipeline.py --cov=.
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ETL_BASE_PATH` | `/data/refinebio` | Base directory for input files |
| `DB_CONNECTION_STRING` | `Server=localhost;Database=BioinformaticsWarehouse;Trusted_Connection=true;` | Database connection string |
| `ETL_BATCH_SIZE` | `50000` | Number of records to process in each batch |
| `ETL_CHUNK_SIZE` | `10000` | Chunk size for memory-efficient processing |
| `ETL_MAX_WORKERS` | `8` | Maximum number of parallel workers |
| `ETL_MEMORY_LIMIT_MB` | `4096` | Memory limit for processing |
| `ETL_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ETL_LOG_FILE` | `None` | Optional log file path |

### Configuration Options

```python
config = ETLConfig(
    base_path="/custom/data/path",
    connection_string="your_connection_string",
    batch_size=25000,  # Smaller batches for memory-constrained environments
    max_workers=4,     # Fewer workers for less powerful systems
    enable_performance_logging=True,
    validate_gene_symbols=True,
    check_data_integrity=True
)
```

## 📈 Performance Optimization

### Memory Management
- **Batch Processing**: Configurable batch sizes prevent memory overflow
- **Chunked Reading**: Large files processed in memory-efficient chunks
- **Garbage Collection**: Automatic cleanup of intermediate data structures

### Parallel Processing
- **Multi-threaded**: Parallel extraction and validation
- **Database Connection Pooling**: Efficient connection management
- **Asynchronous Operations**: Non-blocking I/O operations

### Database Optimization
- **Index Strategy**: Optimized indexes for query performance
- **MERGE Operations**: Efficient upsert operations prevent duplicates
- **Staging Tables**: Minimize production table locking

## 🔍 Monitoring and Logging

### Performance Metrics
- **Throughput**: Records processed per second
- **Memory Usage**: Peak memory consumption
- **Duration**: Step-by-step timing analysis
- **Quality Scores**: Automated data quality assessment

### Audit Trail
- **Execution Logs**: Complete pipeline execution history
- **Data Quality Logs**: Validation results and quality metrics
- **Performance Logs**: Detailed performance analysis
- **Error Tracking**: Comprehensive error reporting

### Monitoring Queries

```sql
-- Get recent ETL execution summary
EXEC sp_get_etl_statistics @start_date = '2023-01-01';

-- Performance analysis
SELECT 
    study_code,
    AVG(duration_seconds) as avg_duration,
    AVG(records_processed) as avg_records,
    AVG(validation_score) as avg_quality
FROM audit.etl_execution_log 
WHERE status = 'SUCCESS'
GROUP BY study_code;
```

## 🛡️ Data Quality

### Validation Checks
- **Format Validation**: File structure and encoding validation
- **Content Validation**: Data ranges and type checking
- **Consistency Validation**: Cross-reference validation between sources
- **Business Rule Validation**: Domain-specific validation rules

### Quality Metrics
- **Completeness**: Missing data percentage
- **Consistency**: Cross-source data alignment
- **Accuracy**: Data range and format validation
- **Timeliness**: Data freshness and update tracking

### Quality Score Calculation
```python
# Quality score ranges from 0-100
score = 100.0
score -= (null_percentage * 100)  # Penalty for missing data
score -= (duplicate_percentage * 50)  # Penalty for duplicates
score -= (consistency_issues * 20)  # Penalty for consistency issues
```

## 🔄 Error Handling

### Error Types
- **Extraction Errors**: File reading and parsing issues
- **Validation Errors**: Data quality and consistency issues
- **Transformation Errors**: Data conversion and mapping issues
- **Loading Errors**: Database connection and constraint violations

### Recovery Mechanisms
- **Retry Logic**: Automatic retry with exponential backoff
- **Graceful Degradation**: Continue processing valid data
- **Comprehensive Logging**: Detailed error context and diagnostics
- **Rollback Support**: Transaction safety and data integrity

## 📚 Database Schema

### Dimension Tables
- `dim_study`: Study-level information and metadata
- `dim_platform`: Sequencing platform and technology details
- `dim_sample`: Sample-specific annotations and classifications
- `dim_gene`: Gene annotations and reference information
- `dim_illness`: Disease/condition classifications

### Fact Table
- `fact_gene_expression`: Gene expression measurements with foreign key references

### Audit Tables
- `audit.etl_execution_log`: Pipeline execution tracking
- `audit.data_quality_log`: Data quality validation results
- `audit.performance_log`: Performance metrics and timing

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Load testing and throughput validation
- **Edge Case Tests**: Error conditions and boundary testing

### Test Data
- **Sample Studies**: SRP049820 with realistic data patterns
- **Edge Cases**: Empty files, invalid formats, large datasets
- **Performance Scenarios**: Memory-constrained and high-throughput testing

## 🔒 Security Considerations

### Data Protection
- **Connection Security**: Encrypted database connections
- **Access Control**: Role-based database permissions
- **Audit Logging**: Complete access and modification tracking
- **Data Masking**: Sensitive data anonymization options

### Best Practices
- **Least Privilege**: Minimal required permissions
- **Secure Configuration**: Environment-based configuration
- **Regular Updates**: Dependency and security patch management
- **Monitoring**: Anomaly detection and alerting

## 📞 Support and Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size: `ETL_BATCH_SIZE=10000`
   - Increase memory limit: `ETL_MEMORY_LIMIT_MB=8192`
   - Enable chunked processing

2. **Database Connection Issues**
   - Verify connection string format
   - Check network connectivity
   - Ensure proper permissions

3. **Data Validation Failures**
   - Review validation report
   - Check input file formats
   - Verify study code patterns

### Getting Help
- Check logs in configured log file or console output
- Review audit tables for execution history
- Run validation-only mode to identify data issues
- Use test suite to verify component functionality

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Run test suite to verify setup

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add tests for new functionality
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **refine.bio team** for providing the data processing platform
- **Bioinformatics community** for schema design patterns
- **Open source contributors** for the excellent libraries used

---

For additional support or questions, please refer to the documentation or create an issue in the repository.