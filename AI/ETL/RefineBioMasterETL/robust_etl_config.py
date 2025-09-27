"""
Production-Grade ETL Configuration Management
Handles all configuration settings, validation, and dynamic study code detection
"""

import os
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Pattern
import logging

@dataclass
class ETLConfig:
    """Production ETL Configuration Settings with dynamic study code support"""

    # Base configuration
    base_path: str
    connection_string: str
    database_name: str = "BioinformaticsWarehouse"
    
    # Dynamic file patterns - support for various study code formats
    study_code_pattern: str = r"[A-Za-z]{2,3}-[A-Za-z]{3}-\d+|[A-Za-z]{3}\d+"
    json_metadata_file: str = "aggregated_metadata.json"
    tsv_metadata_pattern: str = "{study_code}/metadata_{study_code}.tsv"
    expression_data_pattern: str = "{study_code}/{study_code}.tsv"
    
    # Processing settings with performance optimization
    batch_size: int = 50000  # Increased for better throughput
    chunk_size: int = 10000  # Larger chunks for memory efficiency
    max_workers: int = 8     # More workers for parallel processing
    memory_limit_mb: int = 4096  # Increased memory limit
    timeout_seconds: int = 7200  # 2 hours for large datasets
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    
    # Data validation settings
    max_null_percentage: float = 0.1  # Maximum 10% null values allowed
    min_genes_per_sample: int = 1000  # Minimum genes required per sample
    max_duplicate_percentage: float = 0.05  # Maximum 5% duplicates allowed
    
    # Performance monitoring
    enable_performance_logging: bool = True
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 1000
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_rotation: bool = True
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    
    # Scalability settings
    enable_parallel_processing: bool = True
    enable_batch_optimization: bool = True
    use_temp_tables_for_large_inserts: bool = True
    
    # Data quality settings
    validate_gene_symbols: bool = True
    validate_sample_accessions: bool = True
    check_data_integrity: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._setup_logging()
        self._validate_configuration()
        self._compile_patterns()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        if self.log_file:
            log_dir = Path(self.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_configuration(self) -> None:
        """Validate configuration settings"""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.chunk_size <= 0 or self.chunk_size > self.batch_size:
            raise ValueError("Chunk size must be positive and <= batch_size")
        
        if self.max_workers <= 0 or self.max_workers > 16:
            raise ValueError("Max workers must be between 1 and 16")
        
        if not (0 <= self.max_null_percentage <= 1):
            raise ValueError("Max null percentage must be between 0 and 1")
        
        if not (0 <= self.max_duplicate_percentage <= 1):
            raise ValueError("Max duplicate percentage must be between 0 and 1")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for study code detection"""
        self._study_code_regex: Pattern = re.compile(self.study_code_pattern)
    
    @classmethod
    def from_environment(cls) -> "ETLConfig":
        """Create configuration from environment variables with defaults"""
        return cls(
            base_path=os.getenv("ETL_BASE_PATH", "/data/refinebio"),
            connection_string=os.getenv("DB_CONNECTION_STRING", 
                "Server=localhost;Database=BioinformaticsWarehouse;Trusted_Connection=true;"),
            database_name=os.getenv("DB_NAME", "BioinformaticsWarehouse"),
            batch_size=int(os.getenv("ETL_BATCH_SIZE", "50000")),
            chunk_size=int(os.getenv("ETL_CHUNK_SIZE", "10000")),
            max_workers=int(os.getenv("ETL_MAX_WORKERS", "8")),
            memory_limit_mb=int(os.getenv("ETL_MEMORY_LIMIT_MB", "4096")),
            timeout_seconds=int(os.getenv("ETL_TIMEOUT_SECONDS", "7200")),
            retry_attempts=int(os.getenv("ETL_RETRY_ATTEMPTS", "3")),
            retry_delay_seconds=int(os.getenv("ETL_RETRY_DELAY_SECONDS", "5")),
            max_null_percentage=float(os.getenv("ETL_MAX_NULL_PERCENTAGE", "0.1")),
            min_genes_per_sample=int(os.getenv("ETL_MIN_GENES_PER_SAMPLE", "1000")),
            max_duplicate_percentage=float(os.getenv("ETL_MAX_DUPLICATE_PERCENTAGE", "0.05")),
            enable_performance_logging=os.getenv("ETL_ENABLE_PERF_LOGGING", "true").lower() == "true",
            log_slow_queries=os.getenv("ETL_LOG_SLOW_QUERIES", "true").lower() == "true",
            slow_query_threshold_ms=int(os.getenv("ETL_SLOW_QUERY_THRESHOLD_MS", "1000")),
            log_level=os.getenv("ETL_LOG_LEVEL", "INFO"),
            log_file=os.getenv("ETL_LOG_FILE"),
            log_rotation=os.getenv("ETL_LOG_ROTATION", "true").lower() == "true",
            log_max_size_mb=int(os.getenv("ETL_LOG_MAX_SIZE_MB", "100")),
            log_backup_count=int(os.getenv("ETL_LOG_BACKUP_COUNT", "5")),
            enable_parallel_processing=os.getenv("ETL_PARALLEL_PROCESSING", "true").lower() == "true",
            enable_batch_optimization=os.getenv("ETL_BATCH_OPTIMIZATION", "true").lower() == "true",
            use_temp_tables_for_large_inserts=os.getenv("ETL_USE_TEMP_TABLES", "true").lower() == "true",
            validate_gene_symbols=os.getenv("ETL_VALIDATE_GENES", "true").lower() == "true",
            validate_sample_accessions=os.getenv("ETL_VALIDATE_SAMPLES", "true").lower() == "true",
            check_data_integrity=os.getenv("ETL_CHECK_INTEGRITY", "true").lower() == "true"
        )
    
    def validate_paths(self) -> None:
        """Validate that required paths exist"""
        base_path = Path(self.base_path)
        if not base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")
        
        if not (base_path / self.json_metadata_file).exists():
            raise ValueError(f"JSON metadata file not found: {base_path / self.json_metadata_file}")
    
    def discover_study_codes(self) -> List[str]:
        """Dynamically discover study codes from directory structure"""
        base_path = Path(self.base_path)
        study_codes = []
        
        # Look for directories matching study code pattern
        for item in base_path.iterdir():
            if item.is_dir() and self._study_code_regex.match(item.name):
                study_codes.append(item.name)
        
        # Also check JSON metadata for study codes
        try:
            json_path = base_path / self.json_metadata_file
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Extract study codes from experiments
            if 'experiments' in metadata:
                for study_code in metadata['experiments'].keys():
                    if self._study_code_regex.match(study_code) and study_code not in study_codes:
                        study_codes.append(study_code)
        except Exception as e:
            logging.warning(f"Could not extract study codes from JSON: {e}")
        
        if not study_codes:
            raise ValueError(f"No study codes found matching pattern: {self.study_code_pattern}")
        
        logging.info(f"Discovered study codes: {study_codes}")
        return study_codes
    
    def get_study_file_paths(self, study_code: str) -> Dict[str, Path]:
        """Get file paths for a specific study code"""
        base_path = Path(self.base_path)
        
        # Format file patterns with study code
        tsv_metadata_file = self.tsv_metadata_pattern.format(study_code=study_code)
        expression_data_file = self.expression_data_pattern.format(study_code=study_code)
        
        paths = {
            "json_metadata": base_path / self.json_metadata_file,
            "tsv_metadata": base_path / tsv_metadata_file,
            "expression_data": base_path / expression_data_file
        }
        
        # Validate paths exist
        for file_type, path in paths.items():
            if not path.exists():
                raise ValueError(f"File not found for study {study_code}: {path}")
        
        return paths
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for connection pooling"""
        return {
            "connection_string": self.connection_string,
            "database_name": self.database_name,
            "timeout": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay_seconds
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration for data transformation"""
        return {
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "max_workers": self.max_workers,
            "memory_limit_mb": self.memory_limit_mb,
            "enable_parallel_processing": self.enable_parallel_processing,
            "enable_batch_optimization": self.enable_batch_optimization
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration for data quality checks"""
        return {
            "max_null_percentage": self.max_null_percentage,
            "min_genes_per_sample": self.min_genes_per_sample,
            "max_duplicate_percentage": self.max_duplicate_percentage,
            "validate_gene_symbols": self.validate_gene_symbols,
            "validate_sample_accessions": self.validate_sample_accessions,
            "check_data_integrity": self.check_data_integrity
        }


class StudyConfig:
    """Configuration for individual study processing"""
    
    def __init__(self, study_code: str, base_config: ETLConfig):
        self.study_code = study_code
        self.base_config = base_config
        self.file_paths = base_config.get_study_file_paths(study_code)
        
        # Study-specific processing settings
        self.processing_id = f"{study_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.staging_table_suffix = f"_{study_code.lower()}"
        
        # Performance tuning based on expected data size
        self._optimize_for_study_size()
    
    def _optimize_for_study_size(self) -> None:
        """Optimize processing parameters based on study characteristics"""
        # Adjust batch sizes based on study code patterns
        if self.study_code.startswith('SRP'):
            # SRP studies typically have more samples
            self.batch_size = min(self.base_config.batch_size, 25000)
        elif self.study_code.startswith('EX'):
            # EX studies might be smaller
            self.batch_size = min(self.base_config.batch_size, 10000)
        else:
            self.batch_size = self.base_config.batch_size