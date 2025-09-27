"""
Production-Grade Data Loader Module
Handles loading transformed data into database with MERGE statements and performance optimization
"""

import pandas as pd
import pyodbc
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json


class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass


class DataLoader:
    """Load transformed data into database with MERGE statements and performance monitoring"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection = None
        self.load_stats = {
            'tables_loaded': 0,
            'records_inserted': 0,
            'records_updated': 0,
            'records_failed': 0,
            'load_duration_seconds': 0,
            'warnings': [],
            'errors': []
        }
        self.batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def connect(self) -> None:
        """Establish database connection"""
        try:
            self.logger.info("Connecting to database...")
            self.connection = pyodbc.connect(
                self.config.connection_string,
                timeout=self.config.timeout_seconds
            )
            self.connection.autocommit = False
            self.logger.info("Database connection established")
        except Exception as e:
            raise DataLoadError(f"Failed to connect to database: {e}")

    def disconnect(self) -> None:
        """Close database connection"""
        try:
            if self.connection:
                self.connection.close()
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.warning(f"Error closing database connection: {e}")

    def load_all_data(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load all transformed data into database using staging approach
        
        Args:
            transformed_data: Dictionary containing all transformed data
            
        Returns:
            Dictionary containing load statistics
        """
        study_code = transformed_data['study_code']
        self.logger.info(f"Starting data load for study: {study_code}")
        
        start_time = time.time()
        
        try:
            # Create staging tables
            self._create_staging_tables(study_code)
            
            # Load dimension tables
            self._load_dimensions(transformed_data['dimensions'], study_code)
            
            # Load fact table
            self._load_fact_expression(transformed_data['facts']['fact_gene_expression'], study_code)
            
            # Execute MERGE procedures to move from staging to production
            self._execute_merge_procedures(study_code)
            
            # Generate load statistics
            self.load_stats['load_duration_seconds'] = time.time() - start_time
            
            # Log results
            self.logger.info(f"Data load completed for {study_code}")
            self.logger.info(f"Load statistics: {self.get_load_summary()}")
            
            return {
                'study_code': study_code,
                'load_stats': self.load_stats.copy(),
                'success': len(self.load_stats['errors']) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Data load failed for {study_code}: {str(e)}")
            self.connection.rollback()
            raise DataLoadError(f"Failed to load data for {study_code}: {str(e)}")

    def _create_staging_tables(self, study_code: str) -> None:
        """Create staging tables for the study"""
        self.logger.info(f"Creating staging tables for study: {study_code}")
        
        try:
            cursor = self.connection.cursor()
            
            # Create staging schema if not exists
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'staging')
                    EXEC('CREATE SCHEMA staging')
            """)
            
            # Drop existing staging tables
            staging_tables = [
                f'staging.dim_study_{study_code}',
                f'staging.dim_platform_{study_code}',
                f'staging.dim_illness_{study_code}',
                f'staging.dim_samples_{study_code}',
                f'staging.dim_genes_{study_code}',
                f'staging.fact_expression_{study_code}'
            ]
            
            for table in staging_tables:
                cursor.execute(f"IF OBJECT_ID('{table}', 'U') IS NOT NULL DROP TABLE {table}")
            
            # Create staging tables with same structure as production
            cursor.execute(f"""
                SELECT * INTO staging.dim_study_{study_code} 
                FROM dim_study WHERE 1=0
            """)
            
            cursor.execute(f"""
                SELECT * INTO staging.dim_platform_{study_code} 
                FROM dim_platform WHERE 1=0
            """)
            
            cursor.execute(f"""
                SELECT * INTO staging.dim_illness_{study_code} 
                FROM dim_illness WHERE 1=0
            """)
            
            cursor.execute(f"""
                SELECT * INTO staging.dim_samples_{study_code} 
                FROM dim_sample WHERE 1=0
            """)
            
            cursor.execute(f"""
                SELECT * INTO staging.dim_genes_{study_code} 
                FROM dim_gene WHERE 1=0
            """)
            
            cursor.execute(f"""
                SELECT * INTO staging.fact_expression_{study_code} 
                FROM fact_gene_expression WHERE 1=0
            """)
            
            self.connection.commit()
            self.logger.info("Staging tables created successfully")
            
        except Exception as e:
            self.connection.rollback()
            raise DataLoadError(f"Failed to create staging tables: {e}")

    def _load_dimensions(self, dimensions: Dict[str, pd.DataFrame], study_code: str) -> None:
        """Load all dimension tables to staging"""
        self.logger.info("Loading dimension tables to staging")
        
        dimension_loaders = {
            'dim_study': self._load_staging_table,
            'dim_platform': self._load_staging_table,
            'dim_illness': self._load_staging_table,
            'dim_samples': self._load_staging_table,
            'dim_genes': self._load_staging_table
        }
        
        for dim_name, dim_data in dimensions.items():
            staging_table = f"staging.{dim_name}_{study_code}"
            self._load_staging_table(dim_data, staging_table)

    def _load_staging_table(self, data: pd.DataFrame, table_name: str) -> None:
        """Load data into staging table using batch insert"""
        self.logger.info(f"Loading data into {table_name}")
        
        try:
            cursor = self.connection.cursor()
            
            # Prepare data for insert
            columns = list(data.columns)
            placeholders = ', '.join(['?' for _ in columns])
            insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Convert DataFrame to list of tuples
            data_tuples = [tuple(row) for row in data.values]
            
            # Batch insert
            batch_size = self.config.batch_size
            total_records = len(data_tuples)
            
            for i in range(0, total_records, batch_size):
                batch_data = data_tuples[i:i + batch_size]
                cursor.executemany(insert_sql, batch_data)
                
                if i % (batch_size * 10) == 0:
                    self.logger.info(f"Loaded {i + len(batch_data)} of {total_records} records into {table_name}")
            
            self.connection.commit()
            self.load_stats['records_inserted'] += total_records
            self.logger.info(f"Successfully loaded {total_records} records into {table_name}")
            
        except Exception as e:
            self.connection.rollback()
            self.load_stats['records_failed'] += len(data)
            raise DataLoadError(f"Failed to load data into {table_name}: {e}")

    def _load_fact_expression(self, fact_data: pd.DataFrame, study_code: str) -> None:
        """Load fact expression data to staging"""
        self.logger.info("Loading fact expression data to staging")
        
        staging_table = f"staging.fact_expression_{study_code}"
        self._load_staging_table(fact_data, staging_table)

    def _execute_merge_procedures(self, study_code: str) -> None:
        """Execute MERGE procedures to move data from staging to production"""
        self.logger.info("Executing MERGE procedures")
        
        try:
            cursor = self.connection.cursor()
            
            # Execute MERGE for each dimension and fact table
            merge_procedures = [
                ('dim_study', f'staging.dim_study_{study_code}'),
                ('dim_platform', f'staging.dim_platform_{study_code}'),
                ('dim_illness', f'staging.dim_illness_{study_code}'),
                ('dim_sample', f'staging.dim_samples_{study_code}'),
                ('dim_gene', f'staging.dim_genes_{study_code}'),
                ('fact_gene_expression', f'staging.fact_expression_{study_code}')
            ]
            
            for target_table, staging_table in merge_procedures:
                self._execute_merge_operation(cursor, target_table, staging_table, study_code)
            
            self.connection.commit()
            self.logger.info("All MERGE operations completed successfully")
            
        except Exception as e:
            self.connection.rollback()
            raise DataLoadError(f"MERGE operations failed: {e}")

    def _execute_merge_operation(self, cursor, target_table: str, staging_table: str, study_code: str) -> None:
        """Execute MERGE operation for a specific table"""
        self.logger.info(f"Executing MERGE for {target_table}")
        
        try:
            # Build MERGE statement based on table type
            if target_table == 'dim_study':
                merge_sql = self._build_study_merge_sql(target_table, staging_table)
            elif target_table == 'dim_platform':
                merge_sql = self._build_platform_merge_sql(target_table, staging_table)
            elif target_table == 'dim_illness':
                merge_sql = self._build_illness_merge_sql(target_table, staging_table)
            elif target_table == 'dim_sample':
                merge_sql = self._build_sample_merge_sql(target_table, staging_table)
            elif target_table == 'dim_gene':
                merge_sql = self._build_gene_merge_sql(target_table, staging_table)
            elif target_table == 'fact_gene_expression':
                merge_sql = self._build_fact_merge_sql(target_table, staging_table)
            else:
                raise DataLoadError(f"Unknown table type: {target_table}")
            
            # Execute MERGE
            cursor.execute(merge_sql)
            
            # Get MERGE statistics
            records_affected = cursor.rowcount
            self.load_stats['records_inserted'] += records_affected
            
            self.logger.info(f"MERGE completed for {target_table}: {records_affected} records affected")
            
        except Exception as e:
            self.logger.error(f"MERGE failed for {target_table}: {e}")
            raise

    def _build_study_merge_sql(self, target_table: str, staging_table: str) -> str:
        """Build MERGE statement for study dimension"""
        return f"""
        MERGE {target_table} AS target
        USING {staging_table} AS source
        ON target.accession_code = source.accession_code
        
        WHEN MATCHED THEN
            UPDATE SET 
                title = source.title,
                description = source.description,
                technology = source.technology,
                organism = source.organism,
                has_publication = source.has_publication,
                pubmed_id = source.pubmed_id,
                publication_title = source.publication_title,
                source_first_published = source.source_first_published,
                source_last_modified = source.source_last_modified,
                updated_date = GETDATE()
        
        WHEN NOT MATCHED THEN
            INSERT (accession_code, title, description, technology, organism, 
                   has_publication, pubmed_id, publication_title, 
                   source_first_published, source_last_modified)
            VALUES (source.accession_code, source.title, source.description, 
                   source.technology, source.organism, source.has_publication, 
                   source.pubmed_id, source.publication_title, 
                   source.source_first_published, source.source_last_modified);
        """

    def _build_platform_merge_sql(self, target_table: str, staging_table: str) -> str:
        """Build MERGE statement for platform dimension"""
        return f"""
        MERGE {target_table} AS target
        USING {staging_table} AS source
        ON target.platform_name = source.platform_name
        
        WHEN MATCHED THEN
            UPDATE SET 
                platform_type = source.platform_type,
                processor_name = source.processor_name,
                processor_version = source.processor_version,
                processor_id = source.processor_id,
                updated_date = GETDATE()
        
        WHEN NOT MATCHED THEN
            INSERT (platform_name, platform_type, processor_name, processor_version, processor_id)
            VALUES (source.platform_name, source.platform_type, source.processor_name, 
                   source.processor_version, source.processor_id);
        """

    def _build_illness_merge_sql(self, target_table: str, staging_table: str) -> str:
        """Build MERGE statement for illness dimension"""
        return f"""
        MERGE {target_table} AS target
        USING {staging_table} AS source
        ON target.illness_type = source.illness_type
        
        WHEN MATCHED THEN
            UPDATE SET 
                illness_description = source.illness_description,
                updated_date = GETDATE()
        
        WHEN NOT MATCHED THEN
            INSERT (illness_type, illness_description)
            VALUES (source.illness_type, source.illness_description);
        """

    def _build_sample_merge_sql(self, target_table: str, staging_table: str) -> str:
        """Build MERGE statement for sample dimension"""
        return f"""
        MERGE {target_table} AS target
        USING {staging_table} AS source
        ON target.refinebio_accession_code = source.refinebio_accession_code
        
        WHEN MATCHED THEN
            UPDATE SET 
                experiment_accession = source.experiment_accession,
                refinebio_title = source.refinebio_title,
                refinebio_organism = source.refinebio_organism,
                refinebio_processed = source.refinebio_processed,
                refinebio_source_database = source.refinebio_source_database,
                sample_classification = source.sample_classification,
                sample_number = source.sample_number,
                updated_date = GETDATE()
        
        WHEN NOT MATCHED THEN
            INSERT (refinebio_accession_code, experiment_accession, refinebio_title, 
                   refinebio_organism, refinebio_processed, refinebio_source_database,
                   sample_classification, sample_number)
            VALUES (source.refinebio_accession_code, source.experiment_accession, 
                   source.refinebio_title, source.refinebio_organism, source.refinebio_processed,
                   source.refinebio_source_database, source.sample_classification, 
                   source.sample_number);
        """

    def _build_gene_merge_sql(self, target_table: str, staging_table: str) -> str:
        """Build MERGE statement for gene dimension"""
        return f"""
        MERGE {target_table} AS target
        USING {staging_table} AS source
        ON target.gene_symbol = source.gene_symbol
        
        WHEN MATCHED THEN
            UPDATE SET 
                gene_description = source.gene_description,
                gene_type = source.gene_type,
                chromosome = source.chromosome,
                gene_length = source.gene_length,
                strand = source.strand,
                updated_date = GETDATE()
        
        WHEN NOT MATCHED THEN
            INSERT (gene_symbol, gene_description, gene_type, chromosome, gene_length, strand)
            VALUES (source.gene_symbol, source.gene_description, source.gene_type,
                   source.chromosome, source.gene_length, source.strand);
        """

    def _build_fact_merge_sql(self, target_table: str, staging_table: str) -> str:
        """Build MERGE statement for fact table with deduplication"""
        return f"""
        MERGE {target_table} AS target
        USING (
            SELECT sample_key, gene_key, expression_value, etl_batch_id, etl_processed_at,
                   ROW_NUMBER() OVER (PARTITION BY sample_key, gene_key ORDER BY etl_processed_at DESC) as rn
            FROM {staging_table}
        ) AS source
        ON target.sample_key = source.sample_key 
        AND target.gene_key = source.gene_key
        AND source.rn = 1  -- Only take the most recent duplicate
        
        WHEN MATCHED THEN
            UPDATE SET 
                expression_value = source.expression_value,
                etl_batch_id = source.etl_batch_id,
                etl_processed_at = source.etl_processed_at
        
        WHEN NOT MATCHED THEN
            INSERT (sample_key, gene_key, expression_value, etl_batch_id, etl_processed_at)
            VALUES (source.sample_key, source.gene_key, source.expression_value,
                   source.etl_batch_id, source.etl_processed_at);
        """

    def _cleanup_staging_tables(self, study_code: str) -> None:
        """Clean up staging tables after successful load"""
        self.logger.info("Cleaning up staging tables")
        
        try:
            cursor = self.connection.cursor()
            
            staging_tables = [
                f'staging.dim_study_{study_code}',
                f'staging.dim_platform_{study_code}',
                f'staging.dim_illness_{study_code}',
                f'staging.dim_samples_{study_code}',
                f'staging.dim_genes_{study_code}',
                f'staging.fact_expression_{study_code}'
            ]
            
            for table in staging_tables:
                cursor.execute(f"IF OBJECT_ID('{table}', 'U') IS NOT NULL DROP TABLE {table}")
            
            self.connection.commit()
            self.logger.info("Staging tables cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to clean up staging tables: {e}")

    def _log_etl_activity(self, study_code: str, status: str, details: str = None) -> None:
        """Log ETL activity to audit table"""
        try:
            cursor = self.connection.cursor()
            
            # Create audit table if not exists
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'etl_audit_log')
                CREATE TABLE etl_audit_log (
                    log_id INT IDENTITY(1,1) PRIMARY KEY,
                    study_code VARCHAR(50),
                    batch_id VARCHAR(50),
                    activity_type VARCHAR(50),
                    status VARCHAR(20),
                    details NVARCHAR(MAX),
                    timestamp DATETIME2 DEFAULT GETDATE()
                )
            """)
            
            cursor.execute("""
                INSERT INTO etl_audit_log (study_code, batch_id, activity_type, status, details)
                VALUES (?, ?, ?, ?, ?)
            """, (study_code, self.batch_id, 'DATA_LOAD', status, details))
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.warning(f"Failed to log ETL activity: {e}")

    def get_load_summary(self) -> Dict[str, Any]:
        """Get summary of load statistics"""
        return {
            'tables_loaded': self.load_stats['tables_loaded'],
            'records_inserted': self.load_stats['records_inserted'],
            'records_updated': self.load_stats['records_updated'],
            'records_failed': self.load_stats['records_failed'],
            'load_duration_seconds': self.load_stats['load_duration_seconds'],
            'warnings_count': len(self.load_stats['warnings']),
            'errors_count': len(self.load_stats['errors']),
            'warnings': self.load_stats['warnings'],
            'errors': self.load_stats['errors']
        }


class ETLPerformanceMonitor:
    """Monitor ETL performance and generate reports"""
    
    def __init__(self, connection):
        self.connection = connection
        self.logger = logging.getLogger(__name__)
        
    def generate_performance_report(self, study_code: str) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            cursor = self.connection.cursor()
            
            # Get table statistics
            cursor.execute("""
                SELECT 
                    t.name AS table_name,
                    s.row_count,
                    s.used_page_count * 8 AS size_kb
                FROM sys.tables t
                JOIN sys.dm_db_partition_stats s ON t.object_id = s.object_id
                WHERE t.name LIKE ?
            """, f'%{study_code}%')
            
            table_stats = []
            for row in cursor.fetchall():
                table_stats.append({
                    'table_name': row.table_name,
                    'row_count': row.row_count,
                    'size_kb': row.size_kb
                })
            
            # Get index statistics
            cursor.execute("""
                SELECT 
                    t.name AS table_name,
                    i.name AS index_name,
                    i.index_id,
                    dm_ius.user_seeks,
                    dm_ius.user_scans,
                    dm_ius.user_lookups,
                    dm_ius.user_updates
                FROM sys.tables t
                JOIN sys.indexes i ON t.object_id = i.object_id
                JOIN sys.dm_db_index_usage_stats dm_ius ON i.object_id = dm_ius.object_id AND i.index_id = dm_ius.index_id
                WHERE t.name LIKE ?
            """, f'%{study_code}%')
            
            index_stats = []
            for row in cursor.fetchall():
                index_stats.append({
                    'table_name': row.table_name,
                    'index_name': row.index_name,
                    'user_seeks': row.user_seeks,
                    'user_scans': row.user_scans,
                    'user_lookups': row.user_lookups,
                    'user_updates': row.user_updates
                })
            
            return {
                'study_code': study_code,
                'table_statistics': table_stats,
                'index_statistics': index_stats,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}