"""
Database Connector for Gene Expression Data

Handles database connectivity for source data retrieval and processing.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Database connector for gene expression source data."""
    
    DEFAULT_VIEW_SQL = """
    SELECT
      geneA, geneB,
      geneA_ss_sepsis, geneB_ss_sepsis,
      geneA_ss_ctrl,   geneB_ss_ctrl,
      geneA_ss_direction, geneB_ss_direction,
      illness_label,
      rho_spearman, p_value, q_value, n_samples,
      study_key,
      GeneAName, GeneBName, GeneAKey, GeneBKey
    FROM dbo.vw_gene_DE_fact_corr_data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database connector with configuration."""
        self.config = config or {}
        self.engine = None
        self.connection = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load database configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return self.get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.config = config.get('database', {})
            logger.info(f"Loaded database config from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self.get_default_config()
        
        return self.config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default database configuration."""
        return {
            'driver': 'ODBC Driver 17 for SQL Server',
            'server': 'localhost',
            'database': 'BioinformaticsDB',
            'username': 'sa',
            'password': 'password',
            'trusted_connection': False,
            'connection_timeout': 30
        }
    
    def create_connection_string(self) -> str:
        """Create database connection string from configuration."""
        config = self.config
        
        if config.get('trusted_connection', False):
            conn_str = (
                f"DRIVER={{{config['driver']}}};"
                f"SERVER={config['server']};"
                f"DATABASE={config['database']};"
                "Trusted_Connection=yes;"
                "TrustServerCertificate=yes"
            )
        else:
            conn_str = (
                f"DRIVER={{{config['driver']}}};"
                f"SERVER={config['server']};"
                f"DATABASE={config['database']};"
                f"UID={config['username']};"
                f"PWD={config['password']};"
                "TrustServerCertificate=yes"
            )
        
        return conn_str
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            conn_str = self.create_connection_string()
            self.engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={conn_str}",
                future=True,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("Database connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connectivity and return status."""
        result = {
            'connected': False,
            'error': None,
            'server_info': None,
            'database_info': None
        }
        
        try:
            if self.connect():
                with self.engine.connect() as conn:
                    # Get server information
                    server_info = conn.execute(text("SELECT @@VERSION")).scalar()
                    db_info = conn.execute(text("SELECT DB_NAME()")).scalar()
                    
                    result.update({
                        'connected': True,
                        'server_info': server_info,
                        'database_info': db_info
                    })
                    
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Connection test failed: {e}")
        
        return result
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a database table."""
        if not self.connect():
            raise ConnectionError("Cannot connect to database")
        
        try:
            inspector = inspect(self.engine)
            
            # Get columns
            columns = inspector.get_columns(table_name)
            
            # Get row count
            with self.engine.connect() as conn:
                row_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                ).scalar()
            
            return {
                'table_name': table_name,
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable']
                    }
                    for col in columns
                ],
                'row_count': row_count
            }
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        if not self.connect():
            raise ConnectionError("Cannot connect to database")

        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))

                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            logger.info(f"Query executed successfully. Returned {len(df)} rows")
            return df

        except SQLAlchemyError as e:
            logger.error(f"Database query error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            raise

    def get_rho_spearman_distribution(
        self,
        gene_a: str,
        gene_b: str,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Return rho_spearman values for a gene pair across all studies.

        The query pulls correlations for the requested pair in either
        orientation (A/B or B/A) and optionally limits the number of rows
        returned. Results are ordered by correlation value so that callers can
        easily visualise the distribution.
        """

        if not gene_a or not gene_b:
            raise ValueError("Both gene_a and gene_b must be provided")

        params = {
            'gene_a': gene_a,
            'gene_b': gene_b
        }

        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be a positive integer when provided")
            params['limit'] = int(limit)
            query = text(
                """
                SELECT rho_spearman
                FROM dbo.vw_gene_DE_fact_corr_data
                WHERE (GeneAName = :gene_a AND GeneBName = :gene_b)
                   OR (GeneAName = :gene_b AND GeneBName = :gene_a)
                ORDER BY rho_spearman
                OFFSET 0 ROWS FETCH NEXT :limit ROWS ONLY
                """
            )
        else:
            query = text(
                """
                SELECT rho_spearman
                FROM dbo.vw_gene_DE_fact_corr_data
                WHERE (GeneAName = :gene_a AND GeneBName = :gene_b)
                   OR (GeneAName = :gene_b AND GeneBName = :gene_a)
                ORDER BY rho_spearman
                """
            )

        if not self.connect():
            raise ConnectionError("Cannot connect to database")

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, params)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            logger.info(
                "Fetched %d rho_spearman values for gene pair %s/%s",
                len(df),
                gene_a,
                gene_b
            )
            return df

        except SQLAlchemyError as exc:
            logger.error("Error retrieving rho_spearman distribution: %s", exc)
            raise
    
    def get_gene_pair_data(self, 
                          gene_a: Optional[str] = None,
                          gene_b: Optional[str] = None,
                          illness_label: Optional[str] = None,
                          limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve gene pair correlation data from database."""
        
        # Build query
        query_parts = [self.DEFAULT_VIEW_SQL.strip()]
        conditions = []
        params = {}
        
        if gene_a:
            conditions.append("GeneAName = :gene_a")
            params['gene_a'] = gene_a
        
        if gene_b:
            conditions.append("GeneBName = :gene_b")
            params['gene_b'] = gene_b
        
        if illness_label:
            conditions.append("illness_label = :illness_label")
            params['illness_label'] = illness_label
        
        if conditions:
            query_parts.append("WHERE")
            query_parts.append(" AND ".join(conditions))
        
        if limit:
            query_parts.append(f"TOP {limit}")
        
        query = " ".join(query_parts)
        
        return self.execute_query(query, params)
    
    def get_gene_list(self) -> List[str]:
        """Get list of all unique genes in the database."""
        query = """
        SELECT DISTINCT GeneAName as gene_name FROM dbo.vw_gene_DE_fact_corr_data
        UNION
        SELECT DISTINCT GeneBName as gene_name FROM dbo.vw_gene_DE_fact_corr_data
        ORDER BY gene_name
        """
        
        result = self.execute_query(query)
        return result['gene_name'].tolist()
    
    def get_study_summary(self) -> pd.DataFrame:
        """Get summary information about studies in the database."""
        query = """
        SELECT 
            study_key,
            illness_label,
            COUNT(*) as n_pairs,
            AVG(n_samples) as avg_samples,
            MIN(n_samples) as min_samples,
            MAX(n_samples) as max_samples
        FROM dbo.vw_gene_DE_fact_corr_data
        GROUP BY study_key, illness_label
        ORDER BY study_key, illness_label
        """
        
        return self.execute_query(query)
    
    def get_gene_statistics(self, gene_name: str) -> Dict[str, Any]:
        """Get statistical summary for a specific gene."""
        query = """
        SELECT 
            COUNT(*) as total_pairs,
            AVG(rho_spearman) as avg_correlation,
            AVG(p_value) as avg_p_value,
            illness_label,
            COUNT(CASE WHEN p_value < 0.05 THEN 1 END) as significant_pairs
        FROM dbo.vw_gene_DE_fact_corr_data
        WHERE GeneAName = :gene_name OR GeneBName = :gene_name
        GROUP BY illness_label
        """
        
        result = self.execute_query(query, {'gene_name': gene_name})
        
        if result.empty:
            return {'gene_name': gene_name, 'error': 'Gene not found'}
        
        return {
            'gene_name': gene_name,
            'statistics': result.to_dict('records'),
            'total_pairs': int(result['total_pairs'].sum()),
            'significant_pairs': int(result['significant_pairs'].sum())
        }
    
    def export_to_csv(self, output_path: str, query: Optional[str] = None) -> str:
        """Export database data to CSV file."""
        if query is None:
            query = self.DEFAULT_VIEW_SQL
        
        try:
            data = self.execute_query(query)
            data.to_csv(output_path, index=False)
            logger.info(f"Data exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise
    
    def batch_process_gene_pairs(self, 
                               gene_pairs: List[Tuple[str, str]],
                               callback: Optional[callable] = None) -> List[pd.DataFrame]:
        """Process multiple gene pairs in batches."""
        results = []
        
        for i, (gene_a, gene_b) in enumerate(gene_pairs):
            try:
                data = self.get_gene_pair_data(gene_a=gene_a, gene_b=gene_b)
                results.append(data)
                
                if callback:
                    callback(i + 1, len(gene_pairs), gene_a, gene_b)
                    
            except Exception as e:
                logger.error(f"Error processing pair {gene_a}-{gene_b}: {e}")
                # Add empty DataFrame for failed pairs
                results.append(pd.DataFrame())
        
        return results
    
    def create_sample_data(self, n_records: int = 1000) -> pd.DataFrame:
        """Create sample database data for testing."""
        np.random.seed(42)
        
        genes = ['MS4A4A', 'CD86', 'DHRS9', 'SULF2', 'MTMR11', 'RAB20', 'TTYH2', 
                'RGCC', 'PLAC8', 'ARHGEF10L', 'CD82', 'KLF7', 'CKAP4', 'DHCR7']
        
        illness_labels = ['control', 'sepsis', 'septic shock']
        
        sample_data = []
        for i in range(n_records):
            gene_a = np.random.choice(genes)
            gene_b = np.random.choice([g for g in genes if g != gene_a])
            
            sample_data.append({
                'geneA': f'ENSG00000{np.random.randint(100000, 999999)}',
                'geneB': f'ENSG00000{np.random.randint(100000, 999999)}',
                'geneA_ss_sepsis': np.random.normal(100, 20),
                'geneB_ss_sepsis': np.random.normal(-50, 15),
                'geneA_ss_ctrl': np.random.normal(98, 18),
                'geneB_ss_ctrl': np.random.normal(-45, 12),
                'geneA_ss_direction': np.random.choice(['Up', 'Down']),
                'geneB_ss_direction': np.random.choice(['Up', 'Down']),
                'illness_label': np.random.choice(illness_labels),
                'rho_spearman': np.random.uniform(-0.8, 0.8),
                'p_value': np.random.beta(0.5, 5),
                'q_value': np.random.beta(0.5, 3),
                'n_samples': np.random.randint(10, 100),
                'study_key': np.random.randint(1, 10),
                'GeneAName': gene_a,
                'GeneBName': gene_b,
                'GeneAKey': str(np.random.randint(1, 100)),
                'GeneBKey': str(np.random.randint(1, 100))
            })
        
        sample_df = pd.DataFrame(sample_data)
        logger.info(f"Created sample database data with {n_records} records")
        return sample_df