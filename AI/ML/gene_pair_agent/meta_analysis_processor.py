"""
Meta-Analysis Data Processor

Handles file-based meta-analysis data processing and validation.
"""

import logging
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MetaAnalysisProcessor:
    """Processor for meta-analysis data files."""

    REQUIRED_COLUMNS = [
        'pair_id', 'n_studies_ss', 'n_studies_soth',
        'dz_ss_mean', 'dz_ss_se', 'dz_ss_ci_low', 'dz_ss_ci_high',
        'dz_ss_Q', 'dz_ss_I2', 'dz_ss_z', 'p_ss',
        'dz_soth_mean', 'dz_soth_se', 'dz_soth_ci_low', 'dz_soth_ci_high',
        'dz_soth_Q', 'dz_soth_I2', 'dz_soth_z', 'p_soth',
        'kappa_ss', 'kappa_soth', 'abs_dz_ss', 'abs_dz_soth',
        'q_ss', 'q_soth', 'rank_score',
        'GeneAName', 'GeneBName', 'GeneAKey', 'GeneBKey'
    ]

    COLUMN_ALIASES = {
        'gene_a': 'GeneAName',
        'genea': 'GeneAName',
        'gene_a_name': 'GeneAName',
        'gene_a_symbol': 'GeneAName',
        'gene_a_gene': 'GeneAName',
        'gene_b': 'GeneBName',
        'geneb': 'GeneBName',
        'gene_b_name': 'GeneBName',
        'gene_b_symbol': 'GeneBName',
        'gene_b_gene': 'GeneBName',
        'gene_a_key': 'GeneAKey',
        'genea_key': 'GeneAKey',
        'gene_a_id': 'GeneAKey',
        'gene_a_identifier': 'GeneAKey',
        'gene_b_key': 'GeneBKey',
        'geneb_key': 'GeneBKey',
        'gene_b_id': 'GeneBKey',
        'gene_b_identifier': 'GeneBKey',
        'pairid': 'pair_id',
        'pair_identifier': 'pair_id',
        'pair name': 'pair_id',
        'n_studies_septic_shock': 'n_studies_ss',
        'nstudies_ss': 'n_studies_ss',
        'studies_ss': 'n_studies_ss',
        'n_studies_other': 'n_studies_soth',
        'n_studies_sceptic_other': 'n_studies_soth',
        'n_studies_control': 'n_studies_soth',
        'studies_soth': 'n_studies_soth',
        'dz_ss_mean_effect': 'dz_ss_mean',
        'dz_soth_mean_effect': 'dz_soth_mean',
        'pvalue_ss': 'p_ss',
        'p_value_ss': 'p_ss',
        'pvalue_soth': 'p_soth',
        'p_value_soth': 'p_soth',
        'qvalue_ss': 'q_ss',
        'q_value_ss': 'q_ss',
        'qvalue_soth': 'q_soth',
        'q_value_soth': 'q_soth',
        'rankscore': 'rank_score',
        'rank score': 'rank_score'
    }
    
    OPTIONAL_COLUMNS = [
        'study_key', 'illness_label', 'rho_spearman'
    ]
    
    def __init__(self, file_path: Optional[str] = None):
        """Initialize the meta-analysis processor."""
        self.file_path = file_path
        self.data = None
        self.metadata = {}
        
    def load_data(self, file_path: str, file_type: str = 'auto') -> pd.DataFrame:
        """Load meta-analysis data from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        if file_type == 'auto':
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                file_type = 'excel'
            elif file_path.suffix.lower() == '.csv':
                file_type = 'csv'
            elif file_path.suffix.lower() == '.json':
                file_type = 'json'
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Load data based on file type
        if file_type == 'excel':
            data = pd.read_excel(file_path)
        elif file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'json':
            data = pd.read_json(file_path, orient='records')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        data = self._normalize_column_names(data)

        logger.info(f"Loaded {len(data)} records from {file_path}")
        
        # Store data and metadata
        self.data = data
        self.metadata = {
            'file_path': str(file_path),
            'file_type': file_type,
            'load_timestamp': pd.Timestamp.now().isoformat(),
            'original_shape': data.shape
        }

        return data

    @classmethod
    def _normalize_key(cls, column_name: str) -> str:
        """Normalize column names for comparison."""
        normalized = re.sub(r'[^a-z0-9]+', '_', column_name.strip().lower())
        return normalized.strip('_')

    def _normalize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rename columns using known aliases before validation."""
        alias_map = {
            self._normalize_key(col): col for col in self.REQUIRED_COLUMNS
        }

        for alias, target in self.COLUMN_ALIASES.items():
            alias_map[self._normalize_key(alias)] = target

        rename_map = {}
        for column in data.columns:
            normalized = self._normalize_key(column)
            if normalized in alias_map:
                target = alias_map[normalized]
                if target not in data.columns:
                    rename_map[column] = target
                elif column != target:
                    logger.debug(
                        "Skipping renaming of column '%s' to '%s' because target already exists",
                        column,
                        target
                    )

        if rename_map:
            logger.info("Normalizing column names using aliases: %s", rename_map)
            data = data.rename(columns=rename_map)

        return data
    
    def validate_data(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Validate meta-analysis data structure and content."""
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data to validate")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_columns': [],
            'data_quality': {}
        }
        
        # Check required columns
        missing_required = []
        for col in self.REQUIRED_COLUMNS:
            if col not in data.columns:
                missing_required.append(col)
        
        if missing_required:
            validation_results['missing_columns'] = missing_required
            validation_results['errors'].append(f"Missing required columns: {missing_required}")
            validation_results['is_valid'] = False
        
        # Check data types and ranges
        numeric_columns = [
            'dz_ss_mean', 'dz_ss_se', 'dz_ss_ci_low', 'dz_ss_ci_high',
            'dz_soth_mean', 'dz_soth_se', 'dz_soth_ci_low', 'dz_soth_ci_high',
            'p_ss', 'p_soth', 'q_ss', 'q_soth'
        ]
        
        for col in numeric_columns:
            if col in data.columns:
                non_numeric = pd.to_numeric(data[col], errors='coerce').isna().sum()
                if non_numeric > 0:
                    validation_results['warnings'].append(
                        f"Column {col} has {non_numeric} non-numeric values"
                    )
        
        # Check p-value ranges
        if 'p_ss' in data.columns:
            invalid_p = ((data['p_ss'] < 0) | (data['p_ss'] > 1)).sum()
            if invalid_p > 0:
                validation_results['errors'].append(f"Invalid p_ss values: {invalid_p}")
                validation_results['is_valid'] = False
        
        if 'p_soth' in data.columns:
            invalid_p = ((data['p_soth'] < 0) | (data['p_soth'] > 1)).sum()
            if invalid_p > 0:
                validation_results['errors'].append(f"Invalid p_soth values: {invalid_p}")
                validation_results['is_valid'] = False
        
        # Check sample sizes
        if 'n_studies_ss' in data.columns:
            zero_studies = (data['n_studies_ss'] <= 0).sum()
            if zero_studies > 0:
                validation_results['warnings'].append(
                    f"Zero studies for {zero_studies} pairs in septic shock"
                )
        
        # Check effect sizes
        if 'dz_ss_mean' in data.columns:
            extreme_effects = (np.abs(data['dz_ss_mean']) > 10).sum()
            if extreme_effects > 0:
                validation_results['warnings'].append(
                    f"Extreme effect sizes (>10) in {extreme_effects} pairs"
                )
        
        # Data quality metrics
        validation_results['data_quality'] = {
            'total_pairs': len(data),
            'complete_cases': len(data.dropna()),
            'missing_data_pct': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'unique_gene_pairs': len(data[['GeneAName', 'GeneBName']].drop_duplicates())
        }
        
        logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
        
        return validation_results
    
    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean and preprocess meta-analysis data."""
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data to clean")
        
        cleaned_data = data.copy()
        
        # Remove rows with missing critical data
        critical_columns = ['GeneAName', 'GeneBName', 'p_ss', 'p_soth']
        critical_columns = [col for col in critical_columns if col in cleaned_data.columns]
        
        if critical_columns:
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(subset=critical_columns)
            removed_rows = initial_rows - len(cleaned_data)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with missing critical data")
        
        # Handle missing values in statistical measures
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in cleaned_data.columns:
                # Fill missing values with median for robustness
                median_val = cleaned_data[col].median()
                cleaned_data[col] = cleaned_data[col].fillna(median_val)
        
        # Ensure positive values for certain measures
        positive_columns = ['n_studies_ss', 'n_studies_soth', 'abs_dz_ss', 'abs_dz_soth']
        for col in positive_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = np.maximum(0, cleaned_data[col])
        
        # Clip p-values to valid range
        p_columns = ['p_ss', 'p_soth', 'q_ss', 'q_soth']
        for col in p_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = np.clip(cleaned_data[col], 0, 1)
        
        # Create additional useful columns
        if 'dz_ss_mean' in cleaned_data.columns and 'dz_soth_mean' in cleaned_data.columns:
            cleaned_data['effect_size_composite'] = np.sqrt(
                cleaned_data['dz_ss_mean']**2 + cleaned_data['dz_soth_mean']**2
            )
        
        if 'p_ss' in cleaned_data.columns and 'p_soth' in cleaned_data.columns:
            cleaned_data['significance_composite'] = -np.log10(
                cleaned_data['p_ss'] * cleaned_data['p_soth'] + 1e-10
            )
        
        logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
        
        return cleaned_data
    
    def prepare_for_analysis(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare data for machine learning analysis."""
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data to prepare")
        
        # Clean data first
        prepared_data = self.clean_data(data)
        
        # Add metadata columns
        prepared_data['analysis_batch_id'] = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        prepared_data['data_source'] = self.metadata.get('file_path', 'unknown')
        
        # Ensure all required columns exist
        for col in self.REQUIRED_COLUMNS:
            if col not in prepared_data.columns:
                logger.warning(f"Adding missing required column: {col}")
                prepared_data[col] = np.nan
        
        logger.info(f"Data prepared for analysis: {prepared_data.shape}")
        
        return prepared_data
    
    def get_summary_statistics(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data to summarize")
        
        stats = {
            'basic_info': {
                'total_pairs': len(data),
                'unique_genes': len(set(data['GeneAName'].tolist() + data['GeneBName'].tolist())),
                'data_shape': data.shape,
                'columns': data.columns.tolist()
            }
        }
        
        # Statistical measures summary
        numeric_columns = [
            'dz_ss_mean', 'dz_soth_mean', 'p_ss', 'p_soth',
            'abs_dz_ss', 'abs_dz_soth', 'q_ss', 'q_soth'
        ]
        
        stats['statistical_summary'] = {}
        for col in numeric_columns:
            if col in data.columns:
                stats['statistical_summary'][col] = {
                    'mean': float(data[col].mean()),
                    'median': float(data[col].median()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'missing': int(data[col].isnull().sum())
                }
        
        # Significance counts
        if 'p_ss' in data.columns:
            stats['significance_counts'] = {
                'p_ss_significant_0.05': int((data['p_ss'] < 0.05).sum()),
                'p_ss_significant_0.01': int((data['p_ss'] < 0.01).sum()),
                'p_soth_significant_0.05': int((data['p_soth'] < 0.05).sum()) if 'p_soth' in data.columns else 0,
                'p_soth_significant_0.01': int((data['p_soth'] < 0.01).sum()) if 'p_soth' in data.columns else 0
            }
        
        # Quality metrics
        stats['quality_metrics'] = {
            'complete_cases': len(data.dropna()),
            'missing_data_percentage': float((data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100),
            'pairs_with_both_significant': int(
                ((data['p_ss'] < 0.05) & (data['p_soth'] < 0.05)).sum()
            ) if all(col in data.columns for col in ['p_ss', 'p_soth']) else 0
        }
        
        return stats
    
    def export_data(self, 
                   data: pd.DataFrame, 
                   output_path: str, 
                   format: str = 'excel') -> str:
        """Export processed data to file."""
        output_path = Path(output_path)
        
        if format == 'excel':
            data.to_excel(output_path, index=False)
        elif format == 'csv':
            data.to_csv(output_path, index=False)
        elif format == 'json':
            data.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Data exported to {output_path}")
        return str(output_path)
    
    def create_sample_data(self, n_pairs: int = 100) -> pd.DataFrame:
        """Create sample meta-analysis data for testing."""
        np.random.seed(42)
        
        # Sample gene names
        genes = ['MS4A4A', 'CD86', 'DHRS9', 'SULF2', 'MTMR11', 'RAB20', 'TTYH2', 
                'RGCC', 'PLAC8', 'ARHGEF10L', 'CD82', 'KLF7', 'CKAP4', 'DHCR7']
        
        sample_data = []
        for i in range(n_pairs):
            gene_a = np.random.choice(genes)
            gene_b = np.random.choice([g for g in genes if g != gene_a])
            
            # Generate realistic statistical measures
            dz_ss_mean = np.random.normal(-0.5, 0.3)
            dz_soth_mean = np.random.normal(-1.0, 0.5)
            
            p_ss = np.random.beta(0.5, 5)  # Skewed towards small p-values
            p_soth = np.random.beta(0.3, 10)  # More skewed
            
            se_ss = np.random.uniform(0.1, 0.3)
            se_soth = np.random.uniform(0.2, 0.4)
            
            sample_data.append({
                'pair_id': f'{gene_a}_{gene_b}_{i}',
                'n_studies_ss': np.random.randint(1, 5),
                'n_studies_soth': np.random.randint(3, 8),
                'dz_ss_mean': dz_ss_mean,
                'dz_ss_se': se_ss,
                'dz_ss_ci_low': dz_ss_mean - 1.96 * se_ss,
                'dz_ss_ci_high': dz_ss_mean + 1.96 * se_ss,
                'dz_ss_Q': np.random.uniform(0, 10),
                'dz_ss_I2': np.random.uniform(0, 80),
                'dz_ss_z': dz_ss_mean / se_ss,
                'p_ss': p_ss,
                'dz_soth_mean': dz_soth_mean,
                'dz_soth_se': se_soth,
                'dz_soth_ci_low': dz_soth_mean - 1.96 * se_soth,
                'dz_soth_ci_high': dz_soth_mean + 1.96 * se_soth,
                'dz_soth_Q': np.random.uniform(0, 15),
                'dz_soth_I2': np.random.uniform(0, 90),
                'dz_soth_z': dz_soth_mean / se_soth,
                'p_soth': p_soth,
                'kappa_ss': np.random.uniform(0.7, 1.0),
                'kappa_soth': np.random.uniform(0.7, 1.0),
                'abs_dz_ss': abs(dz_ss_mean),
                'abs_dz_soth': abs(dz_soth_mean),
                'q_ss': p_ss * 2,  # Simplified FDR
                'q_soth': p_soth * 2,
                'rank_score': np.random.uniform(0.5, 2.0),
                'GeneAName': gene_a,
                'GeneBName': gene_b,
                'GeneAKey': str(np.random.randint(1, 100)),
                'GeneBKey': str(np.random.randint(1, 100))
            })
        
        sample_df = pd.DataFrame(sample_data)
        self.data = sample_df
        
        logger.info(f"Created sample data with {n_pairs} gene pairs")
        return sample_df