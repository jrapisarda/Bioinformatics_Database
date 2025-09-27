"""
Production-Grade Data Validation Module
Comprehensive data validation and quality checks for ETL pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import re
from datetime import datetime
import hashlib
import json


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class DataValidator:
    """Comprehensive data validation and quality assurance for ETL pipeline"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_results = {
            'validation_passed': False,
            'validation_timestamp': None,
            'checks_performed': [],
            'warnings': [],
            'errors': [],
            'quality_metrics': {}
        }
        
        # Validation patterns
        self._study_code_pattern = re.compile(r'^[A-Za-z]{2,3}-[A-Za-z]{3}-\d+|[A-Za-z]{3}\d+$')
        self._gene_symbol_pattern = re.compile(r'^[A-Za-z0-9_-]+$')
        self._sample_accession_pattern = re.compile(r'^SRR\d+$')
        
        # Data quality thresholds
        self.quality_thresholds = {
            'max_null_percentage': config.max_null_percentage,
            'max_duplicate_percentage': config.max_duplicate_percentage,
            'min_genes_per_sample': config.min_genes_per_sample,
            'max_expression_range': 1000000,  # Reasonable upper limit for expression values
            'min_samples_per_study': 3,
            'max_missing_samples_percentage': 0.2
        }

    def validate_all_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive validation of all extracted data
        
        Args:
            extracted_data: Dictionary containing all extracted data
            
        Returns:
            Validation results dictionary
        """
        study_code = extracted_data['study_code']
        self.logger.info(f"Starting comprehensive data validation for study: {study_code}")
        
        try:
            # Reset validation results
            self.validation_results = {
                'validation_passed': False,
                'validation_timestamp': datetime.now().isoformat(),
                'checks_performed': [],
                'warnings': [],
                'errors': [],
                'quality_metrics': {}
            }
            
            # Perform validation checks
            self._validate_study_code(study_code)
            self._validate_json_metadata(extracted_data['json_metadata'])
            self._validate_tsv_metadata(extracted_data['tsv_metadata'])
            self._validate_expression_data(extracted_data['expression_data'])
            self._validate_data_consistency(extracted_data)
            self._validate_data_quality(extracted_data)
            
            # Determine overall validation result
            self.validation_results['validation_passed'] = len(self.validation_results['errors']) == 0
            
            # Log validation summary
            self._log_validation_summary(study_code)
            
            return self.validation_results.copy()
            
        except Exception as e:
            self.logger.error(f"Validation process failed for {study_code}: {e}")
            self.validation_results['errors'].append(f"Validation process error: {str(e)}")
            return self.validation_results

    def _validate_study_code(self, study_code: str) -> None:
        """Validate study code format"""
        check_name = "Study Code Format"
        
        try:
            if not study_code:
                self.validation_results['errors'].append(f"{check_name}: Study code is empty")
                return
            
            if not self._study_code_pattern.match(study_code):
                self.validation_results['warnings'].append(
                    f"{check_name}: Study code '{study_code}' doesn't match expected pattern"
                )
            
            self.validation_results['checks_performed'].append(check_name)
            
        except Exception as e:
            self.validation_results['errors'].append(f"{check_name}: Error during validation - {e}")

    def _validate_json_metadata(self, json_data: Dict[str, Any]) -> None:
        """Validate JSON metadata structure and content"""
        check_name = "JSON Metadata Validation"
        
        try:
            # Check required top-level structure
            required_keys = ['experiment', 'samples', 'metadata']
            missing_keys = [key for key in required_keys if key not in json_data]
            
            if missing_keys:
                self.validation_results['errors'].append(
                    f"{check_name}: Missing required keys: {missing_keys}"
                )
                return
            
            # Validate experiment data
            experiment = json_data['experiment']
            required_exp_fields = ['accession_code', 'title', 'organism']
            missing_exp_fields = [field for field in required_exp_fields if field not in experiment]
            
            if missing_exp_fields:
                self.validation_results['errors'].append(
                    f"{check_name}: Missing experiment fields: {missing_exp_fields}"
                )
            
            # Validate samples data
            samples = json_data['samples']
            if not isinstance(samples, dict) or len(samples) == 0:
                self.validation_results['errors'].append(
                    f"{check_name}: Samples data is empty or invalid"
                )
            
            # Validate sample structure
            invalid_samples = []
            for sample_id, sample_data in samples.items():
                if not isinstance(sample_data, dict):
                    invalid_samples.append(sample_id)
                elif 'accession_code' not in sample_data:
                    invalid_samples.append(sample_id)
            
            if invalid_samples:
                self.validation_results['warnings'].append(
                    f"{check_name}: Invalid sample records: {invalid_samples[:10]}..."
                )
            
            # Store metadata metrics
            self.validation_results['quality_metrics']['json_samples_count'] = len(samples)
            self.validation_results['checks_performed'].append(check_name)
            
        except Exception as e:
            self.validation_results['errors'].append(f"{check_name}: Error during validation - {e}")

    def _validate_tsv_metadata(self, tsv_data: pd.DataFrame) -> None:
        """Validate TSV metadata structure and content"""
        check_name = "TSV Metadata Validation"
        
        try:
            if tsv_data.empty:
                self.validation_results['errors'].append(f"{check_name}: TSV data is empty")
                return
            
            # Check required columns
            required_columns = ['refinebio_accession_code']
            missing_columns = [col for col in required_columns if col not in tsv_data.columns]
            
            if missing_columns:
                self.validation_results['errors'].append(
                    f"{check_name}: Missing required columns: {missing_columns}"
                )
            
            # Validate accession codes
            if 'refinebio_accession_code' in tsv_data.columns:
                accession_codes = tsv_data['refinebio_accession_code'].dropna()
                invalid_accessions = [
                    code for code in accession_codes 
                    if not self._sample_accession_pattern.match(str(code))
                ]
                
                if invalid_accessions:
                    self.validation_results['warnings'].append(
                        f"{check_name}: Invalid accession codes: {invalid_accessions[:10]}..."
                    )
            
            # Check for duplicate rows
            duplicate_count = tsv_data.duplicated().sum()
            if duplicate_count > 0:
                duplicate_percentage = duplicate_count / len(tsv_data)
                if duplicate_percentage > self.quality_thresholds['max_duplicate_percentage']:
                    self.validation_results['warnings'].append(
                        f"{check_name}: High duplicate percentage: {duplicate_percentage:.2%}"
                    )
            
            # Check for null values
            null_counts = tsv_data.isnull().sum()
            total_cells = len(tsv_data) * len(tsv_data.columns)
            null_percentage = null_counts.sum() / total_cells
            
            if null_percentage > self.quality_thresholds['max_null_percentage']:
                self.validation_results['warnings'].append(
                    f"{check_name}: High null value percentage: {null_percentage:.2%}"
                )
            
            # Store TSV metrics
            self.validation_results['quality_metrics'].update({
                'tsv_rows': len(tsv_data),
                'tsv_columns': len(tsv_data.columns),
                'tsv_duplicate_count': duplicate_count,
                'tsv_null_percentage': null_percentage
            })
            
            self.validation_results['checks_performed'].append(check_name)
            
        except Exception as e:
            self.validation_results['errors'].append(f"{check_name}: Error during validation - {e}")

    def _validate_expression_data(self, expression_data: pd.DataFrame) -> None:
        """Validate gene expression matrix data"""
        check_name = "Expression Data Validation"
        
        try:
            if expression_data.empty:
                self.validation_results['errors'].append(f"{check_name}: Expression data is empty")
                return
            
            # Check matrix dimensions
            rows, cols = expression_data.shape
            
            if rows < self.quality_thresholds['min_genes_per_sample']:
                self.validation_results['warnings'].append(
                    f"{check_name}: Low gene count: {rows} (minimum: {self.quality_thresholds['min_genes_per_sample']})"
                )
            
            if cols < self.quality_thresholds['min_samples_per_study']:
                self.validation_results['warnings'].append(
                    f"{check_name}: Low sample count: {cols} (minimum: {self.quality_thresholds['min_samples_per_study']})"
                )
            
            # Check for null values
            null_percentage = expression_data.isnull().sum().sum() / (rows * cols)
            
            if null_percentage > 0.1:  # 10% threshold for expression data
                self.validation_results['warnings'].append(
                    f"{check_name}: High null percentage in expression data: {null_percentage:.2%}"
                )
            
            # Check expression value ranges
            numeric_data = expression_data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                max_value = numeric_data.max().max()
                min_value = numeric_data.min().min()
                
                if max_value > self.quality_thresholds['max_expression_range']:
                    self.validation_results['warnings'].append(
                        f"{check_name}: Unusually high expression values: {max_value}"
                    )
                
                if min_value < 0:
                    self.validation_results['warnings'].append(
                        f"{check_name}: Negative expression values found: {min_value}"
                    )
            
            # Check for duplicate gene symbols
            duplicate_genes = expression_data.index.duplicated().sum()
            if duplicate_genes > 0:
                self.validation_results['warnings'].append(
                    f"{check_name}: Duplicate gene symbols: {duplicate_genes}"
                )
            
            # Store expression metrics
            self.validation_results['quality_metrics'].update({
                'expression_genes': rows,
                'expression_samples': cols,
                'expression_null_percentage': null_percentage,
                'expression_max_value': numeric_data.max().max() if not numeric_data.empty else None,
                'expression_min_value': numeric_data.min().min() if not numeric_data.empty else None
            })
            
            self.validation_results['checks_performed'].append(check_name)
            
        except Exception as e:
            self.validation_results['errors'].append(f"{check_name}: Error during validation - {e}")

    def _validate_data_consistency(self, extracted_data: Dict[str, Any]) -> None:
        """Validate consistency across all data sources"""
        check_name = "Data Consistency Validation"
        
        try:
            # Get sample sets from different sources
            json_samples = set(extracted_data['json_metadata']['samples'].keys())
            
            tsv_samples = set()
            if 'refinebio_accession_code' in extracted_data['tsv_metadata'].columns:
                tsv_samples = set(extracted_data['tsv_metadata']['refinebio_accession_code'].dropna().astype(str))
            
            expression_samples = set(extracted_data['expression_data'].columns.astype(str))
            
            # Calculate overlaps
            all_samples = json_samples.union(tsv_samples).union(expression_samples)
            
            if len(all_samples) == 0:
                self.validation_results['errors'].append(f"{check_name}: No samples found in any data source")
                return
            
            json_tsv_overlap = len(json_samples.intersection(tsv_samples)) / len(all_samples) if all_samples else 0
            tsv_expr_overlap = len(tsv_samples.intersection(expression_samples)) / len(all_samples) if all_samples else 0
            json_expr_overlap = len(json_samples.intersection(expression_samples)) / len(all_samples) if all_samples else 0
            
            consistency_threshold = 0.8  # 80% overlap required
            
            if json_tsv_overlap < consistency_threshold:
                self.validation_results['warnings'].append(
                    f"{check_name}: Low JSON-TSV sample overlap: {json_tsv_overlap:.2%}"
                )
            
            if tsv_expr_overlap < consistency_threshold:
                self.validation_results['warnings'].append(
                    f"{check_name}: Low TSV-Expression sample overlap: {tsv_expr_overlap:.2%}"
                )
            
            if json_expr_overlap < consistency_threshold:
                self.validation_results['warnings'].append(
                    f"{check_name}: Low JSON-Expression sample overlap: {json_expr_overlap:.2%}"
                )
            
            # Check for missing critical samples
            missing_samples_count = len(all_samples - json_samples - tsv_samples - expression_samples)
            missing_percentage = missing_samples_count / len(all_samples) if all_samples else 0
            
            if missing_percentage > self.quality_thresholds['max_missing_samples_percentage']:
                self.validation_results['warnings'].append(
                    f"{check_name}: High missing samples percentage: {missing_percentage:.2%}"
                )
            
            # Store consistency metrics
            self.validation_results['quality_metrics'].update({
                'consistency_json_tsv_overlap': json_tsv_overlap,
                'consistency_tsv_expression_overlap': tsv_expr_overlap,
                'consistency_json_expression_overlap': json_expr_overlap,
                'consistency_missing_samples_percentage': missing_percentage
            })
            
            self.validation_results['checks_performed'].append(check_name)
            
        except Exception as e:
            self.validation_results['errors'].append(f"{check_name}: Error during validation - {e}")

    def _validate_data_quality(self, extracted_data: Dict[str, Any]) -> None:
        """Perform comprehensive data quality assessment"""
        check_name = "Data Quality Assessment"
        
        try:
            # Gene symbol validation
            gene_symbols = extracted_data['expression_data'].index.tolist()
            invalid_genes = [gene for gene in gene_symbols[:1000] if not self._is_valid_gene_symbol(str(gene))]
            
            if len(invalid_genes) > len(gene_symbols) * 0.1:  # More than 10% invalid
                self.validation_results['warnings'].append(
                    f"{check_name}: High percentage of invalid gene symbols: {len(invalid_genes)}/{len(gene_symbols)}"
                )
            
            # Sample accession validation
            if 'refinebio_accession_code' in extracted_data['tsv_metadata'].columns:
                accessions = extracted_data['tsv_metadata']['refinebio_accession_code'].dropna().astype(str)
                invalid_accessions = [
                    acc for acc in accessions 
                    if not self._sample_accession_pattern.match(acc)
                ]
                
                if invalid_accessions:
                    self.validation_results['warnings'].append(
                        f"{check_name}: Invalid sample accessions: {invalid_accessions[:10]}..."
                    )
            
            # Cross-reference validation
            self._validate_cross_references(extracted_data)
            
            # Business rule validation
            self._validate_business_rules(extracted_data)
            
            self.validation_results['checks_performed'].append(check_name)
            
        except Exception as e:
            self.validation_results['errors'].append(f"{check_name}: Error during validation - {e}")

    def _validate_cross_references(self, extracted_data: Dict[str, Any]) -> None:
        """Validate cross-references between data sources"""
        try:
            # Validate experiment accession consistency
            json_experiment = extracted_data['json_metadata']['experiment'].get('accession_code', '')
            
            if 'experiment_accession_code' in extracted_data['tsv_metadata'].columns:
                tsv_experiments = set(extracted_data['tsv_metadata']['experiment_accession_code'].dropna().astype(str))
                
                if json_experiment and tsv_experiments and json_experiment not in tsv_experiments:
                    self.validation_results['warnings'].append(
                        "Cross-reference validation: Experiment accession mismatch between JSON and TSV"
                    )
            
            # Validate organism consistency
            json_organism = extracted_data['json_metadata']['experiment'].get('organism', '')
            
            if 'refinebio_organism' in extracted_data['tsv_metadata'].columns:
                tsv_organisms = set(extracted_data['tsv_metadata']['refinebio_organism'].dropna().astype(str))
                
                if json_organism and tsv_organisms and json_organism not in tsv_organisms:
                    self.validation_results['warnings'].append(
                        "Cross-reference validation: Organism mismatch between JSON and TSV"
                    )
            
        except Exception as e:
            self.validation_results['errors'].append(f"Cross-reference validation error: {e}")

    def _validate_business_rules(self, extracted_data: Dict[str, Any]) -> None:
        """Validate business rules and data integrity"""
        try:
            # Check for reasonable sample counts
            sample_count = len(extracted_data['expression_data'].columns)
            
            if sample_count > 1000:
                self.validation_results['warnings'].append(
                    f"Business rule validation: Unusually high sample count: {sample_count}"
                )
            
            # Check for reasonable gene counts
            gene_count = len(extracted_data['expression_data'])
            
            if gene_count > 50000:
                self.validation_results['warnings'].append(
                    f"Business rule validation: Unusually high gene count: {gene_count}"
                )
            
            # Validate technology field
            technology = extracted_data['json_metadata']['experiment'].get('technology', '')
            valid_technologies = ['RNA-SEQ', 'MICROARRAY', 'OTHER']
            
            if technology and technology not in valid_technologies:
                self.validation_results['warnings'].append(
                    f"Business rule validation: Invalid technology: {technology}"
                )
            
        except Exception as e:
            self.validation_results['errors'].append(f"Business rule validation error: {e}")

    def _is_valid_gene_symbol(self, gene_symbol: str) -> bool:
        """Validate gene symbol format"""
        return bool(self._gene_symbol_pattern.match(str(gene_symbol)))

    def _log_validation_summary(self, study_code: str) -> None:
        """Log validation summary"""
        passed = self.validation_results['validation_passed']
        warnings = len(self.validation_results['warnings'])
        errors = len(self.validation_results['errors'])
        checks = len(self.validation_results['checks_performed'])
        
        self.logger.info(
            f"Validation summary for {study_code}: "
            f"{'PASSED' if passed else 'FAILED'} - "
            f"{checks} checks performed, {warnings} warnings, {errors} errors"
        )
        
        if self.validation_results['warnings']:
            self.logger.warning(f"Validation warnings: {self.validation_results['warnings']}")
        
        if self.validation_results['errors']:
            self.logger.error(f"Validation errors: {self.validation_results['errors']}")

    def get_validation_report(self) -> str:
        """Generate detailed validation report"""
        report = []
        report.append("=" * 60)
        report.append("ETL DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Validation Timestamp: {self.validation_results['validation_timestamp']}")
        report.append(f"Overall Result: {'PASSED' if self.validation_results['validation_passed'] else 'FAILED'}")
        report.append("")
        
        report.append("CHECKS PERFORMED:")
        for check in self.validation_results['checks_performed']:
            report.append(f"  ✓ {check}")
        report.append("")
        
        if self.validation_results['warnings']:
            report.append("WARNINGS:")
            for warning in self.validation_results['warnings']:
                report.append(f"  ⚠ {warning}")
            report.append("")
        
        if self.validation_results['errors']:
            report.append("ERRORS:")
            for error in self.validation_results['errors']:
                report.append(f"  ✗ {error}")
            report.append("")
        
        if self.validation_results['quality_metrics']:
            report.append("QUALITY METRICS:")
            for metric, value in self.validation_results['quality_metrics'].items():
                report.append(f"  {metric}: {value}")
            report.append("")
        
        return "\n".join(report)

    def save_validation_report(self, file_path: str) -> None:
        """Save validation report to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.get_validation_report())
                
                # Also save as JSON for programmatic access
                json_file = file_path.replace('.txt', '.json')
                with open(json_file, 'w', encoding='utf-8') as json_f:
                    json.dump(self.validation_results, json_f, indent=2, default=str)
                    
            self.logger.info(f"Validation report saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")

    def get_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        if not self.validation_results['validation_passed']:
            return 0.0
        
        # Start with perfect score
        score = 100.0
        
        # Deduct for warnings
        score -= len(self.validation_results['warnings']) * 5
        
        # Deduct for quality issues
        metrics = self.validation_results['quality_metrics']
        
        # Null percentage penalty
        null_pct = metrics.get('tsv_null_percentage', 0)
        if null_pct > 0.05:  # More than 5%
            score -= 10
        
        # Duplicate percentage penalty
        dup_pct = metrics.get('tsv_duplicate_count', 0) / metrics.get('tsv_rows', 1)
        if dup_pct > 0.02:  # More than 2%
            score -= 5
        
        # Expression data quality penalty
        expr_null_pct = metrics.get('expression_null_percentage', 0)
        if expr_null_pct > 0.1:  # More than 10%
            score -= 15
        
        # Consistency penalty
        consistency_scores = [
            metrics.get('consistency_json_tsv_overlap', 1.0),
            metrics.get('consistency_tsv_expression_overlap', 1.0),
            metrics.get('consistency_json_expression_overlap', 1.0)
        ]
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        if avg_consistency < 0.8:  # Less than 80%
            score -= 20
        
        return max(0.0, min(100.0, score))