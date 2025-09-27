"""
Production-Grade Data Extraction Module
Handles reading and parsing of source data files with comprehensive error handling
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Tuple
import logging
import hashlib
from datetime import datetime
import chardet


class DataExtractionError(Exception):
    """Custom exception for data extraction errors"""
    pass


class DataExtractor:
    """Extract data from various source formats with robust error handling and validation"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.extraction_stats = {
            'files_processed': 0,
            'records_extracted': 0,
            'warnings': [],
            'errors': []
        }

    def extract_all_sources(self, study_code: str) -> Dict[str, Any]:
        """
        Extract all data sources for a specific study code
        
        Args:
            study_code: The study code to process
            
        Returns:
            Dictionary containing all extracted data
        """
        self.logger.info(f"Starting data extraction for study: {study_code}")
        
        try:
            file_paths = self.config.get_study_file_paths(study_code)
            
            # Extract data from all sources
            json_data = self.extract_json_metadata(file_paths["json_metadata"], study_code)
            tsv_metadata = self.extract_tsv_metadata(file_paths["tsv_metadata"])
            expression_data = self.extract_expression_data(file_paths["expression_data"])
            
            # Validate data consistency
            self._validate_data_consistency(json_data, tsv_metadata, expression_data, study_code)
            
            result = {
                'study_code': study_code,
                'json_metadata': json_data,
                'tsv_metadata': tsv_metadata,
                'expression_data': expression_data,
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_stats': self.extraction_stats.copy()
            }
            
            self.logger.info(f"Data extraction completed for {study_code}")
            return result
            
        except Exception as e:
            self.logger.error(f"Data extraction failed for {study_code}: {str(e)}")
            raise DataExtractionError(f"Failed to extract data for {study_code}: {str(e)}")

    def extract_json_metadata(self, json_path: Path, study_code: str) -> Dict[str, Any]:
        """
        Extract experiment and sample metadata from JSON file
        
        Args:
            json_path: Path to JSON metadata file
            study_code: Study code to extract data for
            
        Returns:
            Dictionary containing experiment and samples data
        """
        try:
            self.logger.info(f"Extracting JSON metadata from: {json_path}")
            
            # Detect encoding
            encoding = self._detect_encoding(json_path)
            
            with open(json_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            
            # Extract specific study data
            if 'experiments' not in data or study_code not in data['experiments']:
                raise DataExtractionError(f"Study code {study_code} not found in JSON metadata")
            
            experiment_data = data['experiments'][study_code]
            samples_data = data.get('samples', {})
            
            # Filter samples for this specific study
            study_samples = {
                k: v for k, v in samples_data.items() 
                if v.get('experiment_accession_code') == study_code
            }
            
            self.extraction_stats['records_extracted'] += len(study_samples)
            
            # Validate required fields
            self._validate_json_structure(experiment_data, study_samples)
            
            result = {
                'experiment': experiment_data,
                'samples': study_samples,
                'metadata': {
                    'num_experiments': 1,
                    'num_samples': len(study_samples),
                    'created_at': data.get('created_at'),
                    'quantile_normalized': data.get('quantile_normalized', False),
                    'file_hash': self._calculate_file_hash(json_path)
                }
            }
            
            self.logger.info(f"Extracted metadata for {len(study_samples)} samples")
            return result
            
        except FileNotFoundError:
            raise DataExtractionError(f"JSON metadata file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise DataExtractionError(f"Invalid JSON format in {json_path}: {str(e)}")
        except Exception as e:
            raise DataExtractionError(f"Failed to extract JSON metadata: {str(e)}")

    def extract_tsv_metadata(self, tsv_path: Path) -> pd.DataFrame:
        """
        Extract sample metadata from TSV file with comprehensive validation
        
        Args:
            tsv_path: Path to TSV metadata file
            
        Returns:
            DataFrame containing sample metadata
        """
        try:
            self.logger.info(f"Extracting TSV metadata from: {tsv_path}")
            
            # Detect encoding
            encoding = self._detect_encoding(tsv_path)
            
            # Read TSV with robust error handling
            df = pd.read_csv(
                tsv_path,
                sep='\t',
                dtype=str,  # Read all as string initially
                encoding=encoding,
                na_values=['', 'NA', 'NULL', 'null', 'NaN', 'nan', 'N/A'],
                keep_default_na=True,
                quoting=3,  # csv.QUOTE_NONE
                error_bad_lines=False,
                warn_bad_lines=True
            )
            
            self.extraction_stats['records_extracted'] += len(df)
            
            # Log basic information
            self.logger.info(f"Loaded TSV with {len(df)} rows and {len(df.columns)} columns")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Data quality assessment
            self._assess_tsv_quality(df, tsv_path)
            
            # Convert data types appropriately
            df = self._convert_tsv_data_types(df)
            
            # Add file metadata
            df.attrs['file_path'] = str(tsv_path)
            df.attrs['file_hash'] = self._calculate_file_hash(tsv_path)
            df.attrs['extraction_timestamp'] = datetime.now().isoformat()
            
            return df
            
        except FileNotFoundError:
            raise DataExtractionError(f"TSV metadata file not found: {tsv_path}")
        except pd.errors.EmptyDataError:
            raise DataExtractionError(f"TSV file is empty: {tsv_path}")
        except Exception as e:
            raise DataExtractionError(f"Failed to extract TSV metadata: {str(e)}")

    def extract_expression_data(self, expression_path: Path) -> pd.DataFrame:
        """
        Extract gene expression matrix data with memory optimization
        
        Args:
            expression_path: Path to expression data TSV file
            
        Returns:
            DataFrame containing expression matrix (genes × samples)
        """
        try:
            self.logger.info(f"Extracting expression data from: {expression_path}")
            
            # Check file size for memory optimization
            file_size_mb = expression_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Expression data file size: {file_size_mb:.2f} MB")
            
            # Detect encoding
            encoding = self._detect_encoding(expression_path)
            
            # Read expression data with optimized parameters
            if file_size_mb > 500:  # Large files
                self.logger.info("Using memory-efficient reading for large file")
                df = self._read_large_expression_file(expression_path, encoding)
            else:
                df = pd.read_csv(
                    expression_path,
                    sep='\t',
                    index_col=0,  # First column is gene names
                    encoding=encoding,
                    na_values=['', 'NA', 'NULL', 'null', 'NaN', 'nan'],
                    keep_default_na=True
                )
            
            self.extraction_stats['records_extracted'] += len(df)
            
            # Log expression data statistics
            self.logger.info(f"Loaded expression matrix: {len(df)} genes × {len(df.columns)} samples")
            self.logger.info(f"Memory usage: ~{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            
            # Validate expression data structure
            self._validate_expression_data(df, expression_path)
            
            # Add file metadata
            df.attrs['file_path'] = str(expression_path)
            df.attrs['file_hash'] = self._calculate_file_hash(expression_path)
            df.attrs['extraction_timestamp'] = datetime.now().isoformat()
            df.attrs['matrix_shape'] = (len(df), len(df.columns))
            
            return df
            
        except FileNotFoundError:
            raise DataExtractionError(f"Expression data file not found: {expression_path}")
        except pd.errors.EmptyDataError:
            raise DataExtractionError(f"Expression data file is empty: {expression_path}")
        except Exception as e:
            raise DataExtractionError(f"Failed to extract expression data: {str(e)}")

    def _read_large_expression_file(self, file_path: Path, encoding: str) -> pd.DataFrame:
        """Read large expression files in chunks for memory efficiency"""
        chunks = []
        
        for chunk in pd.read_csv(
            file_path,
            sep='\t',
            index_col=0,
            encoding=encoding,
            chunksize=self.config.chunk_size
        ):
            chunks.append(chunk)
            
            # Log progress for very large files
            if len(chunks) % 10 == 0:
                self.logger.info(f"Read {len(chunks) * self.config.chunk_size} rows...")
        
        return pd.concat(chunks, ignore_index=False)

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                self.logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding if confidence > 0.7 else 'utf-8'
        except Exception as e:
            self.logger.warning(f"Could not detect encoding, using utf-8: {e}")
            return 'utf-8'

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for change detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not calculate file hash: {e}")
            return "unknown"

    def _validate_json_structure(self, experiment_data: Dict, samples_data: Dict) -> None:
        """Validate JSON structure and required fields"""
        required_experiment_fields = ['accession_code', 'title', 'organism']
        required_sample_fields = ['accession_code', 'organism', 'platform_name']
        
        # Validate experiment data
        missing_fields = [field for field in required_experiment_fields if field not in experiment_data]
        if missing_fields:
            self.extraction_stats['warnings'].append(f"Missing experiment fields: {missing_fields}")
        
        # Validate samples data
        for sample_id, sample_data in samples_data.items():
            missing_fields = [field for field in required_sample_fields if field not in sample_data]
            if missing_fields:
                self.extraction_stats['warnings'].append(f"Sample {sample_id} missing fields: {missing_fields}")

    def _assess_tsv_quality(self, df: pd.DataFrame, file_path: Path) -> None:
        """Assess TSV data quality and log warnings"""
        # Check for null values
        null_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        null_percentage = null_counts.sum() / total_cells
        
        if null_percentage > self.config.max_null_percentage:
            self.extraction_stats['warnings'].append(
                f"High null percentage in TSV: {null_percentage:.2%}"
            )
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_percentage = duplicate_rows / len(df)
            if duplicate_percentage > self.config.max_duplicate_percentage:
                self.extraction_stats['warnings'].append(
                    f"High duplicate percentage in TSV: {duplicate_percentage:.2%}"
                )
        
        # Check for empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            self.extraction_stats['warnings'].append(f"Empty columns found: {empty_columns}")
        
        self.logger.info(f"TSV quality assessment - Nulls: {null_percentage:.2%}, Duplicates: {duplicate_rows}")

    def _convert_tsv_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert TSV data types appropriately"""
        # Try to convert numeric columns
        numeric_columns = []
        for col in df.columns:
            try:
                # Try to convert to numeric, but keep as object if it fails
                pd.to_numeric(df[col], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                continue
        
        # Convert numeric columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        bool_columns = [col for col in df.columns if 'processed' in col.lower() or 'public' in col.lower()]
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({'True': True, 'False': False, '1': True, '0': False})
        
        return df

    def _validate_expression_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """Validate expression data structure and quality"""
        # Check for minimum genes
        if len(df) < self.config.min_genes_per_sample:
            self.extraction_stats['warnings'].append(
                f"Low gene count in expression data: {len(df)}"
            )
        
        # Check for null values in expression data
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_percentage > 0.1:  # 10% threshold for expression data
            self.extraction_stats['warnings'].append(
                f"High null percentage in expression data: {null_percentage:.2%}"
            )
        
        # Check for negative values (shouldn't exist in gene expression)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            negative_count = (numeric_df < 0).sum().sum()
            if negative_count > 0:
                self.extraction_stats['warnings'].append(
                    f"Found {negative_count} negative values in expression data"
                )

    def _validate_data_consistency(self, json_data: Dict, tsv_metadata: pd.DataFrame, 
                                 expression_data: pd.DataFrame, study_code: str) -> None:
        """Validate consistency across all data sources"""
        # Get sample accessions from different sources
        json_samples = set(json_data['samples'].keys())
        tsv_samples = set(tsv_metadata.get('refinebio_accession_code', []))
        expression_samples = set(expression_data.columns)
        
        # Check for mismatches
        all_samples = json_samples.union(tsv_samples).union(expression_samples)
        
        if len(all_samples) == 0:
            self.extraction_stats['errors'].append("No samples found in any data source")
            return
        
        # Calculate consistency metrics
        json_tsv_overlap = len(json_samples.intersection(tsv_samples)) / len(all_samples)
        tsv_expr_overlap = len(tsv_samples.intersection(expression_samples)) / len(all_samples)
        json_expr_overlap = len(json_samples.intersection(expression_samples)) / len(all_samples)
        
        consistency_threshold = 0.8  # 80% overlap required
        
        if json_tsv_overlap < consistency_threshold:
            self.extraction_stats['warnings'].append(
                f"Low JSON-TSV sample overlap: {json_tsv_overlap:.2%}"
            )
        
        if tsv_expr_overlap < consistency_threshold:
            self.extraction_stats['warnings'].append(
                f"Low TSV-Expression sample overlap: {tsv_expr_overlap:.2%}"
            )
        
        if json_expr_overlap < consistency_threshold:
            self.extraction_stats['warnings'].append(
                f"Low JSON-Expression sample overlap: {json_expr_overlap:.2%}"
            )
        
        self.logger.info(
            f"Data consistency - JSON-TSV: {json_tsv_overlap:.2%}, "
            f"TSV-Expr: {tsv_expr_overlap:.2%}, JSON-Expr: {json_expr_overlap:.2%}"
        )

    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extraction statistics"""
        return {
            'files_processed': self.extraction_stats['files_processed'],
            'total_records': self.extraction_stats['records_extracted'],
            'warnings_count': len(self.extraction_stats['warnings']),
            'errors_count': len(self.extraction_stats['errors']),
            'warnings': self.extraction_stats['warnings'],
            'errors': self.extraction_stats['errors']
        }