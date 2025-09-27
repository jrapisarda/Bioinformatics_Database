"""
Production-Grade Data Transformation Module
Handles transformation of gene expression matrix to long format with comprehensive validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Iterator
import logging
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


class DataTransformationError(Exception):
    """Custom exception for data transformation errors"""
    pass


class DataTransformer:
    """Transform extracted data into warehouse-ready format with performance optimization"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.transformation_stats = {
            'records_transformed': 0,
            'genes_processed': 0,
            'samples_processed': 0,
            'warnings': [],
            'errors': []
        }
        
        # Gene symbol validation patterns
        self._gene_symbol_pattern = re.compile(r'^[A-Za-z0-9_-]+$')
        self._valid_gene_types = {'protein_coding', 'lncRNA', 'miRNA', 'rRNA', 'tRNA', 'pseudogene'}

    def transform_all_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform all extracted data into warehouse-ready format
        
        Args:
            extracted_data: Dictionary containing all extracted data
            
        Returns:
            Dictionary containing transformed data ready for loading
        """
        study_code = extracted_data['study_code']
        self.logger.info(f"Starting data transformation for study: {study_code}")
        
        try:
            json_metadata = extracted_data.get('json_metadata', {})

            experiment_data = json_metadata.get('experiment')
            if not isinstance(experiment_data, dict):
                experiments_map = json_metadata.get('experiments', {})
                experiment_data = experiments_map.get(study_code, {}) if isinstance(experiments_map, dict) else {}

            def _is_sample_mapping(candidate: Any) -> bool:
                return isinstance(candidate, dict) and candidate and all(isinstance(v, dict) for v in candidate.values())

            raw_samples = json_metadata.get('samples')
            if _is_sample_mapping(raw_samples):
                samples_data = raw_samples
            elif isinstance(raw_samples, dict) and study_code in raw_samples and _is_sample_mapping(raw_samples[study_code]):
                samples_data = raw_samples[study_code]
            else:
                experiment_samples = experiment_data.get('samples') if isinstance(experiment_data, dict) else {}
                samples_data = experiment_samples if _is_sample_mapping(experiment_samples) else {}

            # Transform dimension data
            dim_study = self._transform_dim_study(experiment_data)
            dim_platform = self._transform_dim_platform(samples_data)
            dim_illness = self._transform_dim_illness(extracted_data['tsv_metadata'])
            dim_samples = self._transform_dim_samples(extracted_data['tsv_metadata'], samples_data)
            dim_genes = self._transform_dim_genes(extracted_data['expression_data'])
            
            # Transform fact data (gene expression)
            fact_expression = self._transform_fact_expression(
                extracted_data['expression_data'],
                dim_samples,
                dim_genes
            )
            
            # Validate transformed data
            self._validate_transformed_data(
                dim_study, dim_platform, dim_illness, dim_samples, dim_genes, fact_expression
            )
            
            result = {
                'study_code': study_code,
                'dimensions': {
                    'dim_study': dim_study,
                    'dim_platform': dim_platform,
                    'dim_illness': dim_illness,
                    'dim_samples': dim_samples,
                    'dim_genes': dim_genes
                },
                'facts': {
                    'fact_gene_expression': fact_expression
                },
                'transformation_timestamp': datetime.now().isoformat(),
                'transformation_stats': self.transformation_stats.copy()
            }
            
            self.logger.info(f"Data transformation completed for {study_code}")
            return result
            
        except Exception as e:
            self.logger.error(f"Data transformation failed for {study_code}: {str(e)}")
            raise DataTransformationError(f"Failed to transform data for {study_code}: {str(e)}")

    def _transform_dim_study(self, experiment_data: Dict[str, Any]) -> pd.DataFrame:
        """Transform experiment data into study dimension format"""
        self.logger.info("Transforming study dimension data")
        
        try:
            # Extract study attributes
            study_record = {
                'study_key': 1,  # Will be replaced by identity column
                'accession_code': experiment_data.get('accession_code', ''),
                'title': experiment_data.get('title', '')[:500],  # Truncate to fit column
                'description': experiment_data.get('description', '')[:4000],  # Limit description
                'technology': self._standardize_technology(experiment_data.get('technology', 'RNA-SEQ')),
                'organism': experiment_data.get('organism', ''),
                'has_publication': bool(experiment_data.get('has_publication', False)),
                'pubmed_id': experiment_data.get('pubmed_id', ''),
                'publication_title': experiment_data.get('publication_title', '')[:500],
                'source_first_published': self._parse_date(experiment_data.get('source_first_published')),
                'source_last_modified': self._parse_date(experiment_data.get('source_last_modified')),
                'created_date': datetime.now(),
                'updated_date': datetime.now()
            }
            
            # Create DataFrame
            df = pd.DataFrame([study_record])
            
            # Validate required fields
            if not study_record['accession_code']:
                self.transformation_stats['errors'].append("Missing accession_code in study data")
            
            self.transformation_stats['records_transformed'] += 1
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to transform study dimension: {e}")
            raise DataTransformationError(f"Study dimension transformation failed: {e}")

    def _transform_dim_platform(self, samples_data: Dict[str, Any]) -> pd.DataFrame:
        """Transform platform data from samples"""
        self.logger.info("Transforming platform dimension data")
        
        try:
            platforms = {}
            
            for sample_id, sample_data in samples_data.items():
                platform_name = sample_data.get('platform_name', 'Unknown')
                
                if platform_name not in platforms:
                    platforms[platform_name] = {
                        'platform_key': len(platforms) + 1,
                        'platform_name': platform_name,
                        'platform_type': self._infer_platform_type(platform_name),
                        'processor_name': sample_data.get('processor_name', ''),
                        'processor_version': sample_data.get('processor_version', ''),
                        'processor_id': sample_data.get('processor_id'),
                        'created_date': datetime.now(),
                        'updated_date': datetime.now()
                    }
            
            # Create DataFrame
            if platforms:
                df = pd.DataFrame(list(platforms.values()))
            else:
                # Default platform record
                df = pd.DataFrame([{
                    'platform_key': 1,
                    'platform_name': 'Unknown',
                    'platform_type': 'OTHER',
                    'processor_name': '',
                    'processor_version': '',
                    'processor_id': None,
                    'created_date': datetime.now(),
                    'updated_date': datetime.now()
                }])
            
            self.transformation_stats['records_transformed'] += len(df)
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to transform platform dimension: {e}")
            raise DataTransformationError(f"Platform dimension transformation failed: {e}")

    def _transform_dim_illness(self, tsv_metadata: pd.DataFrame) -> pd.DataFrame:
        """Transform illness dimension from sample titles"""
        self.logger.info("Transforming illness dimension data")
        
        try:
            illness_types = set()
            
            # Extract illness types from sample titles
            if 'refinebio_title' in tsv_metadata.columns:
                for title in tsv_metadata['refinebio_title'].dropna():
                    illness_type = self._classify_illness_type(str(title))
                    illness_types.add(illness_type)
            
            # Create illness records
            illness_records = []
            for idx, illness_type in enumerate(sorted(illness_types), 1):
                illness_records.append({
                    'illness_key': idx,
                    'illness_type': illness_type,
                    'illness_description': self._get_illness_description(illness_type),
                    'created_date': datetime.now(),
                    'updated_date': datetime.now()
                })
            
            # Ensure we have at least control and sepsis
            if not illness_records:
                illness_records = [
                    {
                        'illness_key': 1,
                        'illness_type': 'control',
                        'illness_description': 'Control sample',
                        'created_date': datetime.now(),
                        'updated_date': datetime.now()
                    },
                    {
                        'illness_key': 2,
                        'illness_type': 'sepsis',
                        'illness_description': 'Sepsis sample',
                        'created_date': datetime.now(),
                        'updated_date': datetime.now()
                    }
                ]
            
            df = pd.DataFrame(illness_records)
            self.transformation_stats['records_transformed'] += len(df)
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to transform illness dimension: {e}")
            raise DataTransformationError(f"Illness dimension transformation failed: {e}")

    def _transform_dim_samples(self, tsv_metadata: pd.DataFrame, json_samples: Dict[str, Any]) -> pd.DataFrame:
        """Transform sample dimension data"""
        self.logger.info("Transforming sample dimension data")
        
        try:
            samples_records = []
            
            # Process TSV metadata
            for idx, row in tsv_metadata.iterrows():
                accession_code = row.get('refinebio_accession_code', '')
                
                if not accession_code:
                    continue
                
                # Get additional data from JSON if available
                json_sample = json_samples.get(accession_code, {})
                
                # Classify illness type
                title = str(row.get('refinebio_title', ''))
                illness_classification = self._classify_illness_type(title)
                sample_number = self._extract_sample_number(title)
                
                sample_record = {
                    'sample_key': len(samples_records) + 1,
                    'refinebio_accession_code': accession_code,
                    'experiment_accession': row.get('experiment_accession_code', ''),
                    'refinebio_title': title[:200],  # Truncate to fit column
                    'refinebio_organism': row.get('refinebio_organism', json_sample.get('organism', '')),
                    'refinebio_processed': self._parse_boolean(row.get('refinebio_processed', False)),
                    'refinebio_source_database': row.get('refinebio_source_database', ''),
                    'sample_classification': illness_classification,
                    'sample_number': sample_number,
                    'created_date': datetime.now(),
                    'updated_date': datetime.now()
                }
                
                samples_records.append(sample_record)
            
            if not samples_records:
                raise DataTransformationError("No valid sample records found")
            
            df = pd.DataFrame(samples_records)
            self.transformation_stats['samples_processed'] = len(df)
            self.transformation_stats['records_transformed'] += len(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to transform samples dimension: {e}")
            raise DataTransformationError(f"Samples dimension transformation failed: {e}")

    def _transform_dim_genes(self, expression_data: pd.DataFrame) -> pd.DataFrame:
        """Transform gene dimension from expression matrix"""
        self.logger.info("Transforming gene dimension data")
        
        try:
            gene_records = []
            
            # Get gene symbols from index
            gene_symbols = expression_data.index.tolist()
            
            for idx, gene_symbol in enumerate(gene_symbols, 1):
                # Validate gene symbol
                if not self._is_valid_gene_symbol(gene_symbol):
                    self.transformation_stats['warnings'].append(f"Invalid gene symbol: {gene_symbol}")
                
                gene_record = {
                    'gene_key': idx,
                    'gene_symbol': gene_symbol,
                    'gene_description': self._get_gene_description(gene_symbol),
                    'gene_type': self._infer_gene_type(gene_symbol),
                    'chromosome': None,  # Would need additional annotation file
                    'gene_length': None,  # Would need additional annotation file
                    'strand': None,  # Would need additional annotation file
                    'created_date': datetime.now(),
                    'updated_date': datetime.now()
                }
                
                gene_records.append(gene_record)
            
            if not gene_records:
                raise DataTransformationError("No gene records found in expression data")
            
            df = pd.DataFrame(gene_records)
            self.transformation_stats['genes_processed'] = len(df)
            self.transformation_stats['records_transformed'] += len(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to transform genes dimension: {e}")
            raise DataTransformationError(f"Genes dimension transformation failed: {e}")

    def _transform_fact_expression(self, expression_data: pd.DataFrame, dim_samples: pd.DataFrame, 
                                 dim_genes: pd.DataFrame) -> pd.DataFrame:
        """Transform expression matrix to long format fact table"""
        self.logger.info("Transforming gene expression fact data")
        
        try:
            # Create mapping dictionaries
            sample_key_map = dict(zip(dim_samples['refinebio_accession_code'], dim_samples['sample_key']))
            gene_key_map = dict(zip(dim_genes['gene_symbol'], dim_genes['gene_key']))
            
            # Convert matrix to long format
            self.logger.info("Converting expression matrix to long format...")
            
            # Reset index to make gene_symbol a column
            expression_long = expression_data.reset_index()
            expression_long = expression_long.rename(
                columns={expression_long.columns[0]: 'gene_symbol'}
            )
            
            # Melt the dataframe to long format
            expression_long = pd.melt(
                expression_long,
                id_vars=['gene_symbol'],
                var_name='sample_accession',
                value_name='expression_value'
            )
            
            # Remove null values
            initial_count = len(expression_long)
            expression_long = expression_long.dropna(subset=['expression_value'])
            removed_count = initial_count - len(expression_long)
            
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} null expression values")
            
            # Map keys
            expression_long['sample_key'] = expression_long['sample_accession'].map(sample_key_map)
            expression_long['gene_key'] = expression_long['gene_symbol'].map(gene_key_map)
            
            # Remove records where keys couldn't be mapped
            unmapped_samples = expression_long['sample_key'].isnull().sum()
            unmapped_genes = expression_long['gene_key'].isnull().sum()
            
            if unmapped_samples > 0:
                self.transformation_stats['warnings'].append(f"Unmapped samples: {unmapped_samples}")
            
            if unmapped_genes > 0:
                self.transformation_stats['warnings'].append(f"Unmapped genes: {unmapped_genes}")
            
            # Filter out unmapped records
            expression_long = expression_long.dropna(subset=['sample_key', 'gene_key'])
            
            # Convert keys to integers
            expression_long['sample_key'] = expression_long['sample_key'].astype(int)
            expression_long['gene_key'] = expression_long['gene_key'].astype(int)
            
            # Ensure expression values are numeric
            expression_long['expression_value'] = pd.to_numeric(
                expression_long['expression_value'], errors='coerce'
            )
            
            # Remove any remaining null values
            expression_long = expression_long.dropna()
            
            # Add fact table columns
            expression_long['fact_key'] = range(1, len(expression_long) + 1)
            expression_long['etl_batch_id'] = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            expression_long['etl_processed_at'] = datetime.now()
            
            # Reorder columns
            expression_long = expression_long[[
                'fact_key', 'sample_key', 'gene_key', 'expression_value',
                'etl_batch_id', 'etl_processed_at'
            ]]
            
            self.transformation_stats['records_transformed'] += len(expression_long)
            
            self.logger.info(f"Transformed {len(expression_long)} expression records")
            return expression_long
            
        except Exception as e:
            self.logger.error(f"Failed to transform expression fact data: {e}")
            raise DataTransformationError(f"Expression fact transformation failed: {e}")

    def _validate_transformed_data(self, dim_study: pd.DataFrame, dim_platform: pd.DataFrame,
                                 dim_illness: pd.DataFrame, dim_samples: pd.DataFrame,
                                 dim_genes: pd.DataFrame, fact_expression: pd.DataFrame) -> None:
        """Validate all transformed data for consistency and quality"""
        self.logger.info("Validating transformed data")
        
        try:
            # Check for required data
            if dim_study.empty:
                self.transformation_stats['errors'].append("Study dimension is empty")
            
            if dim_samples.empty:
                self.transformation_stats['errors'].append("Samples dimension is empty")
            
            if dim_genes.empty:
                self.transformation_stats['errors'].append("Genes dimension is empty")
            
            if fact_expression.empty:
                self.transformation_stats['errors'].append("Expression fact table is empty")
            
            # Validate referential integrity
            expression_sample_keys = set(fact_expression['sample_key'].unique())
            sample_keys = set(dim_samples['sample_key'].unique())
            
            missing_samples = expression_sample_keys - sample_keys
            if missing_samples:
                self.transformation_stats['errors'].append(f"Missing samples in dimension: {missing_samples}")
            
            expression_gene_keys = set(fact_expression['gene_key'].unique())
            gene_keys = set(dim_genes['gene_key'].unique())
            
            missing_genes = expression_gene_keys - gene_keys
            if missing_genes:
                self.transformation_stats['errors'].append(f"Missing genes in dimension: {missing_genes}")
            
            # Check for data quality issues
            if len(self.transformation_stats['errors']) > 0:
                raise DataTransformationError(f"Data validation failed: {self.transformation_stats['errors']}")
            
            self.logger.info("Data validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise

    # Helper methods
    def _standardize_technology(self, technology: str) -> str:
        """Standardize technology names"""
        tech_upper = str(technology).upper()
        if 'RNA-SEQ' in tech_upper or 'RNA SEQ' in tech_upper:
            return 'RNA-SEQ'
        elif 'MICROARRAY' in tech_upper or 'ARRAY' in tech_upper:
            return 'MICROARRAY'
        else:
            return 'OTHER'

    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if not date_value:
            return None
        
        try:
            if isinstance(date_value, datetime):
                return date_value
            
            # Try common date formats
            date_str = str(date_value)
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return None
        except Exception:
            return None

    def _parse_boolean(self, value: Any) -> bool:
        """Parse boolean from various formats"""
        if isinstance(value, bool):
            return value
        
        str_value = str(value).lower()
        return str_value in ['true', '1', 'yes', 'y', 'on']

    def _infer_platform_type(self, platform_name: str) -> str:
        """Infer platform type from name"""
        name_upper = str(platform_name).upper()
        if 'ILLUMINA' in name_upper:
            return 'ILLUMINA'
        elif 'AFFYMETRIX' in name_upper or 'AFFY' in name_upper:
            return 'AFFYMETRIX'
        elif 'AGILENT' in name_upper:
            return 'AGILENT'
        else:
            return 'OTHER'

    def _classify_illness_type(self, title: str) -> str:
        """Classify illness type from sample title"""
        title_lower = str(title).lower()
        
        if any(word in title_lower for word in ['control', 'ctrl', 'healthy', 'normal']):
            return 'control'
        elif any(word in title_lower for word in ['sepsis', 'infection', 'infected']):
            return 'sepsis'
        else:
            return 'unknown'

    def _get_illness_description(self, illness_type: str) -> str:
        """Get description for illness type"""
        descriptions = {
            'control': 'Control/Healthy sample',
            'sepsis': 'Sepsis/Infected sample',
            'unknown': 'Unknown/Unclassified sample'
        }
        return descriptions.get(illness_type, 'Unknown sample type')

    def _extract_sample_number(self, title: str) -> Optional[int]:
        """Extract sample number from title"""
        try:
            # Look for patterns like "Sample 1", "S1", "_1", etc.
            patterns = [
                r'Sample\s*(\d+)',
                r'S(\d+)',
                r'_(")\d+',
                r'(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, str(title), re.IGNORECASE)
                if match:
                    return int(match.group(1))
            
            return None
        except Exception:
            return None

    def _is_valid_gene_symbol(self, gene_symbol: str) -> bool:
        """Validate gene symbol format"""
        return bool(self._gene_symbol_pattern.match(str(gene_symbol)))

    def _get_gene_description(self, gene_symbol: str) -> str:
        """Get gene description (placeholder - would need annotation database)"""
        return f"Gene: {gene_symbol}"

    def _infer_gene_type(self, gene_symbol: str) -> str:
        """Infer gene type from symbol (placeholder - would need annotation database)"""
        # Simple heuristics for common patterns
        symbol_upper = str(gene_symbol).upper()
        
        if symbol_upper.startswith('MIR') or symbol_upper.startswith('LET'):
            return 'miRNA'
        elif 'LINC' in symbol_upper or symbol_upper.startswith('LNC'):
            return 'lncRNA'
        elif symbol_upper.startswith('RPL') or symbol_upper.startswith('RPS'):
            return 'rRNA'
        elif symbol_upper.startswith('TRNA'):
            return 'tRNA'
        elif symbol_upper.endswith('P'):
            return 'pseudogene'
        else:
            return 'protein_coding'

    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of transformation statistics"""
        return {
            'records_transformed': self.transformation_stats['records_transformed'],
            'genes_processed': self.transformation_stats['genes_processed'],
            'samples_processed': self.transformation_stats['samples_processed'],
            'warnings_count': len(self.transformation_stats['warnings']),
            'errors_count': len(self.transformation_stats['errors']),
            'warnings': self.transformation_stats['warnings'],
            'errors': self.transformation_stats['errors']
        }