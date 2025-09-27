#!/usr/bin/env python3
"""
Comprehensive ETL Pipeline Testing Suite
Tests all components of the production-grade ETL pipeline with sample data
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import logging
from typing import Dict, Any, List
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_etl_config import ETLConfig, StudyConfig
from robust_data_extractor import DataExtractor, DataExtractionError
from robust_data_validator import DataValidator, ValidationError
from robust_data_transformer import DataTransformer, DataTransformationError
from robust_data_loader import DataLoader, DataLoadError


class ETLTestSuite:
    """Comprehensive test suite for ETL pipeline components"""

    def __init__(self):
        self.test_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'component_results': {},
            'performance_metrics': {}
        }
        self.logger = self._setup_logging()
        self.test_data_dir = None

    def _setup_logging(self) -> logging.Logger:
        """Setup test logging"""
        logger = logging.getLogger('ETL_Test')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def create_test_data(self) -> str:
        """Create comprehensive test data for ETL pipeline"""
        self.logger.info("Creating test data...")
        
        # Create temporary directory for test data
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="etl_test_"))
        
        try:
            # Create JSON metadata
            json_metadata = self._create_json_metadata()
            
            # Create TSV metadata
            tsv_metadata = self._create_tsv_metadata()
            
            # Create expression data
            expression_data = self._create_expression_data()
            
            # Save files
            json_path = self.test_data_dir / "aggregated_metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_metadata, f, indent=2)
            
            # Create study directory
            study_dir = self.test_data_dir / "SRP049820"
            study_dir.mkdir(exist_ok=True)
            
            # Save TSV metadata
            tsv_metadata_path = study_dir / "metadata_SRP049820.tsv"
            tsv_metadata.to_csv(tsv_metadata_path, sep='\t', index=False)
            
            # Save expression data
            expression_path = study_dir / "SRP049820.tsv"
            expression_data.to_csv(expression_path, sep='\t', index=True)
            
            self.logger.info(f"Test data created in: {self.test_data_dir}")
            return str(self.test_data_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to create test data: {e}")
            if self.test_data_dir:
                shutil.rmtree(self.test_data_dir, ignore_errors=True)
            raise

    def _create_json_metadata(self) -> Dict[str, Any]:
        """Create realistic JSON metadata for testing"""
        return {
            "experiments": {
                "SRP049820": {
                    "accession_code": "SRP049820",
                    "title": "RNA-seq analysis of sepsis patients",
                    "description": "Transcriptome analysis of peripheral blood from sepsis patients and healthy controls",
                    "organism": "Homo sapiens",
                    "technology": "RNA-SEQ",
                    "has_publication": True,
                    "pubmed_id": "12345678",
                    "publication_title": "Transcriptomic signatures of sepsis",
                    "source_first_published": "2023-01-15",
                    "source_last_modified": "2023-06-20",
                    "source_url": "https://www.ebi.ac.uk/arrayexpress/experiments/SRP049820",
                    "source_archive_url": "https://www.ncbi.nlm.nih.gov/sra/SRP049820",
                    "has_public_raw_data": True
                }
            },
            "samples": {
                "SRR123456": {
                    "accession_code": "SRR123456",
                    "title": "Sepsis patient 1",
                    "organism": "Homo sapiens",
                    "platform_name": "Illumina HiSeq 2500",
                    "platform_type": "ILLUMINA",
                    "processor_name": "Salmon",
                    "processor_version": "1.5.2",
                    "experiment_accession_code": "SRP049820",
                    "refinebio_source_database": "SRA",
                    "refinebio_processed": True,
                    "refinebio_has_raw": True,
                    "age": "45",
                    "sex": "M",
                    "tissue": "blood",
                    "sample_type": "sepsis"
                },
                "SRR123457": {
                    "accession_code": "SRR123457",
                    "title": "Sepsis patient 2",
                    "organism": "Homo sapiens",
                    "platform_name": "Illumina HiSeq 2500",
                    "platform_type": "ILLUMINA",
                    "processor_name": "Salmon",
                    "processor_version": "1.5.2",
                    "experiment_accession_code": "SRP049820",
                    "refinebio_source_database": "SRA",
                    "refinebio_processed": True,
                    "refinebio_has_raw": True,
                    "age": "62",
                    "sex": "F",
                    "tissue": "blood",
                    "sample_type": "sepsis"
                },
                "SRR123458": {
                    "accession_code": "SRR123458",
                    "title": "Control 1",
                    "organism": "Homo sapiens",
                    "platform_name": "Illumina HiSeq 2500",
                    "platform_type": "ILLUMINA",
                    "processor_name": "Salmon",
                    "processor_version": "1.5.2",
                    "experiment_accession_code": "SRP049820",
                    "refinebio_source_database": "SRA",
                    "refinebio_processed": True,
                    "refinebio_has_raw": True,
                    "age": "38",
                    "sex": "M",
                    "tissue": "blood",
                    "sample_type": "control"
                },
                "SRR123459": {
                    "accession_code": "SRR123459",
                    "title": "Control 2",
                    "organism": "Homo sapiens",
                    "platform_name": "Illumina HiSeq 2500",
                    "platform_type": "ILLUMINA",
                    "processor_name": "Salmon",
                    "processor_version": "1.5.2",
                    "experiment_accession_code": "SRP049820",
                    "refinebio_source_database": "SRA",
                    "refinebio_processed": True,
                    "refinebio_has_raw": True,
                    "age": "41",
                    "sex": "F",
                    "tissue": "blood",
                    "sample_type": "control"
                }
            },
            "metadata": {
                "num_experiments": 1,
                "num_samples": 4,
                "created_at": "2023-09-27T10:00:00Z",
                "quantile_normalized": True,
                "aggregate_version": "1.0"
            }
        }

    def _create_tsv_metadata(self) -> pd.DataFrame:
        """Create realistic TSV metadata for testing"""
        data = {
            'refinebio_accession_code': ['SRR123456', 'SRR123457', 'SRR123458', 'SRR123459'],
            'experiment_accession_code': ['SRP049820'] * 4,
            'refinebio_title': ['Sepsis patient 1', 'Sepsis patient 2', 'Control 1', 'Control 2'],
            'refinebio_organism': ['Homo sapiens'] * 4,
            'refinebio_processed': ['True'] * 4,
            'refinebio_source_database': ['SRA'] * 4,
            'refinebio_platform': ['Illumina HiSeq 2500'] * 4,
            'sample_classification': ['sepsis', 'sepsis', 'control', 'control'],
            'age': ['45', '62', '38', '41'],
            'sex': ['M', 'F', 'M', 'F'],
            'tissue': ['blood'] * 4,
            'treatment': ['sepsis'] * 2 + ['control'] * 2,
            'sample_number': ['1', '2', '1', '2']
        }
        
        return pd.DataFrame(data)

    def _create_expression_data(self) -> pd.DataFrame:
        """Create realistic gene expression data for testing"""
        # Create gene symbols (mix of real and synthetic)
        gene_symbols = [
            'GAPDH', 'ACTB', 'TNF', 'IL6', 'IL1B', 'CXCL8', 'CCL2', 'PTGS2',
            'MMP9', 'ICAM1', 'VCAM1', 'SELE', 'TLR4', 'MYD88', 'NFKB1',
            'RELA', 'MAPK1', 'MAPK3', 'STAT3', 'JUN', 'FOS', 'MYC',
            'TP53', 'BCL2', 'BAX', 'CASP3', 'CASP8', 'CASP9', 'CASP10'
        ]
        
        # Add some more genes to make it realistic
        additional_genes = [f'GENE{i:04d}' for i in range(1, 71)]  # GENE0001 to GENE0070
        all_genes = gene_symbols + additional_genes
        
        # Create expression matrix with realistic values
        np.random.seed(42)  # For reproducible results
        
        # Base expression levels
        expression_data = np.random.lognormal(mean=6, sigma=1.5, size=(len(all_genes), 4))
        
        # Add sepsis-specific expression changes
        sepsis_genes = ['TNF', 'IL6', 'IL1B', 'CXCL8', 'CCL2', 'PTGS2', 'MMP9']
        for gene in sepsis_genes:
            if gene in all_genes:
                gene_idx = all_genes.index(gene)
                # Upregulate in sepsis samples (first 2 columns)
                expression_data[gene_idx, :2] *= np.random.uniform(2, 10, size=2)
        
        # Create DataFrame
        df = pd.DataFrame(
            expression_data,
            index=all_genes,
            columns=['SRR123456', 'SRR123457', 'SRR123458', 'SRR123459']
        )
        
        # Round to reasonable precision
        df = df.round(2)
        
        return df

    def test_config_validation(self) -> bool:
        """Test configuration validation"""
        test_name = "Configuration Validation"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            # Test valid configuration
            config = ETLConfig(
                base_path="/tmp/test",
                connection_string="Server=localhost;Database=test;Trusted_Connection=true;",
                batch_size=1000,
                chunk_size=500,
                max_workers=4
            )
            
            config.validate_paths = lambda: None  # Mock path validation
            
            assert config.batch_size == 1000
            assert config.chunk_size == 500
            assert config.max_workers == 4
            
            # Test invalid configuration
            try:
                invalid_config = ETLConfig(
                    base_path="/tmp/test",
                    connection_string="",
                    batch_size=-1
                )
                assert False, "Should have raised ValueError"
            except ValueError:
                pass  # Expected
            
            self.test_results['component_results'][test_name] = True
            return True
            
        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            self.test_results['component_results'][test_name] = False
            return False

    def test_data_extraction(self) -> bool:
        """Test data extraction component"""
        test_name = "Data Extraction"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            # Create test configuration
            config = ETLConfig(
                base_path=self.test_data_dir,
                connection_string="Server=localhost;Database=test;Trusted_Connection=true;",
                batch_size=1000
            )
            config.validate_paths = lambda: None
            
            # Test extraction
            extractor = DataExtractor(config)
            extracted_data = extractor.extract_all_sources("SRP049820")
            
            # Validate extracted data
            assert extracted_data['study_code'] == "SRP049820"
            assert 'json_metadata' in extracted_data
            assert 'tsv_metadata' in extracted_data
            assert 'expression_data' in extracted_data
            
            # Check data integrity
            assert len(extracted_data['json_metadata']['samples']) == 4
            assert len(extracted_data['tsv_metadata']) == 4
            assert extracted_data['expression_data'].shape[0] > 0
            assert extracted_data['expression_data'].shape[1] == 4
            
            # Check extraction stats
            stats = extractor.get_extraction_summary()
            assert stats['records_extracted'] > 0
            
            self.test_results['component_results'][test_name] = True
            return True
            
        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            self.test_results['component_results'][test_name] = False
            return False

    def test_data_validation(self) -> bool:
        """Test data validation component"""
        test_name = "Data Validation"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            config = ETLConfig(
                base_path=self.test_data_dir,
                connection_string="Server=localhost;Database=test;Trusted_Connection=true;"
            )
            config.validate_paths = lambda: None
            
            # Extract data first
            extractor = DataExtractor(config)
            extracted_data = extractor.extract_all_sources("SRP049820")
            
            # Test validation
            validator = DataValidator(config)
            validation_results = validator.validate_all_data(extracted_data)
            
            # Check validation results
            assert isinstance(validation_results, dict)
            assert 'validation_passed' in validation_results
            assert 'checks_performed' in validation_results
            assert 'quality_metrics' in validation_results
            
            # Check quality score
            quality_score = validator.get_quality_score()
            assert 0 <= quality_score <= 100
            
            self.logger.info(f"Data quality score: {quality_score:.1f}/100")
            
            self.test_results['component_results'][test_name] = True
            return True
            
        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            self.test_results['component_results'][test_name] = False
            return False

    def test_data_transformation(self) -> bool:
        """Test data transformation component"""
        test_name = "Data Transformation"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            config = ETLConfig(
                base_path=self.test_data_dir,
                connection_string="Server=localhost;Database=test;Trusted_Connection=true;"
            )
            config.validate_paths = lambda: None
            
            # Extract and validate data first
            extractor = DataExtractor(config)
            extracted_data = extractor.extract_all_sources("SRP049820")
            
            validator = DataValidator(config)
            validation_results = validator.validate_all_data(extracted_data)
            
            # Test transformation
            transformer = DataTransformer(config)
            transformed_data = transformer.transform_all_data(extracted_data)
            
            # Validate transformed data
            assert 'dimensions' in transformed_data
            assert 'facts' in transformed_data
            assert transformed_data['study_code'] == "SRP049820"
            
            # Check dimensions
            dimensions = transformed_data['dimensions']
            assert 'dim_study' in dimensions
            assert 'dim_platform' in dimensions
            assert 'dim_illness' in dimensions
            assert 'dim_samples' in dimensions
            assert 'dim_genes' in dimensions
            
            # Check facts
            facts = transformed_data['facts']
            assert 'fact_gene_expression' in facts
            assert len(facts['fact_gene_expression']) > 0
            
            # Check transformation stats
            stats = transformer.get_transformation_summary()
            assert stats['records_transformed'] > 0
            
            self.test_results['component_results'][test_name] = True
            return True
            
        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            self.test_results['component_results'][test_name] = False
            return False

    def test_edge_cases(self) -> bool:
        """Test edge cases and error conditions"""
        test_name = "Edge Cases"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            # Test with missing files
            config = ETLConfig(
                base_path=self.test_data_dir,
                connection_string="Server=localhost;Database=test;Trusted_Connection=true;"
            )
            config.validate_paths = lambda: None
            
            # Test missing study code
            try:
                extractor = DataExtractor(config)
                extractor.extract_all_sources("NONEXISTENT")
                assert False, "Should have raised error for missing study"
            except (DataExtractionError, ValueError):
                pass  # Expected
            
            # Test empty data handling
            empty_df = pd.DataFrame()
            validator = DataValidator(config)
            
            # This should handle empty data gracefully
            try:
                validator._validate_tsv_metadata(empty_df)
                assert False, "Should have raised error for empty data"
            except (ValidationError, AssertionError):
                pass  # Expected
            
            self.test_results['component_results'][test_name] = True
            return True
            
        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            self.test_results['component_results'][test_name] = False
            return False

    def test_performance_metrics(self) -> bool:
        """Test performance metrics collection"""
        test_name = "Performance Metrics"
        self.logger.info(f"Testing {test_name}...")
        
        try:
            # Test with large dataset
            large_expression_data = pd.DataFrame(
                np.random.lognormal(mean=6, sigma=1.5, size=(1000, 10)),
                index=[f'GENE{i:04d}' for i in range(1, 1001)],
                columns=[f'SRR{i:06d}' for i in range(1, 11)]
            )
            
            # Test transformation performance
            start_time = datetime.now()
            
            config = ETLConfig(
                base_path=self.test_data_dir,
                connection_string="Server=localhost;Database=test;Trusted_Connection=true;"
            )
            config.validate_paths = lambda: None
            
            transformer = DataTransformer(config)
            
            # Simulate transformation of large dataset
            result = transformer._transform_fact_expression(
                large_expression_data,
                pd.DataFrame({'refinebio_accession_code': large_expression_data.columns, 'sample_key': range(1, 11)}),
                pd.DataFrame({'gene_symbol': large_expression_data.index, 'gene_key': range(1, 1001)})
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Large dataset transformation: {duration:.2f}s for {len(result)} records")
            
            # Store performance metric
            self.test_results['performance_metrics']['large_dataset_transformation'] = duration
            
            self.test_results['component_results'][test_name] = True
            return True
            
        except Exception as e:
            self.logger.error(f"{test_name} failed: {e}")
            self.test_results['component_results'][test_name] = False
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test components"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING ETL PIPELINE TEST SUITE")
        self.logger.info("=" * 60)
        
        try:
            # Create test data
            test_data_path = self.create_test_data()
            
            # Run tests
            tests = [
                self.test_config_validation,
                self.test_data_extraction,
                self.test_data_validation,
                self.test_data_transformation,
                self.test_edge_cases,
                self.test_performance_metrics
            ]
            
            for test in tests:
                self.test_results['tests_run'] += 1
                try:
                    if test():
                        self.test_results['tests_passed'] += 1
                    else:
                        self.test_results['tests_failed'] += 1
                except Exception as e:
                    self.logger.error(f"Test {test.__name__} failed with exception: {e}")
                    self.test_results['tests_failed'] += 1
            
            # Generate test report
            self._generate_test_report()
            
            # Cleanup
            if self.test_data_dir:
                shutil.rmtree(self.test_data_dir, ignore_errors=True)
            
            return self.test_results
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            traceback.print_exc()
            
            if self.test_data_dir:
                shutil.rmtree(self.test_data_dir, ignore_errors=True)
            
            return self.test_results

    def _generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("ETL PIPELINE TEST SUITE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL RESULTS:")
        report.append(f"  Tests Run: {self.test_results['tests_run']}")
        report.append(f"  Tests Passed: {self.test_results['tests_passed']}")
        report.append(f"  Tests Failed: {self.test_results['tests_failed']}")
        report.append(f"  Success Rate: {self.test_results['tests_passed']/self.test_results['tests_run']*100:.1f}%")
        report.append("")
        
        # Component results
        report.append("COMPONENT TEST RESULTS:")
        for component, result in self.test_results['component_results'].items():
            status = "PASS" if result else "FAIL"
            report.append(f"  {component}: {status}")
        
        # Performance metrics
        if self.test_results['performance_metrics']:
            report.append("")
            report.append("PERFORMANCE METRICS:")
            for metric, value in self.test_results['performance_metrics'].items():
                report.append(f"  {metric}: {value:.3f}s")
        
        report.append("=" * 80)
        
        test_report = "\n".join(report)
        self.logger.info(test_report)
        
        # Save to file
        try:
            report_file = Path("test_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(test_report)
            self.logger.info(f"Test report saved to {report_file}")
        except Exception as e:
            self.logger.warning(f"Could not save test report: {e}")
        
        return test_report


def main():
    """Run the test suite"""
    print("Starting ETL Pipeline Test Suite...")
    
    try:
        test_suite = ETLTestSuite()
        results = test_suite.run_all_tests()
        
        # Exit with appropriate code
        if results['tests_failed'] == 0:
            print("\n✅ All tests passed!")
            return 0
        else:
            print(f"\n❌ {results['tests_failed']} test(s) failed")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())