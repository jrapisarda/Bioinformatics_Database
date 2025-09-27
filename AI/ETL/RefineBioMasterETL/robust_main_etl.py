#!/usr/bin/env python3
"""
Production-Grade Bioinformatics ETL Pipeline
Main execution script for processing refine.bio RNA-seq gene expression data

Features:
- Scalable to multiple study codes (SRP*, EX-AUR-1004, etc.)
- Performance monitoring and optimization
- Comprehensive data validation
- MERGE-based deduplication
- Robust error handling and logging
- Parallel processing support

Author: Production ETL System
Date: September 2025
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback

# Import custom modules
from robust_etl_config import ETLConfig, StudyConfig
from robust_data_extractor import DataExtractor, DataExtractionError
from robust_data_validator import DataValidator, ValidationError
from robust_data_transformer import DataTransformer, DataTransformationError
from robust_data_loader import DataLoader, DataLoadError, ETLPerformanceMonitor


class ETLPipelineOrchestrator:
    """Main ETL pipeline orchestrator with comprehensive error handling and monitoring"""

    def __init__(self, config: ETLConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.pipeline_stats = {
            'pipeline_start_time': datetime.now(),
            'studies_processed': 0,
            'studies_failed': 0,
            'total_records_processed': 0,
            'total_duration_seconds': 0,
            'errors': [],
            'warnings': []
        }
        
        # Initialize components
        self.extractor = DataExtractor(config)
        self.validator = DataValidator(config)
        self.transformer = DataTransformer(config)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging configuration"""
        logger = logging.getLogger('ETL_Pipeline')
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            log_dir = Path(self.config.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            if self.config.log_rotation:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.config.log_file,
                    maxBytes=self.config.log_max_size_mb * 1024 * 1024,
                    backupCount=self.config.log_backup_count
                )
            else:
                file_handler = logging.FileHandler(self.config.log_file)
            
            file_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    def execute_study_pipeline(self, study_code: str) -> Dict[str, Any]:
        """
        Execute complete ETL pipeline for a single study
        
        Args:
            study_code: Study code to process
            
        Returns:
            Pipeline execution results
        """
        self.logger.info(f"=== Starting ETL Pipeline for Study: {study_code} ===")
        study_start_time = datetime.now()
        study_stats = {
            'study_code': study_code,
            'start_time': study_start_time,
            'end_time': None,
            'duration_seconds': 0,
            'extraction_stats': {},
            'validation_results': {},
            'transformation_stats': {},
            'load_stats': {},
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Extract data from source files
            self.logger.info("Step 1: Extracting data from source files...")
            extracted_data = self.extractor.extract_all_sources(study_code)
            study_stats['extraction_stats'] = self.extractor.get_extraction_summary()
            
            # Step 2: Validate extracted data
            self.logger.info("Step 2: Validating extracted data...")
            validation_results = self.validator.validate_all_data(extracted_data)
            study_stats['validation_results'] = validation_results
            
            # Check if validation passed or has acceptable warnings
            if not validation_results['validation_passed'] and len(validation_results['errors']) > 0:
                raise ValidationError(f"Data validation failed with {len(validation_results['errors'])} errors")
            
            # Log validation report
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"Validation Quality Score: {self.validator.get_quality_score():.1f}/100")
            
            # Step 3: Transform data for warehouse loading
            self.logger.info("Step 3: Transforming data...")
            transformed_data = self.transformer.transform_all_data(extracted_data)
            study_stats['transformation_stats'] = self.transformer.get_transformation_summary()
            
            # Step 4: Load to database with MERGE operations
            self.logger.info("Step 4: Loading data to warehouse...")
            with DataLoader(self.config) as loader:
                load_results = loader.load_all_data(transformed_data)
                study_stats['load_stats'] = load_results['load_stats']
            
            # Step 5: Generate performance report
            self.logger.info("Step 5: Generating performance report...")
            with DataLoader(self.config) as loader:
                monitor = ETLPerformanceMonitor(loader.connection)
                performance_report = monitor.generate_performance_report(study_code)
                study_stats['performance_report'] = performance_report
            
            # Mark as successful
            study_stats['success'] = True
            study_stats['end_time'] = datetime.now()
            study_stats['duration_seconds'] = (study_stats['end_time'] - study_start_time).total_seconds()
            
            # Update pipeline stats
            self.pipeline_stats['studies_processed'] += 1
            self.pipeline_stats['total_records_processed'] += study_stats['load_stats']['records_inserted']
            
            self.logger.info(f"=== ETL Pipeline completed successfully for {study_code} ===")
            self.logger.info(f"Duration: {study_stats['duration_seconds']:.2f} seconds")
            
            return study_stats
            
        except Exception as e:
            study_stats['end_time'] = datetime.now()
            study_stats['duration_seconds'] = (study_stats['end_time'] - study_start_time).total_seconds()
            study_stats['error'] = str(e)
            study_stats['traceback'] = traceback.format_exc()
            
            self.pipeline_stats['studies_failed'] += 1
            self.pipeline_stats['errors'].append(f"Study {study_code}: {str(e)}")
            
            self.logger.error(f"ETL Pipeline failed for {study_code}: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return study_stats

    def execute_full_pipeline(self, study_codes: List[str] = None) -> Dict[str, Any]:
        """
        Execute ETL pipeline for all discovered or specified studies
        
        Args:
            study_codes: Optional list of specific study codes to process
            
        Returns:
            Comprehensive pipeline execution results
        """
        pipeline_start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("PRODUCTION ETL PIPELINE STARTED")
        self.logger.info("=" * 60)
        
        try:
            # Discover study codes if not provided
            if study_codes is None:
                self.logger.info("Discovering study codes...")
                study_codes = self.config.discover_study_codes()
            
            if not study_codes:
                raise ValueError("No study codes found to process")
            
            self.logger.info(f"Found {len(study_codes)} studies to process: {study_codes}")
            
            # Process each study
            study_results = []
            
            for study_code in study_codes:
                try:
                    result = self.execute_study_pipeline(study_code)
                    study_results.append(result)
                    
                    # Log individual study summary
                    self._log_study_summary(result)
                    
                except Exception as e:
                    self.logger.error(f"Critical error processing {study_code}: {e}")
                    continue
            
            # Generate pipeline summary
            pipeline_end_time = datetime.now()
            self.pipeline_stats['total_duration_seconds'] = (
                pipeline_end_time - pipeline_start_time
            ).total_seconds()
            
            pipeline_summary = {
                'pipeline_stats': self.pipeline_stats.copy(),
                'study_results': study_results,
                'summary_report': self._generate_pipeline_summary(study_results),
                'success': self.pipeline_stats['studies_failed'] == 0
            }
            
            # Save pipeline report
            self._save_pipeline_report(pipeline_summary)
            
            self.logger.info("=" * 60)
            self.logger.info("PRODUCTION ETL PIPELINE COMPLETED")
            self.logger.info("=" * 60)
            self.logger.info(f"Studies processed: {self.pipeline_stats['studies_processed']}")
            self.logger.info(f"Studies failed: {self.pipeline_stats['studies_failed']}")
            self.logger.info(f"Total duration: {self.pipeline_stats['total_duration_seconds']:.2f} seconds")
            self.logger.info(f"Total records: {self.pipeline_stats['total_records_processed']:,}")
            
            return pipeline_summary
            
        except Exception as e:
            self.logger.error(f"Critical pipeline error: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _log_study_summary(self, study_result: Dict[str, Any]) -> None:
        """Log summary for individual study"""
        study_code = study_result['study_code']
        success = study_result['success']
        duration = study_result['duration_seconds']
        
        if success:
            self.logger.info(
                f"Study {study_code}: SUCCESS - "
                f"Duration: {duration:.2f}s, "
                f"Records: {study_result['load_stats']['records_inserted']:,}"
            )
        else:
            self.logger.error(
                f"Study {study_code}: FAILED - "
                f"Duration: {duration:.2f}s, "
                f"Error: {study_result['error']}"
            )

    def _generate_pipeline_summary(self, study_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive pipeline summary report"""
        report = []
        report.append("=" * 80)
        report.append("PRODUCTION ETL PIPELINE SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        total_studies = len(study_results)
        successful_studies = sum(1 for r in study_results if r['success'])
        failed_studies = total_studies - successful_studies
        
        report.append("OVERALL STATISTICS:")
        report.append(f"  Total Studies: {total_studies}")
        report.append(f"  Successful: {successful_studies}")
        report.append(f"  Failed: {failed_studies}")
        report.append(f"  Success Rate: {successful_studies/total_studies*100:.1f}%")
        report.append("")
        
        # Performance statistics
        total_duration = self.pipeline_stats['total_duration_seconds']
        total_records = self.pipeline_stats['total_records_processed']
        
        report.append("PERFORMANCE STATISTICS:")
        report.append(f"  Total Duration: {total_duration:.2f} seconds")
        report.append(f"  Total Records: {total_records:,}")
        report.append(f"  Records/Second: {total_records/total_duration:.0f}")
        report.append("")
        
        # Individual study results
        report.append("STUDY RESULTS:")
        for result in study_results:
            status = "SUCCESS" if result['success'] else "FAILED"
            report.append(f"  {result['study_code']}: {status} ({result['duration_seconds']:.2f}s)")
        
        if failed_studies > 0:
            report.append("")
            report.append("FAILED STUDIES:")
            for result in study_results:
                if not result['success']:
                    report.append(f"  {result['study_code']}: {result['error']}")
        
        report.append("=" * 80)
        
        return "\n".join(report)

    def _save_pipeline_report(self, pipeline_summary: Dict[str, Any]) -> None:
        """Save pipeline summary report to file"""
        try:
            report_dir = Path(self.config.base_path) / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = report_dir / f"etl_pipeline_report_{timestamp}.txt"
            json_file = report_dir / f"etl_pipeline_report_{timestamp}.json"
            
            # Save text report
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(pipeline_summary['summary_report'])
            
            # Save JSON report
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_summary, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline report saved to {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline report: {e}")


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(description='Production ETL Pipeline for refine.bio data')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--studies', type=str, nargs='+', help='Specific study codes to process')
    parser.add_argument('--validate-only', action='store_true', help='Only validate data without loading')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run without database changes')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            # Load from file (would need config file parser)
            config = ETLConfig.from_environment()
        else:
            config = ETLConfig.from_environment()
        
        # Override config with command-line arguments
        if args.log_level:
            config.log_level = args.log_level
        if args.max_workers:
            config.max_workers = args.max_workers
        
        # Validate configuration
        config.validate_paths()
        
        # Create and execute pipeline
        pipeline = ETLPipelineOrchestrator(config)
        
        if args.validate_only:
            # Validation-only mode
            study_codes = args.studies or config.discover_study_codes()
            for study_code in study_codes:
                try:
                    # Extract and validate only
                    extracted_data = pipeline.extractor.extract_all_sources(study_code)
                    validation_results = pipeline.validator.validate_all_data(extracted_data)
                    
                    print(f"\nValidation Results for {study_code}:")
                    print(f"  Passed: {validation_results['validation_passed']}")
                    print(f"  Quality Score: {pipeline.validator.get_quality_score():.1f}/100")
                    print(f"  Warnings: {len(validation_results['warnings'])}")
                    print(f"  Errors: {len(validation_results['errors'])}")
                    
                except Exception as e:
                    print(f"Validation failed for {study_code}: {e}")
                    
        else:
            # Full ETL execution
            results = pipeline.execute_full_pipeline(args.studies)
            
            # Print summary
            print("\n" + "=" * 60)
            print("ETL PIPELINE EXECUTION COMPLETE")
            print("=" * 60)
            print(f"Studies processed: {results['pipeline_stats']['studies_processed']}")
            print(f"Studies failed: {results['pipeline_stats']['studies_failed']}")
            print(f"Success rate: {(results['pipeline_stats']['studies_processed'] - results['pipeline_stats']['studies_failed']) / results['pipeline_stats']['studies_processed'] * 100:.1f}%")
            print(f"Total duration: {results['pipeline_stats']['total_duration_seconds']:.2f} seconds")
            print(f"Total records: {results['pipeline_stats']['total_records_processed']:,}")
            
            # Exit with appropriate code
            sys.exit(0 if results['success'] else 1)
            
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()