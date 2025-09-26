#!/usr/bin/env python3
"""
Gene Pair ML Analysis System - Main Entry Point

This script provides a command-line interface for running the gene pair analysis
system with various options for data input, analysis configuration, and output.
"""

import argparse
import sys
import logging
import json

from pathlib import Path
from typing import Dict, Any, Optional
  

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from gene_pair_agent import GenePairAnalyzer, RulesEngine, MetaAnalysisProcessor, DatabaseConnector
from visualization import ChartGenerator

def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('gene_pair_analysis.log')
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)

def run_file_analysis(file_path: str, output_path: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Run analysis on file-based data."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load and process data
        processor = MetaAnalysisProcessor()
        data = processor.load_data(file_path)
        
        # Validate data
        validation = processor.validate_data(data)
        if not validation['is_valid']:
            logger.error(f"Data validation failed: {validation['errors']}")
            sys.exit(1)
        
        # Clean and prepare data
        cleaned_data = processor.clean_data(data)
        prepared_data = processor.prepare_for_analysis(cleaned_data)
        
        # Initialize analyzer
        analyzer = GenePairAnalyzer(
            n_features=config.get('n_features', 5) if config else 5,
            contamination=config.get('contamination', 0.1) if config else 0.1
        )
        
        # Configure rules if provided
        if config and 'rules' in config:
            for rule_data in config['rules']:
                from gene_pair_agent.rules_engine import Rule
                rule = Rule(**rule_data)
                analyzer.rules_engine.add_rule(rule)
        
        # Run analysis
        logger.info("Starting gene pair analysis...")
        analyzer.fit(prepared_data)
        results = analyzer.predict(prepared_data)
        
        # Generate visualizations
        chart_gen = ChartGenerator()
        dashboard = chart_gen.create_summary_dashboard(results)
        
        # Save results
        analyzer.save_results(output_path)
        
        # Print summary
        summary = analyzer.get_analysis_summary()
        logger.info("Analysis completed successfully!")
        logger.info(f"Total recommendations: {summary.get('recommendations_count', 0)}")
        logger.info(f"Positive control validation: {summary.get('summary_stats', {}).get('positive_control_validated', 'Unknown')}")
        
        # Print top recommendations
        if results.get('recommendations'):
            logger.info("\nTop 10 Recommendations:")
            for i, rec in enumerate(results['recommendations'][:10]):
                logger.info(f"{i+1:2d}. {rec['gene_a']}-{rec['gene_b']} (Score: {rec['combined_score']:.3f})")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

def run_database_analysis(config: Dict[str, Any], output_path: str) -> None:
    """Run analysis on database data."""
    logger = logging.getLogger(__name__)
    
    try:
        # Connect to database
        connector = DatabaseConnector(config.get('database', {}))
        
        if not connector.connect():
            logger.error("Failed to connect to database")
            sys.exit(1)
        
        # Fetch data
        logger.info("Fetching data from database...")
        data = connector.get_gene_pair_data(
            gene_a=config.get('filters', {}).get('gene_a'),
            gene_b=config.get('filters', {}).get('gene_b'),
            illness_label=config.get('filters', {}).get('illness_label'),
            limit=config.get('filters', {}).get('limit', 1000)
        )
        
        if data.empty:
            logger.error("No data retrieved from database")
            sys.exit(1)
        
        # Process data
        processor = MetaAnalysisProcessor()
        prepared_data = processor.prepare_for_analysis(data)
        
        # Initialize and run analyzer
        analyzer = GenePairAnalyzer(
            n_features=config.get('n_features', 5),
            contamination=config.get('contamination', 0.1)
        )
        
        # Configure rules if provided
        if 'rules' in config:
            for rule_data in config['rules']:
                from gene_pair_agent.rules_engine import Rule
                rule = Rule(**rule_data)
                analyzer.rules_engine.add_rule(rule)
        
        # Run analysis
        logger.info("Starting analysis...")
        analyzer.fit(prepared_data)
        results = analyzer.predict(prepared_data)
        
        # Save results
        analyzer.save_results(output_path)
        
        logger.info("Database analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Database analysis failed: {e}")
        sys.exit(1)

def create_sample_data(output_path: str, n_pairs: int = 100) -> None:
    """Create sample data for testing."""
    logger = logging.getLogger(__name__)
    
    try:
        processor = MetaAnalysisProcessor()
        sample_data = processor.create_sample_data(n_pairs)
        
        # Export sample data
        if output_path.endswith('.csv'):
            sample_data.to_csv(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            sample_data.to_excel(output_path, index=False)
        else:
            sample_data.to_json(output_path, orient='records', indent=2)
        
        logger.info(f"Sample data created with {n_pairs} gene pairs: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        sys.exit(1)

def main():
    """Main function to handle command-line arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description='Gene Pair ML Analysis System - Identify biologically significant gene pairs'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f', help='Input data file (Excel, CSV, or JSON)')
    input_group.add_argument('--database', '-d', help='Run database analysis (requires config)')
    input_group.add_argument('--sample', '-s', help='Create sample data', action='store_true')
    
    # Output options
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--config', '-c', help='Configuration file path (JSON)')
    
    # Analysis parameters
    parser.add_argument('--n-features', type=int, default=5, help='Number of PCA features (default: 5)')
    parser.add_argument('--contamination', type=float, default=0.1, help='Outlier contamination rate (default: 0.1)')
    parser.add_argument('--n-pairs', type=int, default=100, help='Number of sample pairs (default: 100)')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
        # Override config with command-line arguments
        if args.n_features:
            config['n_features'] = args.n_features
        if args.contamination:
            config['contamination'] = args.contamination
    else:
        config = {
            'n_features': args.n_features,
            'contamination': args.contamination
        }
    
    # Run appropriate analysis
    if args.sample:
        logger.info("Creating sample data...")
        create_sample_data(args.output, args.n_pairs)
        
    elif args.file:
        logger.info(f"Running file analysis on: {args.file}")
        run_file_analysis(args.file, args.output, config)
        
    elif args.database:
        if not args.config:
            logger.error("Database analysis requires a configuration file")
            sys.exit(1)
        
        logger.info("Running database analysis...")
        run_database_analysis(config, args.output)
    
    logger.info("Process completed successfully!")

if __name__ == '__main__':
    main()