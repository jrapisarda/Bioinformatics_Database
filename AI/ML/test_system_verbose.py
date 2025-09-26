#!/usr/bin/env python3
"""
Verbose System Test Script for Gene Pair ML Analysis

This script tests the core functionality with detailed logging to identify issues.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_verbose.log')
    ]
)

logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("=" * 60)
    print("Testing imports with detailed logging...")
    print("=" * 60)
    
    try:
        logger.info("Testing gene_pair_agent imports...")
        from gene_pair_agent import GenePairAnalyzer, RulesEngine, MetaAnalysisProcessor, DatabaseConnector
        print("✓ Gene pair agent modules imported successfully")
        
        logger.info("Testing visualization imports...")
        from visualization import ChartGenerator, InteractivePlotter, ResultsDashboard
        print("✓ Visualization modules imported successfully")
        
        logger.info("Testing config imports...")
        from config import settings
        print("✓ Configuration module imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_rules_engine():
    """Test the rules engine functionality."""
    print("\n" + "=" * 60)
    print("Testing rules engine with detailed logging...")
    print("=" * 60)
    
    try:
        logger.info("Creating rules engine...")
        from gene_pair_agent import RulesEngine, Rule
        
        # Create rules engine
        rules_engine = RulesEngine()
        print(f"✓ Created rules engine with {len(rules_engine.rules)} default rules")
        
        # Test default rules
        logger.info("Getting rule summary...")
        summary = rules_engine.get_rule_summary()
        print(f"✓ Rule summary: {summary['total_rules']} total, {summary['enabled_rules']} enabled")
        
        # Test custom rule addition
        logger.info("Adding custom rule...")
        custom_rule = Rule(
            name="Test Rule",
            condition="p_ss < 0.05",
            weight=0.5,
            description="Test custom rule"
        )
        rules_engine.add_rule(custom_rule)
        print("✓ Added custom rule successfully")
        
        # Test rule evaluation with sample data
        logger.info("Testing rule evaluation...")
        sample_data = {
            'p_ss': 0.01,
            'p_soth': 0.005,
            'abs_dz_ss': 0.8,
            'abs_dz_soth': 1.2,
            'dz_ss_z': 2.5,
            'dz_soth_z': 3.5,
            'q_ss': 0.02,
            'q_soth': 0.01,
            'dz_ss_I2': 25,
            'dz_soth_I2': 60
        }
        
        logger.info(f"Sample data: {sample_data}")
        score = rules_engine.calculate_ranking_score(sample_data)
        print(f"✓ Rule evaluation successful, score: {score:.3f}")
        
        # Test individual rule evaluation
        logger.info("Testing individual rule evaluation...")
        for rule in rules_engine.rules[:3]:  # Test first 3 rules
            rule_score = rules_engine.evaluate_rule(rule, sample_data)
            logger.info(f"Rule '{rule.name}': {rule_score:.3f} (condition: {rule.condition})")
        
        return True
        
    except Exception as e:
        logger.error(f"Rules engine test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality."""
    print("\n" + "=" * 60)
    print("Testing feature engineering with detailed logging...")
    print("=" * 60)
    
    try:
        logger.info("Importing FeatureEngineering...")
        from gene_pair_agent import FeatureEngineering
        import pandas as pd
        import numpy as np
        
        # Create sample data with realistic bioinformatics values
        logger.info("Creating sample data...")
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'dz_ss_mean': np.random.normal(-0.5, 0.3, 50),  # Negative effect sizes are normal
            'dz_soth_mean': np.random.normal(-1.0, 0.5, 50),
            'p_ss': np.random.beta(0.5, 5, 50),
            'p_soth': np.random.beta(0.3, 10, 50),
            'dz_ss_ci_low': np.random.normal(-0.8, 0.3, 50),
            'dz_ss_ci_high': np.random.normal(-0.2, 0.3, 50),
            'dz_soth_ci_low': np.random.normal(-1.3, 0.5, 50),
            'dz_soth_ci_high': np.random.normal(-0.7, 0.5, 50),
            'dz_ss_Q': np.random.uniform(0, 10, 50),
            'dz_ss_I2': np.random.uniform(0, 80, 50),
            'dz_soth_I2': np.random.uniform(0, 90, 50),
            'dz_ss_z': np.random.normal(-2, 1, 50),  # Z-scores can be negative
            'dz_soth_z': np.random.normal(-3, 1.5, 50),
            'q_ss': np.random.beta(0.5, 3, 50),
            'q_soth': np.random.beta(0.3, 5, 50)
        })
        
        print(f"✓ Created sample data: {sample_data.shape}")
        print(f"Sample data statistics:")
        print(sample_data.describe())
        
        # Create feature engineering instance
        logger.info("Creating FeatureEngineering instance...")
        fe = FeatureEngineering(n_components=5)
        
        # Fit and transform
        logger.info("Running fit_transform...")
        features = fe.fit_transform(sample_data)
        print(f"✓ Feature engineering successful: {sample_data.shape[1]} -> {features.shape[1]} features")
        print(f"✓ Feature engineering output shape: {features.shape}")
        
        # Test transform on new data
        logger.info("Testing transform method...")
        new_data = sample_data.iloc[:10].copy()
        new_features = fe.transform(new_data)
        print(f"✓ Feature transformation successful: {new_features.shape}")
        
        # Get feature importance
        logger.info("Getting feature importance...")
        importance = fe.get_feature_importance()
        print(f"✓ Feature importance calculated: {importance.shape}")
        print("Top 5 most important features for PC1:")
        pc1_importance = importance['PC1'].abs().sort_values(ascending=False)
        for i, (feature, importance_val) in enumerate(pc1_importance.head().items()):
            print(f"  {i+1}. {feature}: {importance_val:.3f}")
        
        # Test clustering
        logger.info("Testing clustering...")
        cluster_labels = fe.cluster_features(features, eps=0.3, min_samples=5)
        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"✓ Created {unique_clusters} clusters")
        
        # Test outlier detection
        logger.info("Testing outlier detection...")
        outliers = fe.detect_outliers(features, contamination=0.1)
        print(f"✓ Detected {sum(outliers)} outliers")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_meta_analysis_processor():
    """Test meta-analysis data processing."""
    print("\n" + "=" * 60)
    print("Testing meta-analysis processor...")
    print("=" * 60)
    
    try:
        logger.info("Importing MetaAnalysisProcessor...")
        from gene_pair_agent import MetaAnalysisProcessor
        
        # Create processor
        logger.info("Creating MetaAnalysisProcessor...")
        processor = MetaAnalysisProcessor()
        
        # Create sample data
        logger.info("Creating sample data...")
        sample_data = processor.create_sample_data(50)
        print(f"✓ Created sample data: {sample_data.shape}")
        
        # Log sample data info
        logger.info("Sample data info:")
        logger.info(f"Columns: {list(sample_data.columns)}")
        logger.info(f"Data types:\n{sample_data.dtypes}")
        logger.info(f"First few rows:\n{sample_data.head()}")
        logger.info(f"Data summary:\n{sample_data.describe()}")
        
        # Test data validation
        logger.info("Validating data...")
        validation = processor.validate_data(sample_data)
        print(f"✓ Data validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        
        if not validation['is_valid']:
            logger.error(f"Validation errors: {validation['errors']}")
            logger.error(f"Validation warnings: {validation['warnings']}")
        
        # Test data cleaning
        logger.info("Cleaning data...")
        cleaned_data = processor.clean_data(sample_data)
        print(f"✓ Data cleaning: {cleaned_data.shape}")
        
        # Test summary statistics
        logger.info("Getting summary statistics...")
        stats = processor.get_summary_statistics(cleaned_data)
        print(f"✓ Summary statistics: {stats['basic_info']['total_pairs']} pairs")
        logger.info(f"Summary stats: {stats['basic_info']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Meta-analysis processor test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_gene_pair_analyzer():
    """Test the main gene pair analyzer."""
    print("\n" + "=" * 60)
    print("Testing gene pair analyzer...")
    print("=" * 60)
    
    try:
        logger.info("Importing necessary modules...")
        from gene_pair_agent import MetaAnalysisProcessor, GenePairAnalyzer
        
        # Create sample data with detailed logging
        logger.info("Creating sample data...")
        processor = MetaAnalysisProcessor()
        sample_data = processor.create_sample_data(100)
        
        print(f"✓ Created sample data: {sample_data.shape}")
        logger.info(f"Sample data columns: {list(sample_data.columns)}")
        logger.info(f"Sample data dtypes:\n{sample_data.dtypes}")
        logger.info(f"Sample data head:\n{sample_data.head()}")
        
        # Check for negative values (which are normal in bioinformatics)
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            neg_count = (sample_data[col] < 0).sum()
            if neg_count > 0:
                logger.info(f"Column {col} has {neg_count} negative values (normal for bioinformatics)")
        
        # Initialize analyzer
        logger.info("Initializing GenePairAnalyzer...")
        analyzer = GenePairAnalyzer(
            n_features=5,
            contamination=0.1,
            random_state=42
        )
        
        # Run the analysis
        logger.info("Fitting analyzer...")
        analyzer.fit(sample_data)
        print("✓ Analyzer fitted successfully")
        
        logger.info("Running prediction...")
        results = analyzer.predict(sample_data)
        print(f"✓ Analysis completed: {len(results['recommendations'])} recommendations")
        
        # Check results structure
        logger.info("Checking results structure...")
        if 'recommendations' in results and results['recommendations']:
            rec = results['recommendations'][0]
            required_keys = ['gene_a', 'gene_b', 'combined_score', 'rules_score', 'ml_confidence']
            
            logger.info(f"First recommendation: {rec}")
            
            if all(key in rec for key in required_keys):
                print("✓ Recommendation structure is correct")
            else:
                print(f"✗ Recommendation structure is incorrect. Missing keys: {[k for k in required_keys if k not in rec]}")
                return False
        
        # Test summary
        logger.info("Getting analysis summary...")
        summary = analyzer.get_analysis_summary()
        print(f"✓ Analysis summary: {summary.get('recommendations_count', 0)} recommendations")
        logger.info(f"Full summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Gene pair analyzer test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_visualization():
    """Test visualization components."""
    print("\n" + "=" * 60)
    print("Testing visualization...")
    print("=" * 60)
    
    try:
        logger.info("Importing visualization modules...")
        from gene_pair_agent import MetaAnalysisProcessor, GenePairAnalyzer
        from visualization import ChartGenerator
        
        # Create and analyze data
        logger.info("Creating and analyzing data...")
        processor = MetaAnalysisProcessor()
        sample_data = processor.create_sample_data(50)
        
        analyzer = GenePairAnalyzer()
        analyzer.fit(sample_data)
        results = analyzer.predict(sample_data)
        
        # Create chart generator
        logger.info("Creating ChartGenerator...")
        chart_gen = ChartGenerator()
        
        # Create visualizations with error handling
        charts = {}
        
        try:
            logger.info("Creating box plot...")
            boxplot = chart_gen.create_boxplot(sample_data, results)
            charts['boxplot'] = boxplot
            print(f"✓ Box plot created: {len(boxplot.get('data', []))} traces")
        except Exception as e:
            logger.warning(f"Box plot creation failed: {e}")
        
        try:
            logger.info("Creating scatter plot...")
            scatter = chart_gen.create_scatter_plot(sample_data, results)
            charts['scatter'] = scatter
            print(f"✓ Scatter plot created: {len(scatter.get('data', []))} traces")
        except Exception as e:
            logger.warning(f"Scatter plot creation failed: {e}")
        
        try:
            logger.info("Creating clustering visualization...")
            clustering = chart_gen.create_clustering_viz(results)
            charts['clustering'] = clustering
            print(f"✓ Clustering visualization created")
        except Exception as e:
            logger.warning(f"Clustering visualization failed: {e}")
        
        try:
            logger.info("Creating ranking chart...")
            ranking = chart_gen.create_ranking_chart(results)
            charts['ranking'] = ranking
            print(f"✓ Ranking chart created: {len(ranking.get('data', []))} traces")
        except Exception as e:
            logger.warning(f"Ranking chart creation failed: {e}")
        
        # Save charts as JSON
        if charts:
            with open('test_charts.json', 'w') as f:
                json.dump(charts, f, indent=2)
            print("Charts saved to: test_charts.json")
        
        return len(charts) > 0
        
    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\n" + "=" * 60)
    print("Testing configuration...")
    print("=" * 60)
    
    try:
        logger.info("Importing config settings...")
        from config import settings
        
        # Test default config
        logger.info("Testing default config...")
        default_config = settings.DEFAULT_CONFIG
        print(f"✓ Default config loaded: {len(default_config)} sections")
        
        # Test settings functions
        logger.info("Testing analysis config...")
        analysis_config = settings.get_analysis_config()
        print(f"✓ Analysis config: n_features={analysis_config['n_features']}")
        
        logger.info("Testing web config...")
        web_config = settings.get_web_config()
        print(f"✓ Web config: max_file_size={web_config['max_file_size']}")
        
        logger.info("Testing database config...")
        db_config = settings.get_database_config()
        print(f"✓ Database config: driver={db_config.get('driver', 'default')}")
        
        # Test loading and saving
        logger.info("Testing config save/load...")
        test_config = {'test': 'value', 'number': 42}
        settings.save_config(test_config, 'test_config.json')
        loaded_config = settings.load_config('test_config.json')
        
        if loaded_config.get('test') == 'value' and loaded_config.get('number') == 42:
            print("✓ Config save/load working")
        else:
            print("✗ Config save/load failed")
            return False
        
        # Cleanup
        if os.path.exists('test_config.json'):
            os.remove('test_config.json')
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def create_sample_config():
    """Create a sample configuration file."""
    logger.info("Creating sample configuration file...")
    
    config = {
        "analysis": {
            "n_features": 5,
            "contamination": 0.1,
            "random_state": 42
        },
        "rules": [
            {
                "name": "High Significance",
                "condition": "p_ss < 0.01 and p_soth < 0.001",
                "weight": 0.4,
                "description": "Very high statistical significance"
            },
            {
                "name": "Large Negative Effect",
                "condition": "dz_ss_mean < -0.5 and dz_soth_mean < -1.0",
                "weight": 0.3,
                "description": "Large negative effect sizes (common in sepsis)"
            }
        ],
        "database": {
            "server": "localhost",
            "database": "BioinformaticsDB",
            "username": "sa",
            "password": "password"
        }
    }
    
    with open('sample_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Sample configuration file created: sample_config.json")
    print("\nSample config contents:")
    print(json.dumps(config, indent=2))

def analyze_test_results(results):
    """Analyze test results and provide recommendations."""
    print("\n" + "=" * 60)
    print("Test Analysis and Recommendations")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Overall success rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    failed_tests = [name for name, result in results if not result]
    if failed_tests:
        print(f"\nFailed tests: {', '.join(failed_tests)}")
        
        # Provide specific recommendations
        if 'Imports' in failed_tests:
            print("\nImports Issues:")
            print("- Check Python version (should be 3.8+)")
            print("- Install missing dependencies: pip install -r requirements.txt")
            print("- Check for typing module issues")
        
        if 'Gene Pair Analyzer' in failed_tests or 'Visualization' in failed_tests:
            print("\nML/Analysis Issues:")
            print("- Negative values in bioinformatics data are NORMAL")
            print("- The system has been updated to handle negative values")
            print("- Check that RobustScaler is being used instead of StandardScaler")
            print("- Verify PCA is using svd_solver='full' for stability")
        
        if 'Configuration' in failed_tests:
            print("\nConfiguration Issues:")
            print("- Check typing imports in config/settings.py")
            print("- Verify Dict import from typing module")
    
    # Check for common bioinformatics data issues
    print(f"\nBioinformatics Data Notes:")
    print(f"- Negative effect sizes (dz_ss_mean < 0) are NORMAL in sepsis research")
    print(f"- Negative z-scores indicate downregulation, which is expected")
    print(f"- The system has been designed to handle negative values properly")
    print(f"- RobustScaler is used instead of StandardScaler for better outlier handling")

def main():
    """Run all tests with verbose output."""
    print("Gene Pair ML Analysis System - Verbose Test Suite")
    print("=" * 60)
    print("Logging detailed information to test_verbose.log")
    
    tests = [
        ("Imports", test_imports),
        ("Rules Engine", test_rules_engine),
        ("Feature Engineering", test_feature_engineering),
        ("Meta-Analysis Processor", test_meta_analysis_processor),
        ("Gene Pair Analyzer", test_gene_pair_analyzer),
        ("Visualization", test_visualization),
        ("Configuration", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            result = test_func()
            results.append((test_name, result))
            print(f"\n{'='*60}")
            print(f"{test_name}: {'PASSED' if result else 'FAILED'}")
            print(f"{'='*60}")
        except Exception as e:
            logger.error(f"{test_name} test crashed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:25} [{status}]")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Analyze results and provide recommendations
    analyze_test_results(results)
    
    # Create sample configuration
    if passed > 0:
        create_sample_config()
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)

if __name__ == '__main__':
    main()