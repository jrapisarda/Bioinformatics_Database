#!/usr/bin/env python3
"""
System Test Script for Gene Pair ML Analysis

This script tests the core functionality of the system to ensure everything is working correctly.
"""

import sys
import os
import json
import pandas as pd
import traceback, sys, sklearn
from pathlib import Path

def excepthook(exc_type, exc_value, exc_tb):
    print("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    print("\n>>>  LAST ESTIMATOR ON THE STACK  <<<")
    for frame, line in traceback.walk_tb(exc_tb):
        if "sklearn" in frame.f_code.co_filename and "fit" in frame.f_code.co_name:
            print(frame.f_locals.get("self", "???"))
    sys.exit(1)
sys.excepthook = excepthook

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from gene_pair_agent import GenePairAnalyzer, RulesEngine, MetaAnalysisProcessor, DatabaseConnector
        print("✓ Gene pair agent modules imported successfully")
        
        from visualization import ChartGenerator, InteractivePlotter, ResultsDashboard
        print("✓ Visualization modules imported successfully")
        
        from config import settings
        print("✓ Configuration module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_rules_engine():
    """Test the rules engine functionality."""
    print("\nTesting rules engine...")
    
    try:
        from gene_pair_agent import RulesEngine, Rule
        
        # Create rules engine
        rules_engine = RulesEngine()
        print(f"✓ Created rules engine with {len(rules_engine.rules)} default rules")
        
        # Test default rules
        summary = rules_engine.get_rule_summary()
        print(f"✓ Rule summary: {summary['total_rules']} total, {summary['enabled_rules']} enabled")
        
        # Test custom rule addition
        custom_rule = Rule(
            name="Test Rule",
            condition="p_ss < 0.05",
            weight=0.5,
            description="Test custom rule"
        )
        rules_engine.add_rule(custom_rule)
        print("✓ Added custom rule successfully")
        
        # Test rule evaluation with sample data
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
        
        score = rules_engine.calculate_ranking_score(sample_data)
        print(f"✓ Rule evaluation successful, score: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Rules engine test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality."""
    print("\nTesting feature engineering...")
    
    try:
        from gene_pair_agent import FeatureEngineering
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_data = pd.DataFrame({
            'dz_ss_mean': np.random.normal(-0.5, 0.3, 50),
            'dz_soth_mean': np.random.normal(-1.0, 0.5, 50),
            'p_ss': np.random.beta(0.5, 5, 50),
            'p_soth': np.random.beta(0.3, 10, 50),
            'dz_ss_ci_low': np.random.normal(-0.8, 0.3, 50),
            'dz_ss_ci_high': np.random.normal(-0.2, 0.3, 50),
            'dz_soth_ci_low': np.random.normal(-1.3, 0.5, 50),
            'dz_soth_ci_high': np.random.normal(-0.7, 0.5, 50),
            'dz_ss_I2': np.random.uniform(0, 80, 50),
            'dz_soth_I2': np.random.uniform(0, 90, 50),
            'dz_ss_z': np.random.normal(-2, 1, 50),
            'dz_soth_z': np.random.normal(-3, 1.5, 50),
            'q_ss': np.random.beta(0.5, 3, 50),
            'q_soth': np.random.beta(0.3, 5, 50)
        })
        
        # Create feature engineering instance
        fe = FeatureEngineering(n_components=5)
        
        # Fit and transform
        features = fe.fit_transform(sample_data)
        print(f"✓ Feature engineering successful: {sample_data.shape[1]} -> {features.shape[1]} features")
        
        # Test transform on new data
        new_data = sample_data.iloc[:10].copy()
        new_features = fe.transform(new_data)
        print(f"✓ Feature transformation successful: {new_features.shape}")
        
        # Get feature importance
        importance = fe.get_feature_importance()
        print(f"✓ Feature importance calculated: {importance.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature engineering test failed: {e}")
        return False

def test_meta_analysis_processor():
    """Test meta-analysis data processing."""
    print("\nTesting meta-analysis processor...")
    
    try:
        from gene_pair_agent import MetaAnalysisProcessor
        
        # Create processor
        processor = MetaAnalysisProcessor()
        
        # Create sample data
        sample_data = processor.create_sample_data(50)
        print(f"✓ Created sample data: {sample_data.shape}")
        
        # Test data validation
        validation = processor.validate_data(sample_data)
        print(f"✓ Data validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
        
        # Test data cleaning
        cleaned_data = processor.clean_data(sample_data)
        print(f"✓ Data cleaning: {cleaned_data.shape}")
        
        # Test summary statistics
        stats = processor.get_summary_statistics(cleaned_data)
        print(f"✓ Summary statistics: {stats['basic_info']['total_pairs']} pairs")
        
        return True
        
    except Exception as e:
        print(f"✗ Meta-analysis processor test failed: {e}")
        return False

def test_gene_pair_analyzer():
    """Test the main gene pair analyzer."""
    print("\nTesting gene pair analyzer...")
    
    try:
        from gene_pair_agent import GenePairAnalyzer
        from gene_pair_agent import MetaAnalysisProcessor
        
        # Create sample data
        processor = MetaAnalysisProcessor()
        sample_data = processor.create_sample_data(100)
        
        # Initialize analyzer
        analyzer = GenePairAnalyzer(n_features=5, contamination=0.1)
        
        # Fit the analyzer
        analyzer.fit(sample_data)
        print("✓ Analyzer fitted successfully")
        
        # Run analysis
        results = analyzer.predict(sample_data)
        print(f"✓ Analysis completed: {len(results.get('recommendations', []))} recommendations")
        
        # Check results structure
        if 'recommendations' in results:
            rec = results['recommendations'][0]
            required_keys = ['gene_a', 'gene_b', 'combined_score', 'rules_score', 'ml_confidence']
            if all(key in rec for key in required_keys):
                print("✓ Recommendation structure is correct")
            else:
                print("✗ Recommendation structure is incorrect")
                return False
        
        # Test summary
        summary = analyzer.get_analysis_summary()
        print(f"✓ Analysis summary: {summary.get('recommendations_count', 0)} recommendations")
        
        return True
        
    except Exception as e:
        print(f"✗ Gene pair analyzer test failed: {e}")
        return False

def test_visualization():
    """Test visualization components."""
    print("\nTesting visualization...")
    
    try:
        from visualization import ChartGenerator
        from gene_pair_agent import MetaAnalysisProcessor, GenePairAnalyzer
        
        # Create sample data and run analysis
        processor = MetaAnalysisProcessor()
        sample_data = processor.create_sample_data(50)
        
        analyzer = GenePairAnalyzer()
        analyzer.fit(sample_data)
        results = analyzer.predict(sample_data)
        
        # Create chart generator
        chart_gen = ChartGenerator()
        
        # Test chart creation
        boxplot = chart_gen.create_boxplot(sample_data, results)
        print("✓ Box plot chart created")
        
        scatter = chart_gen.create_scatter_plot(sample_data, results)
        print("✓ Scatter plot chart created")
        
        clustering = chart_gen.create_clustering_viz(results)
        print("✓ Clustering visualization created")
        
        ranking = chart_gen.create_ranking_chart(results)
        print("✓ Ranking chart created")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from config import settings
        
        # Test default config
        default_config = settings.DEFAULT_CONFIG
        print(f"✓ Default config loaded: {len(default_config)} sections")
        
        # Test settings functions
        analysis_config = settings.get_analysis_config()
        print(f"✓ Analysis config: n_features={analysis_config['n_features']}")
        
        web_config = settings.get_web_config()
        print(f"✓ Web config: max_file_size={web_config['max_file_size']}")
        
        db_config = settings.get_database_config()
        print(f"✓ Database config: driver={db_config.get('driver', 'default')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "analysis": {
            "n_features": 5,
            "contamination": 0.1,
            "random_state": 42
        },
        "rules": [
            {
                "name": "High Significance",
                "condition": "p_ss < 0.01 AND p_soth < 0.001",
                "weight": 0.4,
                "description": "Very high statistical significance"
            }
        ]
    }
    
    with open('sample_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Sample configuration file created: sample_config.json")

def main():
    """Run all tests."""
    print("Gene Pair ML Analysis System - Test Suite")
    print("=" * 50)
    
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
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:25} [{status}]")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Create sample configuration
    if passed > 0:
        create_sample_config()
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)

if __name__ == '__main__':
    main()