#!/usr/bin/env python3
"""
Example Usage of Gene Pair ML Analysis System

This script demonstrates how to use the system programmatically.
"""

import sys
import json
import pandas as pd
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def example_1_basic_analysis():
    """Example 1: Basic file-based analysis."""
    print("Example 1: Basic File Analysis")
    print("-" * 40)
    
    # Import the necessary modules
    from gene_pair_agent import MetaAnalysisProcessor, GenePairAnalyzer
    
    # Create sample data
    print("Creating sample meta-analysis data...")
    processor = MetaAnalysisProcessor()
    sample_data = processor.create_sample_data(100)
    
    print(f"Created {len(sample_data)} gene pairs with {len(sample_data.columns)} statistical measures")
    
    # Initialize the analyzer
    print("Initializing gene pair analyzer...")
    analyzer = GenePairAnalyzer(
        n_features=5,           # Number of PCA components
        contamination=0.1,      # Expected proportion of outliers
        random_state=42         # For reproducibility
    )
    
    # Run the analysis
    print("Running ensemble analysis...")
    analyzer.fit(sample_data)
    results = analyzer.predict(sample_data)
    
    # Display results
    print("\nAnalysis Results:")
    print(f"- Total recommendations: {len(results['recommendations'])}")
    print(f"- High confidence pairs: {sum(1 for r in results['recommendations'] if r['is_high_confidence'])}")
    print(f"- Outlier pairs: {sum(1 for r in results['recommendations'] if r['is_outlier'])}")
    
    # Show top 5 recommendations
    print("\nTop 5 Recommendations:")
    for i, rec in enumerate(results['recommendations'][:5]):
        print(f"{i+1:2d}. {rec['gene_a']}-{rec['gene_b']} "
              f"(Score: {rec['combined_score']:.3f}, "
              f"Rules: {rec['rules_score']:.3f}, "
              f"ML: {rec['ml_confidence']:.3f})")
    
    # Save results
    output_file = "example_results.json"
    analyzer.save_results(output_file)
    print(f"\nResults saved to: {output_file}")
    
    return results

def example_2_custom_rules():
    """Example 2: Analysis with custom rules."""
    print("\n\nExample 2: Analysis with Custom Rules")
    print("-" * 40)
    
    from gene_pair_agent import MetaAnalysisProcessor, GenePairAnalyzer, RulesEngine, Rule
    
    # Create sample data
    processor = MetaAnalysisProcessor()
    sample_data = processor.create_sample_data(50)
    
    # Create custom rules engine
    rules_engine = RulesEngine()
    
    # Add custom rules
    custom_rules = [
        Rule(
            name="Very High Significance",
            condition="p_ss < 0.001 AND p_soth < 0.0001",
            weight=0.5,
            description="Extremely significant results"
        ),
        Rule(
            name="Large Effect Size",
            condition="abs_dz_ss > 1.0 AND abs_dz_soth > 1.5",
            weight=0.4,
            description="Large biological effects"
        ),
        Rule(
            name="Consistent Studies",
            condition="dz_ss_I2 < 25 AND dz_soth_I2 < 50",
            weight=0.3,
            description="Low heterogeneity across studies"
        )
    ]
    
    for rule in custom_rules:
        rules_engine.add_rule(rule)
    
    print(f"Added {len(custom_rules)} custom rules")
    
    # Initialize analyzer with custom rules
    analyzer = GenePairAnalyzer()
    analyzer.rules_engine = rules_engine  # Replace default rules engine
    
    # Run analysis
    analyzer.fit(sample_data)
    results = analyzer.predict(sample_data)
    
    print(f"Analysis with custom rules completed: {len(results['recommendations'])} recommendations")
    
    return results

def example_3_feature_engineering():
    """Example 3: Advanced feature engineering."""
    print("\n\nExample 3: Feature Engineering")
    print("-" * 40)
    
    from gene_pair_agent import MetaAnalysisProcessor, FeatureEngineering
    
    # Create sample data
    processor = MetaAnalysisProcessor()
    sample_data = processor.create_sample_data(100)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Original columns: {list(sample_data.columns)[:10]}...")
    
    # Initialize feature engineering
    fe = FeatureEngineering(n_components=5, random_state=42)
    
    # Create derived features
    features = fe.fit_transform(sample_data)
    
    print(f"Feature engineered shape: {features.shape}")
    
    # Get feature importance
    importance = fe.get_feature_importance()
    print(f"Feature importance shape: {importance.shape}")
    print(f"Explained variance ratio: {fe.get_explained_variance_ratio()}")
    
    # Test clustering
    cluster_labels = fe.cluster_features(features, eps=0.3, min_samples=5)
    print(f"Created {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
    
    # Test outlier detection
    outliers = fe.detect_outliers(features, contamination=0.1)
    print(f"Detected {sum(outliers)} outliers")
    
    return features, cluster_labels, outliers

def example_4_visualization():
    """Example 4: Creating visualizations."""
    print("\n\nExample 4: Visualization")
    print("-" * 40)
    
    from gene_pair_agent import MetaAnalysisProcessor, GenePairAnalyzer
    from visualization import ChartGenerator
    
    # Create and analyze data
    processor = MetaAnalysisProcessor()
    sample_data = processor.create_sample_data(50)
    
    analyzer = GenePairAnalyzer()
    analyzer.fit(sample_data)
    results = analyzer.predict(sample_data)
    
    # Create visualizations
    chart_gen = ChartGenerator()
    
    print("Creating visualizations...")
    
    # Box plot
    boxplot = chart_gen.create_boxplot(sample_data, results)
    print(f"✓ Box plot created: {len(boxplot.get('data', []))} traces")
    
    # Scatter plot
    scatter = chart_gen.create_scatter_plot(sample_data, results)
    print(f"✓ Scatter plot created: {len(scatter.get('data', []))} traces")
    
    # Clustering visualization
    clustering = chart_gen.create_clustering_viz(results)
    print(f"✓ Clustering visualization created")
    
    # Ranking chart
    ranking = chart_gen.create_ranking_chart(results)
    print(f"✓ Ranking chart created: {len(ranking.get('data', []))} traces")
    
    # Save charts as HTML
    charts = {
        'boxplot': boxplot,
        'scatter': scatter,
        'clustering': clustering,
        'ranking': ranking
    }
    
    with open('example_charts.json', 'w') as f:
        json.dump(charts, f, indent=2)
    
    print("Charts saved to: example_charts.json")
    
    return charts

def example_5_database_simulation():
    """Example 5: Database operations simulation."""
    print("\n\nExample 5: Database Operations")
    print("-" * 40)
    
    from gene_pair_agent import DatabaseConnector
    
    # Create database connector (would normally connect to real database)
    config = {
        'driver': 'ODBC Driver 17 for SQL Server',
        'server': 'localhost',
        'database': 'BioinformaticsDB',
        'username': 'sa',
        'password': 'password'
    }
    
    connector = DatabaseConnector(config)
    
    # Simulate database operations
    print("Database operations would include:")
    print("- Connecting to bioinformatics database")
    print("- Querying gene pair correlation data")
    print("- Filtering by conditions (control, sepsis, septic shock)")
    print("- Fetching data for specific gene pairs")
    print("- Exporting results to various formats")
    
    # In a real implementation:
    # data = connector.get_gene_pair_data(gene_a='MS4A4A', illness_label='septic shock')
    # print(f"Retrieved {len(data)} records from database")
    
    return True

def main():
    """Run all examples."""
    print("Gene Pair ML Analysis System - Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Analysis", example_1_basic_analysis),
        ("Custom Rules", example_2_custom_rules),
        ("Feature Engineering", example_3_feature_engineering),
        ("Visualization", example_4_visualization),
        ("Database Operations", example_5_database_simulation)
    ]
    
    results = []
    
    for example_name, example_func in examples:
        try:
            print(f"\n{'='*50}")
            result = example_func()
            results.append((example_name, True))
            print(f"✓ {example_name} completed successfully")
        except Exception as e:
            print(f"✗ {example_name} failed: {e}")
            results.append((example_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("Examples Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for example_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{example_name:25} [{status}]")
    
    print(f"\nOverall: {passed}/{total} examples completed ({passed/total*100:.1f}%)")
    
    print("\nGenerated files:")
    print("- example_results.json: Analysis results")
    print("- example_charts.json: Visualization charts")
    print("- sample_config.json: Sample configuration")

if __name__ == '__main__':
    main()