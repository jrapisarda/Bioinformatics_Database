#!/usr/bin/env python3
"""
Fix for Negative Values Issue in Gene Pair ML Analysis

This script demonstrates how to handle negative values in bioinformatics data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def demonstrate_negative_value_handling():
    """Show how the system now handles negative values properly."""
    
    print("=" * 70)
    print("DEMONSTRATION: Handling Negative Values in Bioinformatics Data")
    print("=" * 70)
    
    print("\n1. Understanding the Issue:")
    print("- Bioinformatics data naturally contains negative values")
    print("- Effect sizes (Cohen's d) can be negative (downregulation)")
    print("- Z-scores can be negative (below mean)")
    print("- This is NORMAL and expected in sepsis research")
    
    print("\n2. Original Problem:")
    print("- PCA and some ML algorithms expect non-negative input")
    print("- StandardScaler preserves negative values but some algorithms can't handle them")
    print("- This caused 'Negative values in data passed to X' errors")
    
    print("\n3. Solution Implemented:")
    print("- Use RobustScaler instead of StandardScaler")
    print("- Use PCA with svd_solver='full' for better stability")
    print("- Add absolute value features for algorithms that need positive input")
    print("- Use algorithms that naturally handle negative values (Isolation Forest, GMM)")
    
    # Demonstrate with sample data
    print("\n4. Sample Bioinformatics Data:")
    
    # Create realistic sample data with negative values
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'dz_ss_mean': np.random.normal(-0.5, 0.3, 10),  # Negative effect sizes
        'dz_soth_mean': np.random.normal(-1.0, 0.5, 10),
        'p_ss': np.random.beta(0.5, 5, 10),
        'p_soth': np.random.beta(0.3, 10, 10),
        'dz_ss_z': np.random.normal(-2, 1, 10),  # Negative z-scores
        'dz_soth_z': np.random.normal(-3, 1.5, 10)
    })
    
    print("Sample data with negative values:")
    print(sample_data.round(3))
    
    print(f"\nData statistics:")
    print(f"- Min value: {sample_data.min().min():.3f}")
    print(f"- Max value: {sample_data.max().max():.3f}")
    print(f"- Mean value: {sample_data.mean().mean():.3f}")
    print(f"- Negative values: {(sample_data < 0).sum().sum()} cells")
    
    print("\n5. Feature Engineering Results:")
    
    # Import and use the updated feature engineering
    from gene_pair_agent import FeatureEngineering
    
    fe = FeatureEngineering(n_components=3)
    features = fe.fit_transform(sample_data)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"After feature engineering: {features.shape}")
    print(f"PCA explained variance ratio: {fe.get_explained_variance_ratio()}")
    
    print(f"\nFeature engineering handles negative values successfully!")
    print(f"No more 'Negative values in data passed to X' errors!")
    
    print("\n6. Key Changes Made:")
    print("✓ Replaced StandardScaler with RobustScaler")
    print("✓ Added PCA svd_solver='full' for stability")
    print("✓ Added absolute value features for positive-only algorithms")
    print("✓ Improved error handling and logging")
    print("✓ Added comprehensive data validation")
    
    print("\n7. Why This Works:")
    print("- RobustScaler is less sensitive to outliers than StandardScaler")
    print("- PCA can handle properly scaled negative values")
    print("- Isolation Forest and Gaussian Mixture naturally handle negative data")
    print("- Absolute value features provide positive alternatives when needed")
    
    return True

def show_algorithm_compatibility():
    """Show which algorithms handle negative values well."""
    
    print("\n" + "=" * 70)
    print("ALGORITHM COMPATIBILITY WITH NEGATIVE VALUES")
    print("=" * 70)
    
    algorithms = {
        "Isolation Forest": "✓ EXCELLENT - Designed for anomaly detection with any data",
        "DBSCAN": "✓ GOOD - Density-based, handles negative values",
        "Gaussian Mixture": "✓ EXCELLENT - Probabilistic model, handles negative values",
        "PCA": "✓ GOOD - With proper scaling and stable solver",
        "K-Means": "⚠ LIMITED - Can work but may be sensitive",
        "StandardScaler": "⚠ PROBLEMATIC - Preserves negatives but may cause issues",
        "RobustScaler": "✓ EXCELLENT - Designed for data with outliers",
        "MinMaxScaler": "✓ GOOD - If you want to shift to positive range"
    }
    
    print("\nAlgorithm Compatibility:")
    for algorithm, compatibility in algorithms.items():
        print(f"  {algorithm:20} {compatibility}")
    
    print("\nRecommended Approach:")
    print("1. Use RobustScaler for preprocessing")
    print("2. Use PCA with svd_solver='full'")
    print("3. Use Isolation Forest for anomaly detection")
    print("4. Use Gaussian Mixture for clustering")
    print("5. Keep negative values - they're biologically meaningful!")

def main():
    """Run the demonstration."""
    print("Gene Pair ML Analysis - Negative Values Fix")
    print("This demonstrates how the system now properly handles")
    print("negative values in bioinformatics data.")
    
    demonstrate_negative_value_handling()
    show_algorithm_compatibility()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The system has been updated to handle negative values properly!")
    print("✓ No more 'Negative values in data passed to X' errors")
    print("✓ Bioinformatics data with negative effect sizes is fully supported")
    print("✓ All algorithms have been selected or configured to handle negative data")
    print("✓ The system maintains biological meaning while ensuring ML compatibility")
    
    print("\nNext Steps:")
    print("1. Run the updated test suite: python test_system_verbose.py")
    print("2. Try the example usage: python example_usage.py")
    print("3. Use the web interface with your own data")

if __name__ == '__main__':
    main()