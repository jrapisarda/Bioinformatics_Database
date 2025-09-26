"""
Gene Pair Analyzer - Main ML Analysis Engine

Implements ensemble machine learning approach with unsupervised pattern discovery
and configurable rules-based ranking for gene pair correlation analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings

from .rules_engine import RulesEngine
from .feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)


class GenePairAnalyzer:
    """Main analysis engine for gene pair correlation analysis."""
    
    def __init__(self, 
                 n_features: int = 5,
                 contamination: float = 0.1,
                 random_state: int = 42):
        """Initialize the gene pair analyzer with ensemble methods."""
        self.n_features = n_features
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize components
        self.feature_engineering = FeatureEngineering(
            n_components=n_features,
            random_state=random_state
        )
        self.rules_engine = RulesEngine()
        
        # ML models - using algorithms that handle negative values
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        # Use StandardScaler to handle negative values properly
        self.scaler = StandardScaler()
        
        # Analysis results
        self.analysis_results = {}
        self.is_fitted = False
        
        logger.info(f"Initialized GenePairAnalyzer with n_features={n_features}, contamination={contamination}")
    
    def fit(self, data: pd.DataFrame) -> 'GenePairAnalyzer':
        """Fit the analyzer to the gene pair data."""
        logger.info(f"Fitting GenePairAnalyzer to {len(data)} gene pairs")
        
        try:
            # Create derived features and fit feature engineering
            features = self.feature_engineering.fit_transform(data)
            logger.info(f"Feature engineering completed: {features.shape}")
            
            # Scale features to handle negative values - but preserve for algorithms that need them
            scaled_features = self.scaler.fit_transform(features)
            logger.info(f"Feature scaling completed: {scaled_features.shape}")
            
            # Fit ML models that can handle negative values
            logger.info("Fitting Isolation Forest...")
            self.isolation_forest.fit(scaled_features)
            
            # For algorithms that need positive values, we'll handle this in predict
            logger.info("GenePairAnalyzer fitted successfully")
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error during fitting: {e}")
            logger.error(f"Data shape: {data.shape}")
            logger.error(f"Data columns: {list(data.columns)}")
            logger.error(f"Data sample:\n{data.head()}")
            raise
        
        return self
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze gene pairs and return comprehensive results."""
        if not self.is_fitted:
            raise ValueError("GenePairAnalyzer must be fitted before prediction")
        
        logger.info(f"Analyzing {len(data)} gene pairs")
        
        try:
            # Transform features
            features = self.feature_engineering.transform(data)
            logger.info(f"Features transformed: {features.shape}")
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            logger.info(f"Features scaled: {scaled_features.shape}")
            
            # Check for any remaining issues
            if np.any(np.isnan(scaled_features)):
                logger.warning("NaN values detected in scaled features, replacing with zeros")
                scaled_features = np.nan_to_num(scaled_features)
            
            if np.any(np.isinf(scaled_features)):
                logger.warning("Infinite values detected in scaled features, replacing with zeros")
                scaled_features = np.nan_to_num(scaled_features, posinf=0, neginf=0)
            
            # Run ensemble analysis
            results = self._run_ensemble_analysis(data, features, scaled_features)
            
            # Apply rules-based ranking
            ranked_pairs = self._apply_rules_ranking(data)
            
            # Combine results
            final_results = {
                'ensemble_analysis': results,
                'rules_ranking': ranked_pairs,
                'recommendations': self._generate_recommendations(results, ranked_pairs),
                'summary_stats': self._calculate_summary_stats(data, results)
            }
            
            self.analysis_results = final_results
            logger.info("Analysis completed successfully")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(f"Data shape: {data.shape}")
            logger.error(f"Features shape: {features.shape if 'features' in locals() else 'unknown'}")
            logger.error(f"Scaled features shape: {scaled_features.shape if 'scaled_features' in locals() else 'unknown'}")
            raise
    
    def _run_ensemble_analysis(self, data: pd.DataFrame, features: np.ndarray, scaled_features: np.ndarray) -> Dict[str, Any]:
        """Run ensemble ML analysis on gene pairs."""
        logger.info("Running ensemble analysis...")
        results = {}
        
        # 1. Anomaly Detection with Isolation Forest (handles negative values well)
        logger.info("Running Isolation Forest...")
        anomaly_scores = self.isolation_forest.decision_function(scaled_features)
        anomaly_labels = self.isolation_forest.predict(scaled_features)
        
        results['anomaly_detection'] = {
            'scores': anomaly_scores,
            'labels': anomaly_labels,
            'outliers': np.sum(anomaly_labels == -1),
            'outlier_indices': np.where(anomaly_labels == -1)[0].tolist()
        }
        
        logger.info(f"Anomaly detection completed: {results['anomaly_detection']['outliers']} outliers detected")
        
        # 2. For algorithms that need non-negative data, use absolute values or shift
        logger.info("Preparing non-negative features for algorithms that require them...")
        
        # Method 1: Use absolute values (preserves magnitude)
        abs_features = np.abs(scaled_features)
        
        # Method 2: Shift to positive range (preserves relative relationships)
        min_val = np.min(scaled_features)
        if min_val < 0:
            shifted_features = scaled_features - min_val + 1e-10  # Small offset to avoid zeros
        else:
            shifted_features = scaled_features
        
        # 3. Density-based Clustering with DBSCAN
        logger.info("Running DBSCAN clustering...")
        try:
            distance_matrix = self.feature_engineering.create_similarity_matrix(scaled_features)
            dbscan_labels = DBSCAN(eps=0.3, min_samples=5, metric='precomputed').fit_predict(distance_matrix)
            
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = np.sum(dbscan_labels == -1)
            
            results['dbscan_clustering'] = {
                'labels': dbscan_labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'cluster_indices': self._get_cluster_indices(dbscan_labels)
            }
            
            logger.info(f"DBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
            
        except Exception as e:
            logger.warning(f"DBSCAN failed: {e}")
            results['dbscan_clustering'] = {
                'labels': np.full(len(scaled_features), -1),
                'n_clusters': 0,
                'n_noise': len(scaled_features),
                'cluster_indices': {}
            }
        
        # 4. Gaussian Mixture Model (can handle negative values)
        logger.info("Running Gaussian Mixture Model...")
        try:
            # Use the original scaled features since GMM can handle negative values
            gmm = GaussianMixture(
                n_components=3,
                random_state=self.random_state,
                max_iter=100
            )
            gmm_labels = gmm.fit_predict(scaled_features)
            gmm_probs = gmm.predict_proba(scaled_features)
            gmm_scores = np.max(gmm_probs, axis=1)
            
            results['gmm_clustering'] = {
                'labels': gmm_labels,
                'probabilities': gmm_probs,
                'confidence_scores': gmm_scores,
                'cluster_indices': self._get_cluster_indices(gmm_labels)
            }
            
            logger.info(f"GMM completed: {len(set(gmm_labels))} components")
            
        except Exception as e:
            logger.warning(f"GMM failed: {e}")
            results['gmm_clustering'] = {
                'labels': np.full(len(scaled_features), 0),
                'probabilities': np.ones((len(scaled_features), 1)),
                'confidence_scores': np.ones(len(scaled_features)),
                'cluster_indices': {'0': list(range(len(scaled_features)))}
            }
        
        # 5. Clustering Quality Metrics
        logger.info("Calculating clustering quality metrics...")
        try:
            if results['dbscan_clustering']['n_clusters'] > 1:
                silhouette_avg = silhouette_score(scaled_features, dbscan_labels)
                calinski_score = calinski_harabasz_score(scaled_features, dbscan_labels)
            else:
                silhouette_avg = -1
                calinski_score = -1
            
            results['clustering_quality'] = {
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_score,
                'n_clusters_found': results['dbscan_clustering']['n_clusters']
            }
            
            logger.info(f"Quality metrics - Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_score:.1f}")
            
        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            results['clustering_quality'] = {
                'silhouette_score': -1,
                'calinski_harabasz_score': -1,
                'n_clusters_found': 0
            }
        
        # 6. Ensemble Consensus
        logger.info("Calculating ensemble consensus...")
        consensus = self._compute_ensemble_consensus(
            anomaly_labels, 
            results['dbscan_clustering']['labels'], 
            results['gmm_clustering']['labels'], 
            results['gmm_clustering']['confidence_scores']
        )
        results['ensemble_consensus'] = consensus
        
        return results

    def _get_cluster_indices(self, labels: np.ndarray) -> Dict[int, List[int]]:
        """Get indices of data points in each cluster."""
        cluster_indices = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label != -1:  # Skip noise points
                indices = np.where(labels == label)[0].tolist()
                cluster_indices[int(label)] = indices
        
        return cluster_indices
    
    def _compute_ensemble_consensus(self, 
                                  anomaly_labels: np.ndarray,
                                  dbscan_labels: np.ndarray, 
                                  gmm_labels: np.ndarray,
                                  gmm_scores: np.ndarray) -> Dict[str, Any]:
        """Compute consensus across ensemble methods."""
        n_samples = len(anomaly_labels)
        
        # Create consensus matrix
        consensus_matrix = np.zeros((n_samples, n_samples))
        
        # Add clustering agreement from each method
        methods = [
            ('anomaly', anomaly_labels),
            ('dbscan', dbscan_labels), 
            ('gmm', gmm_labels)
        ]
        
        for method_name, labels in methods:
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if labels[i] == labels[j] and labels[i] != -1:
                        consensus_matrix[i, j] += 1
                        consensus_matrix[j, i] += 1
        
        # Normalize consensus matrix
        consensus_matrix /= len(methods)
        
        # Identify high-confidence pairs (agreement across multiple methods)
        high_confidence_threshold = 0.67  # Agreement in 2+ methods
        high_confidence_pairs = []
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if consensus_matrix[i, j] >= high_confidence_threshold:
                    high_confidence_pairs.append((i, j, consensus_matrix[i, j]))
        
        return {
            'consensus_matrix': consensus_matrix,
            'high_confidence_pairs': high_confidence_pairs,
            'mean_consensus': np.mean(consensus_matrix),
            'consensus_distribution': np.bincount(consensus_matrix.astype(int).flatten())
        }
    
    def _apply_rules_ranking(self, data: pd.DataFrame) -> List[Tuple[Dict[str, Any], float]]:
        """Apply rules-based ranking to gene pairs."""
        # Convert DataFrame to list of dictionaries for rules engine
        pairs_data = data.to_dict('records')
        
        # Rank pairs using rules engine
        ranked_pairs = self.rules_engine.rank_gene_pairs(pairs_data)
        
        return ranked_pairs
    
    def _generate_recommendations(self, 
                                ensemble_results: Dict[str, Any],
                                rules_ranking: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
        """Generate final recommendations combining ML and rules-based approaches."""
        recommendations = []
        
        # Get top-ranked pairs from rules
        top_rules_pairs = rules_ranking[:20]  # Top 20 from rules
        
        # Get high-confidence pairs from ensemble
        high_confidence_indices = set()
        for i, j, confidence in ensemble_results['ensemble_consensus']['high_confidence_pairs']:
            high_confidence_indices.add(i)
            high_confidence_indices.add(j)
        
        # Get anomaly outliers (potential interesting pairs)
        outlier_indices = set(ensemble_results['anomaly_detection']['outlier_indices'])
        
        # Combine approaches
        for idx, (pair_data, rules_score) in enumerate(top_rules_pairs):
            gene_a = pair_data.get('GeneAName', 'Unknown')
            gene_b = pair_data.get('GeneBName', 'Unknown')
            
            # Check if this pair is also identified by ML methods
            ml_confidence = 0.0
            is_high_confidence = idx in high_confidence_indices
            is_outlier = idx in outlier_indices
            
            if is_high_confidence:
                ml_confidence = 0.8
            elif is_outlier:
                ml_confidence = 0.6
            
            # Combined score (weighted average)
            combined_score = 0.7 * rules_score + 0.3 * ml_confidence
            
            recommendation = {
                'gene_a': gene_a,
                'gene_b': gene_b,
                'rules_score': rules_score,
                'ml_confidence': ml_confidence,
                'combined_score': combined_score,
                'is_high_confidence': is_high_confidence,
                'is_outlier': is_outlier,
                'rank': idx + 1,
                'statistical_measures': {
                    'p_ss': pair_data.get('p_ss'),
                    'p_soth': pair_data.get('p_soth'),
                    'dz_ss_mean': pair_data.get('dz_ss_mean'),
                    'dz_soth_mean': pair_data.get('dz_soth_mean'),
                    'q_ss': pair_data.get('q_ss'),
                    'q_soth': pair_data.get('q_soth')
                }
            }
            
            recommendations.append(recommendation)
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Update ranks
        for i, rec in enumerate(recommendations):
            rec['rank'] = i + 1
        
        return recommendations[:50]  # Top 50 recommendations
    
    def _calculate_summary_stats(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the analysis."""
        stats = {
            'total_pairs': len(data),
            'outliers_detected': results['anomaly_detection']['outliers'],
            'clusters_found': results['dbscan_clustering']['n_clusters'],
            'silhouette_score': results['clustering_quality']['silhouette_score'],
            'positive_control_validated': self.rules_engine.validate_positive_control(
                self.rules_engine.rank_gene_pairs(data.to_dict('records'))
            )
        }
        
        # Add basic statistics from data
        if 'p_ss' in data.columns:
            stats['significant_pairs_ss'] = np.sum(data['p_ss'] < 0.05)
        
        if 'p_soth' in data.columns:
            stats['significant_pairs_soth'] = np.sum(data['p_soth'] < 0.01)
        
        if 'dz_ss_mean' in data.columns and 'dz_soth_mean' in data.columns:
            stats['large_effect_pairs'] = np.sum(
                (np.abs(data['dz_ss_mean']) > 0.5) & (np.abs(data['dz_soth_mean']) > 1.0)
            )
        
        return stats
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the analysis."""
        if not self.analysis_results:
            return {'status': 'no_analysis'}
        
        return {
            'status': 'completed',
            'summary_stats': self.analysis_results['summary_stats'],
            'recommendations_count': len(self.analysis_results['recommendations']),
            'top_recommendations': self.analysis_results['recommendations'][:10],
            'clustering_quality': self.analysis_results['ensemble_analysis']['clustering_quality'],
            'feature_engineering_summary': self.feature_engineering.get_feature_summary()
        }
    
    def save_results(self, filepath: str) -> None:
        """Save analysis results to file."""
        import json
        
        results_to_save = {
            'analysis_summary': self.get_analysis_summary(),
            'recommendations': self.analysis_results.get('recommendations', []),
            'summary_stats': self.analysis_results.get('summary_stats', {}),
            'rules_engine_config': self.rules_engine.get_rule_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load analysis results from file."""
        import json
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.analysis_results = {
            'recommendations': results.get('recommendations', []),
            'summary_stats': results.get('summary_stats', {})
        }
        
        logger.info(f"Analysis results loaded from {filepath}")
        return results