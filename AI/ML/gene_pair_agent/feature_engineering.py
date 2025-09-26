"""
Feature Engineering for Gene Pair Analysis

Creates derived features and performs dimensionality reduction to enhance ML performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Feature engineering and dimensionality reduction for gene pair analysis."""
    
    def __init__(self, n_components: int = 5, random_state: int = 42):
        """Initialize feature engineering with PCA and scaling."""
        self.n_components = n_components
        self.random_state = random_state
        
        # Use RobustScaler which is better for data with outliers
        self.scaler = RobustScaler()
        
        # Use PCA with different solver for better numerical stability
        self.pca = PCA(
            n_components=n_components, 
            random_state=random_state,
            svd_solver='full'  # More stable for small datasets
        )
        
        self.feature_names = []
        self.is_fitted = False
        
        logger.info(f"Initialized FeatureEngineering with n_components={n_components}")
    
    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from base statistical measures."""
        logger.info("Creating derived features...")
        features_df = data.copy()
        
        # Handle missing values first
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in features_df.columns:
                # Replace infinite values with NaN
                features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Fill missing values with median (robust to outliers)
                if features_df[col].isnull().sum() > 0:
                    median_val = features_df[col].median()
                    features_df[col] = features_df[col].fillna(median_val)
                    logger.debug(f"Filled {features_df[col].isnull().sum()} missing values in {col} with median {median_val}")
        
        # Effect size ratios and differences
        if 'dz_ss_mean' in data.columns and 'dz_soth_mean' in data.columns:
            features_df['effect_size_ratio'] = self._safe_ratio(
                data['dz_ss_mean'], data['dz_soth_mean']
            )
            features_df['effect_size_diff'] = data['dz_ss_mean'] - data['dz_soth_mean']
            features_df['effect_size_sum'] = data['dz_ss_mean'] + data['dz_soth_mean']
            features_df['effect_size_product'] = data['dz_ss_mean'] * data['dz_soth_mean']
        
        # Statistical significance composite measures
        if 'p_ss' in data.columns and 'p_soth' in data.columns:
            # Handle very small p-values
            p_ss_safe = np.maximum(data['p_ss'], 1e-10)
            p_soth_safe = np.maximum(data['p_soth'], 1e-10)
            
            features_df['log_p_ss'] = -np.log10(p_ss_safe)
            features_df['log_p_soth'] = -np.log10(p_soth_safe)
            features_df['p_composite'] = self._safe_p_composite(p_ss_safe, p_soth_safe)
            features_df['p_harmonic_mean'] = 2 / (1/p_ss_safe + 1/p_soth_safe)
        
        # Confidence interval features
        if all(col in data.columns for col in ['dz_ss_ci_low', 'dz_ss_ci_high']):
            features_df['ci_ss_width'] = data['dz_ss_ci_high'] - data['dz_ss_ci_low']
            features_df['ci_ss_center'] = (data['dz_ss_ci_high'] + data['dz_ss_ci_low']) / 2
            features_df['ci_ss_relative_width'] = features_df['ci_ss_width'] / (np.abs(features_df['ci_ss_center']) + 1e-10)
        
        if all(col in data.columns for col in ['dz_soth_ci_low', 'dz_soth_ci_high']):
            features_df['ci_soth_width'] = data['dz_soth_ci_high'] - data['dz_soth_ci_low']
            features_df['ci_soth_center'] = (data['dz_soth_ci_high'] + data['dz_soth_ci_low']) / 2
            features_df['ci_soth_relative_width'] = features_df['ci_soth_width'] / (np.abs(features_df['ci_soth_center']) + 1e-10)
        
        # Heterogeneity features
        if 'dz_ss_I2' in data.columns and 'dz_soth_I2' in data.columns:
            features_df['i2_ratio'] = self._safe_ratio(
                data['dz_ss_I2'], data['dz_soth_I2']
            )
            features_df['i2_max'] = np.maximum(data['dz_ss_I2'], data['dz_soth_I2'])
            features_df['i2_min'] = np.minimum(data['dz_ss_I2'], data['dz_soth_I2'])
            features_df['i2_mean'] = (data['dz_ss_I2'] + data['dz_soth_I2']) / 2
        
        # Z-score features
        if 'dz_ss_z' in data.columns and 'dz_soth_z' in data.columns:
            features_df['z_composite'] = np.sqrt(
                data['dz_ss_z']**2 + data['dz_soth_z']**2
            )
            features_df['z_ratio'] = self._safe_ratio(
                np.abs(data['dz_ss_z']), np.abs(data['dz_soth_z'])
            )
            features_df['z_sum'] = data['dz_ss_z'] + data['dz_soth_z']
            features_df['z_product'] = data['dz_ss_z'] * data['dz_soth_z']
        
        # Sample size features
        if 'n_studies_ss' in data.columns and 'n_studies_soth' in data.columns:
            features_df['total_studies'] = data['n_studies_ss'] + data['n_studies_soth']
            features_df['study_ratio'] = self._safe_ratio(
                data['n_studies_ss'], data['n_studies_soth']
            )
            features_df['study_log_ratio'] = np.log(features_df['study_ratio'] + 1e-10)
        
        # FDR-adjusted features
        if 'q_ss' in data.columns and 'q_soth' in data.columns:
            q_ss_safe = np.maximum(data['q_ss'], 1e-10)
            q_soth_safe = np.maximum(data['q_soth'], 1e-10)
            
            features_df['log_q_ss'] = -np.log10(q_ss_safe)
            features_df['log_q_soth'] = -np.log10(q_soth_safe)
            features_df['q_composite'] = self._safe_q_composite(q_ss_safe, q_soth_safe)
            features_df['q_max'] = np.maximum(q_ss_safe, q_soth_safe)
            features_df['q_min'] = np.minimum(q_ss_safe, q_soth_safe)
        
        # Add absolute value features for algorithms that need positive input
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['pair_id', 'GeneAKey', 'GeneBKey']:
                features_df[f'{col}_abs'] = np.abs(features_df[col])
        
        logger.info(f"Created {len(features_df.columns) - len(data.columns)} derived features")
        return features_df
    
    def _safe_ratio(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Calculate ratio with protection against division by zero."""
        result = np.where(
            denominator != 0,
            numerator / denominator,
            np.where(numerator > 0, np.inf, np.where(numerator < 0, -np.inf, 0))
        )
        # Replace infinite values with reasonable bounds
        result = np.where(np.isinf(result), np.sign(result) * 1000, result)
        return result
    
    def _safe_p_composite(self, p1: pd.Series, p2: pd.Series) -> pd.Series:
        """Create composite p-value measure using Fisher's method."""
        # Fisher's method: -2 * sum(log(p_values))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            log_p1 = np.log(p1)
            log_p2 = np.log(p2)
            fisher_stat = -2 * (log_p1 + log_p2)
            # Convert back to p-value approximation
            result = np.exp(-fisher_stat / 2)
            # Handle any remaining issues
            result = np.where(np.isnan(result), 1.0, result)
            result = np.where(np.isinf(result), 0.0, result)
            return result
    
    def _safe_q_composite(self, q1: pd.Series, q2: pd.Series) -> pd.Series:
        """Create composite q-value using harmonic mean."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            result = 2 / (1/(q1 + 1e-10) + 1/(q2 + 1e-10))
            result = np.where(np.isnan(result), 1.0, result)
            result = np.where(np.isinf(result), 1.0, result)
            return result
    
    def select_features(self, data: pd.DataFrame, target_correlation: Optional[pd.Series] = None) -> pd.DataFrame:
        """Select most informative features using correlation analysis."""
        logger.info("Selecting informative features...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_data = data[numeric_cols].copy()
        
        # Remove features with too many missing values (>50%)
        missing_ratio = feature_data.isnull().sum() / len(feature_data)
        good_features = missing_ratio[missing_ratio < 0.5].index
        feature_data = feature_data[good_features]
        
        # Fill remaining missing values with median
        feature_data = feature_data.fillna(feature_data.median())
        
        # Remove features with zero variance
        zero_var_features = feature_data.columns[feature_data.var() == 0]
        feature_data = feature_data.drop(columns=zero_var_features)
        
        # Previously, features with variance below a small threshold (default 0.001) were removed.
        # To ensure that all informative columns, including those with low variance (such as binary or near-constant
        # features), are retained in the analysis, we no longer drop features based solely on low variance. Zero-variance
        # features (constant columns) are still removed above to avoid degenerate cases. Retaining low-variance
        # features allows new columns introduced for meta-analysis, such as direction consistency (often 0 or 1) and
        # combined effect sizes, to be considered by the ML model.
        
        logger.info(f"Selected {len(feature_data.columns)} features after cleaning")
        
        if target_correlation is not None and len(target_correlation) == len(feature_data):
            # Calculate correlation with target and select top features
            correlations = []
            for col in feature_data.columns:
                if feature_data[col].var() > 0:
                    try:
                        corr = abs(np.corrcoef(feature_data[col], target_correlation)[0, 1])
                        if not np.isnan(corr):
                            correlations.append((col, corr))
                    except:
                        pass
            
            # Sort by absolute correlation and select top features
            if correlations:
                correlations.sort(key=lambda x: x[1], reverse=True)
                top_features = [col for col, _ in correlations[:20]]  # Top 20 features
                
                feature_data = feature_data[top_features]
                logger.info(f"Selected top {len(top_features)} features by correlation")
        
        self.feature_names = feature_data.columns.tolist()
        return feature_data
    
    def fit_transform(self, data: pd.DataFrame, target_correlation: Optional[pd.Series] = None) -> np.ndarray:
        """Fit the feature engineering pipeline and transform data."""
        logger.info(f"Fitting feature engineering pipeline to data with shape {data.shape}")
        
        # Create derived features
        features_df = self.create_derived_features(data)
        logger.info(f"Derived features created: {features_df.shape}")
        
        # Select informative features
        features_df = self.select_features(features_df, target_correlation)
        logger.info(f"Features selected: {features_df.shape}")
        
        # Check for any remaining issues
        if features_df.isnull().sum().sum() > 0:
            logger.warning(f"Still have {features_df.isnull().sum().sum()} missing values, filling with zeros")
            features_df = features_df.fillna(0)
        
        if np.any(np.isinf(features_df.values)):
            logger.warning("Infinite values detected, replacing with zeros")
            features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Log feature statistics
        logger.info(f"Feature statistics - Min: {features_df.min().min():.3f}, Max: {features_df.max().max():.3f}")
        logger.info(f"Feature statistics - Mean: {features_df.mean().mean():.3f}, Std: {features_df.std().mean():.3f}")
        
        # Scale features
        try:
            scaled_features = self.scaler.fit_transform(features_df)
            logger.info(f"Features scaled successfully: {scaled_features.shape}")
            logger.info(f"Scaled features - Min: {scaled_features.min():.3f}, Max: {scaled_features.max():.3f}")
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            logger.error(f"Features shape: {features_df.shape}")
            logger.error(f"Features sample:\n{features_df.head()}")
            raise
        
        # Apply PCA
        try:
            pca_features = self.pca.fit_transform(scaled_features)
            logger.info(f"PCA completed: {pca_features.shape}")
            logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        except Exception as e:
            logger.error(f"Error in PCA: {e}")
            logger.error(f"Scaled features shape: {scaled_features.shape}")
            logger.error(f"Scaled features sample:\n{scaled_features[:5]}")
            raise
        
        self.is_fitted = True
        logger.info(f"Feature engineering fitted: {features_df.shape[1]} -> {self.n_components} components")
        
        return pca_features
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineering must be fitted before transform")
        
        logger.info(f"Transforming data with shape {data.shape}")
        
        # Create derived features
        features_df = self.create_derived_features(data)
        
        # Ensure we have the same features as during fitting
        available_features = [f for f in self.feature_names if f in features_df.columns]
        features_df = features_df[available_features]
        
        # Fill missing features with zeros if any
        missing_features = set(self.feature_names) - set(available_features)
        if missing_features:
            for feature in missing_features:
                features_df[feature] = 0.0
            features_df = features_df[self.feature_names]
        
        # Handle any remaining data issues
        if features_df.isnull().sum().sum() > 0:
            logger.warning(f"Filling {features_df.isnull().sum().sum()} missing values with zeros")
            features_df = features_df.fillna(0)
        
        if np.any(np.isinf(features_df.values)):
            logger.warning("Replacing infinite values with zeros")
            features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Apply PCA
        pca_features = self.pca.transform(scaled_features)
        
        return pca_features
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from PCA components."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineering must be fitted before getting importance")
        
        # Get PCA component loadings
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        
        importance_df = pd.DataFrame(
            loadings,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.feature_names
        )
        
        return importance_df
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each PCA component."""
        if not self.is_fitted:
            raise ValueError("FeatureEngineering must be fitted before getting variance ratio")
        
        return self.pca.explained_variance_ratio_
    
    def create_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Create pairwise similarity matrix for clustering."""
        # Use cosine similarity for robustness
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(features)
        
        # Convert to distance matrix for clustering
        distance_matrix = 1 - similarity_matrix
       
        distance_matrix = np.clip(distance_matrix, 0, 2)  # force ∈ [0, 2]
        np.fill_diagonal(distance_matrix, 0)              # exact zeros on diag   
        
        return distance_matrix
    
    def detect_outliers(self, features: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using isolation forest."""
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.random_state
        )
        
        outlier_labels = iso_forest.fit_predict(features)
        
        # Convert to boolean array (True for outliers)
        return outlier_labels == -1
    
    def cluster_features(self, features: np.ndarray, eps: float = 0.3, min_samples: int = 5) -> np.ndarray:
        """Cluster features using DBSCAN."""
        distance_matrix = self.create_similarity_matrix(features)
        
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed'
        )
        
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        return cluster_labels
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering process."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        explained_variance = self.get_explained_variance_ratio()
        
        return {
            'status': 'fitted',
            'n_original_features': len(self.feature_names),
            'n_components': self.n_components,
            'explained_variance_ratio': explained_variance.tolist(),
            'total_explained_variance': float(np.sum(explained_variance)),
            'feature_names': self.feature_names
        }