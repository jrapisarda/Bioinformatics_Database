"""
System Configuration Settings

Global configuration for the Gene Pair ML Analysis System.
"""

import os
from pathlib import Path
from typing import Dict, Any 

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
CONFIG_DIR = BASE_DIR / 'config'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    'system': {
        'name': 'Gene Pair ML Analysis System',
        'version': '1.0.0',
        'debug': False,
        'log_level': 'INFO'
    },
    'analysis': {
        'n_features': 5,
        'contamination': 0.1,
        'random_state': 42,
        'positive_control': ['MS4A4A', 'CD86'],
        'min_significant_pairs': 10
    },
    'rules_engine': {
        'default_rules_enabled': True,
        'allow_custom_rules': True,
        'max_custom_rules': 20,
        'validation_threshold': 0.8  # Top 20% for positive control
    },
    'visualization': {
        'chart_height': 400,
        'chart_width': 800,
        'color_palette': {
            'primary': '#2E8B8B',
            'secondary': '#F5F5DC',
            'accent': '#CD853F',
            'highlight': '#FF6B6B',
            'success': '#4ECDC4',
            'warning': '#FFE66D'
        }
    },
    'database': {
        'default_driver': 'ODBC Driver 17 for SQL Server',
        'connection_timeout': 30,
        'pool_size': 5,
        'max_overflow': 10
    },
    'web_interface': {
        'max_file_size': 16 * 1024 * 1024,  # 16MB
        'allowed_extensions': ['.xlsx', '.xls', '.csv', '.json'],
        'session_timeout': 3600,  # 1 hour
        'max_recommendations_display': 100
    }
}

# Database configuration template
DATABASE_CONFIG_TEMPLATE = {
    'driver': 'ODBC Driver 17 for SQL Server',
    'server': 'localhost',
    'database': 'BioinformaticsDB',
    'trusted_connection': False,
    'username': 'sa',
    'password': 'password',
    'connection_timeout': 30
}

# Rules configuration template
RULES_CONFIG_TEMPLATE = {
    'positive_control': ['MS4A4A', 'CD86'],
    'rules': [
        {
            'name': 'Statistical Significance',
            'condition': '(p_ss < 0.1) AND (p_soth < 0.01)',
            'weight': 0.25,
            'description': 'Strong statistical significance in both conditions',
            'enabled': True
        },
        {
            'name': 'Effect Size Strength',
            'condition': '(abs_dz_ss > 0.3) AND (abs_dz_soth > 1.0)',
            'weight': 0.30,
            'description': 'Substantial effect sizes indicating biological relevance',
            'enabled': True
        },
        {
            'name': 'Z-Score Strength',
            'condition': '(abs(dz_ss_z) > 1.5) AND (abs(dz_soth_z) > 3.0)',
            'weight': 0.20,
            'description': 'Strong standardized effect sizes',
            'enabled': True
        },
        {
            'name': 'FDR Correction',
            'condition': '(q_ss < 0.2) AND (q_soth < 0.01)',
            'weight': 0.15,
            'description': 'False discovery rate controlled results',
            'enabled': True
        },
        {
            'name': 'Consistency',
            'condition': '(dz_ss_I2 < 50) OR (dz_soth_I2 < 75)',
            'weight': 0.10,
            'description': 'Low heterogeneity indicating consistent effects',
            'enabled': True
        }
    ]
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file or return defaults."""
    if config_path and os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to file."""
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    config = load_config()
    return config.get('database', DATABASE_CONFIG_TEMPLATE.copy())


def get_analysis_config() -> Dict[str, Any]:
    """Get analysis configuration."""
    config = load_config()
    return config.get('analysis', DEFAULT_CONFIG['analysis'].copy())


def get_web_config() -> Dict[str, Any]:
    """Get web interface configuration."""
    config = load_config()
    return config.get('web_interface', DEFAULT_CONFIG['web_interface'].copy())