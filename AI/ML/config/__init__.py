"""
Configuration Module

Provides centralized configuration management for the Gene Pair ML Analysis System.
"""

from .settings import (
    DEFAULT_CONFIG,
    DATABASE_CONFIG_TEMPLATE,
    RULES_CONFIG_TEMPLATE,
    load_config,
    save_config,
    get_database_config,
    get_analysis_config,
    get_web_config
)

__all__ = [
    'DEFAULT_CONFIG',
    'DATABASE_CONFIG_TEMPLATE',
    'RULES_CONFIG_TEMPLATE',
    'load_config',
    'save_config',
    'get_database_config',
    'get_analysis_config',
    'get_web_config'
]