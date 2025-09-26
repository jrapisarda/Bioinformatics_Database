"""
Gene Pair ML Analysis Agent

A comprehensive system for analyzing gene pair correlations using unsupervised machine learning
and rules-based ranking for bioinformatics research in sepsis and septic shock contexts.
"""

__version__ = "1.0.0"
__author__ = "AI Agent System"

from .gene_pair_analyzer import GenePairAnalyzer
from .rules_engine import RulesEngine, Rule
from .feature_engineering import FeatureEngineering
from .meta_analysis_processor import MetaAnalysisProcessor
from .database_connector import DatabaseConnector

__all__ = [
    'GenePairAnalyzer',
    'RulesEngine', 
    'Rule',
    'FeatureEngineering',
    'MetaAnalysisProcessor',
    'DatabaseConnector'
]