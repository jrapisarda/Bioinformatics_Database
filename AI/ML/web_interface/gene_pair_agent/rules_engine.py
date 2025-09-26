"""
Rules Engine for Gene Pair Ranking

Implements a configurable rules-based ranking system with visual rule management.
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """Represents a single ranking rule with condition and weight."""
    name: str
    condition: str
    weight: float
    description: str = ""
    enabled: bool = True
    
    def __post_init__(self):
        """Validate rule parameters."""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Rule weight must be between 0 and 1, got {self.weight}")
        if not self.name:
            raise ValueError("Rule name cannot be empty")
        if not self.condition:
            raise ValueError("Rule condition cannot be empty")


class RulesEngine:
    """Configurable rules-based ranking system for gene pairs."""
    
    DEFAULT_RULES = [
        Rule(
            name="Statistical Significance",
            condition="(p_ss < 0.1) and (p_soth < 0.01)",
            weight=0.25,
            description="Strong statistical significance in both conditions"
        ),
        Rule(
            name="Effect Size Strength", 
            condition="(abs_dz_ss > 0.3) and (abs_dz_soth > 1.0)",
            weight=0.30,
            description="Substantial effect sizes indicating biological relevance"
        ),
        Rule(
            name="Z-Score Strength",
            condition="(abs(dz_ss_z) > 1.5) and (abs(dz_soth_z) > 3.0)", 
            weight=0.20,
            description="Strong standardized effect sizes"
        ),
        Rule(
            name="FDR Correction",
            condition="(q_ss < 0.2) and (q_soth < 0.01)",
            weight=0.15,
            description="False discovery rate controlled results"
        ),
        Rule(
            name="Consistency",
            condition="(dz_ss_I2 < 50) or (dz_soth_I2 < 75)",
            weight=0.10,
            description="Low heterogeneity indicating consistent effects"
        )
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the rules engine with default or custom rules."""
        self.rules: List[Rule] = []
        self.positive_control = ("MS4A4A", "CD86")  # Configurable positive control
        
        if config_path:
            self.load_rules_from_config(config_path)
        else:
            self.rules = self.DEFAULT_RULES.copy()
            
        logger.info(f"Initialized RulesEngine with {len(self.rules)} rules")
    
    def load_rules_from_config(self, config_path: str) -> None:
        """Load rules from a JSON configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.rules = []
            for rule_data in config.get('rules', []):
                rule = Rule(**rule_data)
                self.rules.append(rule)
                
            # Load positive control if specified
            if 'positive_control' in config:
                self.positive_control = tuple(config['positive_control'])
                
            logger.info(f"Loaded {len(self.rules)} rules from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load rules from {config_path}: {e}")
            logger.info("Falling back to default rules")
            self.rules = self.DEFAULT_RULES.copy()
    
    def save_rules_to_config(self, config_path: str) -> None:
        """Save current rules to a JSON configuration file."""
        config = {
            'rules': [
                {
                    'name': rule.name,
                    'condition': rule.condition,
                    'weight': rule.weight,
                    'description': rule.description,
                    'enabled': rule.enabled
                }
                for rule in self.rules
            ],
            'positive_control': list(self.positive_control)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Saved {len(self.rules)} rules to {config_path}")
    
    def add_rule(self, rule: Rule) -> None:
        """Add a new rule to the engine."""
        if any(r.name == rule.name for r in self.rules):
            raise ValueError(f"Rule with name '{rule.name}' already exists")
        
        self.rules.append(rule)
        logger.info(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name. Returns True if removed."""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        removed = len(self.rules) < initial_count
        
        if removed:
            logger.info(f"Removed rule: {rule_name}")
        else:
            logger.warning(f"Rule not found: {rule_name}")
            
        return removed
    
    def update_rule(self, rule_name: str, **kwargs) -> bool:
        """Update an existing rule. Returns True if updated."""
        for rule in self.rules:
            if rule.name == rule_name:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                    else:
                        raise ValueError(f"Rule has no attribute '{key}'")
                
                logger.info(f"Updated rule: {rule_name}")
                return True
        
        logger.warning(f"Rule not found for update: {rule_name}")
        return False
    
    def evaluate_rule(self, rule: Rule, data: Dict[str, Any]) -> float:
        """Evaluate a single rule against data and return score."""
        if not rule.enabled:
            return 0.0
            
        try:
            # Replace Python operators with safer alternatives
            condition = rule.condition
            
            # Replace 'AND' with 'and' and 'OR' with 'or' for Python syntax
            condition = condition.replace('AND', 'and').replace('OR', 'or')
            
            # Create safe evaluation environment
            eval_env = {
                'abs': abs,
                'min': min,
                'max': max,
                'log': np.log,
                'sqrt': np.sqrt,
                'exp': np.exp,
                'np': np,
                'pd': pd,
            }
            
            # Add data variables to environment
            for key, value in data.items():
                if isinstance(value, (int, float, np.number)):
                    eval_env[key] = float(value)
                else:
                    eval_env[key] = value
            
            # Evaluate condition
            result = eval(condition, {"__builtins__": {}}, eval_env)
            
            # Return weighted score if condition is met
            return rule.weight if bool(result) else 0.0
            
        except Exception as e:
            logger.warning(f"Error evaluating rule '{rule.name}': {e}")
            logger.warning(f"Condition: {rule.condition}")
            logger.warning(f"Data keys: {list(data.keys())}")
            return 0.0
    
    def calculate_ranking_score(self, data: Dict[str, Any]) -> float:
        """Calculate total ranking score for a gene pair."""
        total_score = 0.0
        enabled_rules = [r for r in self.rules if r.enabled]
        
        if not enabled_rules:
            logger.warning("No enabled rules to evaluate")
            return 0.0
        
        for rule in enabled_rules:
            rule_score = self.evaluate_rule(rule, data)
            total_score += rule_score
            
        # Normalize by number of enabled rules
        normalized_score = total_score / len(enabled_rules)
        
        return min(1.0, max(0.0, normalized_score))  # Ensure [0,1] range
    
    def rank_gene_pairs(self, pairs_data: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """Rank multiple gene pairs using all enabled rules."""
        ranked_pairs = []
        
        for pair_data in pairs_data:
            score = self.calculate_ranking_score(pair_data)
            ranked_pairs.append((pair_data, score))
        
        # Sort by score in descending order
        ranked_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_pairs
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of all rules and their status."""
        return {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'positive_control': self.positive_control,
            'rules': [
                {
                    'name': rule.name,
                    'condition': rule.condition,
                    'weight': rule.weight,
                    'enabled': rule.enabled,
                    'description': rule.description
                }
                for rule in self.rules
            ]
        }
    
    def validate_positive_control(self, ranked_pairs: List[Tuple[Dict[str, Any], float]]) -> bool:
        """Validate that positive control ranks in top 20% as per requirements."""
        if not ranked_pairs:
            return False
            
        # Find positive control pair
        positive_control_data = None
        positive_control_score = None
        
        for pair_data, score in ranked_pairs:
            gene_a = pair_data.get('GeneAName', '')
            gene_b = pair_data.get('GeneBName', '')
            
            if (gene_a == self.positive_control[0] and gene_b == self.positive_control[1]) or \
               (gene_a == self.positive_control[1] and gene_b == self.positive_control[0]):
                positive_control_data = pair_data
                positive_control_score = score
                break
        
        if positive_control_score is None:
            logger.warning(f"Positive control {self.positive_control} not found in ranked pairs")
            return False
        
        # Calculate percentile rank
        total_pairs = len(ranked_pairs)
        better_pairs = sum(1 for _, score in ranked_pairs if score > positive_control_score)
        percentile_rank = (total_pairs - better_pairs) / total_pairs
        
        logger.info(f"Positive control {self.positive_control} ranks at {percentile_rank:.1%} percentile")
        
        # Requirement: should rank in top 20%
        return percentile_rank >= 0.80