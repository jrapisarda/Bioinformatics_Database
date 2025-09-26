"""
Interactive Plots Module

Provides interactive visualization components for the web interface.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class InteractivePlotter:
    """Create interactive plots for web interface."""
    
    def __init__(self):
        """Initialize the interactive plotter."""
        self.color_palette = {
            'primary': '#2E8B8B',
            'secondary': '#F5F5DC',
            'accent': '#CD853F',
            'highlight': '#FF6B6B',
            'success': '#4ECDC4',
            'warning': '#FFE66D'
        }
    
    def create_recommendation_card(self, recommendation: Dict[str, Any]) -> str:
        """Create HTML card for a single recommendation."""
        rank = recommendation.get('rank', 0)
        gene_a = recommendation.get('gene_a', 'Unknown')
        gene_b = recommendation.get('gene_b', 'Unknown')
        combined_score = recommendation.get('combined_score', 0.0)
        rules_score = recommendation.get('rules_score', 0.0)
        ml_confidence = recommendation.get('ml_confidence', 0.0)
        is_high_confidence = recommendation.get('is_high_confidence', False)
        is_outlier = recommendation.get('is_outlier', False)
        
        # Create badges
        badges = []
        if is_high_confidence:
            badges.append('<span class="badge bg-success">High Confidence</span>')
        if is_outlier:
            badges.append('<span class="badge bg-warning">Outlier</span>')
        
        badges_html = ' '.join(badges)
        
        html = f"""
        <div class="col-md-6 col-lg-4 mb-3">
            <div class="card h-100 recommendation-card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <span class="recommendation-rank">#{rank}</span>
                    <div class="gene-names">{gene_a} - {gene_b}</div>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-6">
                            <small class="text-muted">Combined Score</small>
                            <div class="score-display">{combined_score:.3f}</div>
                        </div>
                        <div class="col-6">
                            <small class="text-muted">Rules Score</small>
                            <div class="score-display-secondary">{rules_score:.3f}</div>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-6">
                            <small class="text-muted">ML Confidence</small>
                            <div class="score-display-secondary">{ml_confidence:.3f}</div>
                        </div>
                        <div class="col-6">
                            {badges_html}
                        </div>
                    </div>
                </div>
                <div class="card-footer">
                    <button class="btn btn-sm btn-outline-primary w-100" 
                            onclick="showGenePairDetails('{gene_a}', '{gene_b}')">
                        <i class="fas fa-info-circle"></i> View Details
                    </button>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def create_analysis_summary_card(self, summary_stats: Dict[str, Any]) -> str:
        """Create summary statistics card."""
        total_pairs = summary_stats.get('total_pairs', 0)
        outliers_detected = summary_stats.get('outliers_detected', 0)
        clusters_found = summary_stats.get('clusters_found', 0)
        silhouette_score = summary_stats.get('silhouette_score', 0)
        positive_control_validated = summary_stats.get('positive_control_validated', False)
        
        html = f"""
        <div class="card mb-4">
            <div class="card-header">
                <h5 class="mb-0">Analysis Summary</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3 text-center">
                        <div class="stat-number">{total_pairs:,}</div>
                        <div class="stat-label">Gene Pairs</div>
                    </div>
                    <div class="col-md-3 text-center">
                        <div class="stat-number">{outliers_detected}</div>
                        <div class="stat-label">Outliers</div>
                    </div>
                    <div class="col-md-3 text-center">
                        <div class="stat-number">{clusters_found}</div>
                        <div class="stat-label">Clusters</div>
                    </div>
                    <div class="col-md-3 text-center">
                        <div class="stat-number">{silhouette_score:.3f}</div>
                        <div class="stat-label">Silhouette Score</div>
                    </div>
                </div>
                
                <div class="mt-3 text-center">
                    <div class="validation-status {'text-success' if positive_control_validated else 'text-danger'}">
                        <i class="fas fa-{'check-circle' if positive_control_validated else 'times-circle'}"></i>
                        Positive Control Validation: {'PASSED' if positive_control_validated else 'FAILED'}
                    </div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def create_gene_pair_detail_modal(self, gene_a: str, gene_b: str, 
                                    analysis_data: Dict[str, Any]) -> str:
        """Create detailed modal for gene pair."""
        # Find the recommendation data
        recommendation = None
        for rec in analysis_data.get('recommendations', []):
            if rec['gene_a'] == gene_a and rec['gene_b'] == gene_b:
                recommendation = rec
                break
        
        if not recommendation:
            return '<div class="alert alert-warning">Recommendation not found</div>'
        
        stats = recommendation.get('statistical_measures', {})
        
        html = f"""
        <div class="modal fade" id="genePairModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Gene Pair Analysis: {gene_a} - {gene_b}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Ranking Information</h6>
                                <table class="table table-sm">
                                    <tr><td>Rank</td><td>#{recommendation['rank']}</td></tr>
                                    <tr><td>Combined Score</td><td>{recommendation['combined_score']:.4f}</td></tr>
                                    <tr><td>Rules Score</td><td>{recommendation['rules_score']:.4f}</td></tr>
                                    <tr><td>ML Confidence</td><td>{recommendation['ml_confidence']:.4f}</td></tr>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <h6>Statistical Measures</h6>
                                <table class="table table-sm">
                                    <tr><td>P-value (SS)</td><td>{stats.get('p_ss', 'N/A')}</td></tr>
                                    <tr><td>P-value (SOTH)</td><td>{stats.get('p_soth', 'N/A')}</td></tr>
                                    <tr><td>Effect Size (SS)</td><td>{stats.get('dz_ss_mean', 'N/A')}</td></tr>
                                    <tr><td>Effect Size (SOTH)</td><td>{stats.get('dz_soth_mean', 'N/A')}</td></tr>
                                </table>
                            </div>
                        </div>
                        
                        <div class="mt-3">
                            <h6>Analysis Flags</h6>
                            <div class="d-flex gap-2">
                                {('<span class="badge bg-success">High Confidence</span>' if recommendation['is_high_confidence'] else '')}
                                {('<span class="badge bg-warning">Outlier</span>' if recommendation['is_outlier'] else '')}
                            </div>
                        </div>
                        
                        <div class="mt-3">
                            <div id="pair-detail-chart" style="height: 300px;">
                                <!-- Chart will be rendered here -->
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="exportPairData('{gene_a}', '{gene_b}')">
                            <i class="fas fa-download"></i> Export Data
                        </button>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def create_progress_bar(self, progress: int, status: str) -> str:
        """Create progress bar HTML."""
        return f"""
        <div class="progress mb-3">
            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                 role="progressbar" style="width: {progress}%">
                {progress}% - {status}
            </div>
        </div>
        """
    
    def create_loading_spinner(self, message: str = "Loading...") -> str:
        """Create loading spinner HTML."""
        return f"""
        <div class="text-center py-4">
            <div class="loading-spinner"></div>
            <p class="mt-3 text-muted">{message}</p>
        </div>
        """
    
    def create_alert(self, message: str, alert_type: str = 'info') -> str:
        """Create alert HTML."""
        alert_class = {
            'info': 'alert-info',
            'success': 'alert-success',
            'warning': 'alert-warning',
            'danger': 'alert-danger'
        }.get(alert_type, 'alert-info')
        
        return f"""
        <div class="alert {alert_class} alert-dismissible fade show" role="alert">
            {message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
        """
    
    def create_export_options(self, session_id: str) -> str:
        """Create export options HTML."""
        return f"""
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Export Options</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Data Export</h6>
                        <div class="d-grid gap-2">
                            <a href="{{ url_for('export_results', session_id='${session_id}') }}" 
                               class="btn btn-outline-primary">
                                <i class="fas fa-file-excel"></i> Excel Format
                            </a>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6>Analysis Results</h6>
                        <div class="d-grid gap-2">
                            <button class="btn btn-outline-secondary" onclick="exportJSON('${session_id}')">
                                <i class="fas fa-file-code"></i> JSON Format
                            </button>
                            <button class="btn btn-outline-secondary" onclick="exportCharts('${session_id}')">
                                <i class="fas fa-chart-bar"></i> Charts (PNG)
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """


class ResultsDashboard:
    """Create comprehensive results dashboard."""
    
    def __init__(self):
        """Initialize the results dashboard."""
        self.plotter = InteractivePlotter()
    
    def create_dashboard(self, analysis_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Create complete dashboard with all components."""
        
        dashboard = {
            'summary_stats': self._create_summary_stats_html(analysis_data),
            'top_recommendations': self._create_top_recommendations_html(analysis_data),
            'visualization_tabs': self._create_visualization_tabs_html(),
            'export_section': self.plotter.create_export_options(session_id),
            'recommendations_grid': self._create_recommendations_grid_html(analysis_data),
            'modal_template': self._create_modal_template_html()
        }
        
        return dashboard
    
    def _create_summary_stats_html(self, analysis_data: Dict[str, Any]) -> str:
        """Create summary statistics HTML section."""
        summary_stats = analysis_data.get('summary_stats', {})
        return self.plotter.create_analysis_summary_card(summary_stats)
    
    def _create_top_recommendations_html(self, analysis_data: Dict[str, Any]) -> str:
        """Create top recommendations HTML section."""
        recommendations = analysis_data.get('recommendations', [])[:10]
        
        html = '<div class="row">'
        for rec in recommendations:
            html += self.plotter.create_recommendation_card(rec)
        html += '</div>'
        
        return html
    
    def _create_visualization_tabs_html(self) -> str:
        """Create visualization tabs HTML."""
        return """
        <div class="card">
            <div class="card-header">
                <ul class="nav nav-tabs card-header-tabs" id="viz-tabs" role="tablist">
                    <li class="nav-item">
                        <button class="nav-link active" id="boxplot-tab" data-bs-toggle="tab" 
                                data-bs-target="#boxplot" type="button">
                            <i class="fas fa-chart-bar"></i> Effect Sizes
                        </button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" id="scatter-tab" data-bs-toggle="tab" 
                                data-bs-target="#scatter" type="button">
                            <i class="fas fa-braille"></i> Correlation Plot
                        </button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" id="clustering-tab" data-bs-toggle="tab" 
                                data-bs-target="#clustering" type="button">
                            <i class="fas fa-project-diagram"></i> Clustering
                        </button>
                    </li>
                    <li class="nav-item">
                        <button class="nav-link" id="ranking-tab" data-bs-toggle="tab" 
                                data-bs-target="#ranking" type="button">
                            <i class="fas fa-trophy"></i> Rankings
                        </button>
                    </li>
                </ul>
            </div>
            <div class="card-body">
                <div class="tab-content">
                    <div class="tab-pane fade show active" id="boxplot" role="tabpanel">
                        <div class="chart-container">
                            <div id="boxplot-chart"></div>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="scatter" role="tabpanel">
                        <div class="chart-container">
                            <div id="scatter-chart"></div>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="clustering" role="tabpanel">
                        <div class="chart-container">
                            <div id="clustering-chart"></div>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="ranking" role="tabpanel">
                        <div class="chart-container">
                            <div id="ranking-chart"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _create_recommendations_grid_html(self, analysis_data: Dict[str, Any]) -> str:
        """Create recommendations grid HTML."""
        recommendations = analysis_data.get('recommendations', [])
        
        html = '<div class="row" id="recommendations-grid">'
        for rec in recommendations:
            html += self.plotter.create_recommendation_card(rec)
        html += '</div>'
        
        return html
    
    def _create_modal_template_html(self) -> str:
        """Create modal template HTML."""
        return """
        <!-- Gene Pair Detail Modal -->
        <div class="modal fade" id="genePairModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Gene Pair Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="modal-body-content">
                        <!-- Content will be populated dynamically -->
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
        """