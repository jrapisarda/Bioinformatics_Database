"""
Chart Generator for Gene Pair Analysis

Creates interactive Plotly charts for visualizing analysis results.
"""

import json
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate interactive charts for gene pair analysis results."""
    
    def __init__(self):
        """Initialize chart generator with default styling."""
        self.color_palette = {
            'primary': '#2E8B8B',      # Deep teal
            'secondary': '#F5F5DC',    # Warm ivory
            'accent': '#CD853F',       # Peru/sandy brown
            'highlight': '#FF6B6B',    # Coral red for outliers
            'success': '#4ECDC4',      # Turquoise
            'warning': '#FFE66D',      # Yellow
            'background': '#FFFFFF'
        }
        
        self.default_layout = {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': self.color_palette['background'],
            'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60}
        }
    
    def create_boxplot(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create box plots for effect sizes and statistical measures."""
        
        # Prepare data for box plots
        plot_data = []
        
        if 'dz_ss_mean' in data.columns:
            plot_data.append({
                'y': data['dz_ss_mean'].dropna(),
                'name': 'Septic Shock Effect Size',
                'marker_color': self.color_palette['primary'],
                'boxpoints': 'outliers'
            })
        
        if 'dz_soth_mean' in data.columns:
            plot_data.append({
                'y': data['dz_soth_mean'].dropna(),
                'name': 'Other Sepsis Effect Size',
                'marker_color': self.color_palette['accent'],
                'boxpoints': 'outliers'
            })
        
        fig = go.Figure()
        
        for trace_data in plot_data:
            fig.add_trace(go.Box(**trace_data))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Distribution of Effect Sizes by Condition',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Georgia, serif'}
            },
            xaxis_title='Condition',
            yaxis_title='Effect Size (Cohen\'s d)',
            **self.default_layout
        )
        
        # Add annotation for interpretation
        fig.add_annotation(
            text="Box plots show median, quartiles, and outliers.<br>Negative values indicate downregulation.",
            xref="paper", yref="paper",
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        return json.loads(fig.to_json())
    
    def create_scatter_plot(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create scatter plot of effect sizes with significance highlighting."""
        
        if 'dz_ss_mean' not in data.columns or 'dz_soth_mean' not in data.columns:
            logger.warning("Required columns not found for scatter plot")
            return self._empty_chart("Scatter plot requires effect size columns")
        
        # Prepare data
        x_data = data['dz_ss_mean']
        y_data = data['dz_soth_mean']
        
        # Color by significance
        colors = []
        if 'p_ss' in data.columns and 'p_soth' in data.columns:
            for i in range(len(data)):
                p_ss = data.iloc[i]['p_ss']
                p_soth = data.iloc[i]['p_soth']
                
                if p_ss < 0.05 and p_soth < 0.01:
                    colors.append(self.color_palette['highlight'])
                elif p_ss < 0.1 or p_soth < 0.05:
                    colors.append(self.color_palette['success'])
                else:
                    colors.append(self.color_palette['primary'])
        else:
            colors = [self.color_palette['primary']] * len(data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"Gene A: {row['GeneAName']}<br>Gene B: {row['GeneBName']}" 
                  for _, row in data.iterrows()],
            hovertemplate='<b>%{text}</b><br>' +
                         'Septic Shock: %{x:.3f}<br>' +
                         'Other Sepsis: %{y:.3f}<br>' +
                         '<extra></extra>',
            name='Gene Pairs'
        ))
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Gene Pair Effect Sizes: Septic Shock vs Other Sepsis',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Georgia, serif'}
            },
            xaxis_title='Septic Shock Effect Size',
            yaxis_title='Other Sepsis Effect Size',
            **self.default_layout
        )
        
        # Add legend for significance
        fig.add_annotation(
            text="<b>Legend:</b><br>🔴 Both conditions significant<br>🟢 One condition significant<br>🔵 Not significant",
            xref="paper", yref="paper",
            x=0.98, y=0.02, xanchor="right", yanchor="bottom",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        return json.loads(fig.to_json())
    
    def create_clustering_viz(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create clustering visualization."""
        
        if 'ensemble_analysis' not in results:
            return self._empty_chart("Clustering visualization requires analysis results")
        
        ensemble_results = results['ensemble_analysis']
        
        # Create subplots for different clustering methods
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Anomaly Detection', 'DBSCAN Clustering', 
                          'Gaussian Mixture', 'Consensus Matrix'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # 1. Anomaly detection results
        anomaly_labels = ensemble_results['anomaly_detection']['labels']
        anomaly_counts = pd.Series(anomaly_labels).value_counts()
        
        fig.add_trace(
            go.Bar(x=['Normal', 'Outlier'], 
                   y=[anomaly_counts.get(1, 0), anomaly_counts.get(-1, 0)],
                   marker_color=[self.color_palette['primary'], self.color_palette['highlight']]),
            row=1, col=1
        )
        
        # 2. DBSCAN clustering results
        dbscan_labels = ensemble_results['dbscan_clustering']['labels']
        dbscan_counts = pd.Series(dbscan_labels)
        cluster_counts = dbscan_counts[dbscan_counts != -1].value_counts().sort_index()
        noise_count = (dbscan_counts == -1).sum()
        
        fig.add_trace(
            go.Bar(x=list(cluster_counts.index) + ['Noise'],
                   y=list(cluster_counts.values) + [noise_count],
                   marker_color=self.color_palette['success']),
            row=1, col=2
        )
        
        # 3. GMM clustering results
        gmm_labels = ensemble_results['gmm_clustering']['labels']
        gmm_counts = pd.Series(gmm_labels).value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(x=[f'Cluster {i}' for i in gmm_counts.index],
                   y=list(gmm_counts.values),
                   marker_color=self.color_palette['accent']),
            row=2, col=1
        )
        
        # 4. Consensus matrix heatmap
        consensus_matrix = ensemble_results['ensemble_consensus']['consensus_matrix']
        
        fig.add_trace(
            go.Heatmap(z=consensus_matrix,
                       colorscale='RdYlBu',
                       showscale=True),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': 'Ensemble Clustering Analysis Results',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Georgia, serif'}
            },
            height=600,
            showlegend=False
        )
        
        return json.loads(fig.to_json())
    
    def create_ranking_chart(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create ranking visualization."""
        
        if 'recommendations' not in results:
            return self._empty_chart("Ranking chart requires recommendations")
        
        recommendations = results['recommendations'][:20]  # Top 20
        
        # Prepare data
        ranks = [r['rank'] for r in recommendations]
        combined_scores = [r['combined_score'] for r in recommendations]
        rules_scores = [r['rules_score'] for r in recommendations]
        ml_confidence = [r['ml_confidence'] for r in recommendations]
        
        gene_pairs = [f"{r['gene_a']}-{r['gene_b']}" for r in recommendations]
        
        fig = go.Figure()
        
        # Combined score bars
        fig.add_trace(go.Bar(
            x=ranks,
            y=combined_scores,
            name='Combined Score',
            marker_color=self.color_palette['primary'],
            text=gene_pairs,
            textposition='outside',
            hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Score: %{y:.3f}<extra></extra>'
        ))
        
        # Add rules scores as scatter points
        fig.add_trace(go.Scatter(
            x=ranks,
            y=rules_scores,
            mode='markers',
            name='Rules Score',
            marker=dict(
                size=10,
                color=self.color_palette['accent'],
                symbol='diamond'
            ),
            hovertemplate='Rules Score: %{y:.3f}<extra></extra>'
        ))
        
        # Add ML confidence as scatter points
        fig.add_trace(go.Scatter(
            x=ranks,
            y=ml_confidence,
            mode='markers',
            name='ML Confidence',
            marker=dict(
                size=8,
                color=self.color_palette['success'],
                symbol='circle'
            ),
            hovertemplate='ML Confidence: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Top Gene Pair Recommendations',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Georgia, serif'}
            },
            xaxis_title='Rank',
            yaxis_title='Score',
            **self.default_layout
        )
        
        return json.loads(fig.to_json())
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation heatmap of statistical measures."""
        
        # Select numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_data = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_data.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title={
                'text': 'Correlation Matrix of Statistical Measures',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Georgia, serif'}
            },
            width=800,
            height=600,
            **self.default_layout
        )
        
        return json.loads(fig.to_json())
    
    def create_feature_importance_chart(self, feature_importance: pd.DataFrame) -> Dict[str, Any]:
        """Create feature importance visualization."""
        
        fig = go.Figure()
        
        for i, pc in enumerate(feature_importance.columns):
            fig.add_trace(go.Bar(
                x=feature_importance.index,
                y=feature_importance[pc],
                name=pc,
                marker_color=self.color_palette['primary'] if i == 0 else self.color_palette['accent']
            ))
        
        fig.update_layout(
            title={
                'text': 'Feature Importance in PCA Components',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Georgia, serif'}
            },
            xaxis_title='Features',
            yaxis_title='Loading Score',
            barmode='group',
            **self.default_layout
        )
        
        return json.loads(fig.to_json())
    
    def _empty_chart(self, message: str) -> Dict[str, Any]:
        """Create an empty chart with a message."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        
        fig.update_layout(
            title="Chart Unavailable",
            xaxis={'visible': False},
            yaxis={'visible': False},
            **self.default_layout
        )
        
        return json.loads(fig.to_json())
    
    def create_summary_dashboard(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive summary dashboard."""
        
        dashboard = {
            'charts': {},
            'metrics': {},
            'summary_stats': results.get('summary_stats', {})
        }
        
        # Add key metrics
        if 'recommendations' in results:
            recommendations = results['recommendations']
            dashboard['metrics'] = {
                'total_recommendations': len(recommendations),
                'high_confidence_pairs': sum(1 for r in recommendations if r['is_high_confidence']),
                'outlier_pairs': sum(1 for r in recommendations if r['is_outlier']),
                'avg_combined_score': np.mean([r['combined_score'] for r in recommendations]) if recommendations else 0
            }
        
        return dashboard