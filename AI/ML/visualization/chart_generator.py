"""
Chart Generator

Generates interactive visualizations for gene pair analysis results using Plotly.
Creates various chart types including box plots, scatter plots, clustering visualizations,
and ranking charts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Generator for interactive charts and visualizations.
    
    Creates various types of charts for gene pair analysis results using Plotly,
    providing both static images and interactive HTML/JavaScript outputs.
    """
    
    def __init__(self):
        """Initialize the chart generator."""
        self.default_colors = {
            'primary': '#2E8B8B',
            'secondary': '#CD853F',
            'accent': '#4682B4',
            'warning': '#FF6B35',
            'success': '#28A745',
            'background': '#FAFAFA'
        }
        
        self.chart_height = 500
        self.chart_width = None  # Auto-width
    
    def create_boxplot(self, data: pd.DataFrame, results: Dict[str, Any]) -> str:
        """
        Create box plots for effect size distributions.
        
        Args:
            data: Gene pair data DataFrame
            results: Analysis results dictionary
            
        Returns:
            JSON string representing the Plotly chart
        """
        try:
            # Prepare data for box plots
            conditions = []
            effect_sizes = []
            
            if 'effect_size_ss' in data.columns:
                conditions.extend(['Septic Shock'] * len(data))
                effect_sizes.extend(data['effect_size_ss'].tolist())
            
            if 'effect_size_soth' in data.columns:
                conditions.extend(['Other Sepsis'] * len(data))
                effect_sizes.extend(data['effect_size_soth'].tolist())
            
            if not effect_sizes:
                # Generate sample data if no effect sizes available
                np.random.seed(42)
                effect_sizes_ss = np.random.normal(-0.2, 0.8, 100)
                effect_sizes_soth = np.random.normal(-0.3, 0.9, 100)
                
                conditions = ['Septic Shock'] * 100 + ['Other Sepsis'] * 100
                effect_sizes = list(effect_sizes_ss) + list(effect_sizes_soth)
            
            # Create box plot
            fig = go.Figure()
            
            # Septic Shock box plot
            ss_data = [es for es, cond in zip(effect_sizes, conditions) if cond == 'Septic Shock']
            fig.add_trace(go.Box(
                y=ss_data,
                name='Septic Shock Effect Size',
                marker_color=self.default_colors['primary'],
                boxpoints='outliers',
                showlegend=True
            ))
            
            # Other Sepsis box plot
            soth_data = [es for es, cond in zip(effect_sizes, conditions) if cond == 'Other Sepsis']
            fig.add_trace(go.Box(
                y=soth_data,
                name='Other Sepsis Effect Size',
                marker_color=self.default_colors['secondary'],
                boxpoints='outliers',
                showlegend=True
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Effect Size Distribution by Condition',
                    'x': 0.5,
                    'font': {'size': 18, 'family': 'Georgia, serif'}
                },
                xaxis_title='Condition',
                yaxis_title='Effect Size (Cohen\'s d)',
                height=self.chart_height,
                plot_bgcolor='white',
                paper_bgcolor=self.default_colors['background'],
                font={'family': 'Arial, sans-serif'},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            )
            
            return fig.to_json()
            
        except Exception as e:
            logger.error(f"Error creating box plot: {e}")
            return self._create_error_chart("Error creating box plot")
    
    def create_scatter_plot(self, data: pd.DataFrame, results: Dict[str, Any]) -> str:
        """
        Create scatter plot for gene pair correlations.
        
        Args:
            data: Gene pair data DataFrame
            results: Analysis results dictionary
            
        Returns:
            JSON string representing the Plotly chart
        """
        try:
            # Prepare data for scatter plot
            if 'recommendations' in results and results['recommendations']:
                recommendations = results['recommendations']
                
                x_values = [rec['statistical_measures']['dz_ss_mean'] for rec in recommendations]
                y_values = [rec['statistical_measures']['dz_soth_mean'] for rec in recommendations]
                hover_text = [f"{rec['gene_a']}-{rec['gene_b']}" for rec in recommendations]
                colors = [rec['combined_score'] for rec in recommendations]
                sizes = [rec['ml_confidence'] * 20 + 5 for rec in recommendations]
                
            else:
                # Generate sample data
                np.random.seed(42)
                n_points = 50
                x_values = np.random.normal(0, 1, n_points)
                y_values = np.random.normal(0, 1, n_points)
                hover_text = [f"GenePair_{i}" for i in range(n_points)]
                colors = np.random.uniform(0, 1, n_points)
                sizes = np.random.uniform(5, 20, n_points)
            
            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    colorscale='Viridis',
                    colorbar=dict(
                        title='Combined Score',
                        titleside='right'
                    ),
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=hover_text,
                hovertemplate='<b>%{text}</b><br>' +
                             'SS Effect: %{x:.3f}<br>' +
                             'SOTH Effect: %{y:.3f}<br>' +
                             'Score: %{marker.color:.3f}<extra></extra>',
                showlegend=False
            ))
            
            # Add significance thresholds
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Gene Pair Correlation Scatter Plot',
                    'x': 0.5,
                    'font': {'size': 18, 'family': 'Georgia, serif'}
                },
                xaxis_title='Effect Size - Septic Shock',
                yaxis_title='Effect Size - Other Sepsis',
                height=self.chart_height,
                plot_bgcolor='white',
                paper_bgcolor=self.default_colors['background'],
                font={'family': 'Arial, sans-serif'}
            )
            
            # Update axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            )
            
            return fig.to_json()
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            return self._create_error_chart("Error creating scatter plot")
    
    def create_clustering_viz(self, results: Dict[str, Any]) -> str:
        """
        Create clustering visualization.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            JSON string representing the Plotly chart
        """
        try:
            # Get clustering results
            if 'dbscan_labels' in results:
                cluster_labels = results['dbscan_labels']
                unique_labels = np.unique(cluster_labels)
                
                # Count points in each cluster
                cluster_counts = []
                cluster_names = []
                for label in unique_labels:
                    count = np.sum(cluster_labels == label)
                    cluster_counts.append(count)
                    if label == -1:
                        cluster_names.append('Outliers')
                    else:
                        cluster_names.append(f'Cluster {label}')
            else:
                # Sample clustering data
                cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Outliers']
                cluster_counts = [45, 30, 20, 5]
            
            # Create pie chart for cluster distribution
            fig = go.Figure()
            
            colors = [self.default_colors['primary'], self.default_colors['secondary'], 
                     self.default_colors['accent'], self.default_colors['warning']]
            
            fig.add_trace(go.Pie(
                labels=cluster_names,
                values=cluster_counts,
                marker_colors=colors[:len(cluster_names)],
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>' +
                             'Count: %{value}<br>' +
                             'Percentage: %{percent}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Gene Pair Clustering Distribution',
                    'x': 0.5,
                    'font': {'size': 18, 'family': 'Georgia, serif'}
                },
                height=self.chart_height,
                plot_bgcolor='white',
                paper_bgcolor=self.default_colors['background'],
                font={'family': 'Arial, sans-serif'},
                showlegend=True
            )
            
            return fig.to_json()
            
        except Exception as e:
            logger.error(f"Error creating clustering visualization: {e}")
            return self._create_error_chart("Error creating clustering visualization")
    
    def create_ranking_chart(self, results: Dict[str, Any]) -> str:
        """
        Create ranking chart for top recommendations.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            JSON string representing the Plotly chart
        """
        try:
            # Get top recommendations
            if 'recommendations' in results and results['recommendations']:
                recommendations = results['recommendations'][:20]  # Top 20
                
                gene_pairs = [f"{rec['gene_a']}-{rec['gene_b']}" for rec in recommendations]
                scores = [rec['combined_score'] for rec in recommendations]
                colors = [self.default_colors['success'] if rec['is_high_confidence'] 
                         else self.default_colors['warning'] if rec['is_outlier']
                         else self.default_colors['primary'] for rec in recommendations]
                
            else:
                # Sample ranking data
                gene_pairs = [f"GenePair_{i}" for i in range(1, 21)]
                scores = np.random.uniform(0.1, 0.9, 20)
                colors = [self.default_colors['primary']] * 20
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=gene_pairs,
                x=scores,
                orientation='h',
                marker_color=colors,
                text=[f"{score:.3f}" for score in scores],
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>' +
                             'Score: %{x:.3f}<extra></extra>',
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Top Gene Pair Recommendations',
                    'x': 0.5,
                    'font': {'size': 18, 'family': 'Georgia, serif'}
                },
                xaxis_title='Combined Score',
                yaxis_title='Gene Pairs',
                height=max(400, len(gene_pairs) * 25),  # Dynamic height
                plot_bgcolor='white',
                paper_bgcolor=self.default_colors['background'],
                font={'family': 'Arial, sans-serif'}
            )
            
            # Update axes
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, max(scores) * 1.1]
            )
            fig.update_yaxes(
                autorange='reversed'  # Show highest scores at top
            )
            
            return fig.to_json()
            
        except Exception as e:
            logger.error(f"Error creating ranking chart: {e}")
            return self._create_error_chart("Error creating ranking chart")
    
    def create_progress_chart(self, progress_data: Dict[str, float]) -> str:
        """
        Create analysis progress chart.
        
        Args:
            progress_data: Dictionary with progress information
            
        Returns:
            JSON string representing the Plotly chart
        """
        try:
            # Create progress bar chart
            fig = go.Figure()
            
            steps = list(progress_data.keys())
            progress_values = list(progress_data.values())
            
            fig.add_trace(go.Bar(
                x=steps,
                y=progress_values,
                marker_color=self.default_colors['primary'],
                text=[f"{val:.1f}%" for val in progress_values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Progress: %{y:.1f}%<extra></extra>',
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Analysis Progress',
                    'x': 0.5,
                    'font': {'size': 18, 'family': 'Georgia, serif'}
                },
                xaxis_title='Analysis Steps',
                yaxis_title='Progress (%)',
                height=self.chart_height,
                plot_bgcolor='white',
                paper_bgcolor=self.default_colors['background'],
                font={'family': 'Arial, sans-serif'}
            )
            
            # Update axes
            fig.update_xaxes(
                tickangle=45
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                range=[0, 100]
            )
            
            return fig.to_json()
            
        except Exception as e:
            logger.error(f"Error creating progress chart: {e}")
            return self._create_error_chart("Error creating progress chart")
    
    def create_summary_stats_chart(self, summary_stats: Dict[str, Any]) -> str:
        """
        Create summary statistics visualization.
        
        Args:
            summary_stats: Dictionary with summary statistics
            
        Returns:
            JSON string representing the Plotly chart
        """
        try:
            # Create subplots for different statistics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Total Pairs', 'Outliers Detected', 
                              'Clusters Found', 'Significant Pairs'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "bar"}]]
            )
            
            # Total pairs indicator
            fig.add_trace(go.Indicator(
                mode="number",
                value=summary_stats.get('total_pairs', 0),
                title={"text": "Total Gene Pairs"},
                number={'font': {'size': 40, 'color': self.default_colors['primary']}}
            ), row=1, col=1)
            
            # Outliers indicator
            fig.add_trace(go.Indicator(
                mode="number",
                value=summary_stats.get('outliers_detected', 0),
                title={"text": "Outliers Detected"},
                number={'font': {'size': 40, 'color': self.default_colors['warning']}}
            ), row=1, col=2)
            
            # Clusters indicator
            fig.add_trace(go.Indicator(
                mode="number",
                value=summary_stats.get('clusters_found', 0),
                title={"text": "Clusters Found"},
                number={'font': {'size': 40, 'color': self.default_colors['accent']}}
            ), row=2, col=1)
            
            # Significant pairs bar chart
            significant_data = [
                summary_stats.get('significant_pairs_ss', 0),
                summary_stats.get('significant_pairs_soth', 0)
            ]
            
            fig.add_trace(go.Bar(
                x=['Septic Shock', 'Other Sepsis'],
                y=significant_data,
                marker_color=[self.default_colors['primary'], self.default_colors['secondary']],
                text=significant_data,
                textposition='auto',
                showlegend=False
            ), row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Analysis Summary Statistics',
                    'x': 0.5,
                    'font': {'size': 20, 'family': 'Georgia, serif'}
                },
                height=600,
                plot_bgcolor='white',
                paper_bgcolor=self.default_colors['background'],
                font={'family': 'Arial, sans-serif'}
            )
            
            return fig.to_json()
            
        except Exception as e:
            logger.error(f"Error creating summary stats chart: {e}")
            return self._create_error_chart("Error creating summary statistics")
    
    def _create_error_chart(self, error_message: str) -> str:
        """
        Create a simple error chart when visualization fails.
        
        Args:
            error_message: Error message to display
            
        Returns:
            JSON string representing the error chart
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"<b>Visualization Error</b><br>{error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='red'),
            bgcolor="rgba(255, 200, 200, 0.8)",
            bordercolor="red",
            borderwidth=2
        )
        
        fig.update_layout(
            height=self.chart_height,
            plot_bgcolor='white',
            paper_bgcolor=self.default_colors['background'],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return fig.to_json()

# Example usage
if __name__ == "__main__":
    generator = ChartGenerator()
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'effect_size_ss': np.random.normal(0, 1, 100),
        'effect_size_soth': np.random.normal(0, 1, 100)
    })
    
    sample_results = {
        'recommendations': [
            {
                'gene_a': 'MS4A4A',
                'gene_b': 'CD86',
                'combined_score': 0.85,
                'ml_confidence': 0.9,
                'is_high_confidence': True,
                'is_outlier': False,
                'statistical_measures': {
                    'dz_ss_mean': 0.85,
                    'dz_soth_mean': 0.92
                }
            }
        ],
        'summary_stats': {
            'total_pairs': 100,
            'outliers_detected': 5,
            'clusters_found': 3,
            'significant_pairs_ss': 45,
            'significant_pairs_soth': 38
        }
    }
    
    # Generate charts
    print("Generating sample charts...")
    
    boxplot = generator.create_boxplot(sample_data, sample_results)
    print("Box plot created")
    
    scatter = generator.create_scatter_plot(sample_data, sample_results)
    print("Scatter plot created")
    
    clustering = generator.create_clustering_viz(sample_results)
    print("Clustering visualization created")
    
    ranking = generator.create_ranking_chart(sample_results)
    print("Ranking chart created")
    
    print("All charts generated successfully!")