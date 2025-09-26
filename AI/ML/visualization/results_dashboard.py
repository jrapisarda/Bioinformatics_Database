"""
Results Dashboard Module

Provides comprehensive dashboard functionality for displaying analysis results.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from .chart_generator import ChartGenerator
from .interactive_plots import InteractivePlotter

logger = logging.getLogger(__name__)


class ResultsDashboard:
    """Comprehensive results dashboard for gene pair analysis."""
    
    def __init__(self):
        """Initialize the results dashboard."""
        self.chart_generator = ChartGenerator()
        self.interactive_plotter = InteractivePlotter()
        
    def create_comprehensive_dashboard(self, 
                                     analysis_results: Dict[str, Any],
                                     original_data: pd.DataFrame,
                                     session_id: str) -> Dict[str, Any]:
        """Create a comprehensive dashboard with all components."""
        
        dashboard = {
            'metadata': {
                'session_id': session_id,
                'created_at': pd.Timestamp.now().isoformat(),
                'total_pairs': len(original_data),
                'n_recommendations': len(analysis_results.get('recommendations', []))
            },
            'summary_stats': self._create_summary_statistics(analysis_results),
            'top_recommendations': self._get_top_recommendations(analysis_results, n=20),
            'visualizations': self._create_all_visualizations(original_data, analysis_results),
            'interactive_components': self._create_interactive_components(analysis_results, session_id),
            'export_data': self._prepare_export_data(analysis_results, original_data)
        }
        
        return dashboard
    
    def _create_summary_statistics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive summary statistics."""
        summary_stats = analysis_results.get('summary_stats', {})
        recommendations = analysis_results.get('recommendations', [])
        
        # Calculate additional statistics
        stats = {
            'basic_metrics': {
                'total_recommendations': len(recommendations),
                'high_confidence_pairs': sum(1 for r in recommendations if r.get('is_high_confidence', False)),
                'outlier_pairs': sum(1 for r in recommendations if r.get('is_outlier', False)),
                'average_combined_score': np.mean([r.get('combined_score', 0) for r in recommendations]) if recommendations else 0,
                'average_rules_score': np.mean([r.get('rules_score', 0) for r in recommendations]) if recommendations else 0,
                'average_ml_confidence': np.mean([r.get('ml_confidence', 0) for r in recommendations]) if recommendations else 0
            },
            'ensemble_metrics': {
                'outliers_detected': summary_stats.get('outliers_detected', 0),
                'clusters_found': summary_stats.get('clusters_found', 0),
                'silhouette_score': summary_stats.get('silhouette_score', 0),
                'positive_control_validated': summary_stats.get('positive_control_validated', False)
            },
            'quality_metrics': {
                'data_completeness': summary_stats.get('data_completeness', 0),
                'significant_pairs_ss': summary_stats.get('significant_pairs_ss', 0),
                'significant_pairs_soth': summary_stats.get('significant_pairs_soth', 0),
                'large_effect_pairs': summary_stats.get('large_effect_pairs', 0)
            }
        }
        
        return stats
    
    def _get_top_recommendations(self, analysis_results: Dict[str, Any], n: int = 20) -> List[Dict[str, Any]]:
        """Get top N recommendations with additional metadata."""
        recommendations = analysis_results.get('recommendations', [])
        
        # Sort by combined score and take top N
        top_recs = sorted(recommendations, key=lambda x: x.get('combined_score', 0), reverse=True)[:n]
        
        # Add additional metadata
        for i, rec in enumerate(top_recs):
            rec['percentile_rank'] = (i + 1) / len(recommendations) * 100
            rec['score_percentile'] = (rec.get('combined_score', 0) / max(r.get('combined_score', 1) for r in recommendations)) * 100
        
        return top_recs
    
    def _create_all_visualizations(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create all visualizations for the dashboard."""
        visualizations = {}
        
        try:
            # Box plots
            boxplot_chart = self.chart_generator.create_boxplot(data, analysis_results)
            visualizations['boxplot'] = boxplot_chart
            
            # Scatter plot
            scatter_chart = self.chart_generator.create_scatter_plot(data, analysis_results)
            visualizations['scatter'] = scatter_chart
            
            # Clustering visualization
            clustering_chart = self.chart_generator.create_clustering_viz(analysis_results)
            visualizations['clustering'] = clustering_chart
            
            # Ranking chart
            ranking_chart = self.chart_generator.create_ranking_chart(analysis_results)
            visualizations['ranking'] = ranking_chart
            
            # Correlation heatmap
            if len(data.select_dtypes(include=[np.number]).columns) > 2:
                correlation_chart = self.chart_generator.create_correlation_heatmap(data)
                visualizations['correlation'] = correlation_chart
            
            logger.info(f"Created {len(visualizations)} visualizations")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _create_interactive_components(self, analysis_results: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Create interactive HTML components."""
        components = {}
        
        # Top recommendations cards
        top_recs = self._get_top_recommendations(analysis_results, n=10)
        cards_html = []
        for rec in top_recs:
            cards_html.append(self.interactive_plotter.create_recommendation_card(rec))
        
        components['recommendation_cards'] = cards_html
        
        # Analysis summary card
        summary_stats = self._create_summary_statistics(analysis_results)
        components['summary_card'] = self.interactive_plotter.create_analysis_summary_card(summary_stats)
        
        # Export options
        components['export_options'] = self.interactive_plotter.create_export_options(session_id)
        
        return components
    
    def _prepare_export_data(self, analysis_results: Dict[str, Any], original_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for export in various formats."""
        export_data = {
            'recommendations': analysis_results.get('recommendations', []),
            'summary_stats': self._create_summary_statistics(analysis_results),
            'original_data': original_data.to_dict('records'),
            'metadata': {
                'columns': list(original_data.columns),
                'shape': original_data.shape,
                'dtypes': original_data.dtypes.to_dict()
            }
        }
        
        return export_data
    
    def generate_report(self, dashboard: Dict[str, Any], output_path: str, format: str = 'html') -> str:
        """Generate comprehensive analysis report."""
        
        if format == 'html':
            return self._generate_html_report(dashboard, output_path)
        elif format == 'json':
            return self._generate_json_report(dashboard, output_path)
        elif format == 'csv':
            return self._generate_csv_report(dashboard, output_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_html_report(self, dashboard: Dict[str, Any], output_path: str) -> str:
        """Generate HTML report."""
        dashboard_json = json.dumps(dashboard, default=str).replace('</', '<\\/')

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Gene Pair Analysis Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ margin: 20px 0; }}
                .recommendation-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; }}
                .gene-names {{ font-weight: bold; color: #2E8B8B; }}
                .score-display {{ font-size: 1.2em; font-weight: bold; color: #4ECDC4; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #2E8B8B; }}
                .stat-label {{ color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Gene Pair Analysis Report</h1>
                <p class="text-muted">Generated on: {dashboard['metadata']['created_at']}</p>
                <p class="text-muted">Session ID: {dashboard['metadata']['session_id']}</p>
                
                <h2>Summary Statistics</h2>
                {dashboard['interactive_components']['summary_card']}
                
                <h2>Top Recommendations</h2>
                <div class="row">
                    {''.join(dashboard['interactive_components']['recommendation_cards'])}
                </div>
                
                <h2>Visualizations</h2>
                <div class="chart-container">
                    <h3>Effect Size Distribution</h3>
                    <div id="boxplot-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Gene Pair Correlation</h3>
                    <div id="scatter-chart"></div>
                </div>
            </div>

            <div id="gene-detail-modal" class="modal fade" tabindex="-1" aria-labelledby="geneDetailModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-lg modal-dialog-scrollable">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="geneDetailModalLabel">Gene Pair Details</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div id="gene-detail-error" class="alert alert-warning d-none" role="alert">
                                Unable to locate detailed data for the selected gene pair.
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <dl class="row">
                                        <dt class="col-sm-4">Gene A</dt>
                                        <dd class="col-sm-8" id="modal-gene-a">-</dd>
                                        <dt class="col-sm-4">Gene B</dt>
                                        <dd class="col-sm-8" id="modal-gene-b">-</dd>
                                        <dt class="col-sm-4">Combined Score</dt>
                                        <dd class="col-sm-8" id="modal-combined-score">-</dd>
                                        <dt class="col-sm-4">Rules Score</dt>
                                        <dd class="col-sm-8" id="modal-rules-score">-</dd>
                                        <dt class="col-sm-4">ML Confidence</dt>
                                        <dd class="col-sm-8" id="modal-ml-confidence">-</dd>
                                    </dl>
                                </div>
                                <div class="col-md-6">
                                    <h6>Additional Details</h6>
                                    <pre class="bg-light p-3 border rounded" id="modal-full-json" style="max-height: 300px; overflow-y: auto; white-space: pre-wrap;"></pre>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-primary" id="export-pair-btn">Export JSON</button>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                const dashboardData = {dashboard_json};

                // Embed Plotly charts
                const boxplotData = {json.dumps(dashboard['visualizations'].get('boxplot', {}))};
                const scatterData = {json.dumps(dashboard['visualizations'].get('scatter', {}))};

                if (boxplotData.data) {{
                    Plotly.newPlot('boxplot-chart', boxplotData.data, boxplotData.layout);
                }}

                if (scatterData.data) {{
                    Plotly.newPlot('scatter-chart', scatterData.data, scatterData.layout);
                }}

                function getRecommendations() {{
                    const exportData = dashboardData && dashboardData.export_data ? dashboardData.export_data : {{}};
                    const recs = Array.isArray(exportData.recommendations) ? exportData.recommendations : [];
                    return recs;
                }}

                window.showGenePairDetails = function(geneA, geneB) {{
                    const recommendations = getRecommendations();
                    const modalElement = document.getElementById('gene-detail-modal');
                    const modalInstance = bootstrap.Modal.getOrCreateInstance(modalElement);
                    const errorAlert = document.getElementById('gene-detail-error');
                    const exportBtn = document.getElementById('export-pair-btn');

                    const match = recommendations.find(rec => (
                        (rec.gene_a === geneA && rec.gene_b === geneB) ||
                        (rec.gene_a === geneB && rec.gene_b === geneA)
                    ));

                    if (match) {{
                        errorAlert.classList.add('d-none');
                        document.getElementById('modal-gene-a').textContent = match.gene_a || '-';
                        document.getElementById('modal-gene-b').textContent = match.gene_b || '-';
                        document.getElementById('modal-combined-score').textContent = typeof match.combined_score !== 'undefined' ? match.combined_score : '-';
                        document.getElementById('modal-rules-score').textContent = typeof match.rules_score !== 'undefined' ? match.rules_score : '-';
                        document.getElementById('modal-ml-confidence').textContent = typeof match.ml_confidence !== 'undefined' ? match.ml_confidence : '-';
                        document.getElementById('modal-full-json').textContent = JSON.stringify(match, null, 2);

                        if (exportBtn) {{
                            exportBtn.disabled = false;
                            exportBtn.classList.remove('disabled');
                            exportBtn.onclick = function() {{
                                exportPairData(match.gene_a, match.gene_b);
                            }};
                        }}
                    }} else {{
                        document.getElementById('modal-gene-a').textContent = geneA || '-';
                        document.getElementById('modal-gene-b').textContent = geneB || '-';
                        document.getElementById('modal-combined-score').textContent = '-';
                        document.getElementById('modal-rules-score').textContent = '-';
                        document.getElementById('modal-ml-confidence').textContent = '-';
                        document.getElementById('modal-full-json').textContent = 'Recommendation details could not be found in the exported data set.';
                        errorAlert.classList.remove('d-none');

                        if (exportBtn) {{
                            exportBtn.disabled = true;
                            exportBtn.classList.add('disabled');
                            exportBtn.onclick = null;
                        }}
                    }}

                    modalInstance.show();
                }};

                window.exportPairData = function(geneA, geneB) {{
                    const recommendations = getRecommendations();
                    const record = recommendations.find(rec => (
                        (rec.gene_a === geneA && rec.gene_b === geneB) ||
                        (rec.gene_a === geneB && rec.gene_b === geneA)
                    ));

                    const exportBtn = document.getElementById('export-pair-btn');

                    if (!record) {{
                        if (exportBtn) {{
                            exportBtn.disabled = true;
                            exportBtn.classList.add('disabled');
                        }}
                        console.warn('No data available to export for the selected gene pair.');
                        return;
                    }}

                    if (typeof Blob === 'undefined' || typeof URL === 'undefined' || typeof URL.createObjectURL === 'undefined') {{
                        if (exportBtn) {{
                            exportBtn.disabled = true;
                            exportBtn.classList.add('disabled');
                            exportBtn.title = 'Export not supported in this environment.';
                        }}
                        console.warn('Blob downloads are not supported in this environment.');
                        return;
                    }}

                    const jsonString = JSON.stringify(record, null, 2);
                    const blob = new Blob([jsonString], {{ type: 'application/json' }});
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    const safeGeneA = geneA ? geneA.replace(/[^a-z0-9_-]+/gi, '_') : 'geneA';
                    const safeGeneB = geneB ? geneB.replace(/[^a-z0-9_-]+/gi, '_') : 'geneB';
                    link.download = `${{safeGeneA}}_${{safeGeneB}}_recommendation.json`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                }};
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_json_report(self, dashboard: Dict[str, Any], output_path: str) -> str:
        """Generate JSON report."""
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2, default=str)
        
        return output_path
    
    def _generate_csv_report(self, dashboard: Dict[str, Any], output_path: str) -> str:
        """Generate CSV report of recommendations."""
        recommendations = dashboard.get('export_data', {}).get('recommendations', [])
        
        if recommendations:
            df = pd.DataFrame(recommendations)
            df.to_csv(output_path, index=False)
        
        return output_path
    
    def validate_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis results for quality assurance."""
        validation_report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'checks': {},
            'warnings': [],
            'errors': [],
            'overall_status': 'PASSED'
        }
        
        # Check if recommendations exist
        recommendations = analysis_results.get('recommendations', [])
        if not recommendations:
            validation_report['errors'].append('No recommendations generated')
            validation_report['overall_status'] = 'FAILED'
        
        # Check positive control validation
        summary_stats = analysis_results.get('summary_stats', {})
        if not summary_stats.get('positive_control_validated', False):
            validation_report['warnings'].append('Positive control validation failed')
            if validation_report['overall_status'] == 'PASSED':
                validation_report['overall_status'] = 'WARNING'
        
        # Check for reasonable number of outliers
        outliers_detected = summary_stats.get('outliers_detected', 0)
        total_pairs = summary_stats.get('total_pairs', 1)
        outlier_rate = outliers_detected / total_pairs
        
        if outlier_rate > 0.3:
            validation_report['warnings'].append(f'High outlier rate: {outlier_rate:.1%}')
        elif outlier_rate < 0.01:
            validation_report['warnings'].append(f'Very low outlier rate: {outlier_rate:.1%}')
        
        # Check clustering quality
        silhouette_score = summary_stats.get('silhouette_score', -1)
        if silhouette_score >= 0:
            if silhouette_score < 0.3:
                validation_report['warnings'].append(f'Low clustering quality: {silhouette_score:.3f}')
            elif silhouette_score > 0.7:
                validation_report['checks']['good_clustering'] = True
        
        validation_report['checks']['recommendations_count'] = len(recommendations)
        validation_report['checks']['outlier_rate'] = outlier_rate
        validation_report['checks']['silhouette_score'] = silhouette_score
        
        return validation_report