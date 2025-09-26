"""
Flask Web Interface for Gene Pair ML Analysis

Provides web-based interface for data upload, rule configuration, batch processing,
and interactive visualization of results.
"""

import os
import json
import logging
import uuid
from flask import Flask
from flask_moment import Moment
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

app=Flask(__name__)
moment=Moment(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import core modules, fall back to mock implementations if not available
try:
    # Import our gene pair analysis modules
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from gene_pair_agent import GenePairAnalyzer, RulesEngine, MetaAnalysisProcessor, DatabaseConnector, Rule
    from visualization import ChartGenerator, ResultsDashboard
    
    logger.info("Successfully imported core analysis modules")
    USING_MOCK_MODULES = False
    
except ImportError as e:
    logger.warning(f"Could not import core modules: {e}")
    logger.info("Using mock analysis modules for testing")
    
    # Import mock modules
    from mock_analysis_engine import (
        MockGenePairAnalyzer as GenePairAnalyzer,
        MockRulesEngine as RulesEngine,
        MockMetaAnalysisProcessor as MetaAnalysisProcessor,
        MockDatabaseConnector as DatabaseConnector,
        get_mock_analyzer,
        get_mock_rules_engine,
        get_mock_processor
    )

    from dataclasses import dataclass

    @dataclass
    class Rule:
        """Lightweight rule representation for mock engine compatibility."""
        name: str
        condition: str
        weight: float
        description: str = ""
        enabled: bool = True
    
    # Create mock visualization classes
    class MockChartGenerator:
        def create_boxplot(self, data, results):
            return self._create_mock_chart("Box Plot", "Effect Size Distribution")
        
        def create_scatter_plot(self, data, results):
            return self._create_mock_chart("Scatter Plot", "Gene Pair Correlations")
        
        def create_clustering_viz(self, results):
            return self._create_mock_chart("Clustering", "Clustering Results")
        
        def create_ranking_chart(self, results):
            return self._create_mock_chart("Ranking", "Top Recommendations")
        
        def _create_mock_chart(self, chart_type, title):
            mock_chart = {
                "data": [],
                "layout": {
                    "title": f"Mock {chart_type}: {title}",
                    "annotations": [{
                        "text": f"Mock {chart_type} Chart<br>Real module needed for full functionality",
                        "xref": "paper", "yref": "paper",
                        "x": 0.5, "y": 0.5, "xanchor": "center", "yanchor": "middle",
                        "showarrow": False, "font": {"size": 16}
                    }],
                    "xaxis": {"visible": False}, "yaxis": {"visible": False},
                    "height": 400, "plot_bgcolor": "white", "paper_bgcolor": "#FAFAFA"
                }
            }
            return json.dumps(mock_chart)
    
    class MockResultsDashboard:
        def __init__(self):
            self.chart_generator = MockChartGenerator()
            self.current_data = None
            self.current_results = None
        
        def load_results(self, data, results):
            self.current_data = data
            self.current_results = results
        
        def update_charts(self):
            return {
                'boxplot': self.chart_generator.create_boxplot(None, None),
                'scatter': self.chart_generator.create_scatter_plot(None, None),
                'clustering': self.chart_generator.create_clustering_viz(None),
                'ranking': self.chart_generator.create_ranking_chart(None)
            }
    
    ChartGenerator = MockChartGenerator
    ResultsDashboard = MockResultsDashboard
    USING_MOCK_MODULES = True

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'gene-pair-ml-analysis-secret-key'

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
RESULTS_FOLDER = Path(__file__).parent / 'results'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv', 'json'}
RULES_CONFIG_DIR = Path(__file__).parent / 'config'
RULES_CONFIG_PATH = RULES_CONFIG_DIR / 'rules_config.json'

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)
RULES_CONFIG_DIR.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global storage for analysis results (in production, use a proper database)
analysis_results = {}


def _default_rule_names() -> set:
    """Return the set of rule names that ship with the engine by default."""
    if hasattr(RulesEngine, 'DEFAULT_RULES'):
        try:
            return {rule.name for rule in RulesEngine.DEFAULT_RULES}
        except Exception:
            return set()
    return set()


def _load_rules_engine() -> RulesEngine:
    """Instantiate rules engine, loading persisted configuration when available."""
    config_path = str(RULES_CONFIG_PATH) if RULES_CONFIG_PATH.exists() else None

    if config_path:
        try:
            return RulesEngine(config_path=config_path)
        except TypeError:
            logger.debug("RulesEngine does not support config_path parameter; using defaults")

    return RulesEngine()


def _save_rules(engine: RulesEngine) -> None:
    """Persist the current rules configuration to disk."""
    if not hasattr(engine, 'save_rules_to_config'):
        logger.warning("Rules engine does not support persistence; skipping save")
        return

    try:
        engine.save_rules_to_config(str(RULES_CONFIG_PATH))
    except Exception as exc:
        logger.error(f"Failed to save rules configuration: {exc}")
        raise


def _build_rules_response(engine: RulesEngine) -> Dict[str, Any]:
    """Create categorized payload of default and custom rules for the UI."""
    summary = engine.get_rule_summary()
    default_names = _default_rule_names()

    default_rules = []
    custom_rules = []

    for rule in summary.get('rules', []):
        target = default_rules if rule.get('name') in default_names else custom_rules
        target.append(rule)

    summary_payload = {k: v for k, v in summary.items() if k != 'rules'}

    return {
        'summary': summary_payload,
        'default_rules': default_rules,
        'custom_rules': custom_rules,
        'all_rules': summary.get('rules', [])
    }


def _coerce_enabled(value: Any) -> bool:
    """Interpret user-provided values as booleans."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() not in {'false', '0', 'off', 'no', ''}

    return bool(value)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main landing page."""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file uploads for meta-analysis data."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = UPLOAD_FOLDER / unique_filename
            
            file.save(filepath)
            
            # Process the uploaded file
            try:
                processor = MetaAnalysisProcessor()
                data = processor.load_data(str(filepath))
                
                # Validate data
                validation = processor.validate_data(data)
                
                if not validation['is_valid']:
                    flash(f"Data validation failed: {validation['errors']}", 'error')
                    return redirect(request.url)
                
                # Clean and prepare data
                cleaned_data = processor.clean_data(data)
                prepared_data = processor.prepare_for_analysis(cleaned_data)
                
                # Store for analysis
                session_id = str(uuid.uuid4())
                analysis_results[session_id] = {
                    'data': prepared_data,
                    'original_filename': filename,
                    'upload_time': datetime.now().isoformat(),
                    'validation': validation,
                    'filepath': str(filepath)
                }
                
                flash('File uploaded successfully!', 'success')
                # Redirect to analyze page with session_id
                return redirect(url_for('analyze', session_id=session_id))
                
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        
        flash('Invalid file type. Please upload Excel, CSV, or JSON files.', 'error')
        return redirect(request.url)
    
    return render_template('upload.html')


@app.route('/database')
def database_connection():
    """Database connection management page."""
    return render_template('database.html')


@app.route('/api/database/test', methods=['POST'])
def test_database_connection():
    """Test database connection."""
    try:
        config = request.json
        connector = DatabaseConnector(config)
        
        result = connector.test_connection()
        connector.disconnect()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return jsonify({
            'connected': False,
            'error': str(e)
        }), 500


@app.route('/api/database/data', methods=['POST'])
def fetch_database_data():
    """Fetch data from database."""
    try:
        config = request.json.get('config', {})
        filters = request.json.get('filters', {})
        
        connector = DatabaseConnector(config)
        
        if not connector.connect():
            return jsonify({'error': 'Failed to connect to database'}), 500
        
        # Apply filters
        data = connector.get_gene_pair_data(**filters)
        connector.disconnect()
        
        if data.empty:
            return jsonify({'error': 'No data found with specified filters'}), 404
        
        # Store for analysis
        session_id = str(uuid.uuid4())
        analysis_results[session_id] = {
            'data': data,
            'source': 'database',
            'fetch_time': datetime.now().isoformat(),
            'filters': filters
        }
        
        return jsonify({
            'session_id': session_id,
            'row_count': len(data),
            'preview': data.head().to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"Error fetching database data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/<session_id>')
def analyze(session_id):
    """Analysis page for specific session."""
    if session_id not in analysis_results:
        flash('Analysis session not found', 'error')
        return redirect(url_for('index'))
    
    session_data = analysis_results[session_id]
    data = session_data['data']
    
    # Get basic statistics
    stats = {
        'total_pairs': len(data),
        'columns': list(data.columns),
        'preview': data.head(10).to_dict('records')
    }
    
    return render_template('analyze.html', 
                         session_id=session_id, 
                         stats=stats,
                         data_source=session_data.get('source', 'file'))


@app.route('/api/analyze/<session_id>', methods=['POST'])
def run_analysis(session_id):
    """Run gene pair analysis on session data."""
    try:
        if session_id not in analysis_results:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = analysis_results[session_id]
        data = session_data['data']
        
        # Get analysis parameters
        params = request.json or {}
        
        # Initialize analyzer
        analyzer = GenePairAnalyzer(
            n_features=params.get('n_features', 5),
            contamination=params.get('contamination', 0.1)
        )
        
        # Configure rules if provided
        if 'rules' in params:
            for rule_data in params['rules']:
                if rule_data.get('action') == 'add':
                    from gene_pair_agent.rules_engine import Rule
                    rule = Rule(**rule_data['rule'])
                    analyzer.rules_engine.add_rule(rule)
        
        # Fit and predict
        logger.info(f"Starting analysis for session {session_id}")
        analyzer.fit(data)
        results = analyzer.predict(data)
        
        # Store results
        analysis_results[session_id]['analysis_results'] = results
        analysis_results[session_id]['analysis_time'] = datetime.now().isoformat()
        
        logger.info(f"Analysis completed for session {session_id}")
        
        return jsonify({
            'status': 'completed',
            'summary': analyzer.get_analysis_summary(),
            'recommendations_count': len(results.get('recommendations', []))
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/results/<session_id>')
def results(session_id):
    """Display analysis results."""
    if session_id not in analysis_results:
        flash('Analysis session not found', 'error')
        return redirect(url_for('index'))
    
    session_data = analysis_results[session_id]
    
    if 'analysis_results' not in session_data:
        flash('Analysis not yet completed', 'warning')
        return redirect(url_for('analyze', session_id=session_id))
    
    results = session_data['analysis_results']
    
    return render_template('results.html', 
                         session_id=session_id,
                         results=results)


@app.route('/api/results/<session_id>')
def get_results_data(session_id):
    """Get analysis results as JSON."""
    if session_id not in analysis_results:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = analysis_results.get(session_id, {})
    results = session_data.get('analysis_results', {})
    
    return jsonify(results)


@app.route('/api/visualization/<session_id>/<chart_type>')
def get_visualization(session_id, chart_type):
    """Generate visualization for analysis results."""
    try:
        if session_id not in analysis_results:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = analysis_results[session_id]
        
        if 'analysis_results' not in session_data:
            return jsonify({'error': 'Analysis not completed'}), 400
        
        results = session_data['analysis_results']
        data = session_data['data']
        
        # Initialize chart generator
        chart_gen = ChartGenerator()
        
        if chart_type == 'boxplot':
            chart = chart_gen.create_boxplot(data, results)
        elif chart_type == 'scatter':
            chart = chart_gen.create_scatter_plot(data, results)
        elif chart_type == 'clustering':
            chart = chart_gen.create_clustering_viz(results)
        elif chart_type == 'ranking':
            chart = chart_gen.create_ranking_chart(results)
        else:
            return jsonify({'error': 'Unknown chart type'}), 400
        
        return jsonify({'chart': chart})
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rules')
def rules_configuration():
    """Rules configuration page."""
    return render_template('rules.html')


@app.route('/api/rules/default')
def get_default_rules():
    """Get default rules configuration."""
    try:
        rules_engine = _load_rules_engine()
        return jsonify(_build_rules_response(rules_engine))
    except Exception as e:
        logger.error(f"Error getting default rules: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rules', methods=['POST'])
def create_rule():
    """Create a new custom rule and persist configuration."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    payload = request.get_json(force=True)

    missing = {field for field in ('name', 'condition', 'weight') if field not in payload}
    if missing:
        return jsonify({'error': f"Missing required fields: {', '.join(sorted(missing))}"}), 400

    try:
        rule = Rule(
            name=payload['name'],
            condition=payload['condition'],
            weight=float(payload['weight']),
            description=payload.get('description', ''),
            enabled=_coerce_enabled(payload.get('enabled', True))
        )
    except (TypeError, ValueError) as exc:
        logger.error(f"Invalid rule payload: {exc}")
        return jsonify({'error': str(exc)}), 400

    try:
        engine = _load_rules_engine()
        engine.add_rule(rule)
        _save_rules(engine)
        return jsonify(_build_rules_response(engine)), 201
    except ValueError as exc:
        logger.error(f"Rule creation error: {exc}")
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        logger.error(f"Unexpected error creating rule: {exc}")
        return jsonify({'error': str(exc)}), 500


@app.route('/api/rules/<string:rule_name>', methods=['PUT'])
def update_rule(rule_name: str):
    """Update an existing rule identified by name."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    payload = request.get_json(force=True)

    allowed_fields = {'name', 'condition', 'weight', 'description', 'enabled'}
    updates = {k: v for k, v in payload.items() if k in allowed_fields}

    if not updates:
        return jsonify({'error': 'No updatable fields provided'}), 400

    if 'weight' in updates:
        try:
            updates['weight'] = float(updates['weight'])
        except (TypeError, ValueError) as exc:
            return jsonify({'error': f'Invalid weight value: {exc}'}), 400

    if 'enabled' in updates:
        updates['enabled'] = _coerce_enabled(updates['enabled'])

    try:
        engine = _load_rules_engine()
        updated = engine.update_rule(rule_name, **updates)
        if not updated:
            return jsonify({'error': f"Rule '{rule_name}' not found"}), 404

        _save_rules(engine)
        return jsonify(_build_rules_response(engine))
    except ValueError as exc:
        logger.error(f"Rule update error: {exc}")
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        logger.error(f"Unexpected error updating rule '{rule_name}': {exc}")
        return jsonify({'error': str(exc)}), 500


@app.route('/api/rules/<string:rule_name>', methods=['DELETE'])
def delete_rule(rule_name: str):
    """Delete a custom rule by name."""
    default_names = _default_rule_names()
    if rule_name in default_names:
        return jsonify({'error': 'Default rules cannot be deleted'}), 400

    try:
        engine = _load_rules_engine()
        removed = engine.remove_rule(rule_name)
        if not removed:
            return jsonify({'error': f"Rule '{rule_name}' not found"}), 404

        _save_rules(engine)
        return jsonify(_build_rules_response(engine))
    except Exception as exc:
        logger.error(f"Unexpected error deleting rule '{rule_name}': {exc}")
        return jsonify({'error': str(exc)}), 500


@app.route('/api/rules/test', methods=['POST'])
def test_rules():
    """Test rules against sample data."""
    try:
        rules_config = request.json or {}

        # Create sample data for testing
        processor = MetaAnalysisProcessor()
        sample_data = processor.create_sample_data(10)

        # Configure rules engine
        rules_engine = _load_rules_engine()

        if 'custom_rules' in rules_config:
            for rule_data in rules_config['custom_rules']:
                rule = Rule(**rule_data)
                rules_engine.add_rule(rule)
        
        # Test rules
        pairs_data = sample_data.to_dict('records')
        ranked_pairs = rules_engine.rank_gene_pairs(pairs_data)
        
        return jsonify({
            'sample_pairs': len(sample_data),
            'ranked_pairs': len(ranked_pairs),
            'top_scores': [score for _, score in ranked_pairs[:5]],
            'rules_summary': rules_engine.get_rule_summary()
        })
        
    except Exception as e:
        logger.error(f"Rules test error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/export/<session_id>')
def export_results(session_id):
    """Export analysis results."""
    try:
        if session_id not in analysis_results:
            flash('Session not found', 'error')
            return redirect(url_for('index'))
        
        session_data = analysis_results[session_id]
        
        if 'analysis_results' not in session_data:
            flash('Analysis not completed', 'warning')
            return redirect(url_for('analyze', session_id=session_id))
        
        results = session_data['analysis_results']
        
        # Create export DataFrame
        export_data = []
        for rec in results.get('recommendations', []):
            export_data.append({
                'Rank': rec['rank'],
                'Gene_A': rec['gene_a'],
                'Gene_B': rec['gene_b'],
                'Combined_Score': rec['combined_score'],
                'Rules_Score': rec['rules_score'],
                'ML_Confidence': rec['ml_confidence'],
                'High_Confidence': rec['is_high_confidence'],
                'Outlier': rec['is_outlier'],
                'P_Value_SS': rec['statistical_measures']['p_ss'],
                'P_Value_SOTH': rec['statistical_measures']['p_soth'],
                'Effect_Size_SS': rec['statistical_measures']['dz_ss_mean'],
                'Effect_Size_SOTH': rec['statistical_measures']['dz_soth_mean']
            })
        
        export_df = pd.DataFrame(export_data)
        
        # Export to Excel
        filename = f"gene_pair_analysis_{session_id}.xlsx"
        filepath = RESULTS_FOLDER / filename
        
        export_df.to_excel(filepath, index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        flash(f'Export failed: {str(e)}', 'error')
        return redirect(url_for('results', session_id=session_id))


@app.route('/about')
def about():
    """About page with system information."""
    return render_template('about.html')


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('error.html', error_code=404, error_message='Page not found'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', error_code=500, error_message='Internal server error'), 500


@app.route('/api/dashboard/<session_id>')
def get_dashboard_data(session_id):
    """Get dashboard data for a session."""
    try:
        if session_id not in analysis_results:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = analysis_results[session_id]
        
        # If analysis hasn't been run yet, return basic data
        if 'analysis_results' not in session_data:
            return jsonify({
                'session_id': session_id,
                'status': 'uploaded',
                'data_info': {
                    'total_pairs': len(session_data['data']),
                    'columns': list(session_data['data'].columns),
                    'source': session_data.get('source', 'file')
                }
            })
        
        # Return full analysis results
        results = session_data['analysis_results']
        
        dashboard_data = {
            'session_id': session_id,
            'status': 'analyzed',
            'summary_stats': results.get('summary_stats', {}),
            'recommendations': results.get('recommendations', [])[:20],  # Top 20
            'total_recommendations': len(results.get('recommendations', [])),
            'data_source': session_data.get('source', 'file')
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress/<session_id>')
def get_analysis_progress(session_id):
    """Get analysis progress for a session."""
    if session_id not in analysis_results:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = analysis_results[session_id]
    
    # Check if analysis is in progress
    if 'analysis_in_progress' in session_data:
        return jsonify({
            'in_progress': True,
            'progress': session_data.get('analysis_progress', 0),
            'status': session_data.get('analysis_status', 'Processing...')
        })
    
    # Check if analysis is completed
    if 'analysis_results' in session_data:
        return jsonify({
            'in_progress': False,
            'completed': True,
            'redirect_url': url_for('results', session_id=session_id)
        })
    
    # Analysis hasn't started yet
    return jsonify({
        'in_progress': False,
        'completed': False,
        'status': 'Ready for analysis'
    })


if __name__ == '__main__':
    # For development only
    print(f"Using {'mock' if USING_MOCK_MODULES else 'real'} analysis modules")
    print("Starting Gene Pair ML Analysis web application...")
    
    app.run(debug=True, host='0.0.0.0', port=5001)