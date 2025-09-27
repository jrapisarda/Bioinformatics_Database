# Gene Pair ML Analysis - Web Interface

A comprehensive Flask-based web interface for the Gene Pair ML Analysis System.

## 🚀 Features

### Core Functionality
- **File Upload**: Drag-and-drop interface for Excel, CSV, and JSON files
- **Database Connection**: Connect to SQL Server, MySQL, or PostgreSQL databases
- **Rule Configuration**: Visual interface for customizing ranking rules
- **Interactive Analysis**: Real-time analysis with progress tracking
- **Comprehensive Dashboard**: Multiple visualization types and detailed results
- **Export Capabilities**: Download results in Excel format

### Interactive Components
- **Box Plots**: Distribution of effect sizes by condition
- **Scatter Plots**: Gene pair correlations with significance highlighting
- **Clustering Visualizations**: Ensemble clustering results
- **Ranking Charts**: Top recommendations with scores
- **Real-time Progress**: Live updates during analysis

## 📁 Project Structure

```
web_interface/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Landing page
│   ├── upload.html       # File upload interface
│   ├── analyze.html      # Analysis configuration
│   ├── results.html      # Results dashboard
│   ├── database.html     # Database connection
│   ├── rules.html        # Rule configuration
│   ├── about.html        # System information
│   └── error.html        # Error pages
├── static/               # Static assets (CSS, JS, images)
├── uploads/              # Uploaded files storage
├── results/              # Analysis results storage
└── README.md             # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Flask and dependencies (see requirements.txt)
- Modern web browser

### Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Application**
```bash
python app.py
```

3. **Access the Interface**
Open your browser and navigate to: `http://localhost:5000`

## 📊 Usage Guide

### 1. Upload Data
- Navigate to the Upload page
- Drag and drop your meta-analysis file or click to browse
- Supported formats: Excel (.xlsx, .xls), CSV, JSON
- The system will validate your data automatically

### 2. Configure Analysis
- Set machine learning parameters (number of features, contamination rate)
- Enable/disable different analysis components
- Configure custom ranking rules (optional)

### 3. Run Analysis
- Click "Start Analysis" to begin the ensemble ML process
- Monitor progress in real-time
- Analysis typically takes 2-5 minutes depending on data size

### 4. View Results
- Navigate through different visualization tabs
- Examine top gene pair recommendations
- Export results for further analysis

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=development  # For development mode
export FLASK_DEBUG=1          # Enable debug mode
```

### Application Settings
Key settings in `app.py`:
- `MAX_CONTENT_LENGTH`: Maximum file upload size (16MB)
- `UPLOAD_FOLDER`: Directory for uploaded files
- `RESULTS_FOLDER`: Directory for analysis results

## 📈 API Endpoints

### Core Endpoints
- `GET /` - Main landing page
- `GET /upload` - File upload interface
- `POST /upload` - Handle file uploads
- `GET /analyze/<session_id>` - Analysis configuration
- `POST /api/analyze/<session_id>` - Run analysis
- `GET /results/<session_id>` - View results

### API Endpoints
- `POST /api/database/test` - Test database connection
- `POST /api/database/data` - Fetch data from database
- `GET /api/results/<session_id>` - Get results as JSON
- `GET /api/visualization/<session_id>/<type>` - Get chart data
- `GET /api/rules/default` - Get default rules
- `POST /api/rules/test` - Test custom rules

## 🎨 Customization

### Styling
The interface uses Bootstrap 5 with custom CSS. Key style variables:
- Primary color: #2E8B8B (Deep teal)
- Secondary color: #F5F5DC (Warm ivory)
- Accent color: #CD853F (Peru/sandy brown)

### Templates
All templates extend `base.html` for consistent styling and navigation.

### Static Assets
Place custom CSS, JavaScript, and images in the `static/` directory.

## 🔍 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path and virtual environment

2. **"Template not found" errors**
   - Verify template files exist in `templates/` directory
   - Check file permissions

3. **"Port already in use"**
   - Change port: `python app.py --port 5001`
   - Kill existing process: `lsof -ti:5000 | xargs kill -9`

4. **Analysis fails to start**
   - Check data format and required columns
   - Verify file upload completed successfully
   - Check server logs for error messages

### Debug Mode
Enable debug mode for detailed error messages:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📋 Data Requirements

### Required Columns
Uploads must include each of the following fields. Column names are matched case-insensitively, and common legacy aliases (for example, `Gene_A` or `PairID`) are automatically mapped to the canonical names shown here:

- `pair_id`
- `n_studies_ss`
- `n_studies_soth`
- `dz_ss_mean`
- `dz_ss_se`
- `dz_ss_ci_low`
- `dz_ss_ci_high`
- `dz_ss_Q`
- `dz_ss_I2`
- `dz_ss_z`
- `p_ss`
- `dz_soth_mean`
- `dz_soth_se`
- `dz_soth_ci_low`
- `dz_soth_ci_high`
- `dz_soth_Q`
- `dz_soth_I2`
- `dz_soth_z`
- `p_soth`
- `kappa_ss`
- `kappa_soth`
- `abs_dz_ss`
- `abs_dz_soth`
- `q_ss`
- `q_soth`
- `rank_score`
- `GeneAName`
- `GeneBName`
- `GeneAKey`
- `GeneBKey`

### Optional Columns
- `study_key`
- `illness_label`
- `rho_spearman`

## 🔒 Security Considerations

### Production Deployment
- Set `debug=False` in production
- Use environment variables for sensitive configuration
- Implement proper authentication
- Set up HTTPS
- Configure CORS if needed

### File Upload Security
- File type validation
- Size limits enforced
- Secure filename handling
- Temporary file cleanup

## 🧪 Testing

### Unit Tests
```bash
# Test Flask app initialization
python test_flask.py

# Test core functionality
python test_system_verbose.py
```

### Manual Testing
1. Upload different file formats
2. Test with various data sizes
3. Verify all visualization types work
4. Check export functionality

## 📱 Browser Compatibility

### Supported Browsers
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

### Mobile Support
- Responsive design for tablets and phones
- Touch-friendly interface elements
- Optimized for mobile viewing

## 🚀 Performance Tips

### For Large Datasets
- Increase `MAX_CONTENT_LENGTH` for larger files
- Consider using a production WSGI server (Gunicorn, uWSGI)
- Implement pagination for large result sets
- Use database instead of in-memory storage

### Optimization
- Enable template caching in production
- Minimize static assets
- Use CDN for external libraries
- Implement result caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the Gene Pair ML Analysis System. See the main project license for details.

## 🔗 Related Resources

- [Main Project README](../README.md)
- [Gene Pair Agent Documentation](../gene_pair_agent/)
- [Visualization Module](../visualization/)

## 📞 Support

For technical support:
1. Check this README and main project documentation
2. Review server logs for error messages
3. Create an issue in the project repository
4. Include relevant error messages and steps to reproduce

---

**Built with Flask, Bootstrap, and Plotly.js**