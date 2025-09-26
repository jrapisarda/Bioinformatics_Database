# AI Agent for Gene Pair Correlation Analysis - Technical and Functional Requirements

## Executive Summary

This document defines the requirements for an AI Agent system that learns from meta-analysis data of gene pairs and recommends strongly correlated gene pairs likely to have meaningful biological relationships. The system combines unsupervised machine learning with rules-based ranking to provide interpretable recommendations for bioinformatics research.

## 1. Project Overview

### 1.1 Purpose
Develop an AI Agent that analyzes meta-analysis statistics from gene expression studies to identify and rank gene pairs with strong correlations and potential biological significance, particularly in sepsis and septic shock research contexts.

### 1.2 Scope
- **Input**: Gene pair meta-analysis data with effect sizes, p-values, confidence intervals, and heterogeneity statistics
- **Output**: Ranked recommendations of gene pairs with explanatory rules and confidence scores
- **Domain**: Bioinformatics, specifically gene expression meta-analysis in sepsis/septic shock research
- **Learning Type**: Unsupervised learning with hybrid rules-based ranking

### 1.3 Known Baseline
- **Positive Control**: MS4A4A-CD86 pair (confirmed good relationship)
- **Data Location**: `C:\venvs\ref310\Projects\Bioinformatics_Database\AI\`
- **Training Data**: Meta-analysis results from multiple independent studies

## 2. Functional Requirements

### 2.1 Core Learning Capabilities

#### 2.1.1 Unsupervised Pattern Discovery
- **Requirement ID**: FR-001
- **Description**: The system shall identify patterns in gene pair relationships without labeled training data
- **Implementation**: Use clustering algorithms (K-means, DBSCAN, hierarchical clustering) to group similar gene pairs
- **Success Criteria**: Discover meaningful clusters that separate high-potential pairs from low-potential pairs

#### 2.1.2 Multi-Dimensional Feature Analysis
- **Requirement ID**: FR-002  
- **Description**: The system shall analyze multiple statistical dimensions simultaneously
- **Features to Analyze**:
  - Effect sizes (dz_ss_mean, dz_soth_mean)
  - Statistical significance (p_ss, p_soth)
  - Confidence intervals (ci_low, ci_high)
  - Heterogeneity measures (I2, Q statistics)
  - Z-scores and FDR-adjusted q-values
- **Success Criteria**: Create composite similarity measures incorporating all relevant statistical dimensions

#### 2.1.3 Baseline Calibration
- **Requirement ID**: FR-003
- **Description**: The system shall use the MS4A4A-CD86 pair as a positive control for calibration
- **Implementation**: 
  - Use MS4A4A-CD86 characteristics as reference patterns
  - Calculate similarity scores relative to this known good pair
- **Success Criteria**: MS4A4A-CD86 should rank in top 20% of all recommendations

### 2.2 Rules-Based Ranking System

#### 2.2.1 Configurable Rule Engine
- **Requirement ID**: FR-004
- **Description**: The system shall provide a configurable rule-based ranking mechanism
- **Components**:
  - Rule definition interface
  - Rule weight adjustment
  - Rule combination logic
  - Rule impact visualization
- **Success Criteria**: Users can add, modify, and weight rules affecting pair rankings

#### 2.2.2 Default Rule Set
- **Requirement ID**: FR-005
- **Description**: The system shall implement default ranking rules based on statistical best practices
- **Default Rules**:
  1. **Statistical Significance Rule**: `(p_ss < 0.1) AND (p_soth < 0.01)` → Weight: 0.25
  2. **Effect Size Rule**: `(abs_dz_ss > 0.3) AND (abs_dz_soth > 1.0)` → Weight: 0.30
  3. **Z-Score Strength Rule**: `(abs(dz_ss_z) > 1.5) AND (abs(dz_soth_z) > 3.0)` → Weight: 0.20
  4. **FDR Correction Rule**: `(q_ss < 0.2) AND (q_soth < 0.01)` → Weight: 0.15
  5. **Consistency Rule**: `(dz_ss_I2 < 50) OR (dz_soth_I2 < 75)` → Weight: 0.10
- **Success Criteria**: Rules can be independently enabled/disabled and weighted

#### 2.2.3 Custom Rule Addition
- **Requirement ID**: FR-006
- **Description**: The system shall allow users to add custom rules using simple conditional logic
- **Syntax Support**:
  - Comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
  - Logical operators: `AND`, `OR`, `NOT`
  - Mathematical functions: `abs()`, `sqrt()`, `log()`
  - Example: `dz_ss_mean > -0.5 OR p_ss < 0.05`
- **Success Criteria**: Custom rules integrate seamlessly with default rules in ranking calculations

### 2.3 Machine Learning Model Requirements

#### 2.3.1 Ensemble Approach
- **Requirement ID**: FR-007
- **Description**: The system shall implement multiple ML approaches and combine their results
- **Models to Include**:
  - Isolation Forest (anomaly detection for exceptional pairs)
  - DBSCAN clustering (density-based grouping)
  - Principal Component Analysis (dimensionality reduction)
  - Gaussian Mixture Models (probabilistic clustering)
- **Success Criteria**: Ensemble predictions outperform individual model predictions

#### 2.3.2 Feature Engineering
- **Requirement ID**: FR-008
- **Description**: The system shall create derived features to enhance learning
- **Derived Features**:
  - Combined p-value score: `sqrt(p_ss * p_soth)`
  - Combined effect size: `sqrt(abs_dz_ss * abs_dz_soth)`  
  - Statistical power indicator: `combined_z_score / (combined_p_value + 0.001)`
  - Confidence interval widths
  - Effect size ratios between conditions
- **Success Criteria**: Derived features improve model discriminative power

#### 2.3.3 Similarity Metrics
- **Requirement ID**: FR-009
- **Description**: The system shall calculate similarity scores between gene pairs
- **Metrics**:
  - Euclidean distance in normalized feature space
  - Mahalanobis distance accounting for feature covariance
  - Cosine similarity for effect size patterns
  - Weighted Manhattan distance with domain-specific weights
- **Success Criteria**: Similar pairs by biological knowledge show high similarity scores

### 2.4 Recommendation Generation

#### 2.4.1 Ranking Algorithm
- **Requirement ID**: FR-010
- **Description**: The system shall generate ranked recommendations combining ML and rules
- **Algorithm**: 
  1. ML models generate base similarity/anomaly scores
  2. Rules-based system calculates rule compliance scores  
  3. Weighted combination produces final ranking score
  4. Pairs sorted by final score in descending order
- **Success Criteria**: Top-ranked pairs show strong statistical evidence and rule compliance

#### 2.4.2 Explanation Generation
- **Requirement ID**: FR-011
- **Description**: The system shall provide explanations for each recommendation
- **Explanation Components**:
  - Which rules contributed positively/negatively
  - Rule-specific scores and weights
  - ML model confidence scores
  - Comparison to MS4A4A-CD86 baseline
  - Statistical interpretation of key metrics
- **Success Criteria**: Users can understand why specific pairs received their rankings

#### 2.4.3 Confidence Scoring
- **Requirement ID**: FR-012
- **Description**: The system shall provide confidence scores for recommendations
- **Confidence Factors**:
  - ML model consensus (agreement between different models)
  - Rule coverage (how many rules apply to the pair)
  - Statistical strength (Z-scores, p-values)
  - Similarity to known positive control
- **Success Criteria**: High-confidence recommendations correlate with expert validation

## 3. Technical Requirements

### 3.1 Data Processing

#### 3.1.1 Data Validation
- **Requirement ID**: TR-001
- **Description**: The system shall validate input data quality and completeness
- **Validations**:
  - Check for missing critical values
  - Validate statistical consistency (CI bounds, p-values vs Z-scores)
  - Detect outliers and anomalies
  - Verify gene name formatting
- **Success Criteria**: Invalid data is flagged with actionable error messages

#### 3.1.2 Data Normalization  
- **Requirement ID**: TR-002
- **Description**: The system shall normalize features for ML processing
- **Normalization Methods**:
  - Min-max scaling for bounded features
  - Z-score standardization for normally distributed features
  - Log transformation for skewed distributions
  - Robust scaling for features with outliers
- **Success Criteria**: All features contribute equally to distance calculations

#### 3.1.3 Missing Data Handling
- **Requirement ID**: TR-003
- **Description**: The system shall handle missing data appropriately
- **Strategies**:
  - Use statistical imputation for missing numerical values
  - Flag pairs with excessive missing data
  - Adjust confidence scores based on data completeness
- **Success Criteria**: Missing data does not bias recommendations

### 3.2 Model Architecture

#### 3.2.1 Modular Design
- **Requirement ID**: TR-004
- **Description**: The system shall use modular architecture for maintainability
- **Modules**:
  - Data preprocessing module
  - Feature engineering module  
  - ML model ensemble module
  - Rules engine module
  - Recommendation generation module
  - Explanation generation module
- **Success Criteria**: Modules can be updated independently without system-wide changes

#### 3.2.2 Configuration Management
- **Requirement ID**: TR-005
- **Description**: The system shall support configuration-driven behavior
- **Configuration Options**:
  - Model hyperparameters
  - Rule definitions and weights
  - Feature selection and engineering parameters
  - Output formatting options
- **Success Criteria**: System behavior can be modified without code changes

#### 3.2.3 Performance Requirements
- **Requirement ID**: TR-006
- **Description**: The system shall meet performance benchmarks
- **Performance Targets**:
  - Process 10,000 gene pairs within 60 seconds
  - Generate recommendations within 10 seconds of data loading
  - Memory usage under 4GB for typical datasets
- **Success Criteria**: Performance targets met on standard desktop hardware

### 3.3 Integration Requirements

#### 3.3.1 File I/O Support
- **Requirement ID**: TR-007
- **Description**: The system shall support multiple input/output formats
- **Supported Formats**:
  - **Input**: CSV, Excel, JSON, TSV
  - **Output**: CSV, Excel, JSON, HTML reports
- **Success Criteria**: System handles format conversions transparently

#### 3.3.2 API Interface
- **Requirement ID**: TR-008
- **Description**: The system shall provide programmatic access via API
- **API Endpoints**:
  - `POST /analyze` - Submit data for analysis
  - `GET /recommendations/{analysis_id}` - Retrieve recommendations
  - `POST /rules` - Add/modify ranking rules
  - `GET /explain/{pair_id}` - Get explanation for specific pair
- **Success Criteria**: API supports integration with external bioinformatics workflows

### 3.4 Output Requirements

#### 3.4.1 Recommendation Report
- **Requirement ID**: TR-009
- **Description**: The system shall generate comprehensive recommendation reports
- **Report Contents**:
  - Executive summary with top recommendations
  - Detailed rankings with explanations
  - Statistical summaries and visualizations
  - Rule impact analysis
  - Model performance metrics
- **Success Criteria**: Reports provide actionable insights for biological research

#### 3.4.2 Interactive Visualization
- **Requirement ID**: TR-010
- **Description**: The system shall provide interactive visualizations
- **Visualizations**:
  - Scatter plots of effect sizes with highlighting
  - Heatmaps of correlation matrices
  - Rule impact bar charts
  - Cluster visualization in reduced dimensions
- **Success Criteria**: Visualizations enhance understanding of recommendations

## 4. Data Requirements

### 4.1 Input Data Schema

#### 4.1.1 Required Fields
The system expects input data with the following mandatory fields:

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| pair_id | String | Unique identifier for gene pair |
| GeneAName | String | Name of first gene |
| GeneBName | String | Name of second gene |
| dz_ss_mean | Float | Effect size mean for sepsis condition |
| dz_soth_mean | Float | Effect size mean for septic shock condition |
| p_ss | Float | P-value for sepsis condition |
| p_soth | Float | P-value for septic shock condition |
| abs_dz_ss | Float | Absolute effect size for sepsis |
| abs_dz_soth | Float | Absolute effect size for septic shock |

#### 4.1.2 Optional Enhancement Fields
Additional fields that improve analysis quality:

| Field Name | Data Type | Description |
|------------|-----------|-------------|
| dz_ss_se | Float | Standard error of sepsis effect size |
| dz_soth_se | Float | Standard error of septic shock effect size |
| dz_ss_ci_low | Float | Lower confidence interval bound (sepsis) |
| dz_ss_ci_high | Float | Upper confidence interval bound (sepsis) |
| dz_ss_I2 | Float | Heterogeneity percentage (sepsis) |
| dz_soth_I2 | Float | Heterogeneity percentage (septic shock) |
| q_ss | Float | FDR-adjusted p-value (sepsis) |
| q_soth | Float | FDR-adjusted p-value (septic shock) |
| rank_score | Float | Existing composite ranking score |

## 5. Algorithm Logic and Approach

### 5.1 Overall Strategy

The system employs a hybrid approach combining unsupervised machine learning with configurable rule-based ranking:

1. **Feature Engineering Phase**: Transform raw meta-analysis statistics into ML-suitable features
2. **Unsupervised Learning Phase**: Apply multiple ML algorithms to discover patterns and similarities
3. **Rules Evaluation Phase**: Apply statistical and domain-specific rules to score pairs  
4. **Ensemble Ranking Phase**: Combine ML scores and rule scores into final rankings
5. **Explanation Generation Phase**: Generate interpretable explanations for recommendations

### 5.2 Machine Learning Approach

#### 5.2.1 Clustering Strategy
- **Primary Algorithm**: DBSCAN for density-based clustering of similar gene pairs
- **Secondary Algorithm**: K-means for hard cluster assignments
- **Tertiary Algorithm**: Gaussian Mixture Models for probabilistic cluster membership
- **Validation**: Use MS4A4A-CD86 pair to validate cluster quality

#### 5.2.2 Anomaly Detection
- **Algorithm**: Isolation Forest to identify pairs with unusual patterns
- **Purpose**: Find gene pairs that stand out statistically from the general population
- **Calibration**: Ensure MS4A4A-CD86 is detected as a positive anomaly

#### 5.2.3 Dimensionality Reduction
- **Algorithm**: Principal Component Analysis (PCA) for visualization and noise reduction
- **Purpose**: Identify principal dimensions of variation in the data
- **Application**: Use reduced dimensions for clustering and similarity calculations

### 5.3 Rules-Based Logic

#### 5.3.1 Rule Evaluation Engine
```python
def evaluate_rules(pair_data, rule_set):
    total_score = 0
    explanation = []
    
    for rule in rule_set:
        if rule.condition.evaluate(pair_data):
            score_contribution = rule.weight * rule.score_multiplier
            total_score += score_contribution
            explanation.append(f"Rule '{rule.name}' contributed +{score_contribution}")
        else:
            explanation.append(f"Rule '{rule.name}' not satisfied")
    
    return total_score, explanation
```

#### 5.3.2 Statistical Significance Rules
Based on meta-analysis best practices:
- **Strong Sepsis Effect**: `abs_dz_ss > 0.3 AND p_ss < 0.05`
- **Strong Septic Shock Effect**: `abs_dz_soth > 1.0 AND p_soth < 0.01`  
- **Consistent Direction**: `sign(dz_ss_mean) == sign(dz_soth_mean)`
- **Low Heterogeneity**: `dz_ss_I2 < 50 OR dz_soth_I2 < 75`

#### 5.3.3 Composite Scoring Logic
```python
final_score = (
    0.4 * ml_ensemble_score +
    0.4 * rules_based_score + 
    0.2 * similarity_to_baseline_score
)
```

### 5.4 Learning Without Labels

#### 5.4.1 Self-Supervised Learning Elements
- **Positive Control**: Use MS4A4A-CD86 as single positive example
- **Negative Control**: Use pairs in bottom quartile of rank_score as negative examples
- **Semi-Supervised**: Train anomaly detector to distinguish positive control from random pairs

#### 5.4.2 Pattern Discovery Approach
- **Statistical Profiles**: Create profiles of statistical patterns for effective pairs
- **Similarity Learning**: Learn what makes pairs similar to the known good pair
- **Outlier Detection**: Identify pairs that deviate significantly from typical patterns

### 5.5 Recommendation Algorithm

#### 5.5.1 Scoring Pipeline
1. **Feature Normalization**: Scale all features to [0,1] range
2. **ML Scoring**: Apply ensemble of unsupervised models
3. **Rules Scoring**: Evaluate all active rules  
4. **Baseline Similarity**: Calculate similarity to MS4A4A-CD86
5. **Final Ranking**: Weighted combination of all scores
6. **Confidence Calculation**: Based on model agreement and statistical strength

#### 5.5.2 Ranking Formula
```python
def calculate_final_score(pair):
    # ML component (0-1 scale)
    ml_score = normalize(ensemble_predict(pair))
    
    # Rules component (0-1 scale)  
    rules_score = normalize(evaluate_rules(pair))
    
    # Baseline similarity (0-1 scale)
    baseline_sim = calculate_similarity(pair, ms4a4a_cd86_profile)
    
    # Statistical strength bonus
    stat_bonus = min(1.0, abs(pair.dz_soth_z) / 5.0)
    
    return (
        0.35 * ml_score +
        0.35 * rules_score + 
        0.20 * baseline_sim +
        0.10 * stat_bonus
    )
```

## 6. Quality Assurance Requirements

### 6.1 Validation Strategy

#### 6.1.1 Cross-Validation
- **Requirement ID**: QA-001
- **Description**: The system shall validate ML models using cross-validation
- **Method**: 5-fold cross-validation with stratification based on rank_score quartiles
- **Success Criteria**: Model performance consistent across folds

#### 6.1.2 Positive Control Validation  
- **Requirement ID**: QA-002
- **Description**: The system shall consistently rank MS4A4A-CD86 highly
- **Target**: MS4A4A-CD86 should rank in top 20% of all pairs
- **Success Criteria**: Positive control ranking stability across different parameter settings

#### 6.1.3 Statistical Sanity Checks
- **Requirement ID**: QA-003  
- **Description**: The system shall validate statistical consistency
- **Checks**:
  - P-values correlate with Z-scores appropriately
  - Confidence intervals contain effect size means
  - FDR-adjusted q-values >= corresponding p-values
- **Success Criteria**: No statistical inconsistencies in recommendations

### 6.2 Performance Metrics

#### 6.2.1 Ranking Quality Metrics
- **Requirement ID**: QA-004
- **Description**: The system shall track ranking quality metrics
- **Metrics**:
  - Correlation with existing rank_score
  - Precision@K for top-K recommendations  
  - Mean reciprocal rank of known positive control
- **Success Criteria**: Metrics improve over baseline ranking approaches

#### 6.2.2 Model Agreement Metrics
- **Requirement ID**: QA-005
- **Description**: The system shall measure agreement between ensemble models
- **Metrics**:
  - Kendall's tau correlation between model rankings
  - Percentage of pairs with consensus recommendations
  - Variance in confidence scores
- **Success Criteria**: High agreement indicates stable recommendations

## 7. User Interface Requirements

### 7.1 Rule Configuration Interface

#### 7.1.1 Rule Builder
- **Requirement ID**: UI-001
- **Description**: The system shall provide graphical rule building interface
- **Features**:
  - Dropdown menus for fields and operators
  - Real-time syntax validation
  - Rule testing against sample data
- **Success Criteria**: Non-technical users can create valid rules

#### 7.1.2 Rule Management
- **Requirement ID**: UI-002
- **Description**: The system shall support rule lifecycle management
- **Features**:
  - Save/load rule sets
  - Enable/disable individual rules
  - Adjust rule weights via sliders
  - Import/export rule configurations
- **Success Criteria**: Users can efficiently manage complex rule sets

### 7.2 Results Visualization

#### 7.2.1 Recommendation Dashboard
- **Requirement ID**: UI-003
- **Description**: The system shall provide interactive results dashboard
- **Components**:
  - Sortable/filterable recommendations table
  - Rule impact visualization
  - Statistical distribution plots
  - Cluster visualization
- **Success Criteria**: Users can efficiently explore and understand recommendations

#### 7.2.2 Explanation Views
- **Requirement ID**: UI-004
- **Description**: The system shall provide detailed explanation views for pairs
- **Components**:
  - Rule-by-rule scoring breakdown
  - Comparison to positive control
  - Statistical significance indicators
  - Confidence score visualization
- **Success Criteria**: Users can understand rationale for each recommendation

## 8. System Integration

### 8.1 Bioinformatics Workflow Integration

#### 8.1.1 Common Tool Compatibility
- **Requirement ID**: SI-001
- **Description**: The system shall integrate with common bioinformatics tools
- **Integrations**:
  - R/Bioconductor package compatibility
  - Python scikit-learn pipeline integration
  - Galaxy workflow tool support
- **Success Criteria**: System fits into existing bioinformatics workflows

#### 8.1.2 Database Connectivity
- **Requirement ID**: SI-002
- **Description**: The system shall connect to common biological databases
- **Databases**:
  - Gene expression databases (GEO, ArrayExpress)
  - Gene ontology databases
  - Protein-protein interaction databases
- **Success Criteria**: System can enrich recommendations with external biological context

### 8.2 Deployment Options

#### 8.2.1 Standalone Application
- **Requirement ID**: SI-003
- **Description**: The system shall run as standalone desktop application
- **Features**:
  - Self-contained executable
  - Local data processing
  - No internet dependency for core functions
- **Success Criteria**: Researchers can use system offline on their own data

#### 8.2.2 Cloud Service Option
- **Requirement ID**: SI-004
- **Description**: The system shall optionally deploy as cloud service
- **Features**:
  - Web-based interface
  - Scalable processing for large datasets
  - Collaborative features for research teams
- **Success Criteria**: System scales to handle multiple concurrent users

## 9. Documentation Requirements

### 9.1 User Documentation

#### 9.1.1 User Manual
- **Requirement ID**: DOC-001
- **Description**: Comprehensive user manual covering all system features
- **Contents**:
  - Installation and setup instructions
  - Data preparation guidelines
  - Rule configuration tutorials
  - Interpretation guide for results
- **Success Criteria**: Users can successfully use system with minimal support

#### 9.1.2 API Documentation  
- **Requirement ID**: DOC-002
- **Description**: Complete API reference for programmatic integration
- **Contents**:
  - Endpoint descriptions
  - Request/response schemas
  - Code examples in Python and R
  - Error handling guidance
- **Success Criteria**: Developers can integrate system programmatically

### 9.2 Technical Documentation

#### 9.2.1 Algorithm Documentation
- **Requirement ID**: DOC-003
- **Description**: Detailed documentation of ML algorithms and statistical methods
- **Contents**:
  - Mathematical foundations
  - Implementation details
  - Parameter tuning guidance
  - Validation methodology
- **Success Criteria**: Technical users understand system's analytical approach

#### 9.2.2 Deployment Guide
- **Requirement ID**: DOC-004
- **Description**: Complete deployment and configuration guide
- **Contents**:
  - System requirements
  - Installation procedures
  - Configuration options
  - Troubleshooting guide
- **Success Criteria**: System administrators can deploy and maintain system

## 10. Acceptance Criteria

### 10.1 Functional Acceptance

#### 10.1.1 Recommendation Quality
- **Criteria**: MS4A4A-CD86 consistently ranks in top 20% of recommendations
- **Criteria**: System identifies at least 3 other promising gene pairs from sample data
- **Criteria**: Rules-based scoring provides interpretable explanations for rankings

#### 10.1.2 User Experience
- **Criteria**: Non-technical users can configure rules and interpret results
- **Criteria**: System processes sample dataset (5 pairs) in under 10 seconds
- **Criteria**: Generated explanations correlate with statistical significance

### 10.2 Technical Acceptance

#### 10.2.1 Performance Benchmarks
- **Criteria**: System handles datasets with 1000+ gene pairs
- **Criteria**: Memory usage remains under 2GB for typical datasets
- **Criteria**: API response times under 1 second for standard requests

#### 10.2.2 Integration Testing
- **Criteria**: System successfully imports data from Excel and CSV formats
- **Criteria**: Generated reports are readable by standard bioinformatics tools
- **Criteria**: API endpoints return valid JSON responses

### 10.3 Scientific Validation

#### 10.3.1 Statistical Consistency
- **Criteria**: System recommendations correlate with existing rank_score (r > 0.7)
- **Criteria**: High-confidence pairs show strong statistical evidence (p < 0.05)
- **Criteria**: Rule-based scores align with meta-analysis best practices

#### 10.3.2 Biological Relevance
- **Criteria**: Top recommendations include pairs with known biological relationships
- **Criteria**: System explanations reference relevant statistical measures
- **Criteria**: Confidence scores correlate with strength of statistical evidence

## 11. Risk Mitigation

### 11.1 Technical Risks

#### 11.1.1 Overfitting to Small Dataset
- **Risk**: With only 5 sample pairs, models may overfit
- **Mitigation**: Use simpler models, cross-validation, and statistical rules as primary ranking mechanism
- **Contingency**: Focus on rules-based approach if ML models show poor generalization

#### 11.1.2 Missing Data Handling
- **Risk**: Real datasets may have more missing values than sample
- **Mitigation**: Implement robust missing data strategies and adjust confidence scores
- **Contingency**: Provide clear warnings about data quality issues

### 11.2 Scientific Risks  

#### 11.2.1 Statistical Misinterpretation
- **Risk**: Incorrect interpretation of meta-analysis statistics
- **Mitigation**: Validate statistical logic with domain experts and published literature
- **Contingency**: Provide extensive documentation of statistical assumptions

#### 11.2.2 False Discovery Rate
- **Risk**: High false positive rate in gene pair recommendations  
- **Mitigation**: Use conservative statistical thresholds and FDR correction
- **Contingency**: Clearly communicate uncertainty and recommend experimental validation

## 12. Future Enhancements

### 12.1 Advanced Analytics

#### 12.1.1 Network Analysis
- **Enhancement**: Incorporate gene regulatory network analysis
- **Benefit**: Consider indirect relationships and pathway information
- **Timeline**: Phase 2 development

#### 12.1.2 Temporal Analysis
- **Enhancement**: Analyze changes in gene relationships over time
- **Benefit**: Understand dynamic biological processes
- **Timeline**: Phase 3 development  

### 12.2 Integration Enhancements

#### 12.2.1 Literature Mining
- **Enhancement**: Integrate with biomedical literature databases
- **Benefit**: Validate recommendations against published research
- **Timeline**: Phase 2 development

#### 12.2.2 Experimental Design Support
- **Enhancement**: Suggest experimental protocols for validation
- **Benefit**: Bridge computational predictions with laboratory work
- **Timeline**: Phase 3 development

## Conclusion

This requirements document defines a comprehensive AI Agent system for gene pair correlation analysis that combines the pattern recognition capabilities of unsupervised machine learning with the interpretability and domain knowledge of rules-based systems. The hybrid approach ensures both analytical power and scientific transparency, essential for bioinformatics research applications.

The system will provide actionable recommendations for gene pairs likely to have meaningful biological relationships, with full explanations of the analytical reasoning. By using the MS4A4A-CD86 pair as a positive control and implementing configurable rules, the system maintains scientific rigor while adapting to researcher needs.

Key success factors include:
- Robust handling of meta-analysis statistics
- Interpretable ranking explanations  
- Configurable rule-based scoring
- Integration with bioinformatics workflows
- Scientific validation of recommendations

This foundation will enable researchers to efficiently identify promising gene relationships for further experimental investigation, accelerating biological discovery in sepsis and related disease contexts.

# Updated AI Agent Requirements: Statistical Calculations for Large-Scale Gene Pair Analysis

## Statistical Methods Addendum to Requirements Document

### Priority Statistical Calculations

Based on analysis of your source data structure and research into meta-analysis best practices for gene correlation studies, here are the essential statistical calculations your AI Agent should implement:

## 1. Core Meta-Analysis Calculations (Essential - High Priority)

### A. Fisher's Z-Transformation for Correlations
```python
# Transform Spearman correlations for proper meta-analysis
fisher_z = 0.5 * ln((1 + rho_spearman) / (1 - rho_spearman))
standard_error = 1 / sqrt(n_samples - 3)
weight = 1 / (standard_error^2)
```

**Rationale**: Spearman correlations are not normally distributed and can't be directly averaged across studies. Fisher's Z transformation normalizes the distribution and enables proper statistical combination across studies with different sample sizes.

**ML Integration**: Use Fisher's Z values as primary correlation features rather than raw Spearman correlations.

### B. Weighted Meta-Analysis Effect Sizes
```python
# Calculate weighted mean effect size across studies within each condition
weighted_effect_size = sum(fisher_z * weight) / sum(weight)
se_weighted = 1 / sqrt(sum(weight))
back_transformed_correlation = tanh(weighted_effect_size)
```

**Rationale**: Studies with larger sample sizes (lower standard errors) should contribute more to the overall effect size estimate.

**ML Integration**: These become your condition-specific correlation features (control_weighted_r, sepsis_weighted_r, septic_shock_weighted_r).

### C. Heterogeneity Statistics
```python
# Cochran's Q test for between-study heterogeneity
Q = sum(weight * (fisher_z - weighted_effect_size)^2)
degrees_freedom = n_studies - 1
I_squared = max(0, (Q - degrees_freedom) / Q * 100)  # % heterogeneity
```

**Rationale**: High heterogeneity (I² > 50%) indicates inconsistent results across studies, reducing confidence in the meta-analysis.

**ML Integration**: I² values become reliability features - pairs with low heterogeneity are more trustworthy.

### D. Differential Correlation Analysis
```python
# Effect size of correlation change between conditions
delta_r_control_disease = disease_weighted_r - control_weighted_r
delta_r_sepsis_shock = shock_weighted_r - sepsis_weighted_r

# Statistical significance of correlation differences
se_difference = sqrt(se_control^2 + se_disease^2)
z_difference = delta_r_control_disease / se_difference
p_difference = 2 * (1 - norm_cdf(abs(z_difference)))
```

**Rationale**: The key insight from MS4A4A-CD86 is that correlations flip from positive in controls (+0.64) to negative in disease (-0.30), indicating regulatory disruption.

**ML Integration**: These become your most powerful discriminative features for identifying disease-relevant pairs.

## 2. Composite ML Features (Essential - High Priority)

### A. Statistical Power Indicators
```python
# Combined statistical strength across conditions
combined_p_value = sqrt(p_control * p_disease * p_sepsis)
combined_effect_size = sqrt(abs_weighted_r_control * abs_weighted_r_disease * abs_weighted_r_sepsis)
power_score = combined_z_score^2 / (combined_z_score^2 + total_sample_size)
```

**Rationale**: Pairs significant in multiple conditions with large effect sizes are most likely to represent real biological relationships.

### B. Consistency Scores
```python
# Measure of reliability across studies within conditions
consistency_score = 1 / (1 + mean(I_squared_across_conditions))
direction_consistency = sign(control_weighted_r) == sign(disease_weighted_r)
magnitude_consistency = 1 - coefficient_of_variation(effect_sizes_across_studies)
```

**Rationale**: Consistent findings across multiple independent studies are more reliable than single-study results.

## 3. Expression-Based Enhancements (Medium Priority)

### A. Expression Fold Changes
```python
# Log fold changes for each gene between conditions
geneA_log2fc = log2(geneA_expression_disease / geneA_expression_control)
geneB_log2fc = log2(geneB_expression_disease / geneB_expression_control)

# Magnitude of expression changes
expression_change_magnitude = sqrt(geneA_log2fc^2 + geneB_log2fc^2)
```

**Rationale**: From MS4A4A-CD86 analysis, CD86 shows 2.2x expression change while MS4A4A shows minimal change (1.007x). This asymmetry may be important for pair classification.

### B. Expression-Correlation Interactions
```python
# Does expression level correlate with correlation strength?
expression_correlation_interaction = correlation(expression_levels, correlation_strengths)
high_expression_correlation_bonus = (mean_expression > expression_threshold) * correlation_strength
```

**Rationale**: Highly expressed genes may show different correlation patterns than lowly expressed genes.

## 4. Study Quality Weighting (Medium Priority)

### A. Sample Size Adjustments
```python
# Effective sample size accounting for study quality
effective_n = n_samples * quality_factor
quality_factor = min(1.0, n_samples / median_sample_size) * (1 - publication_bias_indicator)

# Study reliability weights
study_weight = effective_n / (effective_n + reliability_penalty)
```

**Rationale**: Larger studies with better methodology should have more influence on final rankings.

### B. P-Value Distribution Analysis
```python
# Test for p-hacking or publication bias
p_value_uniformity_test = kstest(p_values_under_null, uniform_distribution)
publication_bias_score = excess_of_marginally_significant_p_values  # p ∈ [0.04, 0.06]
```

**Rationale**: Distributions skewed toward just-significant p-values may indicate questionable research practices.

## 5. Advanced Network Features (Lower Priority for MVP)

### A. Gene Centrality Measures
```python
# How often genes appear in high-ranking pairs
gene_degree = count_appearances_in_top_percentile_pairs
gene_betweenness = network_centrality_measure
hub_gene_score = gene_degree * mean_correlation_strength
```

### B. Pathway Enrichment Integration
```python
# Bonus for pairs in known biological pathways
pathway_bonus = (geneA_pathways.intersection(geneB_pathways).size > 0) * pathway_weight
go_term_similarity = semantic_similarity(geneA_go_terms, geneB_go_terms)
```

## 6. ML Model Logic Notes Field Implementation

### A. Decision Explanations
For each ranked pair, generate explanations like:
```
"Pair MS4A4A-CD86 ranked #2 (score: 0.847) because:
• Strong differential correlation: +0.64 (control) → -0.30 (septic shock) = -0.94 change [+0.25 points]
• High statistical power: Combined p-value = 1.2e-05, Effect size magnitude = 0.89 [+0.20 points]  
• Low heterogeneity: I² = 0% (septic shock), 69% (control) - consistent in disease [+0.15 points]
• Expression asymmetry: CD86 shows 2.2x fold change, MS4A4A minimal (1.007x) [+0.10 points]
• Multi-condition significance: Significant in 4/5 septic shock studies, 3/5 control studies [+0.12 points]
• ML similarity to validated pair: Cosine similarity = 1.0 (this IS the validated pair) [+0.15 points]
Rule contributions: Statistical_significance_rule (+0.25), Effect_size_rule (+0.30), Consistency_rule (+0.20)"
```

### B. Feature Attribution
```python
def generate_logic_explanation(pair_data, ml_scores, rule_scores):
    explanation = []
    
    # ML contribution breakdown
    for feature, importance in ml_feature_importance.items():
        if importance > 0.05:  # Only significant features
            explanation.append(f"ML feature '{feature}': {pair_data[feature]:.3f} (importance: {importance:.2f})")
    
    # Rules contribution breakdown  
    for rule_name, rule_score in rule_scores.items():
        if rule_score > 0:
            explanation.append(f"Rule '{rule_name}': +{rule_score:.2f} points - {rule.explanation}")
    
    # Final score composition
    explanation.append(f"Final score: {ml_weight:.1%} ML + {rules_weight:.1%} Rules + {baseline_sim_weight:.1%} Baseline similarity")
    
    return "; ".join(explanation)
```

## Implementation Priority for Large-Scale Analysis

### Phase 1 (MVP - Essential for thousands of pairs):
1. Fisher's Z transformation and weighted meta-analysis
2. Differential correlation calculations
3. Heterogeneity statistics (I², Q)
4. Composite statistical scores
5. Basic ML explanations in notes field

### Phase 2 (Scale optimization - for 10k+ pairs):
1. Efficient matrix operations for all pairwise calculations
2. Parallel processing of statistical computations
3. Memory-optimized storage of correlation matrices
4. Advanced consistency and quality metrics

### Phase 3 (Advanced features - for 100k+ pairs):
1. Network centrality measures
2. Pathway enrichment integration
3. Advanced outlier detection
4. Bootstrap confidence intervals

## Updated Rules Engine Configuration

Based on MS4A4A-CD86 analysis ($5000 validation), update default rules:

```python
DEFAULT_RULES = {
    "differential_correlation_rule": {
        "condition": "abs(control_weighted_r - disease_weighted_r) > 0.5",
        "weight": 0.30,  # Highest weight - this is the key insight
        "explanation": "Strong correlation change between control and disease"
    },
    "statistical_significance_rule": {
        "condition": "(combined_p_value < 0.01) AND (n_significant_studies >= 3)",
        "weight": 0.25,
        "explanation": "Statistically significant across multiple studies"
    },
    "effect_size_rule": {
        "condition": "combined_effect_size > 0.5",
        "weight": 0.20,
        "explanation": "Large effect size magnitude"
    },
    "consistency_rule": {
        "condition": "mean_I_squared < 60",
        "weight": 0.15,
        "explanation": "Low heterogeneity - consistent results across studies"  
    },
    "expression_asymmetry_rule": {
        "condition": "max(abs(geneA_log2fc), abs(geneB_log2fc)) > 1.0",
        "weight": 0.10,
        "explanation": "At least one gene shows substantial expression change"
    }
}
```

This statistical framework will enable your AI Agent to scale from thousands to hundreds of thousands of gene pairs while maintaining the interpretability and biological relevance that made MS4A4A-CD86 valuable to Big Pharma.