# Sales Prediction Analysis - Task 4

## 📊 Project Overview

This comprehensive data science project analyzes advertising spend across different channels (TV, Radio, Newspaper) to predict sales outcomes. The analysis employs multiple machine learning algorithms to find the optimal model for sales forecasting and provides valuable business insights for advertising strategy optimization.

## 🎯 Project Objectives

- **Primary Goal**: Predict sales based on advertising spend across TV, Radio, and Newspaper channels
- **Secondary Goals**: 
  - Identify which advertising channels are most effective
  - Analyze synergy effects between different advertising channels
  - Provide actionable business insights for marketing budget allocation
  - Compare performance of various regression algorithms

## 📋 Dataset Information

- **Dataset**: Advertising.csv
- **Records**: 200 sales observations
- **Features**: 4 columns (TV, Radio, Newspaper, Sales)
- **Type**: Regression problem (continuous target variable)
- **Source**: Marketing spend and corresponding sales data

### Dataset Structure
```
TV        - Advertising spend on TV (in thousands)
Radio     - Advertising spend on Radio (in thousands) 
Newspaper - Advertising spend on Newspaper (in thousands)
Sales     - Sales revenue (in thousands)
```

## 🔬 Analysis Features

### 1. Data Exploration & Visualization
- Comprehensive statistical summary
- Distribution analysis of all variables
- Correlation analysis between advertising channels and sales
- Advanced visualization suite including:
  - Channel performance comparison
  - Advertising mix analysis
  - Sales prediction scatter plots
  - Feature importance rankings
  - Business insights dashboard

### 2. Advanced Feature Engineering
- **Efficiency Ratios**: Sales per dollar spent on each channel
- **Interaction Terms**: TV×Radio, TV×Newspaper, Radio×Newspaper synergies
- **Polynomial Features**: Non-linear relationship modeling
- **Total Spend**: Combined advertising budget analysis
- **Dominant Channel**: Identification of primary advertising focus

### 3. Machine Learning Models
The analysis implements and compares 11 different regression algorithms:

1. **Linear Regression** - Baseline model
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization with feature selection
4. **ElasticNet** - Combined L1/L2 regularization
5. **Decision Tree** - Non-linear decision boundaries
6. **Random Forest** - Ensemble of decision trees
7. **Extra Trees** - Extremely randomized trees
8. **Gradient Boosting** - Sequential learning ensemble
9. **Support Vector Regression** - Kernel-based regression
10. **K-Nearest Neighbors** - Instance-based learning
11. **Neural Network (MLP)** - Multi-layer perceptron

### 4. Model Evaluation & Optimization
- **Cross-validation**: 5-fold CV for robust performance assessment
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Multiple Metrics**: MAE, MSE, RMSE, R², MAPE
- **Statistical Testing**: Model significance and assumptions validation
- **Feature Importance**: Understanding key drivers

### 5. Business Intelligence
- **ROI Analysis**: Return on investment per advertising channel
- **Channel Effectiveness**: Ranking of advertising mediums
- **Budget Optimization**: Recommendations for spend allocation
- **Synergy Analysis**: Multi-channel interaction effects

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation
1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
python task4.py
```

## 📁 Project Structure
```
sales(task4)/
├── task4.py                    # Main analysis script
├── requirements.txt            # Project dependencies
├── README.md                  # Project documentation
├── Advertising.csv            # Source dataset
└── outputs/                   # Generated files (created during execution)
    ├── sales_prediction_analysis_report.txt
    ├── advertising_channel_performance.png
    ├── sales_prediction_results.png
    ├── advertising_mix_analysis.png
    ├── feature_importance_ranking.png
    └── business_insights_dashboard.png
```

## 📈 Expected Results

### Model Performance
The analysis typically achieves high accuracy with ensemble methods:
- **Random Forest**: R² > 0.95
- **Gradient Boosting**: R² > 0.94
- **Extra Trees**: R² > 0.95

### Key Business Insights
1. **TV Advertising** typically shows the strongest correlation with sales
2. **Radio + TV synergy** often produces multiplicative effects
3. **Newspaper advertising** may show diminishing returns
4. **Optimal budget allocation** recommendations based on ROI analysis

## 🔍 Analysis Outputs

### 1. Comprehensive Report
- Detailed statistical analysis
- Model performance comparison
- Business recommendations
- Statistical significance tests

### 2. Visualization Suite
- **Channel Performance**: Comparative effectiveness analysis
- **Prediction Results**: Actual vs predicted sales scatter plots
- **Mix Analysis**: Multi-dimensional advertising spend relationships
- **Feature Importance**: Ranked impact of different variables
- **Business Dashboard**: Executive summary visualizations

### 3. Model Insights
- Best performing algorithm identification
- Feature importance rankings
- Prediction confidence intervals
- Business optimization recommendations

## 🎯 Business Applications

1. **Marketing Budget Planning**: Optimize allocation across TV, Radio, Newspaper
2. **Sales Forecasting**: Predict expected sales from planned advertising spend
3. **Channel Strategy**: Identify most effective advertising mediums
4. **ROI Optimization**: Maximize return on advertising investment
5. **Campaign Planning**: Design multi-channel advertising campaigns

## 📊 Technical Specifications

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Statistics**: SciPy
- **Model Selection**: Cross-validation with grid search
- **Performance Metrics**: Multiple regression evaluation metrics

## 🔧 Customization Options

The analysis can be easily customized by modifying:
- **Model parameters** in the hyperparameter grids
- **Feature engineering** functions for domain-specific variables
- **Visualization themes** and color schemes
- **Evaluation metrics** based on business requirements
- **Cross-validation** strategies (time series, stratified, etc.)

## 📝 Notes

- The analysis assumes linear and non-linear relationships between advertising spend and sales
- Feature engineering includes interaction terms to capture synergy effects
- Model selection is based on cross-validated performance metrics
- Business insights are derived from statistical analysis and model interpretation

## 🏆 Project Highlights

- **Comprehensive**: 11 different ML algorithms compared
- **Business-Focused**: Actionable insights for marketing strategy
- **Robust**: Cross-validation and statistical testing
- **Visual**: Rich visualization suite for presentation
- **Scalable**: Easily adaptable to larger datasets and additional features

---

**Author**: Data Science Intern  
**Project**: Sales Prediction Analysis (Task 4)  
**Dataset**: Advertising.csv (200 records)  
**Completion**: Advanced regression analysis with business intelligence
