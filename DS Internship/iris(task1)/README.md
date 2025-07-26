# 🌸 Iris Flower Classification Project

A comprehensive machine learning project that classifies Iris flowers into three species based on their sepal and petal measurements.

## 📋 Project Overview

This project implements a complete machine learning pipeline to classify Iris flowers into one of three species:
- **Iris Setosa** 🌺
- **Iris Versicolor** 🌻  
- **Iris Virginica** 🌷

The classification is based on four key measurements:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## 🎯 Project Goals

1. **Data Exploration**: Comprehensive analysis of the Iris dataset
2. **Visualization**: Create insightful plots to understand feature relationships
3. **Model Training**: Train multiple machine learning algorithms
4. **Model Evaluation**: Compare performance using various metrics
5. **Hyperparameter Tuning**: Optimize the best-performing models
6. **Prediction**: Make predictions on new flower measurements

## 📊 Dataset Information

- **Total Samples**: 150 (50 samples per species)
- **Features**: 4 numerical measurements
- **Target Classes**: 3 species (perfectly balanced dataset)
- **Missing Values**: None ✅
- **Data Quality**: High-quality, clean dataset

## 🔬 Machine Learning Models

The project evaluates the following algorithms:

1. **Logistic Regression** - Linear classification approach
2. **Decision Tree** - Rule-based classification
3. **Random Forest** - Ensemble of decision trees
4. **Support Vector Machine (SVM)** - Margin-based classification
5. **K-Nearest Neighbors (KNN)** - Instance-based learning
6. **Naive Bayes** - Probabilistic classifier

## 📈 Results Summary

### Best Model Performance
- **Winner**: Support Vector Machine (SVM) 🏆
- **Accuracy**: 96.67%
- **Precision**: 96.97%
- **Recall**: 96.67%
- **F1-Score**: 96.66%

### All Models Performance
| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | SVM | 96.67% |
| 2 | Naive Bayes | 96.67% |
| 3 | Random Forest (Tuned) | 96.67% |
| 4 | SVM (Tuned) | 96.67% |
| 5 | Logistic Regression | 93.33% |
| 6 | Decision Tree | 93.33% |
| 7 | K-Nearest Neighbors | 93.33% |
| 8 | KNN (Tuned) | 93.33% |
| 9 | Random Forest | 90.00% |

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
pip (Python package installer)
```

### Installation
1. Clone or download this project
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project
```bash
python task1.py
```

## 📁 Project Structure

```
iris(task1)/
│
├── task1.py                          # Main Python script
├── Iris.csv                          # Dataset
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
└── Generated Output Files:
    ├── iris_data_exploration.png     # Data exploration visualizations
    ├── iris_pairplot.png             # Feature relationship plots
    ├── model_comparison.png          # Model performance comparison
    └── confusion_matrix_SVM.png      # Best model confusion matrix
```

## 🔍 Key Features

### Data Analysis
- ✅ Comprehensive statistical summary
- ✅ Missing value detection
- ✅ Class distribution analysis
- ✅ Feature correlation analysis

### Visualization
- ✅ Feature distribution plots
- ✅ Box plots for outlier detection
- ✅ Correlation heatmap
- ✅ Pairplot for feature relationships
- ✅ Model performance comparisons

### Machine Learning Pipeline
- ✅ Data preprocessing and scaling
- ✅ Train-test split with stratification
- ✅ Cross-validation for robust evaluation
- ✅ Hyperparameter tuning with Grid Search
- ✅ Multiple evaluation metrics

### Model Evaluation
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Confusion matrices
- ✅ Classification reports
- ✅ Feature importance analysis
- ✅ Cross-validation scores

## 🎨 Visualizations Generated

1. **Data Exploration Dashboard**: Feature distributions and statistics
2. **Pairplot**: Scatter plot matrix showing feature relationships
3. **Model Comparison Charts**: Bar charts comparing model performance
4. **Confusion Matrix**: Detailed classification results for best model

## 🧠 Key Insights

1. **Dataset Quality**: The Iris dataset is exceptionally well-balanced and clean
2. **Feature Importance**: Petal measurements are more discriminative than sepal measurements
3. **Model Performance**: Multiple models achieve >95% accuracy, indicating the problem is well-suited for machine learning
4. **Species Separation**: Iris Setosa is easily separable, while Versicolor and Virginica have some overlap

## 🔮 Sample Predictions

The trained model can classify new flower measurements:

```python
# Example: New flower measurements
Sample: [5.1, 3.5, 1.4, 0.2]
Prediction: Iris-setosa (100% confidence)

Sample: [6.2, 2.8, 4.8, 1.8]  
Prediction: Iris-virginica (72% confidence)
```

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Static plotting library
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning library

## 🎓 Educational Value

This project demonstrates:
- Complete machine learning workflow
- Data preprocessing techniques
- Model selection and evaluation
- Hyperparameter optimization
- Result visualization and interpretation
- Best practices in ML development

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements:
- Add new machine learning algorithms
- Enhance visualizations
- Improve documentation
- Add feature engineering techniques

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Parth** - Data Science Intern

---

*This project was created as part of a Data Science internship to demonstrate proficiency in machine learning, data analysis, and Python programming.*
