"""
Iris Flower Classification - Machine Learning Project

This script performs a comprehensive analysis and classification of Iris flowers
based on their sepal and petal measurements. The dataset contains three species:
- Iris Setosa
- Iris Versicolor  
- Iris Virginica

Author: Parth
Date: May 25, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IrisClassifier:
    """
    A comprehensive Iris flower classification system
    """
    
    def __init__(self, data_path):
        """
        Initialize the classifier with data path
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration
        """
        print("=" * 60)
        print("IRIS FLOWER CLASSIFICATION PROJECT")
        print("=" * 60)
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        print(f"\n📊 Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        
        # Basic information
        print(f"\n📋 Dataset Info:")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Data types:\n{self.data.dtypes}")
        
        # Display first few rows
        print(f"\n🔍 First 5 rows:")
        print(self.data.head())
        
        # Statistical summary
        print(f"\n📈 Statistical Summary:")
        print(self.data.describe())
        
        # Check for missing values
        print(f"\n❓ Missing Values:")
        missing_values = self.data.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing_values)
            
        # Species distribution
        print(f"\n🌸 Species Distribution:")
        species_counts = self.data['Species'].value_counts()
        print(species_counts)
        
        for species, count in species_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"{species}: {count} samples ({percentage:.1f}%)")
            
        return self.data
    
    def visualize_data(self):
        """
        Create comprehensive visualizations of the dataset
        """
        print(f"\n🎨 Creating visualizations...")
        
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Distribution of each feature by species
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        
        for i, feature in enumerate(features, 1):
            plt.subplot(3, 4, i)
            for species in self.data['Species'].unique():
                data_subset = self.data[self.data['Species'] == species][feature]
                plt.hist(data_subset, alpha=0.7, label=species, bins=15)
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {feature}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Box plots for each feature
        for i, feature in enumerate(features, 5):
            plt.subplot(3, 4, i)
            sns.boxplot(data=self.data, x='Species', y=feature)
            plt.title(f'Box Plot: {feature}')
            plt.xticks(rotation=45)
        
        # 3. Correlation heatmap
        plt.subplot(3, 4, 9)
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        # 4. Pairplot (scatter plot matrix)
        plt.subplot(3, 4, 10)
        # Create a simple scatter plot as subplot
        for species in self.data['Species'].unique():
            subset = self.data[self.data['Species'] == species]
            plt.scatter(subset['SepalLengthCm'], subset['PetalLengthCm'], 
                       label=species, alpha=0.7)
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.title('Sepal Length vs Petal Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Species count
        plt.subplot(3, 4, 11)
        species_counts = self.data['Species'].value_counts()
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        plt.pie(species_counts.values, labels=species_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Species Distribution')
        
        # 6. Feature importance (mean values by species)
        plt.subplot(3, 4, 12)
        species_means = self.data.groupby('Species')[features].mean()
        species_means.plot(kind='bar', width=0.8)
        plt.title('Average Feature Values by Species')
        plt.ylabel('Average Value (cm)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('iris_data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional detailed pairplot
        print("Creating detailed pairplot...")
        plt.figure(figsize=(12, 10))
        sns.pairplot(self.data, hue='Species', diag_kind='hist', 
                    plot_kws={'alpha': 0.7}, diag_kws={'alpha': 0.7})
        plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_data(self):
        """
        Prepare data for machine learning
        """
        print(f"\n🔧 Preparing data for machine learning...")
        
        # Separate features and target
        # Drop 'Id' column if it exists and 'Species' for features
        feature_columns = [col for col in self.data.columns if col not in ['Id', 'Species']]
        self.X = self.data[feature_columns]
        self.y = self.data['Species']
        
        print(f"Features: {list(self.X.columns)}")
        print(f"Target classes: {list(self.y.unique())}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Feature scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Data preparation completed!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        print(f"\n🤖 Training multiple machine learning models...")
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }
        
        # Train models and evaluate
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # Use scaled data for models that benefit from it
            if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                # Cross-validation with scaled data
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                # Cross-validation with original data
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"✅ {name} trained successfully!")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning for the best models
        """
        print(f"\n⚙️ Performing hyperparameter tuning...")
        
        # Hyperparameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['rbf', 'linear']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        tuned_models = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"\n🔍 Tuning {model_name}...")
            
            # Get the base model
            if model_name == 'Random Forest':
                base_model = RandomForestClassifier(random_state=42)
                X_train_data, X_test_data = self.X_train, self.X_test
            elif model_name == 'SVM':
                base_model = SVC(random_state=42, probability=True)
                X_train_data, X_test_data = self.X_train_scaled, self.X_test_scaled
            elif model_name == 'K-Nearest Neighbors':
                base_model = KNeighborsClassifier()
                X_train_data, X_test_data = self.X_train_scaled, self.X_test_scaled
            
            # Grid search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_data, self.y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_data)
            
            # Update results
            accuracy = accuracy_score(self.y_test, y_pred)
            self.results[f'{model_name} (Tuned)'] = {
                'accuracy': accuracy,
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                'best_params': grid_search.best_params_,
                'predictions': y_pred
            }
            
            tuned_models[model_name] = best_model
            
            print(f"✅ Best parameters for {model_name}:")
            for param, value in grid_search.best_params_.items():
                print(f"   {param}: {value}")
            print(f"   Best CV Score: {grid_search.best_score_:.4f}")
            print(f"   Test Accuracy: {accuracy:.4f}")
        
        return tuned_models
    
    def evaluate_models(self):
        """
        Comprehensive evaluation of all models
        """
        print(f"\n📊 MODEL EVALUATION RESULTS")
        print("=" * 80)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        # Sort by accuracy
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        print(results_df[['accuracy', 'precision', 'recall', 'f1_score']])
        
        # Find best model
        best_model_name = results_df.index[0]
        best_accuracy = results_df.loc[best_model_name, 'accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"🎯 Best Accuracy: {best_accuracy:.4f}")
        
        # Create comparison visualization
        self.plot_model_comparison()
        
        # Detailed evaluation for best model
        self.detailed_evaluation(best_model_name)
        
        return best_model_name, results_df
    
    def plot_model_comparison(self):
        """
        Create visualizations comparing model performance
        """
        # Prepare data for plotting
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        f1_scores = [self.results[model]['f1_score'] for model in models]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Accuracy comparison
        axes[0, 0].bar(range(len(models)), accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. F1-Score comparison
        axes[0, 1].bar(range(len(models)), f1_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Accuracy vs F1-Score scatter
        axes[1, 0].scatter(accuracies, f1_scores, s=100, alpha=0.7, color='green')
        for i, model in enumerate(models):
            axes[1, 0].annotate(model, (accuracies[i], f1_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Accuracy')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('Accuracy vs F1-Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cross-validation scores (if available)
        cv_means = []
        cv_stds = []
        cv_models = []
        
        for model in models:
            if 'cv_mean' in self.results[model]:
                cv_means.append(self.results[model]['cv_mean'])
                cv_stds.append(self.results[model]['cv_std'])
                cv_models.append(model)
        
        if cv_means:
            x_pos = range(len(cv_models))
            axes[1, 1].bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
                          color='gold', alpha=0.7, error_kw={'elinewidth': 2})
            axes[1, 1].set_title('Cross-Validation Scores')
            axes[1, 1].set_ylabel('CV Score')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(cv_models, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detailed_evaluation(self, best_model_name):
        """
        Detailed evaluation of the best model
        """
        print(f"\n🔍 DETAILED EVALUATION - {best_model_name}")
        print("=" * 60)
        
        y_pred = self.results[best_model_name]['predictions']
        
        # Classification report
        print("📋 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.y.unique(), yticklabels=self.y.unique())
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'confusion_matrix_{best_model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance (if available)
        if 'Random Forest' in best_model_name or 'Decision Tree' in best_model_name:
            self.plot_feature_importance(best_model_name)
    
    def plot_feature_importance(self, model_name):
        """
        Plot feature importance for tree-based models
        """
        # Get the model (this is simplified - in practice you'd store the trained models)
        if 'Random Forest' in model_name:
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(self.X_train, self.y_train)
        elif 'Decision Tree' in model_name:
            model = DecisionTreeClassifier(random_state=42)
            model.fit(self.X_train, self.y_train)
        else:
            return
        
        # Feature importance
        importance = model.feature_importances_
        feature_names = self.X.columns
        
        # Create plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        
        plt.bar(range(len(importance)), importance[indices], alpha=0.7, color='green')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(importance[indices]):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🌟 Feature Importance Rankings:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    def make_predictions(self, best_model_name):
        """
        Make predictions on new data points
        """
        print(f"\n🔮 MAKING PREDICTIONS")
        print("=" * 40)
        
        # Example new data points
        new_samples = np.array([
            [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
            [6.2, 2.8, 4.8, 1.8],  # Likely Versicolor  
            [7.2, 3.0, 5.8, 1.6]   # Likely Virginica
        ])
        
        feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
        
        # Recreate and retrain the best model for predictions
        if 'Random Forest' in best_model_name:
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(new_samples)
            probabilities = model.predict_proba(new_samples)
        elif 'Logistic Regression' in best_model_name:
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(self.X_train_scaled, self.y_train)
            new_samples_scaled = self.scaler.transform(new_samples)
            predictions = model.predict(new_samples_scaled)
            probabilities = model.predict_proba(new_samples_scaled)
        else:
            # Default to Random Forest
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(new_samples)
            probabilities = model.predict_proba(new_samples)
        
        print("Sample predictions on new data:")
        print("-" * 60)
        
        for i, (sample, pred, prob) in enumerate(zip(new_samples, predictions, probabilities)):
            print(f"\nSample {i+1}:")
            for j, (feature, value) in enumerate(zip(feature_names, sample)):
                print(f"  {feature}: {value}")
            print(f"  Predicted Species: {pred}")
            print(f"  Confidence Scores:")
            for species, confidence in zip(model.classes_, prob):
                print(f"    {species}: {confidence:.4f}")
    
    def generate_summary_report(self, best_model_name, results_df):
        """
        Generate a comprehensive summary report
        """
        print(f"\n📄 IRIS CLASSIFICATION PROJECT SUMMARY")
        print("=" * 60)
        
        print(f"🗓️  Date: May 25, 2025")
        print(f"📁  Dataset: {self.data_path}")
        print(f"📊  Total Samples: {len(self.data)}")
        print(f"🏷️  Classes: {len(self.data['Species'].unique())}")
        print(f"🔢  Features: {len(self.X.columns)}")
        
        print(f"\n🏆 BEST MODEL PERFORMANCE:")
        print(f"   Model: {best_model_name}")
        print(f"   Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
        print(f"   Precision: {results_df.loc[best_model_name, 'precision']:.4f}")
        print(f"   Recall: {results_df.loc[best_model_name, 'recall']:.4f}")
        print(f"   F1-Score: {results_df.loc[best_model_name, 'f1_score']:.4f}")
        
        print(f"\n📈 ALL MODELS RANKING:")
        for i, (model, row) in enumerate(results_df.iterrows(), 1):
            print(f"   {i}. {model}: {row['accuracy']:.4f}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   • The Iris dataset is well-balanced with equal samples per species")
        print(f"   • Petal measurements are generally more discriminative than sepal measurements")
        print(f"   • Tree-based models perform exceptionally well on this dataset")
        print(f"   • The classification problem is relatively easy with high accuracy achievable")
        
        print(f"\n📁 Generated Files:")
        print(f"   • iris_data_exploration.png - Data exploration visualizations")
        print(f"   • iris_pairplot.png - Feature relationships")
        print(f"   • model_comparison.png - Model performance comparison")
        print(f"   • confusion_matrix_{best_model_name.replace(' ', '_')}.png - Confusion matrix")
        
        print(f"\n✅ Project completed successfully!")
    
    def run_complete_analysis(self):
        """
        Run the complete machine learning pipeline
        """
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Visualize data
            self.visualize_data()
            
            # Step 3: Prepare data
            self.prepare_data()
            
            # Step 4: Train models
            self.train_models()
            
            # Step 5: Hyperparameter tuning
            self.hyperparameter_tuning()
            
            # Step 6: Evaluate models
            best_model_name, results_df = self.evaluate_models()
            
            # Step 7: Make sample predictions
            self.make_predictions(best_model_name)
            
            # Step 8: Generate summary report
            self.generate_summary_report(best_model_name, results_df)
            
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main function to run the Iris classification project
    """
    # Initialize the classifier
    data_path = "Iris.csv"  # Assuming the CSV is in the same directory
    classifier = IrisClassifier(data_path)
    
    # Run complete analysis
    classifier.run_complete_analysis()


if __name__ == "__main__":
    main()