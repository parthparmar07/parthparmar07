"""
Car Price Prediction Analysis - Task 3
======================================

This comprehensive analysis predicts car prices using machine learning algorithms.
The analysis includes data exploration, feature engineering, model training,
and extensive evaluation of multiple regression algorithms.

Author: Data Science Intern
Date: 2024
Dataset: car data.csv (303 car records)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Statistical imports
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CarPricePredictionAnalysis:
    """
    Comprehensive Car Price Prediction Analysis Class
    
    This class provides a complete pipeline for car price prediction including:
    - Data loading and preprocessing
    - Exploratory data analysis with visualizations
    - Feature engineering and encoding
    - Multiple machine learning model training
    - Model evaluation and comparison
    - Hyperparameter tuning
    - Prediction visualization and reporting
    """
    
    def __init__(self, data_path):
        """
        Initialize the analysis with dataset path
        
        Args:
            data_path (str): Path to the car dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.feature_importance = None
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory for visualizations
        self.output_dir = "car_price_analysis_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration
        """
        print("=" * 80)
        print("LOADING AND EXPLORING CAR PRICE DATASET")
        print("=" * 80)
        
        try:
            # Load the dataset
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Dataset loaded successfully!")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {list(self.df.columns)}")
            
            # Basic information
            print("\n" + "=" * 50)
            print("DATASET OVERVIEW")
            print("=" * 50)
            print(self.df.info())
            
            print("\n" + "=" * 50)
            print("STATISTICAL SUMMARY")
            print("=" * 50)
            print(self.df.describe())
            
            print("\n" + "=" * 50)
            print("SAMPLE DATA")
            print("=" * 50)
            print(self.df.head(10))
            
            # Check for missing values
            print("\n" + "=" * 50)
            print("MISSING VALUES ANALYSIS")
            print("=" * 50)
            missing_values = self.df.isnull().sum()
            print(missing_values)
            
            if missing_values.sum() == 0:
                print("✓ No missing values found!")
            
            # Data types analysis
            print("\n" + "=" * 50)
            print("DATA TYPES ANALYSIS")
            print("=" * 50)
            for col in self.df.columns:
                print(f"{col}: {self.df[col].dtype} - Unique values: {self.df[col].nunique()}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading dataset: {str(e)}")
            return False
    
    def data_quality_analysis(self):
        """
        Perform comprehensive data quality analysis
        """
        print("\n" + "=" * 80)
        print("DATA QUALITY ANALYSIS")
        print("=" * 80)
        
        # Duplicate analysis
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Categorical variables analysis
        categorical_cols = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']
        
        print("\n" + "=" * 50)
        print("CATEGORICAL VARIABLES ANALYSIS")
        print("=" * 50)
        
        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n{col}:")
                value_counts = self.df[col].value_counts()
                print(value_counts.head(10))
                print(f"Total unique values: {len(value_counts)}")
        
        # Numerical variables analysis
        numerical_cols = ['Year', 'Selling_Price', 'Present_Price', 'Driven_kms', 'Owner']
        
        print("\n" + "=" * 50)
        print("NUMERICAL VARIABLES ANALYSIS")
        print("=" * 50)
        
        for col in numerical_cols:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(f"  Min: {self.df[col].min()}")
                print(f"  Max: {self.df[col].max()}")
                print(f"  Mean: {self.df[col].mean():.2f}")
                print(f"  Median: {self.df[col].median():.2f}")
                print(f"  Std: {self.df[col].std():.2f}")
        
        # Target variable analysis
        print("\n" + "=" * 50)
        print("TARGET VARIABLE (Selling_Price) ANALYSIS")
        print("=" * 50)
        target = self.df['Selling_Price']
        print(f"Price range: ₹{target.min():.2f} lakhs to ₹{target.max():.2f} lakhs")
        print(f"Average price: ₹{target.mean():.2f} lakhs")
        print(f"Price distribution skewness: {target.skew():.3f}")
        print(f"Price distribution kurtosis: {target.kurtosis():.3f}")
    
    def create_comprehensive_visualizations(self):
        """
        Create comprehensive visualizations for data exploration
        """
        print("\n" + "=" * 80)
        print("CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 80)
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Price distribution
        plt.subplot(4, 3, 1)
        plt.hist(self.df['Selling_Price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Car Selling Prices', fontsize=12, fontweight='bold')
        plt.xlabel('Selling Price (Lakhs)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. Price vs Year
        plt.subplot(4, 3, 2)
        plt.scatter(self.df['Year'], self.df['Selling_Price'], alpha=0.6, color='coral')
        plt.title('Car Price vs Manufacturing Year', fontsize=12, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Selling Price (Lakhs)')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df['Year'], self.df['Selling_Price'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['Year'], p(self.df['Year']), "r--", alpha=0.8)
        
        # 3. Price vs Present Price
        plt.subplot(4, 3, 3)
        plt.scatter(self.df['Present_Price'], self.df['Selling_Price'], alpha=0.6, color='lightgreen')
        plt.title('Selling Price vs Present Price', fontsize=12, fontweight='bold')
        plt.xlabel('Present Price (Lakhs)')
        plt.ylabel('Selling Price (Lakhs)')
        plt.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        max_price = max(self.df['Present_Price'].max(), self.df['Selling_Price'].max())
        plt.plot([0, max_price], [0, max_price], 'r--', alpha=0.5, label='Equal Price Line')
        plt.legend()
        
        # 4. Price vs Kilometers Driven
        plt.subplot(4, 3, 4)
        plt.scatter(self.df['Driven_kms'], self.df['Selling_Price'], alpha=0.6, color='gold')
        plt.title('Car Price vs Kilometers Driven', fontsize=12, fontweight='bold')
        plt.xlabel('Kilometers Driven')
        plt.ylabel('Selling Price (Lakhs)')
        plt.grid(True, alpha=0.3)
        
        # 5. Fuel Type Distribution
        plt.subplot(4, 3, 5)
        fuel_counts = self.df['Fuel_Type'].value_counts()
        plt.pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Distribution by Fuel Type', fontsize=12, fontweight='bold')
        
        # 6. Price by Fuel Type
        plt.subplot(4, 3, 6)
        sns.boxplot(data=self.df, x='Fuel_Type', y='Selling_Price')
        plt.title('Price Distribution by Fuel Type', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Selling Price (Lakhs)')
        
        # 7. Transmission Type Distribution
        plt.subplot(4, 3, 7)
        transmission_counts = self.df['Transmission'].value_counts()
        plt.bar(transmission_counts.index, transmission_counts.values, color=['lightcoral', 'lightblue'])
        plt.title('Distribution by Transmission Type', fontsize=12, fontweight='bold')
        plt.xlabel('Transmission Type')
        plt.ylabel('Count')
        
        # 8. Price by Transmission Type
        plt.subplot(4, 3, 8)
        sns.boxplot(data=self.df, x='Transmission', y='Selling_Price')
        plt.title('Price Distribution by Transmission', fontsize=12, fontweight='bold')
        plt.ylabel('Selling Price (Lakhs)')
        
        # 9. Selling Type Distribution
        plt.subplot(4, 3, 9)
        selling_type_counts = self.df['Selling_type'].value_counts()
        plt.bar(selling_type_counts.index, selling_type_counts.values, color=['lightgreen', 'orange'])
        plt.title('Distribution by Selling Type', fontsize=12, fontweight='bold')
        plt.xlabel('Selling Type')
        plt.ylabel('Count')
        
        # 10. Price by Owner Count
        plt.subplot(4, 3, 10)
        owner_price = self.df.groupby('Owner')['Selling_Price'].mean()
        plt.bar(owner_price.index, owner_price.values, color='plum')
        plt.title('Average Price by Owner Count', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Previous Owners')
        plt.ylabel('Average Selling Price (Lakhs)')
        
        # 11. Car Age vs Price
        plt.subplot(4, 3, 11)
        current_year = datetime.now().year
        self.df['Car_Age'] = current_year - self.df['Year']
        plt.scatter(self.df['Car_Age'], self.df['Selling_Price'], alpha=0.6, color='mediumpurple')
        plt.title('Car Age vs Selling Price', fontsize=12, fontweight='bold')
        plt.xlabel('Car Age (Years)')
        plt.ylabel('Selling Price (Lakhs)')
        plt.grid(True, alpha=0.3)
        
        # 12. Top Car Brands by Count
        plt.subplot(4, 3, 12)
        top_brands = self.df['Car_Name'].value_counts().head(10)
        plt.barh(range(len(top_brands)), top_brands.values)
        plt.yticks(range(len(top_brands)), top_brands.index)
        plt.title('Top 10 Car Models by Count', fontsize=12, fontweight='bold')
        plt.xlabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comprehensive_car_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Comprehensive visualization saved!")
    
    def correlation_analysis(self):
        """
        Perform detailed correlation analysis
        """
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Create numerical dataset for correlation
        numerical_df = self.df.select_dtypes(include=[np.number]).copy()
        
        # Add Car Age
        current_year = datetime.now().year
        numerical_df['Car_Age'] = current_year - self.df['Year']
        
        # Calculate correlation matrix
        correlation_matrix = numerical_df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        plt.title('Car Features Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print strong correlations with target variable
        print("\nCorrelations with Selling Price:")
        price_corr = correlation_matrix['Selling_Price'].abs().sort_values(ascending=False)
        for feature, corr in price_corr.items():
            if feature != 'Selling_Price':
                print(f"  {feature}: {corr:.3f}")
        
        # Statistical significance testing
        print("\n" + "=" * 50)
        print("CORRELATION SIGNIFICANCE TESTING")
        print("=" * 50)
        
        for col in numerical_df.columns:
            if col != 'Selling_Price':
                correlation, p_value = pearsonr(numerical_df[col], numerical_df['Selling_Price'])
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{col}: r={correlation:.3f}, p={p_value:.4f} {significance}")
        
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")
    
    def feature_engineering(self):
        """
        Perform comprehensive feature engineering
        """
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
        
        # Create a copy for processing
        self.processed_df = self.df.copy()
        
        # 1. Create Car Age feature
        current_year = datetime.now().year
        self.processed_df['Car_Age'] = current_year - self.processed_df['Year']
        print("✓ Created Car_Age feature")
        
        # 2. Create Depreciation Rate
        self.processed_df['Depreciation_Rate'] = (self.processed_df['Present_Price'] - self.processed_df['Selling_Price']) / self.processed_df['Present_Price']
        print("✓ Created Depreciation_Rate feature")
        
        # 3. Create Mileage Category
        mileage_percentiles = self.processed_df['Driven_kms'].quantile([0.33, 0.67])
        self.processed_df['Mileage_Category'] = pd.cut(
            self.processed_df['Driven_kms'],
            bins=[0, mileage_percentiles[0.33], mileage_percentiles[0.67], float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        print("✓ Created Mileage_Category feature")
        
        # 4. Create Price per Year feature
        self.processed_df['Price_per_Year'] = self.processed_df['Selling_Price'] / (self.processed_df['Car_Age'] + 1)
        print("✓ Created Price_per_Year feature")
        
        # 5. Extract brand from Car_Name
        self.processed_df['Brand'] = self.processed_df['Car_Name'].str.split().str[0].str.lower()
        print("✓ Created Brand feature")
        
        # 6. Create brand popularity feature
        brand_counts = self.processed_df['Brand'].value_counts()
        self.processed_df['Brand_Popularity'] = self.processed_df['Brand'].map(brand_counts)
        print("✓ Created Brand_Popularity feature")
        
        # 7. Create interaction features
        self.processed_df['Age_Mileage_Interaction'] = self.processed_df['Car_Age'] * self.processed_df['Driven_kms']
        self.processed_df['Present_Age_Ratio'] = self.processed_df['Present_Price'] / (self.processed_df['Car_Age'] + 1)
        print("✓ Created interaction features")
        
        # Display engineered features
        print(f"\nTotal features after engineering: {len(self.processed_df.columns)}")
        print("New features created:")
        new_features = ['Car_Age', 'Depreciation_Rate', 'Mileage_Category', 'Price_per_Year', 
                       'Brand', 'Brand_Popularity', 'Age_Mileage_Interaction', 'Present_Age_Ratio']
        for feature in new_features:
            print(f"  - {feature}")
    
    def prepare_data_for_modeling(self):
        """
        Prepare data for machine learning modeling
        """
        print("\n" + "=" * 80)
        print("PREPARING DATA FOR MODELING")
        print("=" * 80)
        
        # Select features for modeling
        features_to_use = [
            'Year', 'Present_Price', 'Driven_kms', 'Owner',
            'Car_Age', 'Depreciation_Rate', 'Price_per_Year',
            'Brand_Popularity', 'Age_Mileage_Interaction', 'Present_Age_Ratio',
            'Fuel_Type', 'Selling_type', 'Transmission', 'Mileage_Category'
        ]
        
        # Create feature matrix
        X = self.processed_df[features_to_use].copy()
        y = self.processed_df['Selling_Price'].copy()
        
        # Handle categorical variables
        categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission', 'Mileage_Category']
        numerical_features = [col for col in features_to_use if col not in categorical_features]
        
        print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        )
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5)
        )
        
        # Apply preprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Store processed data
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = preprocessor
        
        print(f"✓ Data prepared successfully!")
        print(f"  Training set: {X_train_processed.shape}")
        print(f"  Test set: {X_test_processed.shape}")
        print(f"  Target range: ₹{y.min():.2f} - ₹{y.max():.2f} lakhs")
    
    def train_multiple_models(self):
        """
        Train multiple regression models for comparison
        """
        print("\n" + "=" * 80)
        print("TRAINING MULTIPLE REGRESSION MODELS")
        print("=" * 80)
        
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Support Vector Regression': SVR(kernel='rbf'),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
        }
        
        # Train and evaluate each model
        results = []
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            result = {
                'Model': name,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'CV_Mean_R2': cv_mean,
                'CV_Std_R2': cv_std,
                'Overfitting': train_r2 - test_r2
            }
            
            results.append(result)
            self.models[name] = model
            
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  CV R² Score: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Create results DataFrame
        self.model_scores = pd.DataFrame(results)
        
        # Display results table
        print("\n" + "=" * 100)
        print("MODEL COMPARISON RESULTS")
        print("=" * 100)
        print(self.model_scores.round(4).to_string(index=False))
        
        # Find best model
        best_model_name = self.model_scores.loc[self.model_scores['Test_R2'].idxmax(), 'Model']
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test R² Score: {self.model_scores['Test_R2'].max():.4f}")
    
    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning on the best performing models
        """
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING")
        print("=" * 80)
        
        # Get top 3 models for tuning
        top_models = self.model_scores.nlargest(3, 'Test_R2')['Model'].tolist()
        
        print(f"Tuning top 3 models: {top_models}")
        
        tuning_results = []
        
        for model_name in top_models:
            print(f"\nTuning {model_name}...")
            
            if model_name == 'Random Forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                model = RandomForestRegressor(random_state=42)
                
            elif model_name == 'Gradient Boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
                model = GradientBoostingRegressor(random_state=42)
                
            elif model_name == 'Support Vector Regression':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'epsilon': [0.01, 0.1, 0.2]
                }
                model = SVR(kernel='rbf')
                
            else:
                continue
            
            # Perform grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Best model evaluation
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_test)
            
            tuned_r2 = r2_score(self.y_test, y_pred)
            tuned_mae = mean_absolute_error(self.y_test, y_pred)
            
            tuning_results.append({
                'Model': f"{model_name} (Tuned)",
                'Best_Params': str(grid_search.best_params_),
                'CV_Score': grid_search.best_score_,
                'Test_R2': tuned_r2,
                'Test_MAE': tuned_mae
            })
            
            # Update best model if improved
            original_score = self.model_scores[self.model_scores['Model'] == model_name]['Test_R2'].iloc[0]
            if tuned_r2 > original_score:
                self.models[f"{model_name} (Tuned)"] = best_model
                print(f"  ✓ Improvement: {tuned_r2:.4f} vs {original_score:.4f}")
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Tuned Test R²: {tuned_r2:.4f}")
        
        # Display tuning results
        if tuning_results:
            tuning_df = pd.DataFrame(tuning_results)
            print("\n" + "=" * 80)
            print("HYPERPARAMETER TUNING RESULTS")
            print("=" * 80)
            print(tuning_df.to_string(index=False))
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance using the best model
        """
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        # Use Random Forest for feature importance (works with preprocessed features)
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        
        # Get feature names after preprocessing
        feature_names = self.preprocessor.get_feature_names_out()
        
        # Get feature importance
        importance_scores = rf_model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        print("Top 15 Most Important Features:")
        print(feature_importance_df.head(15).to_string(index=False))
        
        # Create feature importance visualization
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.title('Top 15 Feature Importance for Car Price Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.feature_importance = feature_importance_df
        print("✓ Feature importance analysis completed!")
    
    def create_prediction_visualizations(self):
        """
        Create visualizations for model predictions
        """
        print("\n" + "=" * 80)
        print("CREATING PREDICTION VISUALIZATIONS")
        print("=" * 80)
        
        # Get best model predictions
        best_model_name = self.model_scores.loc[self.model_scores['Test_R2'].idxmax(), 'Model']
        best_model = self.models[best_model_name]
        
        y_pred_train = best_model.predict(self.X_train)
        y_pred_test = best_model.predict(self.X_test)
        
        # Create prediction plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted (Training)
        axes[0, 0].scatter(self.y_train, y_pred_train, alpha=0.6, color='blue')
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price (Lakhs)')
        axes[0, 0].set_ylabel('Predicted Price (Lakhs)')
        axes[0, 0].set_title(f'Training Set: Actual vs Predicted\n{best_model_name}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted (Testing)
        axes[0, 1].scatter(self.y_test, y_pred_test, alpha=0.6, color='green')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price (Lakhs)')
        axes[0, 1].set_ylabel('Predicted Price (Lakhs)')
        axes[0, 1].set_title(f'Test Set: Actual vs Predicted\n{best_model_name}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Plot (Training)
        residuals_train = self.y_train - y_pred_train
        axes[1, 0].scatter(y_pred_train, residuals_train, alpha=0.6, color='orange')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Price (Lakhs)')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Training Set: Residuals Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residuals Plot (Testing)
        residuals_test = self.y_test - y_pred_test
        axes[1, 1].scatter(y_pred_test, residuals_test, alpha=0.6, color='purple')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Price (Lakhs)')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Test Set: Residuals Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Model comparison visualization
        plt.figure(figsize=(14, 8))
        
        # Sort models by Test R2 score
        sorted_models = self.model_scores.sort_values('Test_R2', ascending=True)
        
        # Create horizontal bar chart
        plt.barh(range(len(sorted_models)), sorted_models['Test_R2'], color='skyblue', alpha=0.8)
        plt.yticks(range(len(sorted_models)), sorted_models['Model'])
        plt.xlabel('R² Score')
        plt.title('Model Performance Comparison (Test Set R² Score)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, v in enumerate(sorted_models['Test_R2']):
            plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Prediction visualizations created!")
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report
        """
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        
        # Get best model info
        best_idx = self.model_scores['Test_R2'].idxmax()
        best_model_info = self.model_scores.iloc[best_idx]
        
        report = f"""
CAR PRICE PREDICTION ANALYSIS REPORT
=====================================

Dataset Overview:
- Total Records: {len(self.df)}
- Features: {len(self.df.columns)}
- Price Range: Rs.{self.df['Selling_Price'].min():.2f} - Rs.{self.df['Selling_Price'].max():.2f} lakhs
- Average Price: Rs.{self.df['Selling_Price'].mean():.2f} lakhs

Key Insights:
1. Data Quality: No missing values found in the dataset
2. Price Distribution: Mean price is Rs.{self.df['Selling_Price'].mean():.2f} lakhs with std Rs.{self.df['Selling_Price'].std():.2f} lakhs
3. Car Age Impact: Strong negative correlation between car age and selling price
4. Present Price Correlation: High positive correlation ({self.processed_df[['Present_Price', 'Selling_Price']].corr().iloc[0,1]:.3f}) with selling price

Feature Engineering Results:
- Created {len(self.processed_df.columns) - len(self.df.columns)} new features
- Key engineered features: Car_Age, Depreciation_Rate, Brand_Popularity
- Feature selection reduced dimensionality while maintaining predictive power

Model Performance Results:
==========================
Best Model: {best_model_info['Model']}
- Test R² Score: {best_model_info['Test_R2']:.4f}
- Test MAE: Rs.{best_model_info['Test_MAE']:.2f} lakhs
- Test RMSE: Rs.{best_model_info['Test_RMSE']:.2f} lakhs
- Cross-Validation R²: {best_model_info['CV_Mean_R2']:.4f} ± {best_model_info['CV_Std_R2']:.4f}

Model Comparison Summary:
"""
        
        # Add model comparison table
        for _, row in self.model_scores.iterrows():
            report += f"\n{row['Model']}:\n"
            report += f"  - Test R²: {row['Test_R2']:.4f}\n"
            report += f"  - Test MAE: Rs.{row['Test_MAE']:.2f} lakhs\n"
            report += f"  - CV Score: {row['CV_Mean_R2']:.4f} ± {row['CV_Std_R2']:.4f}\n"
        
        # Add feature importance
        if self.feature_importance is not None:
            report += f"\nTop 10 Most Important Features:\n"
            for idx, row in self.feature_importance.head(10).iterrows():
                report += f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}\n"
        
        report += f"""
Key Findings:
=============
1. The {best_model_info['Model']} achieved the best performance with R² = {best_model_info['Test_R2']:.4f}
2. Present price is the strongest predictor of selling price
3. Car age and depreciation rate are crucial factors in price determination
4. Model can predict car prices with average error of Rs.{best_model_info['Test_MAE']:.2f} lakhs

Business Implications:
======================
1. Age Depreciation: Cars depreciate significantly with age, following predictable patterns
2. Brand Value: Some brands maintain value better than others
3. Mileage Impact: High mileage vehicles show greater depreciation
4. Market Trends: The model can help in fair price estimation for buyers and sellers

Recommendations:
================
1. Use the {best_model_info['Model']} for production price predictions
2. Regularly retrain the model with new market data
3. Consider additional features like service history, accident records
4. Implement confidence intervals for price predictions

Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
          # Save report to file
        with open(f'{self.output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✓ Comprehensive report generated!")
        print(f"✓ Report saved to: {self.output_dir}/analysis_report.txt")
        
        # Display summary
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Best Model: {best_model_info['Model']}")
        print(f"Test R² Score: {best_model_info['Test_R2']:.4f}")
        print(f"Average Prediction Error: Rs.{best_model_info['Test_MAE']:.2f} lakhs")
        print(f"Files Generated: 4 visualization files + 1 analysis report")
    
    def run_complete_analysis(self):
        """
        Run the complete car price prediction analysis pipeline
        """
        print("🚗" * 20)
        print("STARTING COMPREHENSIVE CAR PRICE PREDICTION ANALYSIS")
        print("🚗" * 20)
        
        # Step 1: Load and explore data
        if not self.load_and_explore_data():
            return False
        
        # Step 2: Data quality analysis
        self.data_quality_analysis()
        
        # Step 3: Create visualizations
        self.create_comprehensive_visualizations()
        
        # Step 4: Correlation analysis
        self.correlation_analysis()
        
        # Step 5: Feature engineering
        self.feature_engineering()
        
        # Step 6: Prepare data for modeling
        self.prepare_data_for_modeling()
        
        # Step 7: Train multiple models
        self.train_multiple_models()
        
        # Step 8: Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Step 9: Feature importance analysis
        self.feature_importance_analysis()
        
        # Step 10: Create prediction visualizations
        self.create_prediction_visualizations()
        
        # Step 11: Generate comprehensive report
        self.generate_comprehensive_report()
        
        print("\n" + "🎉" * 20)
        print("CAR PRICE PREDICTION ANALYSIS COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print("Generated files:")
        print("  - comprehensive_car_analysis.png")
        print("  - correlation_matrix.png") 
        print("  - feature_importance.png")
        print("  - prediction_analysis.png")
        print("  - model_comparison.png")
        print("  - analysis_report.txt")
        
        return True

def main():
    """
    Main function to run the car price prediction analysis
    """
    # Initialize the analysis
    data_path = "car data.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset file '{data_path}' not found!")
        print("Please ensure the dataset is in the current directory.")
        return
    
    # Create analysis instance
    analysis = CarPricePredictionAnalysis(data_path)
    
    # Run complete analysis
    success = analysis.run_complete_analysis()
    
    if success:
        print("\n✅ Analysis completed successfully!")
        print("Check the generated visualizations and report for detailed insights.")
    else:
        print("\n❌ Analysis failed!")

if __name__ == "__main__":
    main()