"""
Unemployment Analysis in India - Data Science Project

This script performs comprehensive analysis of unemployment data in India,
examining trends across different states, time periods, and the impact of COVID-19.
The dataset contains unemployment statistics from various Indian states and union territories.

Key Metrics Analyzed:
- Unemployment Rate (%)
- Estimated Employed Numbers
- Labour Participation Rate (%)
- Regional and temporal variations
- COVID-19 impact analysis

Author: Parth
Date: May 25, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional plotly import
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class UnemploymentAnalyzer:
    """
    A comprehensive unemployment analysis system for India
    """
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with data path
        """
        self.data_path = data_path
        self.data = None
        self.covid_start = datetime(2020, 3, 1)  # COVID-19 lockdown start
        
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration
        """
        print("=" * 80)
        print("UNEMPLOYMENT ANALYSIS IN INDIA - DATA SCIENCE PROJECT")
        print("=" * 80)
        
        # Load the data
        self.data = pd.read_csv(self.data_path)
        
        # Clean column names (remove extra spaces)
        self.data.columns = self.data.columns.str.strip()
        
        print(f"\n📊 Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        
        # Basic information
        print(f"\n📋 Dataset Info:")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Data types:\n{self.data.dtypes}")
        
        # Display first few rows
        print(f"\n🔍 First 5 rows:")
        print(self.data.head())
        
        # Check for missing values
        print(f"\n❓ Missing Values:")
        missing_values = self.data.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        # Basic statistics
        print(f"\n📈 Basic Statistics:")
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_columns].describe())
        
        return self.data
    
    def clean_and_preprocess_data(self):
        """
        Clean and preprocess the unemployment data
        """
        print(f"\n🔧 Cleaning and preprocessing data...")
          # Clean data and handle missing values
        self.data = self.data.dropna()
        
        # Strip whitespace from Date column
        self.data['Date'] = self.data['Date'].astype(str).str.strip()
        
        # Convert date column to datetime with flexible parsing
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d-%m-%Y')
        except ValueError:
            # Try alternative format if the first one fails
            try:
                self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True)
            except:
                self.data['Date'] = pd.to_datetime(self.data['Date'], infer_datetime_format=True)
        
        # Extract year, month, and quarter
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Month_Name'] = self.data['Date'].dt.strftime('%B')
        self.data['Quarter'] = self.data['Date'].dt.quarter
        
        # Create COVID period indicator
        self.data['COVID_Period'] = self.data['Date'] >= self.covid_start
        self.data['Period'] = self.data['COVID_Period'].map({True: 'COVID Period', False: 'Pre-COVID'})
        
        # Clean numeric columns
        numeric_cols = ['Estimated Unemployment Rate (%)', 'Estimated Employed', 
                       'Estimated Labour Participation Rate (%)']
        
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Remove any rows with missing essential data
        initial_rows = len(self.data)
        self.data = self.data.dropna(subset=numeric_cols)
        dropped_rows = initial_rows - len(self.data)
        
        if dropped_rows > 0:
            print(f"⚠️ Dropped {dropped_rows} rows with missing essential data")
        
        # Basic info about unique values
        print(f"\n📍 Unique Regions: {self.data['Region'].nunique()}")
        print(f"📅 Date Range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"🏙️ Areas: {list(self.data['Area'].unique())}")
        print(f"📊 Total Records: {len(self.data)}")
        
        print("✅ Data preprocessing completed!")
        
        return self.data
    
    def exploratory_data_analysis(self):
        """
        Perform comprehensive exploratory data analysis
        """
        print(f"\n🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Overall statistics
        print(f"\n📊 Overall Unemployment Statistics:")
        unemployment_stats = {
            'Average Unemployment Rate': self.data['Estimated Unemployment Rate (%)'].mean(),
            'Median Unemployment Rate': self.data['Estimated Unemployment Rate (%)'].median(),
            'Max Unemployment Rate': self.data['Estimated Unemployment Rate (%)'].max(),
            'Min Unemployment Rate': self.data['Estimated Unemployment Rate (%)'].min(),
            'Standard Deviation': self.data['Estimated Unemployment Rate (%)'].std()
        }
        
        for stat, value in unemployment_stats.items():
            print(f"   {stat}: {value:.2f}%")
        
        # Regional analysis
        print(f"\n🗺️ Regional Analysis:")
        regional_stats = self.data.groupby('Region').agg({
            'Estimated Unemployment Rate (%)': ['mean', 'max', 'min'],
            'Estimated Employed': 'mean',
            'Estimated Labour Participation Rate (%)': 'mean'
        }).round(2)
        
        regional_stats.columns = ['Avg_Unemployment', 'Max_Unemployment', 'Min_Unemployment', 
                                'Avg_Employed', 'Avg_Participation']
        
        # Sort by average unemployment rate
        regional_stats = regional_stats.sort_values('Avg_Unemployment', ascending=False)
        print(f"Top 10 states by average unemployment rate:")
        print(regional_stats.head(10))
        
        # COVID-19 impact analysis
        print(f"\n🦠 COVID-19 Impact Analysis:")
        covid_impact = self.data.groupby('Period').agg({
            'Estimated Unemployment Rate (%)': ['mean', 'median', 'max'],
            'Estimated Labour Participation Rate (%)': 'mean'
        }).round(2)
        
        covid_impact.columns = ['Avg_Unemployment', 'Median_Unemployment', 'Max_Unemployment', 'Avg_Participation']
        print(covid_impact)
        
        # Calculate percentage increase
        pre_covid_avg = covid_impact.loc['Pre-COVID', 'Avg_Unemployment']
        covid_avg = covid_impact.loc['COVID Period', 'Avg_Unemployment']
        percentage_increase = ((covid_avg - pre_covid_avg) / pre_covid_avg) * 100
        
        print(f"\n📈 COVID-19 Impact:")
        print(f"   Pre-COVID Average Unemployment: {pre_covid_avg:.2f}%")
        print(f"   COVID Period Average Unemployment: {covid_avg:.2f}%")
        print(f"   Percentage Increase: {percentage_increase:.2f}%")
        
        return regional_stats, covid_impact
    
    def create_comprehensive_visualizations(self):
        """
        Create comprehensive visualizations of unemployment data
        """
        print(f"\n🎨 Creating comprehensive visualizations...")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall unemployment rate distribution
        plt.subplot(4, 3, 1)
        plt.hist(self.data['Estimated Unemployment Rate (%)'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Unemployment Rate')
        plt.xlabel('Unemployment Rate (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. Unemployment rate by area (Rural vs Urban)
        plt.subplot(4, 3, 2)
        area_unemployment = self.data.groupby('Area')['Estimated Unemployment Rate (%)'].mean()
        area_unemployment.plot(kind='bar', color=['lightcoral', 'lightgreen'])
        plt.title('Average Unemployment Rate by Area')
        plt.xlabel('Area Type')
        plt.ylabel('Unemployment Rate (%)')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # 3. Time series of unemployment rate
        plt.subplot(4, 3, 3)
        monthly_unemployment = self.data.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
        monthly_unemployment.plot(color='red', linewidth=2)
        plt.title('Unemployment Rate Over Time')
        plt.xlabel('Date')
        plt.ylabel('Unemployment Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. Top 10 states by average unemployment
        plt.subplot(4, 3, 4)
        top_states = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().nlargest(10)
        top_states.plot(kind='barh', color='orange')
        plt.title('Top 10 States by Avg Unemployment Rate')
        plt.xlabel('Unemployment Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # 5. COVID-19 impact comparison
        plt.subplot(4, 3, 5)
        covid_comparison = self.data.groupby('Period')['Estimated Unemployment Rate (%)'].mean()
        covid_comparison.plot(kind='bar', color=['green', 'red'], alpha=0.7)
        plt.title('Pre-COVID vs COVID Period Unemployment')
        plt.xlabel('Period')
        plt.ylabel('Average Unemployment Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. Monthly trend analysis
        plt.subplot(4, 3, 6)
        monthly_trend = self.data.groupby('Month_Name')['Estimated Unemployment Rate (%)'].mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_trend = monthly_trend.reindex([m for m in month_order if m in monthly_trend.index])
        monthly_trend.plot(kind='line', marker='o', color='purple', linewidth=2, markersize=6)
        plt.title('Unemployment Rate by Month')
        plt.xlabel('Month')
        plt.ylabel('Unemployment Rate (%)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Labour participation rate distribution
        plt.subplot(4, 3, 7)
        plt.hist(self.data['Estimated Labour Participation Rate (%)'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        plt.title('Distribution of Labour Participation Rate')
        plt.xlabel('Labour Participation Rate (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 8. Correlation heatmap
        plt.subplot(4, 3, 8)
        numeric_data = self.data[['Estimated Unemployment Rate (%)', 'Estimated Employed', 
                                 'Estimated Labour Participation Rate (%)']].corr()
        sns.heatmap(numeric_data, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Matrix')
        
        # 9. Year-wise unemployment trend
        plt.subplot(4, 3, 9)
        yearly_unemployment = self.data.groupby('Year')['Estimated Unemployment Rate (%)'].mean()
        yearly_unemployment.plot(kind='bar', color='gold', alpha=0.8)
        plt.title('Year-wise Average Unemployment Rate')
        plt.xlabel('Year')
        plt.ylabel('Unemployment Rate (%)')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # 10. Scatter plot: Employment vs Participation Rate
        plt.subplot(4, 3, 10)
        plt.scatter(self.data['Estimated Labour Participation Rate (%)'], 
                   self.data['Estimated Unemployment Rate (%)'], 
                   alpha=0.6, c=self.data['COVID_Period'], cmap='viridis')
        plt.xlabel('Labour Participation Rate (%)')
        plt.ylabel('Unemployment Rate (%)')
        plt.title('Unemployment vs Labour Participation')
        plt.colorbar(label='COVID Period')
        plt.grid(True, alpha=0.3)
        
        # 11. Box plot of unemployment by region (top 15 states)
        plt.subplot(4, 3, 11)
        top_15_states = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().nlargest(15).index
        sns.boxplot(data=self.data[self.data['Region'].isin(top_15_states)], 
                   x='Estimated Unemployment Rate (%)', y='Region')
        plt.title('Unemployment Rate Distribution (Top 15 States)')
        plt.xlabel('Unemployment Rate (%)')
        
        # 12. COVID impact timeline
        plt.subplot(4, 3, 12)
        covid_timeline = self.data[self.data['Date'] >= datetime(2020, 1, 1)].groupby('Date')['Estimated Unemployment Rate (%)'].mean()
        covid_timeline.plot(color='red', linewidth=3, marker='o', markersize=4)
        plt.axvline(x=self.covid_start, color='black', linestyle='--', linewidth=2, label='COVID Start')
        plt.title('Unemployment During COVID-19 Period')
        plt.xlabel('Date')
        plt.ylabel('Unemployment Rate (%)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('unemployment_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create additional detailed visualizations
        self.create_regional_analysis_plots()
        self.create_covid_impact_plots()
    
    def create_regional_analysis_plots(self):
        """
        Create detailed regional analysis plots
        """
        print("Creating detailed regional analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Heatmap of unemployment by state and month
        pivot_data = self.data.pivot_table(values='Estimated Unemployment Rate (%)', 
                                         index='Region', columns='Month_Name', aggfunc='mean')
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        pivot_data = pivot_data.reindex(columns=[m for m in month_order if m in pivot_data.columns])
        
        sns.heatmap(pivot_data, cmap='Reds', annot=False, fmt='.1f', ax=axes[0, 0])
        axes[0, 0].set_title('Unemployment Rate Heatmap by State and Month')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('State')
        
        # 2. Rural vs Urban comparison by state
        rural_urban = self.data.groupby(['Region', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack()
        rural_urban.plot(kind='bar', ax=axes[0, 1], color=['green', 'orange'], alpha=0.7)
        axes[0, 1].set_title('Rural vs Urban Unemployment by State')
        axes[0, 1].set_xlabel('State')
        axes[0, 1].set_ylabel('Unemployment Rate (%)')
        axes[0, 1].legend(title='Area')
        axes[0, 1].tick_params(axis='x', rotation=90)
        
        # 3. Employment numbers by state
        employment_by_state = self.data.groupby('Region')['Estimated Employed'].mean().sort_values(ascending=False)
        employment_by_state.head(15).plot(kind='bar', ax=axes[1, 0], color='skyblue')
        axes[1, 0].set_title('Average Employment Numbers (Top 15 States)')
        axes[1, 0].set_xlabel('State')
        axes[1, 0].set_ylabel('Employed (in millions)')
        axes[1, 0].tick_params(axis='x', rotation=90)
        
        # 4. Labour participation rate by state
        participation_by_state = self.data.groupby('Region')['Estimated Labour Participation Rate (%)'].mean().sort_values(ascending=False)
        participation_by_state.head(15).plot(kind='bar', ax=axes[1, 1], color='lightcoral')
        axes[1, 1].set_title('Labour Participation Rate (Top 15 States)')
        axes[1, 1].set_xlabel('State')
        axes[1, 1].set_ylabel('Participation Rate (%)')
        axes[1, 1].tick_params(axis='x', rotation=90)
        
        plt.tight_layout()
        plt.savefig('regional_unemployment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_covid_impact_plots(self):
        """
        Create detailed COVID-19 impact analysis plots
        """
        print("Creating COVID-19 impact analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Before and after COVID comparison by state
        covid_impact_by_state = self.data.groupby(['Region', 'Period'])['Estimated Unemployment Rate (%)'].mean().unstack()
        covid_impact_by_state['Impact'] = covid_impact_by_state['COVID Period'] - covid_impact_by_state['Pre-COVID']
        top_impacted = covid_impact_by_state.nlargest(15, 'Impact')
        
        top_impacted[['Pre-COVID', 'COVID Period']].plot(kind='bar', ax=axes[0, 0], 
                                                        color=['green', 'red'], alpha=0.7)
        axes[0, 0].set_title('COVID-19 Impact: Top 15 Most Affected States')
        axes[0, 0].set_xlabel('State')
        axes[0, 0].set_ylabel('Unemployment Rate (%)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=90)
        
        # 2. Monthly progression during COVID
        covid_data = self.data[self.data['COVID_Period'] == True]
        monthly_covid = covid_data.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
        monthly_covid.plot(ax=axes[0, 1], color='red', linewidth=3, marker='o')
        axes[0, 1].set_title('Unemployment Rate During COVID-19')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Unemployment Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Rural vs Urban impact during COVID
        covid_rural_urban = covid_data.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack()
        covid_rural_urban.plot(ax=axes[1, 0], color=['green', 'orange'], linewidth=2, marker='o')
        axes[1, 0].set_title('Rural vs Urban Unemployment During COVID-19')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Unemployment Rate (%)')
        axes[1, 0].legend(title='Area')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Recovery analysis
        recovery_data = self.data[self.data['Date'] >= datetime(2020, 4, 1)]
        monthly_recovery = recovery_data.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
        monthly_recovery.plot(ax=axes[1, 1], color='blue', linewidth=3, marker='s')
        axes[1, 1].set_title('Unemployment Recovery Trend (Post April 2020)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Unemployment Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('covid_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_analysis(self):
        """
        Perform statistical analysis and hypothesis testing
        """
        print(f"\n📊 STATISTICAL ANALYSIS")
        print("=" * 60)
        
        from scipy import stats
        
        # Test if COVID period has significantly higher unemployment
        pre_covid = self.data[self.data['COVID_Period'] == False]['Estimated Unemployment Rate (%)']
        covid_period = self.data[self.data['COVID_Period'] == True]['Estimated Unemployment Rate (%)']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(covid_period, pre_covid)
        
        print(f"📈 T-test Results (COVID vs Pre-COVID):")
        print(f"   T-statistic: {t_stat:.4f}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Significance level: 0.05")
        
        if p_value < 0.05:
            print("   ✅ Result: Statistically significant difference")
            print("   📊 Conclusion: COVID-19 significantly increased unemployment rates")
        else:
            print("   ❌ Result: No statistically significant difference")
        
        # Rural vs Urban analysis
        rural_unemployment = self.data[self.data['Area'] == 'Rural']['Estimated Unemployment Rate (%)']
        urban_unemployment = self.data[self.data['Area'] == 'Urban']['Estimated Unemployment Rate (%)']
        
        if len(urban_unemployment) > 0:
            t_stat_area, p_value_area = stats.ttest_ind(rural_unemployment, urban_unemployment)
            
            print(f"\n🏙️ T-test Results (Rural vs Urban):")
            print(f"   T-statistic: {t_stat_area:.4f}")
            print(f"   P-value: {p_value_area:.6f}")
            
            if p_value_area < 0.05:
                print("   ✅ Result: Statistically significant difference between Rural and Urban unemployment")
            else:
                print("   ❌ Result: No statistically significant difference between Rural and Urban unemployment")
        
        # Correlation analysis
        print(f"\n🔗 CORRELATION ANALYSIS:")
        correlations = self.data[['Estimated Unemployment Rate (%)', 'Estimated Employed', 
                                 'Estimated Labour Participation Rate (%)']].corr()
        
        print("Correlation Matrix:")
        print(correlations.round(4))
        
        return t_stat, p_value
    
    def identify_key_insights(self):
        """
        Identify and summarize key insights from the analysis
        """
        print(f"\n💡 KEY INSIGHTS AND FINDINGS")
        print("=" * 60)
        
        # Calculate key statistics
        overall_avg = self.data['Estimated Unemployment Rate (%)'].mean()
        pre_covid_avg = self.data[self.data['COVID_Period'] == False]['Estimated Unemployment Rate (%)'].mean()
        covid_avg = self.data[self.data['COVID_Period'] == True]['Estimated Unemployment Rate (%)'].mean()
        max_unemployment = self.data['Estimated Unemployment Rate (%)'].max()
        
        # Find the state and date with highest unemployment
        max_row = self.data.loc[self.data['Estimated Unemployment Rate (%)'].idxmax()]
        
        # Top affected states during COVID
        covid_impact = self.data.groupby(['Region', 'Period'])['Estimated Unemployment Rate (%)'].mean().unstack()
        if 'COVID Period' in covid_impact.columns and 'Pre-COVID' in covid_impact.columns:
            covid_impact['Impact'] = covid_impact['COVID Period'] - covid_impact['Pre-COVID']
            most_affected_state = covid_impact['Impact'].idxmax()
            impact_value = covid_impact['Impact'].max()
        
        # Recovery analysis
        peak_covid = self.data[self.data['COVID_Period'] == True]['Estimated Unemployment Rate (%)'].max()
        latest_data = self.data[self.data['Date'] == self.data['Date'].max()]
        current_unemployment = latest_data['Estimated Unemployment Rate (%)'].mean()
        
        print(f"🎯 UNEMPLOYMENT OVERVIEW:")
        print(f"   • Overall Average Unemployment Rate: {overall_avg:.2f}%")
        print(f"   • Pre-COVID Average: {pre_covid_avg:.2f}%")
        print(f"   • COVID Period Average: {covid_avg:.2f}%")
        print(f"   • Maximum Unemployment Rate: {max_unemployment:.2f}%")
        print(f"   • Peak occurred in: {max_row['Region']}, {max_row['Date'].strftime('%B %Y')}")
        
        print(f"\n🦠 COVID-19 IMPACT:")
        print(f"   • Average increase during COVID: {((covid_avg - pre_covid_avg) / pre_covid_avg * 100):.1f}%")
        print(f"   • Peak unemployment during COVID: {peak_covid:.2f}%")
        if 'most_affected_state' in locals():
            print(f"   • Most affected state: {most_affected_state} (+{impact_value:.2f}% increase)")
        
        print(f"\n📊 REGIONAL INSIGHTS:")
        highest_avg_state = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().idxmax()
        lowest_avg_state = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().idxmin()
        highest_avg_value = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().max()
        lowest_avg_value = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().min()
        
        print(f"   • Highest average unemployment: {highest_avg_state} ({highest_avg_value:.2f}%)")
        print(f"   • Lowest average unemployment: {lowest_avg_state} ({lowest_avg_value:.2f}%)")
        
        # Area analysis
        if len(self.data['Area'].unique()) > 1:
            area_analysis = self.data.groupby('Area')['Estimated Unemployment Rate (%)'].mean()
            print(f"\n🏙️ AREA-WISE ANALYSIS:")
            for area, rate in area_analysis.items():
                print(f"   • {area} areas: {rate:.2f}% average unemployment")
        
                print(f"\n📈 RECOVERY STATUS:")
        print(f"   • Current unemployment rate: {current_unemployment:.2f}%")
        if current_unemployment < peak_covid:
            recovery_percent = ((peak_covid - current_unemployment) / peak_covid) * 100
            print(f"   • Recovery from peak: {recovery_percent:.1f}%")
        
        return {
            'overall_avg': overall_avg,
            'covid_impact': covid_avg - pre_covid_avg,
            'max_unemployment': max_unemployment,
            'current_unemployment': current_unemployment
        }
    
    def create_interactive_dashboard(self):
        """
        Create an interactive dashboard using Plotly
        """
        print(f"\n🎨 Creating interactive dashboard...")
        
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available. Skipping interactive dashboard.")
            print("   Install plotly for interactive features: pip install plotly")
            return
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Unemployment Rate Over Time', 'State-wise Average Unemployment',
                              'Rural vs Urban Comparison', 'COVID-19 Impact'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Time series plot
            monthly_data = self.data.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=monthly_data['Date'], y=monthly_data['Estimated Unemployment Rate (%)'],
                          mode='lines+markers', name='Unemployment Rate', line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            # 2. State-wise bar chart
            state_avg = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().nlargest(10)
            fig.add_trace(
                go.Bar(x=state_avg.values, y=state_avg.index, orientation='h',
                      name='Top 10 States', marker_color='orange'),
                row=1, col=2
            )
            
            # 3. Rural vs Urban comparison
            if len(self.data['Area'].unique()) > 1:
                area_data = self.data.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack().reset_index()
                for area in area_data.columns[1:]:
                    fig.add_trace(
                        go.Scatter(x=area_data['Date'], y=area_data[area],
                                  mode='lines', name=f'{area} Area'),
                        row=2, col=1
                    )
            
            # 4. COVID impact
            covid_comparison = self.data.groupby('Period')['Estimated Unemployment Rate (%)'].mean()
            fig.add_trace(
                go.Bar(x=covid_comparison.index, y=covid_comparison.values,
                      name='COVID Impact', marker_color=['green', 'red']),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Unemployment Analysis Dashboard - India",
                title_x=0.5,
                height=800,
                showlegend=True
            )
            
            # Save as HTML
            fig.write_html("unemployment_dashboard.html")
            print("✅ Interactive dashboard saved as 'unemployment_dashboard.html'")
            
        except Exception as e:
            print(f"⚠️ Could not create interactive dashboard: {e}")
            print("   (This requires plotly installation: pip install plotly)")
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report
        """
        print(f"\n📄 COMPREHENSIVE UNEMPLOYMENT ANALYSIS REPORT")
        print("=" * 80)
        
        insights = self.identify_key_insights()
        
        print(f"\n🗓️  Analysis Date: May 25, 2025")
        print(f"📁  Dataset: {self.data_path}")
        print(f"📊  Total Records: {len(self.data)}")
        print(f"🗺️  States Covered: {self.data['Region'].nunique()}")
        print(f"📅  Time Period: {self.data['Date'].min().strftime('%B %Y')} to {self.data['Date'].max().strftime('%B %Y')}")
        
        print(f"\n🎯 EXECUTIVE SUMMARY:")
        print(f"   • India experienced significant unemployment challenges during COVID-19")
        print(f"   • Average unemployment increased by {insights['covid_impact']:.2f}% during the pandemic")
        print(f"   • Peak unemployment reached {insights['max_unemployment']:.2f}%")
        print(f"   • Current unemployment stands at {insights['current_unemployment']:.2f}%")
        
        print(f"\n📊 METHODOLOGY:")
        print(f"   • Data cleaning and preprocessing")
        print(f"   • Exploratory data analysis with 12+ visualizations")
        print(f"   • Statistical hypothesis testing")
        print(f"   • Regional and temporal trend analysis")
        print(f"   • COVID-19 impact assessment")
        
        print(f"\n📈 KEY FINDINGS:")
        print(f"   • COVID-19 caused unprecedented unemployment spikes")
        print(f"   • Regional disparities in unemployment rates")
        print(f"   • Seasonal patterns in employment")
        print(f"   • Recovery trends post-lockdown")
        
        print(f"\n📁 Generated Files:")
        print(f"   • unemployment_comprehensive_analysis.png - Main analysis dashboard")
        print(f"   • regional_unemployment_analysis.png - Regional insights")
        print(f"   • covid_impact_analysis.png - COVID-19 impact study")
        print(f"   • unemployment_dashboard.html - Interactive dashboard (if available)")
        
        print(f"\n💼 POLICY IMPLICATIONS:")
        print(f"   • Need for robust employment support systems")
        print(f"   • Regional focus on high-unemployment states")
        print(f"   • Rural vs urban employment strategies")
        print(f"   • Crisis preparedness for future emergencies")
        
        print(f"\n✅ Analysis completed successfully!")
    
    def run_complete_analysis(self):
        """
        Run the complete unemployment analysis pipeline
        """
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Clean and preprocess
            self.clean_and_preprocess_data()
            
            # Step 3: Exploratory data analysis
            self.exploratory_data_analysis()
            
            # Step 4: Create visualizations
            self.create_comprehensive_visualizations()
            
            # Step 5: Statistical analysis
            self.statistical_analysis()
            
            # Step 6: Create interactive dashboard
            self.create_interactive_dashboard()
            
            # Step 7: Generate comprehensive report
            self.generate_comprehensive_report()
            
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main function to run the unemployment analysis project
    """
    # Initialize the analyzer
    data_path = "Unemployment in India.csv"
    analyzer = UnemploymentAnalyzer(data_path)
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()