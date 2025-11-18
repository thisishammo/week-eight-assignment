import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set the style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("=" * 60)

try:
    # Load the Iris dataset
    iris = load_iris()
    
    # Create a DataFrame
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = iris.target
    
    # Map species numbers to names for better readability
    species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].map(species_names)
    
    print("\n✓ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape} (rows, columns)")
    
    # Display the first few rows
    print("\n1. First 5 rows of the dataset:")
    print(df.head())
    
    # Explore the structure
    print("\n2. Dataset Information:")
    print(df.info())
    
    print("\n3. Data Types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\n4. Missing Values:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("\n✓ No missing values found in the dataset!")
    else:
        print("\n⚠ Missing values detected. Cleaning dataset...")
        # Fill missing values with mean (for numerical columns)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        print("✓ Missing values handled successfully!")
    
except FileNotFoundError as e:
    print(f"✗ Error: File not found - {e}")
except Exception as e:
    print(f"✗ An error occurred while loading the dataset: {e}")

print("\n" + "=" * 60)
print("TASK 2: BASIC DATA ANALYSIS")
print("=" * 60)

try:
    # Compute basic statistics
    print("\n1. Basic Statistics of Numerical Columns:")
    print(df.describe())
    
    # Perform grouping by species
    print("\n2. Group Analysis by Species:")
    print("\nMean values for each species:")
    grouped_means = df.groupby('species').mean()
    print(grouped_means)
    
    print("\n3. Interesting Findings:")
    print("-" * 60)
    
    # Finding 1: Species with largest average measurements
    avg_sepal_length = grouped_means['sepal length (cm)']
    largest_sepal = avg_sepal_length.idxmax()
    print(f"• '{largest_sepal}' has the largest average sepal length: "
          f"{avg_sepal_length.max():.2f} cm")
    
    # Finding 2: Species with smallest average measurements
    avg_petal_width = grouped_means['petal width (cm)']
    smallest_petal = avg_petal_width.idxmin()
    print(f"• '{smallest_petal}' has the smallest average petal width: "
          f"{avg_petal_width.min():.2f} cm")
    
    # Finding 3: Variation analysis
    print("\n• Standard Deviation by Feature (overall variability):")
    print(df.std(numeric_only=True))
    
    # Finding 4: Correlation analysis
    print("\n• Correlation between features:")
    correlation = df.select_dtypes(include=[np.number]).corr()
    print(f"  Strongest correlation: Petal length & Petal width "
          f"({correlation.loc['petal length (cm)', 'petal width (cm)']:.3f})")
    
except Exception as e:
    print(f"✗ Error during analysis: {e}")

print("\n" + "=" * 60)
print("TASK 3: DATA VISUALIZATION")
print("=" * 60)

try:
    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. LINE CHART - Trend of measurements across samples
    plt.subplot(2, 2, 1)
    for column in ['sepal length (cm)', 'petal length (cm)']:
        plt.plot(df.index, df[column], label=column, linewidth=2, alpha=0.7)
    plt.title('Trend of Sepal and Petal Lengths Across Samples', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Length (cm)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 2. BAR CHART - Average petal length per species
    plt.subplot(2, 2, 2)
    avg_petal_length = df.groupby('species')['petal length (cm)'].mean()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(avg_petal_length.index, avg_petal_length.values, 
                   color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Average Petal Length by Species', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Average Petal Length (cm)', fontsize=12)
    plt.xticks(rotation=0)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. HISTOGRAM - Distribution of sepal width
    plt.subplot(2, 2, 3)
    plt.hist(df['sepal width (cm)'], bins=20, color='#95E1D3', 
             edgecolor='black', alpha=0.7)
    plt.title('Distribution of Sepal Width', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sepal Width (cm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(df['sepal width (cm)'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: {df["sepal width (cm)"].mean():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. SCATTER PLOT - Relationship between sepal and petal length
    plt.subplot(2, 2, 4)
    species_list = df['species'].unique()
    colors_scatter = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 
                     'virginica': '#45B7D1'}
    
    for species in species_list:
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal length (cm)'], 
                   species_data['petal length (cm)'],
                   label=species, alpha=0.7, s=100,
                   color=colors_scatter[species],
                   edgecolor='black', linewidth=0.5)
    
    plt.title('Sepal Length vs Petal Length by Species', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sepal Length (cm)', fontsize=12)
    plt.ylabel('Petal Length (cm)', fontsize=12)
    plt.legend(title='Species', loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("\n✓ All visualizations created successfully!")
    print("✓ Plots saved as 'iris_analysis_plots.png'")
    plt.show()
    
    # BONUS: Create an additional correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, fmt='.3f')
    plt.title('Correlation Heatmap of Iris Features', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('iris_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Bonus correlation heatmap created!")
    print("✓ Heatmap saved as 'iris_correlation_heatmap.png'")
    plt.show()
    
except Exception as e:
    print(f"✗ Error creating visualizations: {e}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")