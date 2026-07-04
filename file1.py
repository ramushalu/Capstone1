#Applied AI & ML Essentials Capstone
#Part 1: Data Acquisition, Cleaning and Exploratory Data Analysis

#Student Name: Aashika R
#Dataset: Medical Cost Personal Dataset
#Download the medical cost personal dataset from Kaggle
----------------------------------------------------------------------------
from google.colab import files

print("Please select the PDF file(s) to upload:")
uploaded = files.upload()

for filename in uploaded.keys():
  print(f'Uploaded file: {filename}')

Output:

Please select the PDF file(s) to upload:
No file chosen Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving archive.zip to archive (2).zip
Uploaded file: archive (2).zip
----------------------------------------------------------------------------------
import zipfile
import io

# Assuming the user uploaded a single zip file, we can get its name from the uploaded dictionary keys.
# The `filename` variable from the previous cell also holds the correct name.

if uploaded:
    # Get the first (and likely only) uploaded filename
    zip_filename = list(uploaded.keys())[0]

    if zip_filename.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(uploaded[zip_filename]), 'r') as zip_ref:
            zip_ref.extractall('/content/extracted_data')
        print(f"Archive '{zip_filename}' extracted to '/content/extracted_data/'")
        print("Contents of extracted_data:")
        !ls /content/extracted_data
    else:
        print(f"The uploaded file '{zip_filename}' is not a zip archive. Please upload a .zip file.")
else:
    print("No files were uploaded. Please re-upload your zip archive.")

Output:

Archive 'archive (2).zip' extracted to '/content/extracted_data/'
Contents of extracted_data:
insurance.csv

---------------------------------------------------------------------------
# TASK 1 - IMPORT LIBRARIES & LOAD DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
df = pd.read_csv("/content/extracted_data/insurance.csv")

print("="*60)
print("FIRST FIVE ROWS")
print("="*60)
print(df.head())

print("\n")

print("="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

print("\n")

print("="*60)
print("DATAFRAME SHAPE")
print("="*60)
print(df.shape)

print(f"\nRows : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

Output:

============================================================
FIRST FIVE ROWS
============================================================
   age     sex     bmi  children smoker     region      charges
0   19  female  27.900         0    yes  southwest  16884.92400
1   18    male  33.770         1     no  southeast   1725.55230
2   28    male  33.000         3     no  southeast   4449.46200
3   33    male  22.705         0     no  northwest  21984.47061
4   32    male  28.880         0     no  northwest   3866.85520


============================================================
DATA TYPES
============================================================
age           int64
sex          object
bmi         float64
children      int64
smoker       object
region       object
charges     float64
dtype: object


============================================================
DATAFRAME SHAPE
============================================================
(1338, 7)

Rows : 1338
Columns : 7
---------------------------------------------------------------------------
# TASK 2 - NULL VALUE ANALYSIS

print("\nNULL VALUE COUNT")
print(df.isnull().sum())

print("\nNULL VALUE PERCENTAGE")

null_percentage = (df.isnull().sum()/df.shape[0])*100

print(null_percentage)

print("\nColumns having more than 20% missing values")

high_null = null_percentage[null_percentage>20]

if len(high_null)==0:
    print("No column has more than 20% missing values.")
else:
    print(high_null)

# Fill numeric columns with median

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

for col in numeric_cols:
    if df[col].isnull().sum()>0:
        df[col]=df[col].fillna(df[col].median())

print("\nMissing values after imputation")

print(df.isnull().sum())

Output:

NULL VALUE COUNT
age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
dtype: int64

NULL VALUE PERCENTAGE
age         0.0
sex         0.0
bmi         0.0
children    0.0
smoker      0.0
region      0.0
charges     0.0
dtype: float64

Columns having more than 20% missing values
No column has more than 20% missing values.

Missing values after imputation
age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
dtype: int64
--------------------------------------------------------------
# TASK 3 - DUPLICATE DETECTION

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :",duplicates)

rows_before = df.shape[0]

null_before = (df.isnull().sum()/rows_before)*100

df = df.drop_duplicates().copy()
# Added .copy() here to prevent SettingWithCopyWarning

rows_after = df.shape[0]

null_after = (df.isnull().sum()/rows_after)*100

print("\nRows Before :",rows_before)
print("Rows After :",rows_after)
print("Rows Removed :",rows_before-rows_after)

print("\nNull Percentage Before")
print(null_before)

print("\nNull Percentage After")
print(null_after)

Output:

Duplicate Rows : 0

Rows Before : 1337
Rows After : 1337
Rows Removed : 0

Null Percentage Before
age         0.0
sex         0.0
bmi         0.0
children    0.0
smoker      0.0
region      0.0
charges     0.0
dtype: float64

Null Percentage After
age         0.0
sex         0.0
bmi         0.0
children    0.0
smoker      0.0
region      0.0
charges     0.0
dtype: float64
--------------------------------------------------------------------
# TASK 4 - DATA TYPE CORRECTION

memory_before = df.memory_usage(deep=True).sum()

print("\nMemory Before :",memory_before)

# Convert repetitive string columns

df["sex"]=df["sex"].astype("category")
df["smoker"]=df["smoker"].astype("category")
df["region"]=df["region"].astype("category")

memory_after=df.memory_usage(deep=True).sum()

print("Memory After :",memory_after)

print("Memory Saved :",memory_before-memory_after)

print("\nCurrent Data Types")

print(df.dtypes)

Output:

Memory Before : 58322
Memory After : 58322
Memory Saved : 0

Current Data Types
age            int64
sex         category
bmi          float64
children       int64
smoker      category
region      category
charges      float64
dtype: object
---------------------------------------------------------------------
# TASK 5 - DESCRIPTIVE STATISTICS & SKEWNESS

print(df.describe())

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

print("\nSkewness")

skewness={}

for col in numeric_cols:
    skewness[col]=df[col].skew()

skew_df=pd.DataFrame(skewness.items(),
                     columns=["Column","Skewness"])

print(skew_df)

highest_skew=skew_df.iloc[
    skew_df["Skewness"].abs().idxmax()
]

print("\nHighest Absolute Skew")

print(highest_skew)

Output:
              age          bmi     children       charges
count  1337.000000  1337.000000  1337.000000   1337.000000
mean     39.222139    30.663452     1.095737  13279.121487
std      14.044333     6.100468     1.205571  12110.359656
min      18.000000    15.960000     0.000000   1121.873900
25%      27.000000    26.290000     0.000000   4746.344000
50%      39.000000    30.400000     1.000000   9386.161300
75%      51.000000    34.700000     2.000000  16657.717450
max      64.000000    53.130000     5.000000  63770.428010

Skewness
     Column  Skewness
0       age  0.054781
1       bmi  0.283914
2  children  0.937421
3   charges  1.515391

Highest Absolute Skew
Column       charges
Skewness    1.515391
Name: 3, dtype: object
------------------------------------------------------------------------------

# TASK 6 - OUTLIER DETECTION

columns=["bmi","charges"]

for col in columns:

    print("\nColumn :",col)

    Q1=df[col].quantile(0.25)

    Q3=df[col].quantile(0.75)

    IQR=Q3-Q1

    lower=Q1-1.5*IQR

    upper=Q3+1.5*IQR

    outliers=df[
        (df[col]<lower)|
        (df[col]>upper)
    ]

    print("Q1 :",Q1)

    print("Q3 :",Q3)

    print("IQR :",IQR)

    print("Lower Bound :",lower)

    print("Upper Bound :",upper)

    print("Outliers :",len(outliers))

Output:

Column : bmi
Q1 : 26.29
Q3 : 34.7
IQR : 8.410000000000004
Lower Bound : 13.674999999999994
Upper Bound : 47.31500000000001
Outliers : 9

Column : charges
Q1 : 4746.344
Q3 : 16657.71745
IQR : 11911.37345
Lower Bound : -13120.716174999998
Upper Bound : 34524.777625
Outliers : 139
-------------------------------------------------------------

import matplotlib.pyplot as plt

# TASK 7 - VISUALIZATIONS

# Line Plot

plt.figure(figsize=(10,5))

plt.plot(df.index,df["charges"])

plt.title("Insurance Charges")

plt.xlabel("Index")

plt.ylabel("Charges")

plt.savefig("line_plot.png")

plt.show()

# Bar Chart

plt.figure(figsize=(8,5))

df.groupby("region")["charges"].mean().plot(kind="bar")

plt.title("Average Charges by Region")

plt.xlabel("Region")

plt.ylabel("Average Charges")

plt.savefig("bar_chart.png")

plt.show()

# Histogram

most_skewed=highest_skew["Column"]

plt.figure(figsize=(8,5))

sns.histplot(df[most_skewed],
             bins=20,
             kde=True)

plt.title(f"Histogram of {most_skewed}")

plt.savefig("histogram.png")

plt.show()

# Scatter Plot

plt.figure(figsize=(8,6))

sns.scatterplot(data=df,
                x="bmi",
                y="charges",
                hue="smoker")

plt.title("BMI vs Charges")

plt.savefig("scatter_plot.png")

plt.show()

# Box Plot

plt.figure(figsize=(8,6))

sns.boxplot(data=df,
            x="smoker",
            y="charges")

plt.title("Charges by Smoker")

plt.savefig("box_plot.png")

plt.show()

print("All visualizations created successfully.")

Output:


------------------------------------------------------------------
# TASK 8 - CORRELATION HEATMAP

import numpy as np

numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Pearson Correlation
pearson_corr = numeric_df.corr()

print("\nPearson Correlation Matrix")
print(pearson_corr)

plt.figure(figsize=(8,6))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Highest absolute correlation
corr_abs = pearson_corr.abs()
np.fill_diagonal(corr_abs.values, 0)

highest_pair = corr_abs.stack().idxmax()
highest_value = corr_abs.stack().max()

print("\nHighest Absolute Correlation Pair:")
print(highest_pair)
print("Correlation Value:", highest_value)

Output:

-----------------------------------------------------------------------

# TASK 9(a) - MEAN VS MEDIAN COMPARISON

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

skew_values = {}

for col in numeric_cols:
    skew_values[col] = abs(df[col].skew())

skew_df = pd.DataFrame(skew_values.items(), columns=["Column","AbsSkew"])

top2 = skew_df.sort_values(by="AbsSkew", ascending=False).head(2)

print("\nTop 2 Highest Skewed Columns")
print(top2)

for col in top2["Column"]:

    print("\nColumn:", col)

    mean_value = df[col].mean()
    median_value = df[col].median()

    print("Mean   :", mean_value)
    print("Median :", median_value)

    # Fill remaining nulls using median
    df[col] = df[col].fillna(median_value)

print("\nNull values after imputation")

print(df[top2["Column"]].isnull().sum())

Output:

Top 2 Highest Skewed Columns
     Column   AbsSkew
3   charges  1.515391
2  children  0.937421

Column: charges
Mean   : 13279.121486655948
Median : 9386.1613

Column: children
Mean   : 1.0957367240089753
Median : 1.0

Null values after imputation
charges     0
children    0
dtype: int64
----------------------------------------------------------
# TASK 9(b) - SPEARMAN CORRELATION

spearman_corr = numeric_df.corr(method='spearman')

print("\nSpearman Correlation Matrix")
print(spearman_corr)

difference = abs(spearman_corr - pearson_corr)

pairs = []

cols = difference.columns

for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        pairs.append({
            "Column 1": cols[i],
            "Column 2": cols[j],
            "Pearson": pearson_corr.iloc[i,j],
            "Spearman": spearman_corr.iloc[i,j],
            "|Difference|": difference.iloc[i,j]
        })

difference_table = pd.DataFrame(pairs)

difference_table = difference_table.sort_values(
    by="|Difference|",
    ascending=False
)

print("\nTop 3 Largest Differences")

print(difference_table.head(3))

Output:

Spearman Correlation Matrix
               age       bmi  children   charges
age       1.000000  0.107897  0.055813  0.533523
bmi       0.107897  1.000000  0.015643  0.119585
children  0.055813  0.015643  1.000000  0.132200
charges   0.533523  0.119585  0.132200  1.000000

Top 3 Largest Differences
   Column 1 Column 2   Pearson  Spearman  |Difference|
2       age  charges  0.298308  0.533523      0.235215
4       bmi  charges  0.198401  0.119585      0.078816
5  children  charges  0.067389  0.132200      0.064811

-----------------------------------------------------------------------
# TASK 9(c)- GROUPED AGGREGATION

group_result = df.groupby("region")["charges"].agg(
    ['mean','std','count']
)

print("\nGrouped Aggregation")

print(group_result)

highest_mean = group_result["mean"].idxmax()

highest_std = group_result["std"].idxmax()

print("\nHighest Mean Group :", highest_mean)

print("Highest Standard Deviation Group :", highest_std)

ratio = group_result["mean"].max() / group_result["mean"].min()

print("\nHighest Mean / Lowest Mean Ratio")

print(ratio)

Output:

Grouped Aggregation
                   mean           std  count
region                                      
northeast  13406.384516  11255.803066    324
northwest  12450.840844  11073.125699    324
southeast  14735.411438  13971.098589    364
southwest  12346.937377  11557.179101    325

Highest Mean Group : southeast
Highest Standard Deviation Group : southeast

Highest Mean / Lowest Mean Ratio
1.1934466813373743
/tmp/ipykernel_5814/4083779600.py:5: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  group_result = df.groupby("region")["charges"].agg( 
--------------------------------------------------------------------------------
# Task 10- SAVE CLEAN DATASET

df.to_csv("cleaned_data.csv", index=False)

print("cleaned_data.csv saved successfully.")

Output:
cleaned_data.csv saved successfully.
----------------------------------------------------------------------

