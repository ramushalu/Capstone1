# Capstone1

# Part 1 – Data Acquisition, Cleaning, and Exploratory Data Analysis

## Project Overview

This project is Part 1 of the Applied AI & ML Essentials Capstone Project. The objective is to clean, analyze, and understand a raw dataset before applying machine learning techniques. The Medical Cost Personal Dataset was selected because it contains both numerical and categorical variables and is suitable for regression analysis.

---

# Dataset Description

**Dataset:** Medical Cost Personal Dataset (`insurance.csv`)

Number of Rows: **1338**

Number of Columns: **7**

Features:

* age
* sex
* bmi
* children
* smoker
* region
* charges (Target Variable)

The dataset contains demographic and medical information that can be used to predict individual medical insurance charges.

---

# Software Requirements

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn
```

---

# How to Run

1. Download the dataset (`insurance.csv`).
2. Place it in the project folder.
3. Run the Python script or Jupyter Notebook.
4. The cleaned dataset and plots will be generated automatically.

---

# Data Cleaning Steps

## 1. Data Loading

The dataset was loaded using `pd.read_csv()`.

The following information was displayed:

* First five rows
* Data types
* Dataset shape

---

## 2. Missing Value Analysis

Missing values were calculated using:

* `df.isnull().sum()`
* Percentage of missing values

Columns with more than 20% missing values were identified.

For numeric columns with less than 20% missing values, missing values were replaced using the **median**.

### Why Median Instead of Mean?

The median is resistant to extreme values (outliers). Since insurance charges are highly skewed, the median represents the center of the data more accurately than the mean.

---

## 3. Duplicate Records

Duplicate rows were detected using

```python
df.duplicated().sum()
```

Duplicate rows were removed using

```python
df.drop_duplicates()
```

The number of removed rows was reported.

---

## 4. Data Type Conversion

Categorical columns

* sex
* smoker
* region

were converted from object to category data type.

Memory usage before and after conversion was compared.

---

# Descriptive Statistics

Descriptive statistics were calculated using

```python
df.describe()
```

The following statistics were obtained:

* Mean
* Standard Deviation
* Minimum
* Maximum
* Quartiles

---

# Skewness Analysis

Skewness was calculated for every numerical feature.

The column with the highest absolute skewness was identified.

## Interpretation

Positive skew means the distribution has a long right tail caused by a few very large values.

Negative skew means the distribution has a long left tail caused by a few very small values.

Because skewed distributions pull the mean toward extreme values, the median provides a better estimate of central tendency for missing-value imputation.

---

# Outlier Detection

Outliers were detected using the Interquartile Range (IQR) method.

The following were calculated:

* Q1
* Q3
* IQR
* Lower Bound
* Upper Bound

Outliers were identified for:

* BMI
* Charges

No outliers were removed in Part 1.

These outliers will be evaluated during model building in Part 2.

---

# Visualizations

## Line Plot

Displays the variation of insurance charges across the dataset.

---

## Bar Chart

Shows the average insurance charges across different regions.

---

## Histogram

Displays the distribution of the most skewed numerical variable.

The histogram indicates whether the data follows a normal distribution or is positively/negatively skewed.

---

## Scatter Plot

Shows the relationship between BMI and insurance charges.

A positive relationship indicates that higher BMI tends to be associated with higher insurance charges.

Smoking status further influences this relationship.

---

## Box Plot

Compares insurance charges for smokers and non-smokers.

The box plot clearly shows that smokers have a higher median insurance cost and greater variability.

---

# Correlation Heatmap

Pearson correlation was calculated for all numerical variables.

The heatmap visualizes the strength of linear relationships.

The strongest correlated variables were identified.

A strong correlation does **not** imply causation.

For example, smoking status may influence both BMI and insurance charges, making it a possible third variable.

---

# Spearman Correlation

Spearman correlation was calculated to identify monotonic relationships.

The Pearson and Spearman correlation matrices were compared.

The three variable pairs with the largest absolute differences were identified.

If

```
|Spearman| > |Pearson|
```

the relationship is likely monotonic but non-linear.

If

```
|Pearson| ≥ |Spearman|
```

the relationship is approximately linear.

For feature selection in Part 2, Pearson correlation will be used for linear regression models, while Spearman correlation provides additional insight into non-linear monotonic relationships.

---

# Grouped Aggregation

Insurance charges were grouped by region.

The following statistics were calculated:

* Mean
* Standard Deviation
* Count

The region with the highest average insurance charges was identified.

The region with the highest variability was also identified.

The ratio between the highest and lowest mean values was computed to determine whether the categorical feature carries predictive information.

---

# Output Files

The project automatically generates:

* cleaned_data.csv
* line_plot.png
* bar_chart.png
* histogram.png
* scatter_plot.png
* box_plot.png
* correlation_heatmap.png

---

# Conclusion

The dataset was successfully cleaned and explored.

Major findings include:

* Missing values were handled using median imputation.
* Duplicate records were removed.
* Categorical variables were optimized using category data types.
* Outliers were identified but retained.
* Correlation analysis highlighted relationships among variables.
* Visualizations provided valuable insights into the data distribution and feature relationships.

The cleaned dataset (`cleaned_data.csv`) is ready for feature engineering and machine learning in Part 2.
