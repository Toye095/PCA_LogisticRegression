# Project: Breast Cancer Dataset Analysis with PCA and Logistic Regression

## Table of Contents
* 1. Overview
* 2. Features
* 3. Libraries Used
* 4. Data Flow and Key Steps
    * a. Data Loading
    * b. Standardization
    * c. PCA Implementation
    * d. Logistic Regression
* 5. Visualizations
* 6. Output
* 7. Results
* 7. How to Run the Code

## 1. Overview
This project implemented Principal Component Analysis (PCA) for dimensionality reduction and adopted Logistic Regression to build a classification model on the provided Breast Cancer Dataset. It reduce the dataset's dimensionality and retained the most significant features for predictive model building.

## 2. Features
* Data Loading: Loads the Breast Cancer dataset from sklearn.datasets.
* Standardization: Prepares the data for PCA by standardizing features.
* PCA Implementation:
    * Extracts eigenvalues, eigenvectors, and explained variance.
    * Visualizes the variance explained by each principal component.
    * Reduces the dataset to 2 principal components.
* Logistic Regression:
    * Trains a Logistic Regression model on the reduced dataset.
    * Evaluates the model's accuracy and provides a classification report.
    
## 3. Libraries Used
    * sklearn.datasets: Load Breast Cancer dataset.
    * numpy: Array manipulation and cumulative sum calculations.
    * pandas: Dataframe creation for visualization.
    * matplotlib: Plotting graphs to visualize PCA results.
    * seaborn: Enhanced data visualization for PCA results.
    * sklearn.decomposition: Principal Component Analysis (PCA).
    * sklearn.model_selection: Data splitting for training and testing.
    * sklearn.preprocessing: Data standardization.
    * sklearn.linear_model: Logistic Regression implementation.
    * sklearn.metrics: Model evaluation metrics.    

## 4. Data Flow and Key Steps
### 1. Data Loading
* Load the dataset using load_breast_cancer().
* Separate the independent (X) and dependent (y) variables.
### 2. Standardization
* Standardize the features to ensure equal importance for all variables during PCA.
### 3. PCA Implementation
* Perform PCA with all 30 components to identify the explained variance.
* Visualize eigenvalues against the explained variance.
* Reduce the dimensionality to 2 components for further analysis and modeling.
### 4. Logistic Regression
* Split the reduced dataset into training and testing sets.
* Train a logistic regression model using the training data.
* Evaluate the model's performance on the test set using accuracy and classification metrics.

## 5. Visualizations
* Variance Explained by PCA Components: Bar plot showing the percentage of variance explained by each eigenvalue.
* Cumulative Variance: Step plot showing how cumulative variance increases with more components.
* PCA Component Visualization: Pair plot to visualize the reduced dataset.

## 6. Outputs
* Explained Variance Ratio: Displays the proportion of variance explained by each PCA component.
* Model Evaluation: Accuracy: Percentage of correct predictions.
* Classification Report: Includes precision, recall, and F1-score

## 7. Results
* PCA Explained Variance (First 2 Components)
* Variance explained by Component 1: ~44%
* Variance explained by Component 2: ~25%
* Total Variance: ~69%
* Logistic Regression Results
* Accuracy: 95% 
* Precision: 0.96
* Recall: 0.94
* F1-Score: 0.95

## 8. How to Run the Code
* Clone the repository and ensure Python 3.x is installed.
* Install required libraries
* Run the script in a Jupyter Notebook


```python

```
