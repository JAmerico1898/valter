# Credit Risk Modeling Interactive App

## Overview
This Streamlit application provides an interactive educational tool for understanding credit risk modeling. It allows users to build, evaluate, and apply logistic regression models to predict loan defaults. The app is designed for recently hired interns to learn credit modeling concepts through hands-on exploration.

## Features
The application guides users through the entire credit modeling workflow:

1. **Data Overview**: Explore training and testing datasets with visualizations of class distribution.
2. **Feature Selection**: Choose which features to include in the predictive model.
3. **Model Training**: Build a logistic regression model with 70:30 train-test split.
4. **Model Visualization**:
   - Logistic regression S-curve showing predicted probabilities
   - ROC/AUC curve with performance interpretation
   - Confusion matrix with key classification metrics
   - Detailed regression statistics and coefficient interpretation
   - Visual representation of the logistic regression equation
5. **Borrower Analysis**: Apply the trained model to potential borrowers.
6. **Performance Evaluation**: Compare predictions to actual outcomes with comprehensive analysis.

## Data Requirements
The application works with two main datasets:
- `training_sample.csv`: Historical loan data with loan status (80% repaid, 20% default)
- `testing_sample.csv`: Potential borrower data for predictions
- `testing_sample_true.csv`: Actual outcomes for the testing sample (for evaluation)

### Expected Data Structure
Training data should include:
- `id`: Unique identifier for each loan
- Features: `loan_amnt`, `int_rate`, `annual_inc`, `dti`, `delinq_2yrs`, `fico_range_low`, grade dummies (`grade_A` through `grade_G`)
- `loan_status`: Target variable (0 = repaid, 1 = default)

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Setup
1. Clone the repository or download the application code.
2. Install required packages:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```
3. Place your data files in the same directory as the application.

### Running the App
```bash
streamlit run credit_risk_modeling_app.py
```

## Usage Guide

### 1. Data Exploration
- Review the training data characteristics
- Understand the class distribution

### 2. Feature Selection
- Select relevant features for model building
- Choose both numerical and categorical features

### 3. Model Training
- Click "Train Logistic Regression Model" to build the model
- Review model visualizations and statistics

### 4. Analyzing Potential Borrowers
- Click "Analyze Potential Borrowers" to apply the model to new data
- Review risk tiers and predicted probabilities

### 5. Evaluating Model Performance
- Compare predictions against actual outcomes
- Analyze errors and business impact
- Optimize decision thresholds

## Educational Components

### Key Concepts Covered
- Logistic regression fundamentals
- Feature selection and impact
- Interpreting model coefficients
- Classification metrics (accuracy, precision, recall, F1)
- ROC curves and AUC
- Confusion matrices
- Decision thresholds and business implications
- Cost-benefit analysis of model errors

### Interpretability Features
- Coefficient explanation in terms of odds ratios
- Visual representation of the logistic function
- Error analysis with business context
- Decision threshold optimization

## Demo Mode
If data files aren't available, the application will generate synthetic data for demonstration purposes. This allows the app to function as a standalone educational tool without requiring actual loan data.

## Customization
The application can be extended or modified to:
- Include additional model types (random forest, gradient boosting, etc.)
- Add more advanced visualizations
- Incorporate feature engineering capabilities
- Include model comparison functionality

## Technical Notes
- The app handles data cleaning and preprocessing automatically
- Statsmodels dependency has been removed to improve reliability
- Performance metrics are calculated using scikit-learn
- Visualizations are created with Matplotlib and Seaborn

## Troubleshooting
- If encountering data format issues, ensure CSV files have the expected column structure
- For memory issues with large datasets, consider sampling the data
- Check console logs for detailed error messages
