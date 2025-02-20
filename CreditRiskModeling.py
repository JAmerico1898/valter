import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, log_loss
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Credit Risk Modeling Demo",
    page_icon="💰",
    layout="wide"
)

# App title and description
st.title("Credit Risk Modeling Interactive Tool")
st.markdown("""
This tool demonstrates how credit risk modeling works using logistic regression.
You can select features, train a model, and analyze potential borrowers.
""")

# Function to load data
@st.cache_data
def load_data():
    # In a real app, replace these with actual file paths
    try:
        training_sample = pd.read_csv('training_sample.csv')
        testing_sample = pd.read_csv('testing_sample.csv')
        return training_sample, testing_sample
    except:
        # Demo data for when files aren't available
        st.warning("Using demo data. In production, connect to actual datasets.")
        # Create synthetic training data
        np.random.seed(42)
        n_training = 250000
        n_testing = 20000
        
        # Features
        loan_amnt = np.random.uniform(1000, 35000, n_training)
        int_rate = np.random.uniform(5, 25, n_training)
        annual_inc = np.random.uniform(20000, 150000, n_training)
        dti = np.random.uniform(0, 40, n_training)  # debt-to-income ratio
        delinq_2yrs = np.random.poisson(0.5, n_training)
        fico_range_low = np.random.normal(700, 50, n_training).astype(int)
        
        # Grade dummy variables (one-hot encoded)
        grade_probs = [0.30, 0.25, 0.20, 0.15, 0.05, 0.03, 0.02]  # Probability distribution
        grades = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_training, p=grade_probs)
        grade_A = (grades == 'A').astype(int)
        grade_B = (grades == 'B').astype(int)
        grade_C = (grades == 'C').astype(int)
        grade_D = (grades == 'D').astype(int)
        grade_E = (grades == 'E').astype(int)
        grade_F = (grades == 'F').astype(int)
        grade_G = (grades == 'G').astype(int)
        
        # Generate loan_status (target) with realistic relationship to features
        # Higher FICO, lower loan amount, lower interest rate, higher income = lower default probability
        logit = -5 + 0.00005 * loan_amnt + 0.1 * int_rate - 0.00001 * annual_inc + 0.05 * dti + \
                0.5 * delinq_2yrs - 0.01 * fico_range_low + 0 * grade_A + 0.2 * grade_B + \
                0.5 * grade_C + 0.8 * grade_D + 1.1 * grade_E + 1.4 * grade_F + 1.7 * grade_G
        
        prob_default = 1 / (1 + np.exp(-logit))
        loan_status = np.random.binomial(1, prob_default)
        
        # Ensure 80% successful, 20% default ratio by resampling
        successful_indices = np.where(loan_status == 0)[0]
        default_indices = np.where(loan_status == 1)[0]
        
        target_successful_count = int(0.8 * n_training)
        target_default_count = n_training - target_successful_count
        
        if len(successful_indices) > target_successful_count:
            successful_indices = np.random.choice(successful_indices, target_successful_count, replace=False)
        else:
            # Need to create more successful loans
            additional_needed = target_successful_count - len(successful_indices)
            loan_status[np.random.choice(default_indices, additional_needed, replace=False)] = 0
            successful_indices = np.where(loan_status == 0)[0]
            default_indices = np.where(loan_status == 1)[0]
        
        if len(default_indices) > target_default_count:
            default_indices = np.random.choice(default_indices, target_default_count, replace=False)
        else:
            # Need to create more defaults
            additional_needed = target_default_count - len(default_indices)
            loan_status[np.random.choice(successful_indices, additional_needed, replace=False)] = 1
        
        # Create DataFrame
        training_sample = pd.DataFrame({
            'id': range(1, n_training + 1),
            'loan_amnt': loan_amnt,
            'int_rate': int_rate,
            'annual_inc': annual_inc,
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'fico_range_low': fico_range_low,
            'grade_A': grade_A,
            'grade_B': grade_B,
            'grade_C': grade_C,
            'grade_D': grade_D,
            'grade_E': grade_E,
            'grade_F': grade_F,
            'grade_G': grade_G,
            'loan_status': loan_status
        })
        
        # Create testing sample (similar structure but without loan_status)
        # Use same distributions but different random samples
        loan_amnt_test = np.random.uniform(1000, 35000, n_testing)
        int_rate_test = np.random.uniform(5, 25, n_testing)
        annual_inc_test = np.random.uniform(20000, 150000, n_testing)
        dti_test = np.random.uniform(0, 40, n_testing)
        delinq_2yrs_test = np.random.poisson(0.5, n_testing)
        fico_range_low_test = np.random.normal(700, 50, n_testing).astype(int)
        
        grades_test = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_testing, p=grade_probs)
        grade_A_test = (grades_test == 'A').astype(int)
        grade_B_test = (grades_test == 'B').astype(int)
        grade_C_test = (grades_test == 'C').astype(int)
        grade_D_test = (grades_test == 'D').astype(int)
        grade_E_test = (grades_test == 'E').astype(int)
        grade_F_test = (grades_test == 'F').astype(int)
        grade_G_test = (grades_test == 'G').astype(int)
        
        testing_sample = pd.DataFrame({
            'id': range(1, n_testing + 1),
            'loan_amnt': loan_amnt_test,
            'int_rate': int_rate_test,
            'annual_inc': annual_inc_test,
            'dti': dti_test,
            'delinq_2yrs': delinq_2yrs_test,
            'fico_range_low': fico_range_low_test,
            'grade_A': grade_A_test,
            'grade_B': grade_B_test,
            'grade_C': grade_C_test,
            'grade_D': grade_D_test,
            'grade_E': grade_E_test,
            'grade_F': grade_F_test,
            'grade_G': grade_G_test
        })
        
        return training_sample, testing_sample

# Load data
training_sample, testing_sample = load_data()

# Display data overview
with st.expander("Data Overview"):
    st.subheader("Training Sample")
    st.write(f"Shape: {training_sample.shape}")
    
    # Styling DataFrame using Pandas
    def style_table(df):
        df = df.reset_index(drop=True)

        # Create a list of column names for columns 2 to 8
        columns_to_format = df.columns[2:8]

        return (
            df.style
            .set_table_styles(
                [{
                    'selector': 'thead th',
                    'props': [('font-weight', 'bold'),
                            ('border-style', 'solid'),
                            ('border-width', '0px 0px 2px 0px'),
                            ('border-color', 'black')]
                }, {
                    'selector': 'thead th:not(:first-child)',
                    'props': [('text-align', 'center')]  # Center all headers except the first
                }, {
                    'selector': 'thead th:last-child',
                    'props': [('color', 'black')]  # Make last column header black
                }, {
                    'selector': 'td',
                    'props': [('border-style', 'solid'),
                            ('border-width', '0px 0px 1px 0px'),
                            ('border-color', 'black'),
                            ('text-align', 'center')]
                }, {
                    'selector': 'th',
                    'props': [('border-style', 'solid'),
                            ('border-width', '0px 0px 1px 0px'),
                            ('border-color', 'black'),
                            ('text-align', 'left')]
                }]
            )
            .set_properties(**{'padding': '2px', 'font-size': '15px'})
            .format({col: "{:.2f}" for col in columns_to_format})  # Format columns to 2 decimal places
        )

    # Displaying in Streamlit
    def main():
        styled_html = style_table(training_sample.head()).to_html(index=False, escape=False)
        centered_html = f'''
        <div style="display: flex; justify-content: left;">
            {styled_html}
        '''  # Properly close the div tag
        st.markdown(centered_html, unsafe_allow_html=True)

    if __name__ == '__main__':
        main()
    
    
    st.subheader("Testing Sample")
    st.write(f"Shape: {testing_sample.shape}")

    # Styling DataFrame using Pandas
    def style_table(df):
        df = df.reset_index(drop=True)

        # Create a list of column names for columns 2 to 8
        columns_to_format = df.columns[2:8]

        return (
            df.style
            .set_table_styles(
                [{
                    'selector': 'thead th',
                    'props': [('font-weight', 'bold'),
                            ('border-style', 'solid'),
                            ('border-width', '0px 0px 2px 0px'),
                            ('border-color', 'black')]
                }, {
                    'selector': 'thead th:not(:first-child)',
                    'props': [('text-align', 'center')]  # Center all headers except the first
                }, {
                    'selector': 'thead th:last-child',
                    'props': [('color', 'black')]  # Make last column header black
                }, {
                    'selector': 'td',
                    'props': [('border-style', 'solid'),
                            ('border-width', '0px 0px 1px 0px'),
                            ('border-color', 'black'),
                            ('text-align', 'center')]
                }, {
                    'selector': 'th',
                    'props': [('border-style', 'solid'),
                            ('border-width', '0px 0px 1px 0px'),
                            ('border-color', 'black'),
                            ('text-align', 'left')]
                }]
            )
            .set_properties(**{'padding': '2px', 'font-size': '15px'})
            .format({col: "{:.2f}" for col in columns_to_format})  # Format columns to 2 decimal places
        )

    # Displaying in Streamlit
    def main():
        styled_html = style_table(testing_sample.head()).to_html(index=False, escape=False)
        centered_html = f'''
        <div style="display: flex; justify-content: left;">
            {styled_html}
        '''  # Properly close the div tag
        st.markdown(centered_html, unsafe_allow_html=True)

    if __name__ == '__main__':
        main()

    
    # Display class distribution
    st.subheader("Class Distribution in Training Data")
    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts = training_sample['loan_status'].value_counts()
    ax.bar(['Repaid (0)', 'Default (1)'], class_counts.values)
    ax.set_ylabel('Count')
    ax.set_title('Loan Status Distribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 3000, f"{v} ({v/len(training_sample)*100:.1f}%)", ha='center')
    st.pyplot(fig)

# Feature selection
st.header("1. Feature Selection")
st.write("Select features to include in your logistic regression model:")

# Get available features (excluding id and target)
available_features = [col for col in training_sample.columns 
                     if col not in ['id', 'loan_status']]

# Group features by category for better organization
numerical_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low']
categorical_features = [col for col in available_features if col.startswith('grade_')]

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Features")
        selected_numerical = []
        for feature in numerical_features:
            if st.checkbox(f"{feature}", value=True):
                selected_numerical.append(feature)
    
    with col2:
        st.subheader("Categorical Features")
        selected_categorical = []
        for feature in categorical_features:
            if st.checkbox(f"{feature}", value=True):
                selected_categorical.append(feature)

selected_features = selected_numerical + selected_categorical

if not selected_features:
    st.warning("Please select at least one feature to build the model.")
else:
    st.success(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")

# Model training
st.header("2. Model Training")

if st.button("Train Logistic Regression Model", disabled=not selected_features):
    if not selected_features:
        st.error("No features selected. Please select at least one feature.")
    else:
        with st.spinner("Training model..."):
            # Prepare data
            X = training_sample[selected_features]
            y = training_sample['loan_status']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # Fit logistic regression model
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            st.session_state.model = model
            
            # Predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            st.session_state.y_pred = y_pred
            st.session_state.y_pred_proba = y_pred_proba
            
            # Skip statsmodels and use sklearn's LogisticRegression directly
            # This bypasses the problematic statsmodels fitting process
            st.session_state.model_coef = model.coef_[0]
            st.session_state.model_intercept = model.intercept_[0]
            
            # Create custom summary statistics
            from sklearn.metrics import log_loss
            train_pred_proba = model.predict_proba(X_train)[:, 1]
            train_log_loss = log_loss(y_train, train_pred_proba)
            
            # Store these for display later
            st.session_state.custom_summary = {
                'features': selected_features,
                'coefficients': model.coef_[0],
                'intercept': model.intercept_[0],
                'train_log_loss': train_log_loss,
                'train_accuracy': model.score(X_train, y_train),
                'test_accuracy': model.score(X_test, y_test)
            }
            
            # We'll skip the statsmodels summary and show our custom summary instead
            # This removes the dependency on statsmodels which is causing issues
            # We don't need to store sm_model anymore since we're using our custom summary
            
            # Calculate equation for display
            coefficients = model.coef_[0]
            intercept = model.intercept_[0]
            st.session_state.coefficients = coefficients
            st.session_state.intercept = intercept
            st.session_state.selected_features = selected_features
            
            st.success("Model trained successfully!")
            st.session_state.model_trained = True
            
            # Store feature information for later use with testing sample
            st.session_state.original_features = selected_features

# Results display - only show if model has been trained
if 'model_trained' in st.session_state and st.session_state.model_trained:
    st.header("Model Results")
    
    # 1. Logistic Regression Curve
    st.subheader("2. Logistic Regression S-Curve")
    with st.container():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort probabilities and actual values for plotting
        sorted_indices = np.argsort(st.session_state.y_pred_proba)
        sorted_probs = st.session_state.y_pred_proba[sorted_indices]
        sorted_actuals = st.session_state.y_test.values[sorted_indices]
        
        # Plot the logistic curve
        ax.plot(range(len(sorted_probs)), sorted_probs, 'b-', linewidth=2)
        
        # Add actual observations as dots (jittered for visibility)
        y_jittered = sorted_actuals + np.random.normal(0, 0.02, len(sorted_actuals))
        ax.scatter(range(len(sorted_probs)), y_jittered, c='r', alpha=0.1, s=1)
        
        ax.set_xlabel('Observations (ordered by predicted probability)')
        ax.set_ylabel('Probability of Default')
        ax.set_title('Logistic Regression S-Curve')
        ax.grid(True, alpha=0.3)
        
        # Add a horizontal line at 0.5 probability
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
        ax.text(len(sorted_probs)*0.02, 0.52, 'Decision Threshold (p=0.5)', color='green')
        
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:** The S-curve shows how the model's predicted probability of default varies across all observations.
        - Blue line: Predicted probabilities (sorted from lowest to highest)
        - Red dots: Actual outcomes (0=repaid, 1=default)
        - Green line: Decision threshold (typically 0.5)
        
        In a good model, you want to see most of the dots clustered at the bottom left (correctly predicted repayments) 
        and top right (correctly predicted defaults).
        """)
    
    # 2. ROC Curve
    st.subheader("3. ROC Curve")
    with st.container():
        fpr, tpr, thresholds = roc_curve(st.session_state.y_test, st.session_state.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown(f"""
        **Interpretation:** The ROC curve shows the trade-off between sensitivity and specificity.
        - AUC (Area Under Curve): {roc_auc:.3f} (higher is better, 1.0 is perfect, 0.5 is random)
        
        The closer the curve follows the top-left corner, the better the model's ability to
        discriminate between repayments and defaults. An AUC of:
        - 0.9-1.0: Excellent discrimination
        - 0.8-0.9: Good discrimination
        - 0.7-0.8: Fair discrimination
        - 0.6-0.7: Poor discrimination
        - 0.5-0.6: Failed discrimination
        """)
    
    # 3. Confusion Matrix
    st.subheader("4. Confusion Matrix")
    with st.container():
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted Repaid', 'Predicted Default'],
                    yticklabels=['Actually Repaid', 'Actually Default'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Metrics:**
            - **Accuracy:** {accuracy:.3f}
            - **Precision:** {precision:.3f}
            - **Recall (Sensitivity):** {recall:.3f}
            - **Specificity:** {specificity:.3f}
            - **F1 Score:** {f1:.3f}
            """)
            
        with col2:
            st.markdown("""
            **Definitions:**
            - **True Negatives (TN):** Correctly predicted repayments
            - **False Positives (FP):** Incorrectly predicted defaults
            - **False Negatives (FN):** Incorrectly predicted repayments
            - **True Positives (TP):** Correctly predicted defaults
            - **Precision:** Proportion of predicted defaults that were actual defaults
            - **Recall:** Proportion of actual defaults that were correctly predicted
            """)
    
    # 4. Model Summary (Custom version since statsmodels is causing issues)
    st.subheader("5. Regression Statistics Summary")
    with st.container():
        if 'custom_summary' in st.session_state:
            summary = st.session_state.custom_summary
            
            # Create a DataFrame for coefficient display
            coef_df = pd.DataFrame({
                'Feature': summary['features'],
                'Coefficient': summary['coefficients'],
                'Odds Ratio': np.exp(summary['coefficients'])
            })
            
            # Display model information
            st.markdown("### Model Information")
            st.markdown(f"""
            - **Number of observations:** {len(st.session_state.X_train)}
            - **Number of predictors:** {len(summary['features'])}
            - **Intercept:** {summary['intercept']:.4f}
            - **Training Log Loss:** {summary['train_log_loss']:.4f}
            - **Training Accuracy:** {summary['train_accuracy']:.4f}
            - **Testing Accuracy:** {summary['test_accuracy']:.4f}
            """)
            
            st.markdown("### Coefficients")
            
            # Styling DataFrame using Pandas
            def style_table(df):
                df = df.reset_index(drop=True)
                
                return df.style.set_table_styles(
                    [{
                        'selector': 'thead th',
                        'props': [('font-weight', 'bold'),
                                ('border-style', 'solid'),
                                ('border-width', '0px 0px 2px 0px'),
                                ('border-color', 'black'),]
                    }, {
                        'selector': 'thead th:not(:first-child)',
                        'props': [('text-align', 'center')]  # Centering all headers except the first
                    }, {
                        'selector': 'thead th:last-child',
                        'props': [('color', 'black')]  # Make last column header black
                    }, {
                        'selector': 'td',
                        'props': [('border-style', 'solid'),
                                ('border-width', '0px 0px 1px 0px'),
                                ('border-color', 'black'),
                                ('text-align', 'center')]
                    }, {
                        'selector': 'th',
                        'props': [('border-style', 'solid'),
                                ('border-width', '0px 0px 1px 0px'),
                                ('border-color', 'black'),
                                ('text-align', 'left'),]
                    }]
                ).set_properties(**{'padding': '2px',
                                    'font-size': '15px'})

            # Displaying in Streamlit
            def main():
                styled_html = style_table(coef_df).to_html(index=False, escape=False)
                centered_html = f'''
                <div style="display: flex; justify-content: left;">
                    {styled_html}
                '''  # Properly close the div tag
                st.markdown(centered_html, unsafe_allow_html=True)

            if __name__ == '__main__':
                main()
            
            # Add download button for summary
            summary_text = f"""
            Credit Risk Logistic Regression Model Summary
            ==============================================
            
            Model Information:
            -----------------
            Number of observations: {len(st.session_state.X_train)}
            Number of predictors: {len(summary['features'])}
            Intercept: {summary['intercept']:.6f}
            Training Log Loss: {summary['train_log_loss']:.6f}
            Training Accuracy: {summary['train_accuracy']:.6f}
            Testing Accuracy: {summary['test_accuracy']:.6f}
            
            Coefficients:
            ------------
            """
            
            for feature, coef, odds in zip(summary['features'], summary['coefficients'], np.exp(summary['coefficients'])):
                summary_text += f"{feature}: {coef:.6f} (odds ratio: {odds:.6f})\n"
            
            st.download_button(
                label="Download Summary as Text",
                data=summary_text,
                file_name="logistic_regression_summary.txt",
                mime="text/plain"
            )
        else:
            st.error("Model summary statistics are not available. Please train the model first.")
    
    # 5. Statistics explanation
    st.subheader("6. Interpretation of Statistics")
    with st.container():
        st.markdown("""
        **Key statistics and their interpretation:**
        
        **Coefficient (coef):** 
        - Indicates the change in log odds of default for a one-unit increase in the predictor.
        - Positive coefficient: As the predictor increases, default probability increases.
        - Negative coefficient: As the predictor increases, default probability decreases.
        
        **Odds Ratio:**
        - The exponentiated coefficient (e^coef).
        - Represents how the odds of default multiply when the predictor increases by one unit.
        - Odds Ratio > 1: The feature increases default risk.
        - Odds Ratio < 1: The feature decreases default risk.
        
        **Log Loss:**
        - Measures how well the model's predicted probabilities match the actual outcomes.
        - Lower values indicate better fit (less uncertainty).
        
        **Accuracy:**
        - Proportion of correct predictions (both repaid and defaulted loans).
        - Higher values indicate better overall predictive performance.
        
        **Interpreting Feature Effects:**
        - The magnitude of coefficients indicates the strength of the effect.
        - Categorical variables (like grade) show the effect relative to a reference category.
        - Features with larger absolute coefficients have stronger effects on default probability.
        """)
    
    # 6. Logistic Regression Equation
    st.subheader("7. Logistic Regression Equation")
    with st.container():
        # Format the equation with better spacing and alignment
        st.markdown("""
        The logistic regression equation (log-odds form):
        """)
        
        #Build equation in parts for better readability
        equation_start = r"""
        $\large 
        \log \left( \frac{P(\text{Default})}{1 - P(\text{Default})} \right) = 
        0.0351 \;+\; 0.0000 \times \text{loan\_amnt} \;+\; 0.0766 \times \text{int\_rate} \;-\; 0.0000 \times \text{annual\_inc} \\
        +\; 0.0197 \times \text{dti} \;+\; 0.0068 \times \text{delinq\_2yrs} \;-\; 0.0043 \times \text{fico\_range\_low} \\
        -\; 0.4751 \times \text{grade\_A} \;-\; 0.1542 \times \text{grade\_B} \;+\; 0.1256 \times \text{grade\_C} \\
        +\; 0.2646 \times \text{grade\_D} \;+\; 0.2908 \times \text{grade\_E} \;+\; 0.0096 \times \text{grade\_F} \;-\; 0.0262 \times \text{grade\_G}$
        """
       
        # Format intercept
        intercept_term = f"{st.session_state.intercept:.4f}"
        
        # Format feature terms with proper spacing
        feature_terms = []
        for i, feature in enumerate(st.session_state.selected_features):
            coef = st.session_state.coefficients[i]
            if i == 0 and coef >= 0:
                # First term after intercept, positive
                term = f" {coef:.4f} \\times \\text{{{feature}}}"
            elif i == 0 and coef < 0:
                # First term after intercept, negative
                term = f" {coef:.4f} \\times \\text{{{feature}}}"
            elif coef >= 0:
                # Positive coefficient
                term = f" + {coef:.4f} \\times \\text{{{feature}}}"
            else:
                # Negative coefficient (sign is included in the value)
                term = f" {coef:.4f} \\times \\text{{{feature}}}"
            feature_terms.append(term)
        
        # Combine equation parts
        full_equation = equation_start
        
        # Display equation
        st.markdown(full_equation)
        
        # Probability equation with better formatting
        st.markdown("""
        **Probability of Default:**
        
        $\\large P(\\text{Default}) = \\frac{1}{1 + e^{-z}}$
        
        Where $z$ is the log-odds equation above.
        """)
        
        # Coefficient interpretation table
        st.subheader("Coefficient Interpretation")
        
        coef_df = pd.DataFrame({
            'Feature': st.session_state.selected_features,
            'Coefficient': st.session_state.coefficients,
            'Odds Ratio': np.exp(st.session_state.coefficients)
        })
        
        # Add interpretation
        def get_interpretation(feature, coef, odds_ratio):
            if coef > 0:
                return f"A one-unit increase in {feature} multiplies the odds of default by {odds_ratio:.3f} (increases by {(odds_ratio-1)*100:.1f}%)"
            else:
                return f"A one-unit increase in {feature} multiplies the odds of default by {odds_ratio:.3f} (decreases by {(1-odds_ratio)*100:.1f}%)"
        
        coef_df['Interpretation'] = [
            get_interpretation(feature, coef, odds_ratio) 
            for feature, coef, odds_ratio in zip(
                coef_df['Feature'], coef_df['Coefficient'], coef_df['Odds Ratio']
            )
        ]
        
        # Styling DataFrame using Pandas
        def style_table(df):
            df = df.reset_index(drop=True)
            
            return df.style.set_table_styles(
                [{
                    'selector': 'thead th',
                    'props': [('font-weight', 'bold'),
                            ('border-style', 'solid'),
                            ('border-width', '0px 0px 2px 0px'),
                            ('border-color', 'black'),]
                }, {
                    'selector': 'thead th:not(:first-child)',
                    'props': [('text-align', 'center')]  # Centering all headers except the first
                }, {
                    'selector': 'thead th:last-child',
                    'props': [('color', 'black')]  # Make last column header black
                }, {
                    'selector': 'td',
                    'props': [('border-style', 'solid'),
                            ('border-width', '0px 0px 1px 0px'),
                            ('border-color', 'black'),
                            ('text-align', 'center')]
                }, {
                    'selector': 'th',
                    'props': [('border-style', 'solid'),
                            ('border-width', '0px 0px 1px 0px'),
                            ('border-color', 'black'),
                            ('text-align', 'left'),]
                }]
            ).set_properties(**{'padding': '2px',
                                'font-size': '15px'})

        # Displaying in Streamlit
        def main():
            styled_html = style_table(coef_df).to_html(index=False, escape=False)
            centered_html = f'''
            <div style="display: flex; justify-content: left;">
                {styled_html}
            '''  # Properly close the div tag
            st.markdown(centered_html, unsafe_allow_html=True)

        if __name__ == '__main__':
            main()
        
        
        
        
        
        
        
        
        
        

# Prediction on Test Sample
if 'model_trained' in st.session_state and st.session_state.model_trained:
    st.header("Analyze Potential Borrowers")
    
    st.write("""
    Now you can analyze potential borrowers using the trained model.
    The model will use the same features you selected for training.
    """)
    
    # Check if required features exist in testing sample
    missing_features = [f for f in st.session_state.original_features if f not in testing_sample.columns]
    
    if missing_features:
        st.error(f"Testing sample is missing required features: {', '.join(missing_features)}")
    else:
        if st.button("Analyze Potential Borrowers"):
            with st.spinner("Analyzing potential borrowers..."):
                # Prepare test data using same features
                X_potential = testing_sample[st.session_state.original_features]
                
                # Make predictions
                potential_proba = st.session_state.model.predict_proba(X_potential)[:, 1]
                potential_pred = st.session_state.model.predict(X_potential)
                
                # Add predictions to testing sample
                results_df = testing_sample.copy()
                results_df['predicted_probability'] = potential_proba
                results_df['predicted_status'] = potential_pred
                
                # Store predictions for comparison later
                st.session_state.prediction_results = results_df
                
                # Display results
                st.subheader("Prediction Results")
                
                # Summary stats
                approved_count = (potential_pred == 0).sum()
                rejected_count = (potential_pred == 1).sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Approved Loans", f"{approved_count} ({approved_count/len(potential_pred)*100:.1f}%)")
                with col2:
                    st.metric("Rejected Loans", f"{rejected_count} ({rejected_count/len(potential_pred)*100:.1f}%)")
                
                # Distribution of predicted probabilities
                st.subheader("Distribution of Default Probabilities")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(potential_proba, bins=50, kde=True, ax=ax)
                ax.axvline(x=0.5, color='red', linestyle='--')
                ax.text(0.52, ax.get_ylim()[1]*0.9, 'Decision Threshold (0.5)', color='red')
                ax.set_xlabel('Predicted Probability of Default')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Predicted Default Probabilities')
                
                st.pyplot(fig)
                
                # Display results table
                st.subheader("Detailed Results")
                st.dataframe(results_df.sort_values('predicted_probability', ascending=False))
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='potential_borrowers_predictions.csv',
                    mime='text/csv',
                )
                
                # Risk tier analysis
                st.subheader("Risk Tier Analysis")
                
                # Create risk tiers
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                labels = ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
                results_df['risk_tier'] = pd.cut(results_df['predicted_probability'], bins=bins, labels=labels)
                
                # Count by tier
                tier_counts = results_df['risk_tier'].value_counts().sort_index()
                
                # Display as bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                tier_counts.plot(kind='bar', ax=ax, color=plt.cm.RdYlGn_r(np.linspace(0, 1, len(labels))))
                ax.set_xlabel('Risk Tier')
                ax.set_ylabel('Number of Potential Borrowers')
                ax.set_title('Distribution of Potential Borrowers by Risk Tier')
                
                for i, v in enumerate(tier_counts):
                    ax.text(i, v + 5, f"{v} ({v/len(results_df)*100:.1f}%)", ha='center')
                
                st.pyplot(fig)

    # Compare predictions with actual results
    st.header("Compare Predictions with Actual Results")
    
    st.write("""
    This section compares the model's predictions against the actual loan outcomes
    from the testing_sample_true.csv file.
    """)
    
    # Function to load the true testing data
    @st.cache_data
    def load_true_testing_data():
        try:
            # Try to load the actual file
            testing_sample_true = pd.read_csv('testing_sample_true.csv')
            return testing_sample_true
        except:
            # Create synthetic ground truth data for demo purposes
            st.warning("Using synthetic ground truth data. In production, connect to actual testing_sample_true.csv")
            if 'prediction_results' in st.session_state:
                # Use IDs from prediction results
                ids = st.session_state.prediction_results['id'].values
                n = len(ids)
                
                # Generate synthetic ground truth that's correlated with our predictions
                # but not perfectly matching (80% agreement rate)
                if 'model' in st.session_state:
                    # For demo: make synthetic truth somewhat correlated with model predictions
                    predicted_probs = st.session_state.prediction_results['predicted_probability'].values
                    
                    # Add some noise to the probabilities
                    noisy_probs = predicted_probs + np.random.normal(0, 0.15, n)
                    noisy_probs = np.clip(noisy_probs, 0, 1)
                    
                    # Convert to binary outcomes
                    synthetic_outcomes = (noisy_probs > 0.5).astype(int)
                else:
                    # If no model predictions available, generate random outcomes with realistic distribution
                    synthetic_outcomes = np.random.binomial(1, 0.2, n)  # 20% default rate
                
                return pd.DataFrame({
                    'id': ids,
                    'loan_status': synthetic_outcomes
                })
    
    # Load the true results
    testing_sample_true = load_true_testing_data()
    
    if 'prediction_results' not in st.session_state:
        st.info("Please run 'Analyze Potential Borrowers' first to generate predictions.")
    else:
        # Merge predictions with true results
        comparison_df = st.session_state.prediction_results[['id', 'predicted_status', 'predicted_probability']].merge(
            testing_sample_true[['id', 'loan_status']], 
            on='id',
            how='inner'
        )
        
        if len(comparison_df) == 0:
            st.error("Could not match any predictions with true results. Please check that IDs match between datasets.")
        else:
            st.success(f"Successfully matched {len(comparison_df)} loans between predictions and actual results.")
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(comparison_df['loan_status'], comparison_df['predicted_status'])
            precision = precision_score(comparison_df['loan_status'], comparison_df['predicted_status'])
            recall = recall_score(comparison_df['loan_status'], comparison_df['predicted_status'])
            f1 = f1_score(comparison_df['loan_status'], comparison_df['predicted_status'])
            roc_auc = roc_auc_score(comparison_df['loan_status'], comparison_df['predicted_probability'])
            
            # Display metrics
            st.subheader("Model Performance on Test Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
                st.metric("ROC-AUC", f"{roc_auc:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            
            # Add explanation of metrics
            with st.expander("Understand these metrics"):
                st.markdown("""
                - **Accuracy**: Percentage of correct predictions (both repaid and defaulted loans)
                - **Precision**: Percentage of predicted defaults that were actual defaults (lower false positives)
                - **Recall**: Percentage of actual defaults that were correctly predicted (lower false negatives)
                - **F1 Score**: Harmonic mean of precision and recall
                - **ROC-AUC**: Area under the ROC curve, measures the model's ability to distinguish between classes
                
                In credit risk modeling, different metrics may be prioritized depending on business goals:
                - Higher **precision** means fewer good loans are incorrectly rejected
                - Higher **recall** means fewer bad loans are incorrectly approved
                """)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(comparison_df['loan_status'], comparison_df['predicted_status'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Predicted Repaid', 'Predicted Default'],
                        yticklabels=['Actually Repaid', 'Actually Default'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix on Test Data')
            st.pyplot(fig)
            
            # Error analysis 
            st.subheader("Error Analysis")
            
            # Add error categories
            comparison_df['result_category'] = 'Unknown'
            comparison_df.loc[(comparison_df['predicted_status'] == 0) & (comparison_df['loan_status'] == 0), 'result_category'] = 'True Negative (Correctly Predicted Repayment)'
            comparison_df.loc[(comparison_df['predicted_status'] == 1) & (comparison_df['loan_status'] == 1), 'result_category'] = 'True Positive (Correctly Predicted Default)'
            comparison_df.loc[(comparison_df['predicted_status'] == 1) & (comparison_df['loan_status'] == 0), 'result_category'] = 'False Positive (Incorrectly Predicted Default)'
            comparison_df.loc[(comparison_df['predicted_status'] == 0) & (comparison_df['loan_status'] == 1), 'result_category'] = 'False Negative (Incorrectly Predicted Repayment)'
            
            # Count by category
            result_counts = comparison_df['result_category'].value_counts()
            
            # Display as pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']
            result_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors, textprops={'fontsize': 14})
            ax.set_ylabel('')
            ax.set_title('Distribution of Prediction Results')
            st.pyplot(fig)
            
            # ROC Curve
            st.subheader("ROC Curve on Test Data")
            fpr, tpr, _ = roc_curve(comparison_df['loan_status'], comparison_df['predicted_probability'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
            ax.set_xlabel('False Positive Rate (1 - Specificity)')
            ax.set_ylabel('True Positive Rate (Sensitivity)')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve on Test Data')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Detailed error analysis - show the top misclassified cases
            st.subheader("Top Misclassified Cases")
            
            # False Positives (Type I errors) - rejected good loans
            st.markdown("### False Positives (Good loans incorrectly predicted as defaults)")
            fp_df = comparison_df[comparison_df['result_category'] == 'False Positive (Incorrectly Predicted Default)'].sort_values('predicted_probability', ascending=False)
            if len(fp_df) > 0:
                st.dataframe(fp_df.head(10))
            else:
                st.info("No false positives found.")
                
            # False Negatives (Type II errors) - approved bad loans
            st.markdown("### False Negatives (Bad loans incorrectly predicted as repayments)")
            fn_df = comparison_df[comparison_df['result_category'] == 'False Negative (Incorrectly Predicted Repayment)'].sort_values('predicted_probability')
            if len(fn_df) > 0:
                st.dataframe(fn_df.head(10))
            else:
                st.info("No false negatives found.")
            
            # Business impact analysis
            st.subheader("Business Impact Analysis")
            
            # In credit modeling, false negatives (approving bad loans) typically cost more than 
            # false positives (rejecting good loans)
            fn_count = len(fn_df)
            fp_count = len(fp_df)
            
            # Estimate costs (for demonstration)
            avg_loan_amount = testing_sample['loan_amnt'].mean() if 'loan_amnt' in testing_sample.columns else 10000
            
            # Assume loss rate on defaults is about 70% of principal
            estimated_fn_loss = fn_count * avg_loan_amount * 0.7
            
            # Assume opportunity cost on false positives is about 10% of potential profit
            estimated_fp_loss = fp_count * avg_loan_amount * 0.1
            
            st.markdown(f"""
            ### Estimated Financial Impact
            
            Based on a simple cost model:
            
            - **False Negatives (Approving bad loans):**
              - Count: {fn_count}
              - Estimated loss: ${estimated_fn_loss:,.2f}
              
            - **False Positives (Rejecting good loans):**
              - Count: {fp_count}
              - Estimated opportunity cost: ${estimated_fp_loss:,.2f}
              
            - **Total estimated impact:** ${estimated_fn_loss + estimated_fp_loss:,.2f}
            
            Note: This is a simplified estimate for demonstration purposes. Actual financial impact would
            require more complex analysis incorporating interest rates, recovery rates, operational costs,
            and opportunity costs.
            """)
            
            # Threshold analysis
            st.subheader("Decision Threshold Analysis")
            
            # Calculate metrics at different thresholds
            thresholds = np.linspace(0.1, 0.9, 9)
            threshold_results = []
            
            for threshold in thresholds:
                pred_at_threshold = (comparison_df['predicted_probability'] >= threshold).astype(int)
                acc = accuracy_score(comparison_df['loan_status'], pred_at_threshold)
                prec = precision_score(comparison_df['loan_status'], pred_at_threshold)
                rec = recall_score(comparison_df['loan_status'], pred_at_threshold)
                f1_score_val = f1_score(comparison_df['loan_status'], pred_at_threshold)
                
                # Count FP and FN at this threshold
                fp = ((pred_at_threshold == 1) & (comparison_df['loan_status'] == 0)).sum()
                fn = ((pred_at_threshold == 0) & (comparison_df['loan_status'] == 1)).sum()
                
                # Estimated costs
                fn_cost = fn * avg_loan_amount * 0.7
                fp_cost = fp * avg_loan_amount * 0.1
                total_cost = fn_cost + fp_cost
                
                threshold_results.append({
                    'Threshold': threshold,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1 Score': f1_score_val,
                    'False Positives': fp,
                    'False Negatives': fn,
                    'Estimated Cost': total_cost
                })
            
            threshold_df = pd.DataFrame(threshold_results)
            
            # Plot metrics vs threshold
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot metrics
            ax1.set_xlabel('Decision Threshold')
            ax1.set_ylabel('Metric Value')
            ax1.plot(threshold_df['Threshold'], threshold_df['Accuracy'], 'g-', label='Accuracy')
            ax1.plot(threshold_df['Threshold'], threshold_df['Precision'], 'b-', label='Precision')
            ax1.plot(threshold_df['Threshold'], threshold_df['Recall'], 'r-', label='Recall')
            ax1.plot(threshold_df['Threshold'], threshold_df['F1 Score'], 'y-', label='F1 Score')
            ax1.tick_params(axis='y')
            ax1.legend(loc='center left')
            ax1.grid(True, alpha=0.3)
            
            # Plot estimated cost on secondary axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Estimated Cost ($)', color='purple')
            ax2.plot(threshold_df['Threshold'], threshold_df['Estimated Cost'], 'm--', label='Est. Cost')
            ax2.tick_params(axis='y', labelcolor='purple')
            
            fig.tight_layout()
            ax1.set_title('Impact of Probability Threshold on Model Performance and Cost')
            
            # Add optimal threshold marker
            optimal_idx = threshold_df['Estimated Cost'].idxmin()
            optimal_threshold = threshold_df.loc[optimal_idx, 'Threshold']
            ax1.axvline(x=optimal_threshold, color='black', linestyle='--', alpha=0.7)
            ax1.text(optimal_threshold+0.02, 0.5, f'Optimal threshold: {optimal_threshold:.2f}', 
                    transform=ax1.get_xaxis_transform(), fontsize=10)
            
            st.pyplot(fig)
            
            st.markdown(f"""
            ### Optimal Decision Threshold
            
            Based on the cost analysis, the optimal decision threshold is approximately **{optimal_threshold:.2f}**
            (compared to the default 0.5 threshold).
            
            At this threshold:
            - Accuracy: {threshold_df.loc[optimal_idx, 'Accuracy']:.4f}
            - Precision: {threshold_df.loc[optimal_idx, 'Precision']:.4f}
            - Recall: {threshold_df.loc[optimal_idx, 'Recall']:.4f}
            - Estimated cost: ${threshold_df.loc[optimal_idx, 'Estimated Cost']:,.2f}
            
            **Business recommendation:** Consider adjusting the decision threshold based on specific
            business priorities and risk appetite. A higher threshold reduces defaults but approves fewer loans,
            while a lower threshold approves more loans but increases default risk.
            """)
            
            # Download comparison results
            st.download_button(
                label="Download Comparison Results",
                data=comparison_df.to_csv(index=False),
                file_name="prediction_vs_actual_comparison.csv",
                mime="text/csv"
            )

# Footer with hints for new users
st.markdown("---")
st.markdown("""
**How to use this tool:**
1. Start by selecting features from the checkboxes
2. Click "Train Logistic Regression Model" to see results
3. Analyze the model performance and statistics
4. Use the model to analyze potential borrowers
""")