import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-positive {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .prediction-negative {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = load('pipeline_final.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'pipeline_final.joblib' exists.")
        return None

# Load dataset for reference
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('alzheimers_disease_data.csv')
        return df
    except FileNotFoundError:
        return None

# Title and Description
st.title("🧠 Alzheimer's Disease Prediction System")
st.markdown("---")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "About", "Model Info"])

if page == "Home":
    st.header("Welcome to Alzheimer's Disease Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## About Alzheimer's Disease
        Alzheimer's disease is a progressive neurodegenerative disorder that primarily affects older adults. 
        Key characteristics include:
        
        - **Memory Loss**: Difficulty remembering recent events or conversations
        - **Cognitive Decline**: Problems with thinking, judgment, and learning
        - **Behavioral Changes**: Changes in personality and social withdrawal
        - **Physical Changes**: Loss of coordination and motor control
        
        Early detection is crucial for effective intervention and management.
        """)
    
    with col2:
        st.info("""
        ### Why Early Detection Matters?
        - Enables timely medical intervention
        - Allows for lifestyle modifications
        - Provides family planning opportunities
        - Helps manage symptoms effectively
        - Improves quality of life
        """)
    
    st.markdown("---")
    
    # Statistics
    df = load_dataset()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(df))
        
        with col2:
            alzheimer_count = len(df[df['Diagnosis'] == 1])
            st.metric("Alzheimer's Cases", alzheimer_count)
        
        with col3:
            healthy_count = len(df[df['Diagnosis'] == 0])
            st.metric("Healthy Cases", healthy_count)
        
        with col4:
            percentage = (alzheimer_count / len(df)) * 100
            st.metric("Alzheimer's %", f"{percentage:.1f}%")

elif page == "Prediction":
    st.header("Patient Assessment & Prediction")
    
    model = load_model()
    df = load_dataset()
    
    if model is None:
        st.error("Cannot load prediction model.")
    else:
        # Get feature information from dataset
        feature_cols = df.drop(['PatientID', 'DoctorInCharge', 'Diagnosis'], axis=1).columns.tolist()
        
        st.markdown("### Enter Patient Information")
        
        # Create input columns
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        # Create input fields based on the actual feature columns
        with col1:
            for feature in feature_cols[:len(feature_cols)//3]:
                feature_min = df[feature].min()
                feature_max = df[feature].max()
                feature_mean = df[feature].mean()
                
                input_data[feature] = st.slider(
                    f"{feature}",
                    min_value=float(feature_min),
                    max_value=float(feature_max),
                    value=float(feature_mean),
                    step=0.1
                )
        
        with col2:
            for feature in feature_cols[len(feature_cols)//3:2*len(feature_cols)//3]:
                feature_min = df[feature].min()
                feature_max = df[feature].max()
                feature_mean = df[feature].mean()
                
                input_data[feature] = st.slider(
                    f"{feature}",
                    min_value=float(feature_min),
                    max_value=float(feature_max),
                    value=float(feature_mean),
                    step=0.1
                )
        
        with col3:
            for feature in feature_cols[2*len(feature_cols)//3:]:
                feature_min = df[feature].min()
                feature_max = df[feature].max()
                feature_mean = df[feature].mean()
                
                input_data[feature] = st.slider(
                    f"{feature}",
                    min_value=float(feature_min),
                    max_value=float(feature_max),
                    value=float(feature_mean),
                    step=0.1
                )
        
        st.markdown("---")
        
        # Prediction Button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔍 Make Prediction", use_container_width=True):
                # Prepare data in correct order
                X_input = np.array([input_data[col] for col in feature_cols]).reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(X_input)[0]
                probability = model.predict_proba(X_input)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("### Prediction Results")
                
                if prediction == 1:
                    st.markdown(
                        """<div class="prediction-positive">
                        <h3>⚠️ POSITIVE - Likely Alzheimer's Disease</h3>
                        <p>The model indicates signs of Alzheimer's disease.</p>
                        <p><strong>Confidence:</strong> {:.1f}%</p>
                        </div>""".format(probability[1] * 100),
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """<div class="prediction-negative">
                        <h3>✅ NEGATIVE - No Alzheimer's Disease</h3>
                        <p>The model indicates no signs of Alzheimer's disease.</p>
                        <p><strong>Confidence:</strong> {:.1f}%</p>
                        </div>""".format(probability[0] * 100),
                        unsafe_allow_html=True
                    )
                
                # Detailed Probability Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Healthy Probability", f"{probability[0]*100:.2f}%")
                
                with col2:
                    st.metric("Alzheimer's Probability", f"{probability[1]*100:.2f}%")
                
                # Feature Values Summary
                st.markdown("### Input Features Summary")
                input_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Value': [input_data[col] for col in feature_cols]
                })
                st.dataframe(input_df, use_container_width=True)

elif page == "About":
    st.header("About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Project Overview
        This project develops a machine learning-based prediction system for 
        Alzheimer's disease using clinical and cognitive assessment data.
        
        ### Objectives:
        1. Build predictive models for early Alzheimer's detection
        2. Compare multiple machine learning algorithms
        3. Optimize model performance
        4. Create an easy-to-use prediction interface
        
        ### Data Source:
        - **Dataset**: Alzheimer's Disease Patients Dataset
        - **Samples**: 700+ patient records
        - **Features**: 30+ clinical and cognitive indicators
        """)
    
    with col2:
        st.markdown("""
        ## Key Features Used:
        - **MMSE Score**: Mini-Mental State Examination
        - **Age**: Patient age
        - **Gender**: Patient gender
        - **Cognitive Tests**: Various cognitive assessment measures
        - **Biomarkers**: Blood and CSF biomarkers
        - **MRI Measurements**: Brain imaging metrics
        - **Health Status**: Medical history and conditions
        
        ### Methodology:
        - Data preprocessing and normalization
        - Feature selection and engineering
        - Model training with cross-validation
        - Hyperparameter tuning
        - Model evaluation and comparison
        """)
    
    st.markdown("---")
    st.info("""
    **Disclaimer**: This model is for educational purposes only and should not be used 
    for clinical diagnosis. Always consult with healthcare professionals for medical decisions.
    """)

elif page == "Model Info":
    st.header("Model Information & Performance")
    
    df = load_dataset()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Model Architecture
        
        ### Pipeline Components:
        1. **StandardScaler**: Feature normalization
        2. **GradientBoostingClassifier**: Final prediction model
        
        ### Model Specifications:
        - **Algorithm**: Gradient Boosting
        - **Random State**: 42
        - **Framework**: Scikit-learn
        - **Type**: Binary Classification
        """)
    
    with col2:
        st.markdown("""
        ## Model Performance
        
        ### Training Metrics:
        - **Training Accuracy**: 96.68%
        - **Testing Accuracy**: 95.81%
        
        ### Classification Metrics (Test Set):
        - **Precision**: 95-96%
        - **Recall**: 92-98%
        - **F1-Score**: 94-97%
        - **Support**: 430 test samples
        """)
    
    st.markdown("---")
    
    # Feature Statistics
    st.markdown("### Dataset Statistics")
    
    if df is not None:
        feature_cols = df.drop(['PatientID', 'DoctorInCharge', 'Diagnosis'], axis=1).columns.tolist()
        
        # Display statistics
        stats_df = pd.DataFrame({
            'Feature': feature_cols,
            'Min': [df[col].min() for col in feature_cols],
            'Mean': [df[col].mean() for col in feature_cols],
            'Max': [df[col].max() for col in feature_cols],
            'Std': [df[col].std() for col in feature_cols]
        })
        
        st.dataframe(stats_df, use_container_width=True)
    
    # Model Comparison
    st.markdown("---")
    st.markdown("### Models Tested During Development")
    
    models_comparison = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Random Forest',
            'SVM',
            'Gradient Boosting',
            'XGBoost',
            'CatBoost',
            'LightGBM'
        ],
        'Type': [
            'Linear',
            'Ensemble (Tree)',
            'Kernel-based',
            'Ensemble (Boosting)',
            'Ensemble (Boosting)',
            'Ensemble (Boosting)',
            'Ensemble (Boosting)'
        ],
        'Complexity': [
            'Low',
            'High',
            'Very High',
            'High',
            'Very High',
            'Very High',
            'Very High'
        ]
    })
    
    st.dataframe(models_comparison, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>🧠 Alzheimer's Disease Prediction System | Built with Streamlit</p>
    <p>Last Updated: 2026 | Version 1.0</p>
</div>
""", unsafe_allow_html=True)
