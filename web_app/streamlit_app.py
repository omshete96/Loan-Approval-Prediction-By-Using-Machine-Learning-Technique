import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import create_features

# Load models and preprocessor
@st.cache_resource
def load_models():
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the project root directory (parent of web_app)
        project_root = os.path.dirname(script_dir)
        
        # Construct absolute paths to model files
        models_dir = os.path.join(project_root, 'models')
        logistic_path = os.path.join(models_dir, 'logistic_regression.pkl')
        rf_path = os.path.join(models_dir, 'random_forest.pkl')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Project root: {project_root}")
        st.write(f"Models directory: {models_dir}")
        st.write(f"logistic_regression.pkl exists: {os.path.exists(logistic_path)}")
        st.write(f"random_forest.pkl exists: {os.path.exists(rf_path)}")
        st.write(f"preprocessor.pkl exists: {os.path.exists(preprocessor_path)}")
        
        models = {
            'Logistic Regression': joblib.load(logistic_path),
            'Random Forest': joblib.load(rf_path)
        }
        preprocessor = joblib.load(preprocessor_path)
        return models, preprocessor
    except Exception as e:
        st.error(f"Models not found! Please run main_training.py first. Error: {e}")
        return None, None

def main():
    st.title("🏦 Loan Approval Prediction System")
    st.write("Enter the details below to predict loan approval")
    
    # Load models
    models, preprocessor = load_models()
    
    if models is None:
        st.stop()
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
    with col2:
        st.subheader("Financial Information")
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=100000)
        loan_term = st.selectbox("Loan Amount Term (months)", [360, 180, 240, 300, 480])
        credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good" if x == 1 else "Poor")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    if st.button("🔍 Predict Loan Approval", use_container_width=True):
        # Create input DataFrame
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
        df = pd.DataFrame([input_data])
        
        try:
            # Preprocess data
            df_clean = preprocessor.clean_data(df)
            df_features = create_features(df_clean)
            df_encoded = preprocessor.encode_features(df_features)
            
            # Make predictions
            st.subheader("🎯 Prediction Results")
            
            for model_name, model in models.items():
                pred = model.predict(df_encoded)[0]
                pred_proba = model.predict_proba(df_encoded)[0]
                
                result = "✅ Approved" if pred == 1 else "❌ Rejected"
                probability = pred_proba[1] * 100
                
                # Create columns for better layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.metric(f"{model_name}", result)
                
                with col2:
                    st.metric("Approval Probability", f"{probability:.1f}%")
                
                # Progress bar
                st.progress(probability / 100)
                
                # Color coding based on probability
                if probability > 70:
                    st.success(f"High chance of approval with {model_name}")
                elif probability > 50:
                    st.warning(f"Moderate chance of approval with {model_name}")
                else:
                    st.error(f"Low chance of approval with {model_name}")
                
                st.write("---")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please make sure you've trained the models first by running main_training.py")

if __name__ == "__main__":
    main()
