import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path to import create_features
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.data_preprocessing import create_features
except ImportError:
    st.error("❌ Could not import 'create_features' from src/data_preprocessing.py")
    st.stop()

# Load models and preprocessor from current folder
@st.cache_resource
def load_models():
    try:
        logistic_model = joblib.load("logistic_regression.pkl")
        rf_model = joblib.load("random_forest.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        
        models = {
            "Logistic Regression": logistic_model,
            "Random Forest": rf_model
        }
        return models, preprocessor
    except Exception as e:
        st.error(f"❌ Models failed to load: {e}")
        return None, None

def main():
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("Enter your details below to check your loan approval chances.")

    models, preprocessor = load_models()

    if models is None or preprocessor is None:
        st.stop()

    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col2:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=100000)
        loan_term = st.selectbox("Loan Amount Term (months)", [360, 180, 240, 300, 480])
        credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good" if x == 1 else "Poor")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("🔍 Predict Loan Approval", use_container_width=True):
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
            df_clean = preprocessor.clean_data(df)
            df_features = create_features(df_clean)
            df_encoded = preprocessor.encode_features(df_features)

            st.subheader("🎯 Prediction Results")

            for model_name, model in models.items():
                pred = model.predict(df_encoded)[0]
                pred_proba = model.predict_proba(df_encoded)[0][1]

                result = "✅ Approved" if pred == 1 else "❌ Rejected"
                prob_text = f"{pred_proba * 100:.1f}%"

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.metric(model_name, result)
                with col_b:
                    st.metric("Approval Probability", prob_text)

                st.progress(pred_proba)

                if pred_proba > 0.7:
                    st.success(f"High chance of approval with {model_name}")
                elif pred_proba > 0.5:
                    st.warning(f"Moderate chance of approval with {model_name}")
                else:
                    st.error(f"Low chance of approval with {model_name}")

                st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.info("Please ensure models and preprocessing pipeline are trained and compatible.")

if __name__ == "__main__":
    main()
