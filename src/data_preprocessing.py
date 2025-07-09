import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        """Clean and preprocess the dataset"""
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        # Fill numerical columns with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def encode_features(self, df, target_col=None):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        # Get categorical columns (excluding target if specified)
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        if target_col:
            categorical_cols = [col for col in categorical_cols if col != target_col]
        
        # Label encode categorical features
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        # Encode target variable if specified
        if target_col and target_col in df_encoded.columns:
            if target_col not in self.label_encoders:
                self.label_encoders[target_col] = LabelEncoder()
                df_encoded[target_col] = self.label_encoders[target_col].fit_transform(df_encoded[target_col])
        
        return df_encoded

def create_features(df):
    """Create new features from existing ones"""
    df_features = df.copy()
    
    # Create total income feature
    if 'ApplicantIncome' in df_features.columns and 'CoapplicantIncome' in df_features.columns:
        df_features['TotalIncome'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
    
    # Create income to loan ratio
    if 'TotalIncome' in df_features.columns and 'LoanAmount' in df_features.columns:
        df_features['IncomeToLoanRatio'] = df_features['TotalIncome'] / (df_features['LoanAmount'] + 1)
    
    # Create loan amount per term
    if 'LoanAmount' in df_features.columns and 'Loan_Amount_Term' in df_features.columns:
        df_features['LoanAmountPerTerm'] = df_features['LoanAmount'] / (df_features['Loan_Amount_Term'] + 1)
    
    return df_features