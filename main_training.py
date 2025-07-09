from src.data_preprocessing import DataPreprocessor, create_features
from src.model_training import ModelTrainer
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def main():
    # Load data
    train_df = pd.read_csv('data/train.csv')
    
    print("Dataset loaded successfully!")
    print(f"Shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean and preprocess data
    train_clean = preprocessor.clean_data(train_df)
    train_features = create_features(train_clean)
    
    # Encode features
    train_encoded = preprocessor.encode_features(train_features, target_col='Loan_Status')
    
    # Prepare features and target
    X = train_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = train_encoded['Loan_Status']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Save models and preprocessor
    trainer.save_models()
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    
    # Print results
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Cross-validation: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")

if __name__ == "__main__":
    main()