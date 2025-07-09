import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and evaluate them"""
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Store trained model
            self.trained_models[model_name] = model
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def save_models(self, models_dir='models'):
        """Save trained models"""
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(models_dir, f'{model_name}.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")