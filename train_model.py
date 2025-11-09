import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
from config.config import MODELS_DIR

class ThreatDetectionModel:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42),
            'neural_network': MLPClassifier(random_state=42, max_iter=1000),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        
        self.best_model = None
        self.best_score = 0
        
    def train_models(self, X_train, y_train, cv=5):
        """Train multiple models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            results[name] = {
                'model': model,
                'cv_mean': mean_score,
                'cv_std': std_score,
                'cv_scores': cv_scores
            }
            
            print(f"{name}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
            
            # Train the model on full training data
            model.fit(X_train, y_train)
            
            # Update best model
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = model
                self.best_model_name = name
        
        self.results = results
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for the best model"""
        print("Performing hyperparameter tuning...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 4, 5]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        best_model_name = self.best_model_name
        if best_model_name in param_grids:
            grid_search = GridSearchCV(
                self.models[best_model_name],
                param_grids[best_model_name],
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.best_model = grid_search.best_estimator_
            self.best_score = grid_search.best_score_
            
            print(f"Best parameters for {best_model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    def save_model(self, model_name=None):
        """Save the trained model"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        if model_name is None:
            model_name = self.best_model_name
        
        model_path = os.path.join(MODELS_DIR, f'{model_name}_threat_detector.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save all results
        results_path = os.path.join(MODELS_DIR, 'training_results.pkl')
        joblib.dump(self.results, results_path)
        
        return model_path
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.best_model = joblib.load(model_path)
        return self.best_model