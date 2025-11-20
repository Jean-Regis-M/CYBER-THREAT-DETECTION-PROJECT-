import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from config.config import DATA_DIR, MODEL_CONFIG

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load the dataset"""
        return pd.read_csv(file_path)
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
       
         # Fill numerical columns with median
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Fill categorical columns with mode
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')
            
        return df
    
    def engineer_features(self, df):
        """Create additional features for better detection"""
        # Ratio features
        df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
        df['packet_size_std'] = df[['src_bytes', 'dst_bytes']].std(axis=1)
        
        # Behavioral features
        
        df['high_error_rate'] = (df['serror_rate'] > 0.7) | (df['rerror_rate'] > 0.7)
        df['suspicious_service'] = df['service'].isin([0, 1, 2])  # Common attack services
        
        return df
    
    def preprocess_data(self, df, fit_scaler=True):
        """Preprocess the entire dataset"""
        
        # Handle missing values
        
        df = self.handle_missing_values(df)
        
        # Engineer new features
        df = self.engineer_features(df)
        
        # Separate features and target
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if fit_scaler:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        return train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'], 
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
    
    def save_preprocessor(self, path):
        """Save the fitted preprocessor"""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, path)
    
    def load_preprocessor(self, path):
        """Load a fitted preprocessor"""
        preprocessor = joblib.load(path)
        self.scaler = preprocessor['scaler']
        self.label_encoders = preprocessor['label_encoders']