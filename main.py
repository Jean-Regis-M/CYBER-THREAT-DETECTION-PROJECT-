import pandas as pd
import numpy as np
import os
import sys
from data.download_data import DataDownloader
from data.preprocessing import DataPreprocessor
from features.feature_engineering import FeatureEngineer
from models.train_model import ThreatDetectionModel
from models.model_evaluation import ModelEvaluator
from config.config import DATA_DIR, MODELS_DIR

def main():
    
    print("=== Cyber Threat Detection with Machine Learning ===\n")
    
    # Step 1: Download and prepare data
    print("Step 1: Downloading and preparing data...")
    downloader = DataDownloader()
    
    downloader.create_sample_dataset()
    
    preprocessor = DataPreprocessor()
    data_path = os.path.join(DATA_DIR, 'raw', 'network_traffic.csv')
    df = preprocessor.load_data(data_path)
    
    # Step 2: Preprocess data
    print("Step 2: Preprocessing data...")
    X, y = preprocessor.preprocess_data(df)
    
    # Step 3: Feature engineering
    
    print("Step 3: Feature engineering...")
    feature_engineer = FeatureEngineer()
    selected_features = feature_engineer.select_features(X, y, n_features=15)
    X_selected = X[selected_features]
    
    importance_df = feature_engineer.calculate_feature_importance(X, y)
    print("Top 10 features:")
    print(importance_df.head(10))
    
    # Step 4: Split data
    
    print("Step 4: Splitting data...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_selected, y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Malicious ratio: {y_train.mean():.2%} (train), {y_test.mean():.2%} (test)")
    
    # Step 5: Train models
    print("Step 5: Training machine learning models...")
    model_trainer = ThreatDetectionModel()
    results = model_trainer.train_models(X_train, y_train)
    
    # Step 6: Hyperparameter tuning
    print("Step 6: Hyperparameter tuning...")
    model_trainer.hyperparameter_tuning(X_train, y_train)
    
    # Step 7: Evaluate the best model
    print("Step 7: Evaluating the best model...")
    best_model_name = model_trainer.best_model_name
    print(f"Best model: {best_model_name} with score: {model_trainer.best_score:.4f}")
    
    evaluator = ModelEvaluator(model_trainer.best_model, best_model_name)
    evaluator.comprehensive_evaluation(X_test, y_test)
    
    # Step 8: Save the model
    print("Step 8: Saving the model...")
    model_path = model_trainer.save_model()
    
    # Step 9: Real-time detection simulation
    print("Step 9: Simulating real-time threat detection...")
    simulate_real_time_detection(model_trainer.best_model, preprocessor, selected_features)
    
    print("\n=== Project Complete ===")
    print("The cyber threat detection system is ready!")
    print(f"Best model: {best_model_name}")
    print(f"Model saved at: {model_path}")

def simulate_real_time_detection(model, preprocessor, feature_names):
    """Simulate real-time threat detection on new data"""
    print("\nSimulating real-time threat detection...")
    
    # Generate new synthetic network traffic
    np.random.seed(123)
    n_new_samples = 50
    
    new_data = {
        'duration': np.random.exponential(1, n_new_samples),
        'protocol_type': np.random.choice([0, 1, 2], n_new_samples),
        'service': np.random.randint(0, 10, n_new_samples),
        'flag': np.random.randint(0, 5, n_new_samples),
        'src_bytes': np.random.lognormal(5, 2, n_new_samples),
        'dst_bytes': np.random.lognormal(6, 2, n_new_samples),
        'same_srv_rate': np.random.uniform(0, 1, n_new_samples),
        'diff_srv_rate': np.random.uniform(0, 1, n_new_samples),
        'srv_diff_host_rate': np.random.uniform(0, 1, n_new_samples),
        'count': np.random.poisson(10, n_new_samples),
        'serror_rate': np.random.uniform(0, 1, n_new_samples),
        'rerror_rate': np.random.uniform(0, 1, n_new_samples),
        'hot': np.random.poisson(1, n_new_samples),
        'num_failed_logins': np.random.poisson(0.1, n_new_samples),
        'logged_in': np.random.choice([0, 1], n_new_samples),
        'num_compromised': np.random.poisson(0.05, n_new_samples),
    }
    
    new_df = pd.DataFrame(new_data)
    new_df = preprocessor.engineer_features(new_df)
    
    # Select features and preprocess
    X_new = new_df[feature_names]
    numerical_cols = X_new.select_dtypes(include=[np.number]).columns
    X_new[numerical_cols] = preprocessor.scaler.transform(X_new[numerical_cols])
    
    # Make predictions
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Display results
    threat_count = predictions.sum()
    print(f"Detected {threat_count} potential threats out of {n_new_samples} connections")
    
    if threat_count > 0:
        print("Alert! Potential cyber threats detected:")
        threat_indices = np.where(predictions == 1)[0]
        for idx in threat_indices[:5]:  # Show first 5 threats
            confidence = probabilities[idx] if probabilities is not None else "N/A"
            print(f"  Connection {idx}: Malicious (Confidence: {confidence:.2f})")
    
    return predictions, probabilities

if __name__ == "__main__":
    main()