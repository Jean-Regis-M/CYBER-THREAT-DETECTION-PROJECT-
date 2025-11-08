import os

# Dataset configuration
DATASET_CONFIG = {
    'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
    'local_path': 'data/raw/',
    'processed_path': 'data/processed/'
}

# Feature configuration
FEATURES = {
    'basic': ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes'],
    'time_based': ['same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate'],
    'connection': ['count', 'serror_rate', 'rerror_rate', 'same_srv_rate'],
    'content': ['hot', 'num_failed_logins', 'logged_in', 'num_compromised']
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')