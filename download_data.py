import pandas as pd
import numpy as np
import os
from urllib.request import urlretrieve
import zipfile
from config.config import DATA_DIR, DATASET_CONFIG

class DataDownloader:
    def __init__(self):
        self.raw_path = DATASET_CONFIG['local_path']
        os.makedirs(self.raw_path, exist_ok=True)
    
    def download_sample_data(self):
        """Download sample dataset or use built-in data for demonstration"""
        print("Downloading sample network traffic data...")
        
        # For demonstration, we'll create synthetic data
        # In practice, you would download from CICIDS or KDD
        self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create a synthetic dataset for demonstration"""
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            # Basic connection features
            
            'duration': np.random.exponential(1, n_samples),
            'protocol_type': np.random.choice([0, 1, 2], n_samples),  # 0:TCP, 1:UDP, 2:ICMP
            'service': np.random.randint(0, 10, n_samples),
            'flag': np.random.randint(0, 5, n_samples),
            'src_bytes': np.random.lognormal(5, 2, n_samples),
            'dst_bytes': np.random.lognormal(6, 2, n_samples),
            
            # Time-based features
            
            'same_srv_rate': np.random.uniform(0, 1, n_samples),
            'diff_srv_rate': np.random.uniform(0, 1, n_samples),
            'srv_diff_host_rate': np.random.uniform(0, 1, n_samples),
            
            # Connection features
            
            'count': np.random.poisson(10, n_samples),
            'serror_rate': np.random.uniform(0, 1, n_samples),
            'rerror_rate': np.random.uniform(0, 1, n_samples),
            'same_srv_rate': np.random.uniform(0, 1, n_samples),
            
            # Content features
            
            'hot': np.random.poisson(1, n_samples),
            'num_failed_logins': np.random.poisson(0.1, n_samples),
            'logged_in': np.random.choice([0, 1], n_samples),
            'num_compromised': np.random.poisson(0.05, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create labels (0: normal, 1: malicious)
        # Malicious traffic has different patterns
        malicious_mask = (
            (df['src_bytes'] > 10000) |
            (df['duration'] > 100) |
            (df['num_failed_logins'] > 3) |
            (df['serror_rate'] > 0.8)
        )
        
        df['label'] = malicious_mask.astype(int)
        
        # Add some noise
        
        flip_mask = np.random.random(n_samples) < 0.05
        df.loc[flip_mask, 'label'] = 1 - df.loc[flip_mask, 'label']
        
        # Save the dataset
        output_path = os.path.join(self.raw_path, 'network_traffic.csv')
        df.to_csv(output_path, index=False)
        print(f"Sample dataset created with {len(df)} samples")
        print(f"Malicious samples: {df['label'].sum()} ({df['label'].mean():.2%})")
        
        return df