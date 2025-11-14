import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureEngineer:
    def __init__(self):
        self.selected_features = None
        
        
    def calculate_feature_importance(self, X, y):
        """Calculate feature importance using multiple methods"""
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance_rf': rf.feature_importances_
        })
        
        
        # ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        anova_scores = pd.DataFrame({
            'feature': X.columns,
            'importance_anova': selector.scores_
        })
        
        
        # Combine importance scores
        importance_df = pd.merge(rf_importance, anova_scores, on='feature')
        importance_df['combined_importance'] = (
            importance_df['importance_rf'] + 
            importance_df['importance_anova'] / importance_df['importance_anova'].max()
        )
        
        return importance_df.sort_values('combined_importance', ascending=False)
    
    
    def select_features(self, X, y, n_features=20):
        """Select top features based on importance"""
        importance_df = self.calculate_feature_importance(X, y)
        self.selected_features = importance_df.head(n_features)['feature'].tolist()
        return self.selected_features
    
    def plot_feature_importance(self, importance_df, top_n=15):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=top_features, y='feature', x='importance_rf')
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Importance')
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=top_features, y='feature', x='importance_anova')
        plt.title('ANOVA F-value Importance')
        plt.xlabel('F-value')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_time_based_features(self, df, time_window=60):
        """Create time-based features for temporal analysis"""
        # This would typically be used with timestamped data
        time_features = df.copy()
        
        # Rolling statistics (simplified)
        for col in ['src_bytes', 'dst_bytes', 'duration']:
            time_features[f'{col}_rolling_mean'] = df[col].rolling(window=5, min_periods=1).mean()
            time_features[f'{col}_rolling_std'] = df[col].rolling(window=5, min_periods=1).std()
        
        return time_features