import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import learning_curve

class ModelEvaluator:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
    
    def comprehensive_evaluation(self, X_test, y_test):
        """Perform comprehensive model evaluation"""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # ROC Curve
        if y_pred_proba is not None:
            self.plot_roc_curve(y_test, y_pred_proba)
            self.plot_precision_recall_curve(y_test, y_pred_proba)
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance(X_test.columns)
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Malicious'],
                   yticklabels=['Normal', 'Malicious'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'roc_curve_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        average_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(f'precision_recall_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names, top_n=15):
        """Plot feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Feature Importance - {self.model_name}')
            plt.bar(range(min(top_n, len(importances))), 
                   importances[indices[:top_n]])
            plt.xticks(range(min(top_n, len(importances))), 
                      [feature_names[i] for i in indices[:top_n]], rotation=45)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{self.model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_learning_curve(self, X_train, y_train):
        """Plot learning curve"""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1'
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Learning Curve - {self.model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.legend(loc="best")
        plt.savefig(f'learning_curve_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()