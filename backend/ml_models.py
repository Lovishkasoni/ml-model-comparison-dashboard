import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)
import joblib
import json

class MLModelComparison:
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models based on problem type - OPTIMIZED FOR SPEED"""
        if self.problem_type == 'classification':
            self.models = {
                'Logistic Regression': LogisticRegression(max_iter=100, random_state=42, n_jobs=-1),
                'Random Forest': RandomForestClassifier(n_estimators=5, max_depth=10, random_state=42, n_jobs=-1),
                'SVM': SGDClassifier(max_iter=1000),
                'XGBoost': GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
                'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            }
        else:  # regression
            self.models = {
                'Linear Regression': LinearRegression(n_jobs=-1),
                'Random Forest': RandomForestRegressor(n_estimators=5, max_depth=10, random_state=42, n_jobs=-1),
                'SVM': SVR(kernel='rbf'),
                'XGBoost': GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
                'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
            }
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate"""
        self.results = {}
        self.feature_importance = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            if self.problem_type == 'classification':
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                }
                # ROC-AUC only for binary classification
                try:
                    if len(np.unique(y_test)) == 2:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
                except:
                    pass
            else:  # regression
                metrics = {
                    'MSE': mean_squared_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R² Score': r2_score(y_test, y_pred),
                }
            
            self.results[model_name] = metrics
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                try:
                    self.feature_importance[model_name] = np.abs(model.coef_[0]).tolist()
                except:
                    self.feature_importance[model_name] = np.abs(model.coef_).flatten().tolist()
        
        return self.results
    
    def get_results_sorted(self):
        """Get results sorted by best metric"""
        if self.problem_type == 'classification':
            metric_key = 'Accuracy'
        else:
            metric_key = 'R² Score'
        
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get(metric_key, 0),
            reverse=True
        )
        
        return sorted_results
    
    def get_best_model_name(self):
        """Get best performing model"""
        sorted_results = self.get_results_sorted()
        return sorted_results[0][0] if sorted_results else None
    
    def get_feature_importance(self, model_name):
        """Get feature importance for a specific model"""
        return self.feature_importance.get(model_name, [])
    
    def tune_hyperparameters(self, model_name, params, X_train, X_test, y_train, y_test):
        """Retrain model with new hyperparameters"""
        try:
            model = self.models[model_name]
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if self.problem_type == 'classification':
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                }
            else:
                metrics = {
                    'MSE': mean_squared_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R² Score': r2_score(y_test, y_pred),
                }
            
            return metrics, None
        except Exception as e:
            return None, str(e)
    
    def save_model(self, model_name, filepath):
        """Save trained model"""
        joblib.dump(self.models[model_name], filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        return joblib.load(filepath)