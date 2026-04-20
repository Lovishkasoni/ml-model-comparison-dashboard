import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import io

class DataProcessor:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        self.problem_type = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.removed_features = []  # Track removed features
        
    def load_data(self, file_stream):
        """Load CSV file from upload"""
        try:
            df = pd.read_csv(file_stream)
            return df, None
        except Exception as e:
            return None, str(e)
    
    def _remove_non_predictive_features(self, df):
        """Remove ID columns and non-predictive features"""
        # Common ID column names to remove
        id_patterns = ['id', 'customerid', 'rownumber', 'phone', 'email', 'name', 'address','roll_no','order_no','student_id','employee_id','transaction_id','customer_id','user_id','account_id']
        
        columns_to_remove = []
        for col in df.columns:
            col_lower = col.lower().strip()
            # Check if column name matches ID patterns
            if any(pattern in col_lower for pattern in id_patterns):
                columns_to_remove.append(col)
        
        self.removed_features = columns_to_remove
        
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
        
        return df
    
    def preprocess(self, df, target_column):
        """Preprocess data for ML training"""
        try:
            # Remove non-predictive features
            df = self._remove_non_predictive_features(df)
            
            # Remove missing values
            df = df.dropna()
            
            # Separate features and target
            if target_column not in df.columns:
                return None, f"Target column '{target_column}' not found in dataset"
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Detect problem type
            if y.dtype == 'object' or len(y.unique()) < 20:
                self.problem_type = 'classification'
            else:
                self.problem_type = 'regression'
            
            # Encode categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
            
            # Encode target if classification
            if self.problem_type == 'classification' and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                self.label_encoders['target'] = le
            
            self.feature_names = X.columns.tolist()
            self.target_name = target_column
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            return {
                'success': True,
                'problem_type': self.problem_type,
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'features': self.feature_names,
                'removed_features': self.removed_features  # Return list of removed features
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def get_data(self):
        """Return processed data"""
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': self.feature_names,
            'problem_type': self.problem_type
        }