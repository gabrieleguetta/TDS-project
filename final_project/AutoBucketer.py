import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin


class SimpleAutoBucketer(BaseEstimator, TransformerMixin):
    """
    AutoBucketer that creates KMeans clusters for features.
    Handles all data types by using KMeans on encoded values.
    
    Parameters:
    -----------
    n_buckets : int, default=5
        Maximum number of buckets to create for each feature
    random_state : int, default=42
        Random seed for reproducibility
    passthrough : bool, default=False
        If True, skip KMeans bucketing and use the encoded values directly
    """
    
    def __init__(self, n_buckets=5, random_state=42, passthrough=False):
        self.n_buckets = n_buckets
        self.random_state = random_state
        self.passthrough = passthrough
        self.encoders = {}
        self.models = {}
        self.feature_names = None
    
    def _encode_values(self, values, encoder=None):
        """Helper method to encode feature values"""
        if encoder is None:
            # For numeric features
            try:
                encoded_values = values.astype(float)
                return encoded_values.reshape(-1, 1), None
            except (ValueError, TypeError):
                # If conversion fails, treat as categorical
                pass
                
        # For categorical features
        str_values = np.array([str(v) for v in values])
        
        if encoder is None:
            # Create a new encoder
            unique_values, encoded_values = np.unique(str_values, return_inverse=True)
            encoder = dict(zip(unique_values, range(len(unique_values))))
        else:
            # Use existing encoder
            encoded_values = np.array([encoder.get(v, -1) for v in str_values])
            # Replace -1 with a valid value to avoid errors
            if -1 in encoded_values:
                encoded_values[encoded_values == -1] = 0
                
        return encoded_values.reshape(-1, 1), encoder
    
    def fit(self, X, y=None, X_val=None, y_val=None):
        """
        Fit the bucketing model on the data
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Training data
        y : array-like, default=None
            Target values (not used)
        X_val : array-like or DataFrame, default=None
            Validation data for scoring models. If None, X will be used.
        y_val : array-like, default=None
            Validation targets (not used)
        """
        # Use X for validation if X_val is not provided
        if X_val is None:
            X_val = X
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Process each feature
        for j, col_name in enumerate(self.feature_names):
            # Extract the feature values
            if hasattr(X, 'iloc'):
                train_values = X.iloc[:, j].values
                val_values = X_val.iloc[:, j].values
            else:
                train_values = X[:, j]
                val_values = X_val[:, j]
            
            # Encode training values
            train_encoded, encoder = self._encode_values(train_values)
            self.encoders[col_name] = encoder
            
            # Encode validation values
            val_encoded, _ = self._encode_values(val_values, encoder)
            
            # If passthrough is True, mark this feature for passthrough
            if self.passthrough:
                self.models[col_name] = "passthrough"
                continue
            
            # Try different numbers of buckets and score on validation set
            scores = []
            fitted_models = []
            
            for nb in range(2, self.n_buckets):
                n_buckets = min(nb, len(np.unique(train_encoded)))
                
                if n_buckets < 2:
                    # If only one unique value, no need for bucketing
                    continue
                
                # Fit KMeans on training data
                kmeans = KMeans(
                    n_clusters=n_buckets,
                    random_state=self.random_state,
                    n_init=10
                )
                kmeans.fit(train_encoded)
                
                # Score on validation data
                val_score = kmeans.score(val_encoded)
                scores.append(val_score)
                fitted_models.append(kmeans)
            
            if not scores:
                # If no valid models were fitted
                self.models[col_name] = None
                continue
            
            # Select the best number of buckets based on validation score
            best_idx = np.argmax(scores)
            best_nb = list(range(2, self.n_buckets))[best_idx]
            
            # Combine train and validation data for final fit
            if hasattr(X, 'iloc'):
                combined_values = np.concatenate([train_values, val_values])
            else:
                combined_values = np.concatenate([train_values, val_values])
            
            combined_encoded, _ = self._encode_values(combined_values, encoder)
            
            # Fit final model on combined data
            n_buckets = min(best_nb, len(np.unique(combined_encoded)))
            
            if n_buckets < 2:
                # If only one unique value, no need for bucketing
                self.models[col_name] = None
            else:
                # Fit KMeans on combined data
                kmeans = KMeans(
                    n_clusters=n_buckets,
                    random_state=self.random_state,
                    n_init=10
                )
                kmeans.fit(combined_encoded)
                self.models[col_name] = kmeans
        
        return self
    
    def transform(self, X):
        """Transform the data into bucket assignments"""
        # Create output array
        bucket_array = np.zeros((X.shape[0], len(self.feature_names)))
        
        # Process each feature
        for j, col_name in enumerate(self.feature_names):
            # Extract the feature values
            if hasattr(X, 'iloc'):
                values = X.iloc[:, j].values
            else:
                values = X[:, j]
            
            # Get the encoder and model
            encoder = self.encoders[col_name]
            model = self.models[col_name]
            
            # Encode the values
            encoded_values, _ = self._encode_values(values, encoder)
            
            # Handle different model cases
            if model == "passthrough":
                # Use encoded values directly
                bucket_array[:, j] = encoded_values.flatten()
                continue
            elif model is None:
                # If no model, just use zeros
                bucket_array[:, j] = 0
                continue
            
            # Predict bucket assignments
            try:
                bucket_array[:, j] = model.predict(encoded_values)
            except:
                # If prediction fails, use zeros
                bucket_array[:, j] = 0
        
        return bucket_array
    
    def fit_transform(self, X, y=None, X_val=None, y_val=None):
        """
        Fit the model and transform the data in one step
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Training data
        y : array-like, default=None
            Target values (not used)
        X_val : array-like or DataFrame, default=None
            Validation data for scoring models. If None, X will be used.
        y_val : array-like, default=None
            Validation targets (not used)
        """
        self.fit(X, y, X_val, y_val)
        return self.transform(X)