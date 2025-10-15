import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Auto ANN Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    .column-selector {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .encoding-info {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #2196F3;
    }
    .prediction-input {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .feature-input-group {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 8px 0;
    }
    .slider-value {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
        color: #ffeb3b;
    }
</style>
""", unsafe_allow_html=True)

class AutoANN:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.encoding_types = {}  # Track encoding type for each column
        self.problem_type = None
        self.target_column = None
        self.columns_to_drop = []
        self.feature_mappings = {}  # Store original to encoded feature mappings
        self.original_columns_info = {}  # Store original column information
        self.num_classes = None
        self.label_encoder_target = None
        
    def preprocess_data(self, df, target_column, problem_type, columns_to_drop=None):
        """Automatically preprocess the data with smart encoding"""
        self.problem_type = problem_type
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop or []
        
        # Create a copy of the dataframe
        df_processed = df.copy()
        
        # Drop selected columns
        if self.columns_to_drop:
            columns_to_drop_final = [col for col in self.columns_to_drop if col in df_processed.columns and col != target_column]
            df_processed = df_processed.drop(columns=columns_to_drop_final)
            st.info(f"Dropped columns: {', '.join(columns_to_drop_final)}")
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Handle target variable based on problem type - FIXED REGRESSION ISSUE
        if problem_type == "regression":
            # For regression, ensure target is numeric with better error handling
            try:
                # Try to convert to numeric, coercing errors to NaN
                y_numeric = pd.to_numeric(y, errors='coerce')
                
                # Check for any NaN values after conversion
                nan_count = y_numeric.isna().sum()
                if nan_count > 0:
                    st.warning(f"⚠️ {nan_count} non-numeric values in target column '{target_column}' were converted to NaN")
                    
                    # Show the problematic values
                    problematic_values = y[y_numeric.isna()].unique()
                    st.write(f"Problematic values: {list(problematic_values)}")
                    
                    # Remove rows with NaN target values
                    valid_mask = ~y_numeric.isna()
                    X = X[valid_mask]
                    y_numeric = y_numeric[valid_mask]
                    st.info(f"Removed {nan_count} rows with invalid target values")
                
                y = y_numeric
                st.write(f"🎯 **Target ({target_column})**: Regression target (numeric) - Cleaned {len(y)} samples")
                
            except Exception as e:
                st.error(f"Error converting target column to numeric: {str(e)}")
                return None, None, None, None, None
        
        # Store original column information for prediction interface
        self.original_columns_info = {}
        for col in X.columns:
            self.original_columns_info[col] = {
                'dtype': str(X[col].dtype),
                'min': float(X[col].min()) if X[col].dtype in ['int64', 'float64'] else None,
                'max': float(X[col].max()) if X[col].dtype in ['int64', 'float64'] else None,
                'mean': float(X[col].mean()) if X[col].dtype in ['int64', 'float64'] else None,
                'unique_values': X[col].unique().tolist() if X[col].dtype == 'object' else None,
                'unique_count': X[col].nunique() if X[col].dtype == 'object' else None,
                'is_binary_numeric': False,  # New field to track binary numeric columns
                'is_slider_candidate': False,  # New field to track if column should use slider
                'is_dropdown_candidate': False  # NEW: Track if column should use dropdown (1-10 range)
            }
            
            # Check if column is binary numeric (only 0 and 1)
            if X[col].dtype in ['int64', 'float64']:
                unique_values = sorted(X[col].unique())
                if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                    self.original_columns_info[col]['is_binary_numeric'] = True
                    self.original_columns_info[col]['binary_values'] = [0, 1]
                
                # Check if column is a good candidate for slider (values between 1 and 1000)
                if (X[col].min() >= 1 and X[col].max() <= 1000 and 
                    X[col].dtype in ['int64', 'float64'] and 
                    X[col].nunique() > 10):  # More than 10 unique values
                    self.original_columns_info[col]['is_slider_candidate'] = True
                
                # NEW: Check if column is in 1-10 range (perfect for dropdown) - FIXED CONDITION
                if (X[col].min() >= 1 and X[col].max() <= 10 and 
                    X[col].dtype in ['int64', 'float64'] and
                    X[col].nunique() <= 10):  # ADDED: Maximum 10 unique values for dropdown
                    self.original_columns_info[col]['is_dropdown_candidate'] = True
                    self.original_columns_info[col]['dropdown_values'] = sorted(X[col].unique())
        
        # Handle categorical variables with smart encoding
        categorical_columns = X.select_dtypes(include=['object']).columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        # Store original categorical values for prediction interface
        self.original_categories = {}
        
        # Smart encoding for categorical variables - ALWAYS USE ONEHOT FOR MULTI-CLASS
        for col in categorical_columns:
            unique_values = X[col].nunique()
            self.original_categories[col] = X[col].unique().tolist()
            
            # For multi-class classification problems, always use OneHot encoding
            if self.problem_type == "multiclass_classification":
                # Use OneHot Encoding for all categorical features in multi-class classification
                try:
                    # Try with sparse_output first (newer scikit-learn)
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                except TypeError:
                    # Fall back to sparse (older scikit-learn)
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                
                encoded = encoder.fit_transform(X[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)
                self.encoders[col] = encoder
                self.encoding_types[col] = 'onehot'
                self.feature_mappings[col] = {
                    'type': 'onehot',
                    'categories': encoder.categories_[0].tolist(),
                    'feature_names': encoder.get_feature_names_out([col]).tolist()
                }
                st.write(f"🎯 **{col}**: OneHot Encoded for Multi-class Classification ({unique_values} categories)")
                
            elif unique_values == 2:
                # Binary categorical - use Label Encoding
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                self.encoders[col] = encoder
                self.encoding_types[col] = 'label'
                self.feature_mappings[col] = {
                    'type': 'label',
                    'classes': encoder.classes_.tolist(),
                    'mapping': dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                }
                st.write(f"🔤 **{col}**: Label Encoded ({encoder.classes_[0]} → 0, {encoder.classes_[1]} → 1)")
                
            elif 3 <= unique_values <= 10:
                # Multi-categorical with reasonable cardinality - use OneHot Encoding
                try:
                    # Try with sparse_output first (newer scikit-learn)
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                except TypeError:
                    # Fall back to sparse (older scikit-learn)
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                
                encoded = encoder.fit_transform(X[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)
                self.encoders[col] = encoder
                self.encoding_types[col] = 'onehot'
                self.feature_mappings[col] = {
                    'type': 'onehot',
                    'categories': encoder.categories_[0].tolist(),
                    'feature_names': encoder.get_feature_names_out([col]).tolist()
                }
                st.write(f"🎯 **{col}**: OneHot Encoded ({unique_values} categories)")
                
            else:
                # High cardinality - use Label Encoding with warning
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                self.encoders[col] = encoder
                self.encoding_types[col] = 'label'
                self.feature_mappings[col] = {
                    'type': 'label',
                    'classes': encoder.classes_.tolist(),
                    'mapping': dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                }
                st.warning(f"⚠️ **{col}**: Label Encoded (High cardinality: {unique_values} categories)")
        
        # Encode target for classification - FIXED: Handle mixed data types
        if problem_type == "binary_classification":
            # Convert target to string to handle mixed types
            y = y.astype(str)
            
            # Check if target has exactly 2 unique values
            unique_targets = y.unique()
            if len(unique_targets) != 2:
                st.error(f"Target column '{target_column}' has {len(unique_targets)} unique values. For binary classification, exactly 2 values are required. Found: {list(unique_targets)}")
                return None, None, None, None, None
            
            # Use LabelEncoder for target
            self.label_encoder_target = LabelEncoder()
            y_encoded = self.label_encoder_target.fit_transform(y)
            
            # Show encoding mapping
            class_mapping = dict(zip(self.label_encoder_target.classes_, self.label_encoder_target.transform(self.label_encoder_target.classes_)))
            st.write(f"🎯 **Target ({target_column})**: Label Encoded for Binary Classification")
            for orig, encoded in class_mapping.items():
                st.write(f"   - {orig} → {encoded}")

        elif problem_type == "multiclass_classification":
            # Convert target to string to handle mixed types
            y = y.astype(str)
            
            self.label_encoder_target = LabelEncoder()
            y_encoded = self.label_encoder_target.fit_transform(y)
            
            # Show encoding mapping
            class_mapping = dict(zip(self.label_encoder_target.classes_, self.label_encoder_target.transform(self.label_encoder_target.classes_)))
            st.write(f"🎯 **Target ({target_column})**: Label Encoded for Multi-class Classification ({len(self.label_encoder_target.classes_)} classes)")
            for orig, encoded in class_mapping.items():
                st.write(f"   - {orig} → {encoded}")
        
        else:  # regression
            # For regression, we already handled the conversion above
            y_encoded = y
            st.write(f"🎯 **Target ({target_column})**: Regression target (numeric) - Ready for training")
        
        # Use encoded target
        if problem_type in ["binary_classification", "multiclass_classification"]:
            y = y_encoded
        
        # Check if we have any data left after cleaning
        if len(X) == 0:
            st.error("❌ No valid data remaining after preprocessing. Please check your dataset.")
            return None, None, None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features - apply StandardScaler to all numerical features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        st.info(f"🔧 Applied StandardScaler to all {X_train.shape[1]} features")
        st.info(f"📊 Final dataset size: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    
    def build_model(self, input_dim, problem_type):
        """Build ANN model based on problem type - FIXED VERSION"""
        model = Sequential()
        
        # Input validation
        if input_dim <= 0:
            raise ValueError(f"Invalid input dimension: {input_dim}. Check if features were properly processed.")
        
        # Input layer
        model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.3))
        
        # Hidden layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        
        # Output layer
        if problem_type == "binary_classification":
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif problem_type == "multiclass_classification":
            # For multi-class classification, use the determined number of classes
            if self.num_classes is None:
                raise ValueError("Number of classes not determined for multi-class classification")
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:  # regression
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
        
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100):
        """Train the ANN model"""
        try:
            # For multi-class classification, determine number of classes
            if self.problem_type == "multiclass_classification":
                self.num_classes = len(np.unique(y_train))
                st.info(f"🔢 Multi-class classification detected: {self.num_classes} classes")
            
            # Validate input data
            if X_train is None or len(X_train) == 0:
                st.error("❌ No training data available")
                return None
                
            if X_train.shape[1] == 0:
                st.error("❌ No features available for training")
                return None
            
            st.info(f"🏗️ Building model with {X_train.shape[1]} input features...")
            self.model = self.build_model(X_train.shape[1], self.problem_type)
            
            # Display model summary
            st.subheader("📋 Model Architecture")
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))
            
            early_stopping = EarlyStopping(
                monitor='val_loss' if self.problem_type == 'regression' else 'val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            
            st.info("🚀 Starting training...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            st.success("✅ Training completed successfully!")
            return history
            
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            return None
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        predictions = self.model.predict(X_scaled, verbose=0)
        
        if self.problem_type == "binary_classification":
            # For binary classification, convert probabilities to class labels
            predictions = (predictions > 0.5).astype(int)
            
            # Convert 0/1 to original labels if encoder exists
            if hasattr(self, 'label_encoder_target') and self.label_encoder_target is not None:
                predictions = self.label_encoder_target.inverse_transform(predictions.flatten())
            else:
                # If target was already numeric (0/1), convert to Yes/No with REVERSED mapping
                # 0 → Yes, 1 → No
                predictions = np.array(['Yes' if pred == 0 else 'No' for pred in predictions.flatten()])
        
        elif self.problem_type == "multiclass_classification":
            # For multi-class, get the class with highest probability
            predictions = np.argmax(predictions, axis=1)
            if hasattr(self, 'label_encoder_target') and self.label_encoder_target is not None:
                predictions = self.label_encoder_target.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities (for classification only)"""
        if self.problem_type not in ["binary_classification", "multiclass_classification"]:
            return None
            
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        probabilities = self.model.predict(X_scaled, verbose=0)
        
        if self.problem_type == "binary_classification":
            # For binary classification, return probability of positive class
            return probabilities.flatten()
        else:
            # For multi-class, return all probabilities
            return probabilities
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance with proper metric handling - FIXED REGRESSION DATA TYPE ISSUE"""
        predictions = self.predict(X_test)
        
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            # For classification, ensure proper encoding and data types
            if hasattr(self, 'label_encoder_target') and self.label_encoder_target is not None:
                # Convert y_test to string first to handle mixed types, then encode
                y_test_encoded = self.label_encoder_target.transform(y_test.astype(str))
                
                # Ensure predictions are in the same format as y_test_encoded
                if isinstance(predictions[0], str):
                    # If predictions are strings, convert them back to numerical labels
                    predictions_encoded = self.label_encoder_target.transform(predictions.astype(str))
                else:
                    # If predictions are already numerical, use as is
                    predictions_encoded = predictions.astype(int)
            else:
                # If no label encoder, ensure both are numerical
                y_test_encoded = y_test.astype(int)
                predictions_encoded = predictions.astype(int)
                
            accuracy = accuracy_score(y_test_encoded, predictions_encoded)
            
            # Generate classification report with proper labels
            try:
                if hasattr(self, 'label_encoder_target') and self.label_encoder_target is not None:
                    target_names = self.label_encoder_target.classes_
                    classification_rep = classification_report(y_test_encoded, predictions_encoded, target_names=target_names)
                else:
                    classification_rep = classification_report(y_test_encoded, predictions_encoded)
            except Exception as e:
                st.warning(f"Could not generate detailed classification report: {str(e)}")
                classification_rep = "Basic accuracy only available"
            
            return {
                'accuracy': accuracy,
                'classification_report': classification_rep
            }
        else:
            # REGRESSION EVALUATION - COMPLETELY FIXED VERSION
            try:
                # Debug information
                st.write("🔍 Debug - Regression Evaluation:")
                st.write(f"y_test type: {type(y_test)}, shape: {y_test.shape if hasattr(y_test, 'shape') else 'N/A'}")
                st.write(f"predictions type: {type(predictions)}, shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                
                # Convert both to numpy arrays and ensure they are numeric
                y_test_array = np.array(y_test, dtype=float)
                predictions_array = np.array(predictions, dtype=float)
                
                # Check for any NaN or infinite values
                y_test_valid = np.isfinite(y_test_array)
                predictions_valid = np.isfinite(predictions_array)
                
                # Combine valid masks
                valid_mask = y_test_valid & predictions_valid
                
                # Count invalid values
                invalid_count = len(y_test_array) - np.sum(valid_mask)
                
                if invalid_count > 0:
                    st.warning(f"⚠️ Found {invalid_count} invalid values (NaN or infinite) in regression evaluation")
                
                # Get valid samples
                y_test_clean = y_test_array[valid_mask]
                predictions_clean = predictions_array[valid_mask]
                
                # Check if we have any valid data left
                if len(y_test_clean) == 0:
                    st.error("❌ No valid numeric values for regression evaluation after cleaning")
                    return {
                        'mse': float('nan'),
                        'mae': float('nan'),
                        'rmse': float('nan'),
                        'r2': float('nan'),
                        'valid_samples': 0
                    }
                
                # Calculate metrics
                mse = mean_squared_error(y_test_clean, predictions_clean)
                mae = mean_absolute_error(y_test_clean, predictions_clean)
                r2 = r2_score(y_test_clean, predictions_clean)
                
                st.info(f"📊 Regression evaluation on {len(y_test_clean)} valid samples (removed {invalid_count} invalid samples)")
                
                return {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'valid_samples': len(y_test_clean)
                }
                
            except Exception as e:
                st.error(f"❌ Error in regression evaluation: {str(e)}")
                # Provide more detailed error information
                st.write("Debug info:")
                st.write(f"y_test sample: {y_test[:5] if hasattr(y_test, '__getitem__') else 'N/A'}")
                st.write(f"predictions sample: {predictions[:5] if hasattr(predictions, '__getitem__') else 'N/A'}")
                return {
                    'mse': float('nan'),
                    'mae': float('nan'),
                    'rmse': float('nan'),
                    'r2': float('nan'),
                    'valid_samples': 0
                }
    
    def save_model(self, filepath):
        """Save model and preprocessors"""
        # Use .h5 format for better compatibility
        self.model.save(f'{filepath}_model.h5')
        
        with open(f'{filepath}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(f'{filepath}_encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
            
        with open(f'{filepath}_encoding_types.pkl', 'wb') as f:
            pickle.dump(self.encoding_types, f)
            
        with open(f'{filepath}_feature_mappings.pkl', 'wb') as f:
            pickle.dump(self.feature_mappings, f)
            
        with open(f'{filepath}_original_columns_info.pkl', 'wb') as f:
            pickle.dump(self.original_columns_info, f)
            
        if self.problem_type in ["binary_classification", "multiclass_classification"] and hasattr(self, 'label_encoder_target'):
            with open(f'{filepath}_label_encoder_target.pkl', 'wb') as f:
                pickle.dump(self.label_encoder_target, f)
        
        # Save columns to drop
        with open(f'{filepath}_columns_to_drop.pkl', 'wb') as f:
            pickle.dump(self.columns_to_drop, f)
    
    def load_model(self, filepath, problem_type):
        """Load model and preprocessors"""
        self.problem_type = problem_type
        # Use tf.keras.models.load_model for better compatibility
        self.model = load_model(f'{filepath}_model.h5')
        
        with open(f'{filepath}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(f'{filepath}_encoders.pkl', 'rb') as f:
            self.encoders = pickle.load(f)
            
        with open(f'{filepath}_encoding_types.pkl', 'rb') as f:
            self.encoding_types = pickle.load(f)
            
        with open(f'{filepath}_feature_mappings.pkl', 'rb') as f:
            self.feature_mappings = pickle.load(f)
            
        with open(f'{filepath}_original_columns_info.pkl', 'rb') as f:
            self.original_columns_info = pickle.load(f)
            
        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            try:
                with open(f'{filepath}_label_encoder_target.pkl', 'rb') as f:
                    self.label_encoder_target = pickle.load(f)
            except FileNotFoundError:
                self.label_encoder_target = None
        
        # Load columns to drop
        with open(f'{filepath}_columns_to_drop.pkl', 'rb') as f:
            self.columns_to_drop = pickle.load(f)

def create_column_selector(df, target_column):
    """Create a multi-select widget for column selection"""
    st.subheader("🗑️ Select Columns to Drop")
    st.write("Choose which columns to exclude from the model:")
    
    # Get all columns except target
    available_columns = [col for col in df.columns if col != target_column]
    
    if available_columns:
        # Create two columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="column-selector">', unsafe_allow_html=True)
            selected_columns = []
            
            # Create checkboxes for each column
            for column in available_columns:
                # Show column info
                col_type = df[column].dtype
                unique_count = df[column].nunique() if df[column].dtype == 'object' else ''
                missing_count = df[column].isnull().sum()
                
                col_info = f"**{column}** - {col_type}"
                if unique_count:
                    col_info += f" | {unique_count} unique values"
                if missing_count > 0:
                    col_info += f" | {missing_count} missing"
                
                # Checkbox with column info
                if st.checkbox(col_info, key=f"drop_{column}"):
                    selected_columns.append(column)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Selected to Drop")
            if selected_columns:
                for col in selected_columns:
                    st.write(f"❌ {col}")
                st.info(f"Total columns to drop: {len(selected_columns)}")
            else:
                st.write("No columns selected")
                st.info("All columns will be used for training")
        
        return selected_columns
    else:
        st.warning("No columns available to drop (only target column exists)")
        return []

def create_unified_prediction_inputs(df, feature_names, feature_mappings, original_columns_info, columns_to_drop):
    """Create unified input fields where ONLY original columns are shown, OneHot encoded columns are hidden"""
    input_data = {}
    
    st.markdown("""
    <div class="prediction-input">
    <h4>🎯 Enter Your Values for Prediction</h4>
    <p>Fill in the values below for each feature. The system will automatically handle encoding and scaling.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get original columns that weren't dropped
    original_columns = [col for col in df.columns if col not in columns_to_drop and col != st.session_state.get('target_column', '')]
    
    # Identify which original columns have OneHot encoding
    onehot_original_columns = {}
    for col_name in original_columns:
        if col_name in feature_mappings and feature_mappings[col_name]['type'] == 'onehot':
            onehot_original_columns[col_name] = feature_mappings[col_name]
    
    # Create input widgets for ALL original columns only
    st.subheader("📝 All Input Features")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    # Process all original columns in a single loop
    for i, col_name in enumerate(original_columns):
        # Alternate between columns
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            # Handle OneHot encoded features - show as dropdown with original categories
            if col_name in onehot_original_columns:
                mapping = onehot_original_columns[col_name]
                categories = mapping['categories']
                
                st.markdown('<div class="feature-input-group">', unsafe_allow_html=True)
                
                st.write(f"**{col_name}** 🎯")
                
                selected_value = st.selectbox(
                    f"Select category for {col_name}",
                    options=categories,
                    help=f"OneHot Encoded: {len(categories)} categories",
                    key=f"onehot_{col_name}"
                )
                
                # Store the selected value - we'll handle the OneHot encoding later
                input_data[col_name] = selected_value
                
                st.caption(f"OneHot Encoded: {len(categories)} categories")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # NEW: Check if column is in 1-10 range (perfect for dropdown) - FIXED CONDITION
            elif (col_name in original_columns_info and 
                original_columns_info[col_name].get('is_dropdown_candidate', False) and
                original_columns_info[col_name].get('dropdown_values') and
                len(original_columns_info[col_name]['dropdown_values']) <= 10):  # ADDED: Check number of values
                
                # Use dropdown for this column (1-10 range)
                col_info = original_columns_info[col_name]
                dropdown_values = col_info['dropdown_values']
                
                st.markdown('<div class="feature-input-group">', unsafe_allow_html=True)
                
                st.write(f"**{col_name}** 🔢")
                
                # Create dropdown
                selected_value = st.selectbox(
                    f"Select value for {col_name}",
                    options=dropdown_values,
                    help=f"Dropdown for values between 1-10: {dropdown_values}",
                    key=f"dropdown_{col_name}"
                )
                
                st.caption(f"1-10 Range: {dropdown_values}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                input_data[col_name] = float(selected_value)
                
            # Check if column is a slider candidate (values between 1-1000)
            elif (col_name in original_columns_info and 
                original_columns_info[col_name].get('is_slider_candidate', False)):
                
                # Use slider for this column
                col_info = original_columns_info[col_name]
                min_val = int(col_info['min'])
                max_val = int(col_info['max'])
                mean_val = int(col_info['mean'])
                
                st.markdown('<div class="feature-input-group">', unsafe_allow_html=True)
                
                st.write(f"**{col_name}** 🎚️")
                
                # Create slider
                selected_value = st.slider(
                    f"Select value for {col_name}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=1,
                    key=f"slider_{col_name}",
                    help=f"Slide to select value between {min_val} and {max_val}"
                )
                
                # Display current value prominently
                st.markdown(f'<div class="slider-value">Current Value: {selected_value}</div>', unsafe_allow_html=True)
                
                st.caption(f"Range: {min_val} to {max_val}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                input_data[col_name] = float(selected_value)
                
            # Check if column is binary numeric (only 0 and 1)
            elif (col_name in original_columns_info and 
                original_columns_info[col_name].get('is_binary_numeric', False)):
                
                # Binary numeric column - show as dropdown with Yes/No labels
                binary_options = [0, 1]
                binary_labels = {0: "No", 1: "Yes"}
                
                selected_value = st.selectbox(
                    f"**{col_name}** 🔢",
                    options=binary_options,
                    format_func=lambda x: binary_labels[x],
                    help="Binary numeric feature (No = 0, Yes = 1)",
                    key=f"input_{col_name}"
                )
                input_data[col_name] = selected_value
                st.caption("Binary Numeric: No = 0, Yes = 1")
                
            elif col_name in feature_mappings and feature_mappings[col_name]['type'] == 'label':
                mapping = feature_mappings[col_name]
                
                # Label encoded feature - Binary (Male/Female) or multi-category
                classes = mapping['classes']
                
                # Check if this is a binary categorical feature that should show Yes/No
                if len(classes) == 2:
                    # Check if the classes represent binary choices that could be mapped to Yes/No
                    class0_str = str(classes[0]).lower()
                    class1_str = str(classes[1]).lower()
                    
                    # Common binary patterns that can be mapped to Yes/No
                    yes_no_patterns = [
                        {'no', 'yes'}, {'false', 'true'}, {'0', '1'}, 
                        {'absent', 'present'}, {'negative', 'positive'},
                        {'off', 'on'}, {'closed', 'open'}, {'inactive', 'active'}
                    ]
                    
                    current_set = {class0_str, class1_str}
                    use_yes_no = any(current_set == pattern for pattern in yes_no_patterns)
                    
                    if use_yes_no:
                        # Map to Yes/No display but keep original encoding
                        yes_no_labels = {classes[0]: "No", classes[1]: "Yes"}
                        selected_value = st.selectbox(
                            f"**{col_name}** 🔤",
                            options=classes,
                            format_func=lambda x: yes_no_labels[x],
                            help=f"Binary feature: {classes[0]} = No, {classes[1]} = Yes",
                            key=f"input_{col_name}"
                        )
                        st.caption(f"Binary: {classes[0]} → No (0), {classes[1]} → Yes (1)")
                    else:
                        # Regular binary with original labels
                        selected_value = st.selectbox(
                            f"**{col_name}** 🔤",
                            options=classes,
                            help=f"Binary feature: {classes[0]} → 0, {classes[1]} → 1",
                            key=f"input_{col_name}"
                        )
                        st.caption(f"Binary: {classes[0]} → 0, {classes[1]} → 1")
                else:
                    # Multi-category feature - show all options
                    selected_value = st.selectbox(
                        f"**{col_name}** 🔤",
                        options=classes,
                        help=f"Multi-category feature with {len(classes)} options",
                        key=f"input_{col_name}"
                    )
                    st.caption(f"Multi-category: {len(classes)} options")
                
                # Convert to numerical value using mapping
                mapping_dict = feature_mappings[col_name]['mapping']
                input_data[col_name] = mapping_dict[selected_value]
                    
            else:
                # Numerical feature - use number input
                if col_name in original_columns_info:
                    col_info = original_columns_info[col_name]
                    if col_info['dtype'] in ['int64', 'float64']:
                        min_val = col_info['min']
                        max_val = col_info['max']
                        mean_val = col_info['mean']
                        
                        input_val = st.number_input(
                            f"**{col_name}** 📊",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(mean_val),
                            step=float((max_val - min_val) / 100),
                            help=f"Numerical feature: {min_val:.2f} to {max_val:.2f}",
                            key=f"input_{col_name}"
                        )
                        input_data[col_name] = input_val
                        st.caption(f"Range: {min_val:.2f} to {max_val:.2f}")
                    else:
                        # For other categorical types not caught by feature_mappings
                        unique_vals = df[col_name].unique()
                        selected_val = st.selectbox(
                            f"**{col_name}** ❓", 
                            unique_vals,
                            key=f"input_{col_name}"
                        )
                        input_data[col_name] = selected_val
                        st.caption(f"Categorical: {len(unique_vals)} unique values")
    
    return input_data

def convert_original_input_to_model_format(input_data, feature_mappings, feature_names):
    """Convert original column inputs to model-ready format with OneHot encoding"""
    model_input = {}
    
    # Initialize all feature columns with 0
    for feature in feature_names:
        model_input[feature] = 0.0
    
    # Process each original column
    for col_name, value in input_data.items():
        if col_name in feature_mappings:
            mapping = feature_mappings[col_name]
            
            if mapping['type'] == 'onehot':
                # OneHot encoding: set the corresponding column to 1, others remain 0
                for feature in mapping['feature_names']:
                    # Check if this feature matches the selected value
                    category_name = feature.split('__')[1] if '__' in feature else feature
                    if str(value) == str(category_name):
                        model_input[feature] = 1.0
                    else:
                        model_input[feature] = 0.0
                        
            elif mapping['type'] == 'label':
                # Label encoding: use the numerical value directly
                # Find which feature this corresponds to
                matching_features = [f for f in feature_names if col_name in f]
                if len(matching_features) == 1:
                    model_input[matching_features[0]] = float(value)
        else:
            # Direct numerical feature - find matching feature name
            matching_features = [f for f in feature_names if col_name == f]
            if len(matching_features) == 1:
                model_input[matching_features[0]] = float(value)
    
    return model_input

def main():
    st.markdown('<h1 class="main-header">🧠 Auto ANN Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Smart Neural Network with Automatic Feature Encoding")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'ann_model' not in st.session_state:
        st.session_state.ann_model = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'problem_type' not in st.session_state:
        st.session_state.problem_type = None
    if 'columns_to_drop' not in st.session_state:
        st.session_state.columns_to_drop = []
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
        ["📊 Data Upload & Setup", "🤖 Train Model", "🎯 Make Predictions", "📁 Load Saved Model"])
    
    if app_mode == "📊 Data Upload & Setup":
        handle_data_upload()
    
    elif app_mode == "🤖 Train Model":
        handle_model_training()
    
    elif app_mode == "🎯 Make Predictions":
        handle_predictions()
    
    elif app_mode == "📁 Load Saved Model":
        handle_load_model()

def handle_data_upload():
    st.header("📊 Data Upload & Setup")
    
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Show column information
            st.subheader("🔍 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)
            
            # Target column selection
            st.subheader("🎯 Select Target Column")
            target_column = st.selectbox("Choose the target variable", df.columns)
            st.session_state.target_column = target_column
            
            # Problem type selection
            st.subheader("🎯 Select Problem Type")
            
            # Auto-detect problem type based on target column
            target_dtype = df[target_column].dtype
            unique_targets = df[target_column].nunique()
            
            if target_dtype == 'object':
                if unique_targets == 2:
                    default_problem = "binary_classification"
                    st.info(f"🔍 Auto-detected: Binary Classification (2 unique categories)")
                else:
                    default_problem = "multiclass_classification"
                    st.info(f"🔍 Auto-detected: Multi-class Classification ({unique_targets} unique categories)")
            else:
                default_problem = "regression"
                st.info(f"🔍 Auto-detected: Regression (numeric target)")
            
            problem_type = st.selectbox(
                "Confirm problem type",
                ["binary_classification", "multiclass_classification", "regression"],
                index=["binary_classification", "multiclass_classification", "regression"].index(default_problem)
            )
            st.session_state.problem_type = problem_type
            
            # Show target variable statistics
            st.subheader("📊 Target Variable Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Statistics:**")
                if problem_type == "regression":
                    # For regression, show numeric statistics
                    try:
                        target_numeric = pd.to_numeric(df[target_column], errors='coerce')
                        valid_count = target_numeric.notna().sum()
                        nan_count = target_numeric.isna().sum()
                        
                        st.write(f"Mean: {target_numeric.mean():.2f}")
                        st.write(f"Std: {target_numeric.std():.2f}")
                        st.write(f"Min: {target_numeric.min():.2f}")
                        st.write(f"Max: {target_numeric.max():.2f}")
                        st.write(f"Valid values: {valid_count}")
                        if nan_count > 0:
                            st.warning(f"Non-numeric values: {nan_count}")
                    except:
                        st.error("Cannot compute statistics - non-numeric values present")
                else:
                    value_counts = df[target_column].value_counts()
                    for value, count in value_counts.items():
                        percentage = (count / len(df)) * 100
                        st.write(f"{value}: {count} ({percentage:.1f}%)")
            
            with col2:
                st.write("**Visualization:**")
                if problem_type == "regression":
                    fig, ax = plt.subplots(figsize=(8, 4))
                    target_numeric = pd.to_numeric(df[target_column], errors='coerce')
                    target_clean = target_numeric[target_numeric.notna()]
                    if len(target_clean) > 0:
                        ax.hist(target_clean, bins=30, alpha=0.7, color='skyblue')
                        ax.set_xlabel(target_column)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {target_column}')
                        st.pyplot(fig)
                    else:
                        st.warning("No valid numeric data to plot")
                else:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    value_counts = df[target_column].value_counts()
                    ax.bar(value_counts.index.astype(str), value_counts.values, alpha=0.7, color='lightcoral')
                    ax.set_xlabel(target_column)
                    ax.set_ylabel('Count')
                    ax.set_title(f'Distribution of {target_column}')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # Column selection for dropping
            columns_to_drop = create_column_selector(df, target_column)
            st.session_state.columns_to_drop = columns_to_drop
            
            # Store dataframe in session state
            st.session_state.df = df
            
            st.success("✅ Data setup complete! Proceed to 'Train Model'.")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def handle_model_training():
    st.header("🤖 Train Neural Network Model")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload data first in 'Data Upload & Setup'")
        return
    
    df = st.session_state.df
    target_column = st.session_state.target_column
    problem_type = st.session_state.problem_type
    columns_to_drop = st.session_state.columns_to_drop
    
    # Training parameters
    st.subheader("⚙️ Training Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", 50, 500, 100, 50)
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
    
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)
        enable_early_stopping = st.checkbox("Enable Early Stopping", value=True)
    
    if st.button("🚀 Train Neural Network", type="primary"):
        with st.spinner("Training in progress... This may take a few minutes."):
            try:
                # Initialize and train model
                ann_model = AutoANN()
                
                # Preprocess data
                X_train, X_test, y_train, y_test, feature_names = ann_model.preprocess_data(
                    df, target_column, problem_type, columns_to_drop
                )
                
                if X_train is None:
                    st.error("❌ Data preprocessing failed. Please check your data and try again.")
                    return
                
                # Train model
                history = ann_model.train_model(X_train, y_train, X_test, y_test, epochs=epochs)
                
                if history is None:
                    st.error("❌ Model training failed. Please check the error messages above.")
                    return
                
                # Evaluate model
                evaluation = ann_model.evaluate_model(X_test, y_test)
                
                # Store model in session state
                st.session_state.ann_model = ann_model
                st.session_state.model_trained = True
                st.session_state.feature_names = feature_names
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Display results
                st.success("✅ Model trained successfully!")
                
                # Show training history
                st.subheader("📈 Training History")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Plot loss
                ax1.plot(history.history['loss'], label='Training Loss')
                if 'val_loss' in history.history:
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Model Loss')
                ax1.legend()
                
                # Plot accuracy for classification, MSE for regression
                if problem_type != "regression":
                    ax2.plot(history.history['accuracy'], label='Training Accuracy')
                    if 'val_accuracy' in history.history:
                        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.set_title('Model Accuracy')
                else:
                    ax2.plot(history.history['mae'], label='Training MAE')
                    if 'val_mae' in history.history:
                        ax2.plot(history.history['val_mae'], label='Validation MAE')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('MAE')
                    ax2.set_title('Model MAE')
                ax2.legend()
                
                st.pyplot(fig)
                
                # Show evaluation metrics
                st.subheader("📊 Model Performance")
                if problem_type != "regression":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Accuracy", f"{evaluation['accuracy']:.4f}")
                    with col2:
                        st.write("**Classification Report:**")
                        st.text(evaluation['classification_report'])
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MSE", f"{evaluation['mse']:.4f}")
                    with col2:
                        st.metric("MAE", f"{evaluation['mae']:.4f}")
                    with col3:
                        st.metric("RMSE", f"{evaluation['rmse']:.4f}")
                    with col4:
                        st.metric("R² Score", f"{evaluation['r2']:.4f}")
                    
                    if 'valid_samples' in evaluation:
                        st.info(f"✅ Evaluation performed on {evaluation['valid_samples']} valid samples")
                
                # Model saving
                st.subheader("💾 Save Model")
                model_name = st.text_input("Model name", "my_ann_model")
                
                if st.button("💾 Save Model Files"):
                    ann_model.save_model(model_name)
                    st.success(f"✅ Model saved as '{model_name}'!")
                    st.info("Saved files: model.h5, scaler.pkl, encoders.pkl, encoding_types.pkl, feature_mappings.pkl")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

def handle_predictions():
    st.header("🎯 Make Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in 'Train Model' or load a saved model")
        return
    
    ann_model = st.session_state.ann_model
    df = st.session_state.df
    feature_names = st.session_state.feature_names
    feature_mappings = ann_model.feature_mappings
    original_columns_info = ann_model.original_columns_info
    columns_to_drop = st.session_state.columns_to_drop
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**Prediction Interface**: Enter values for each feature below. The system automatically handles encoding and scaling.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create unified prediction inputs (showing only original columns)
    input_data = create_unified_prediction_inputs(df, feature_names, feature_mappings, original_columns_info, columns_to_drop)
    
    if st.button("🔮 Make Prediction", type="primary"):
        if input_data:
            try:
                # Convert original inputs to model format
                model_input_dict = convert_original_input_to_model_format(input_data, feature_mappings, feature_names)
                
                # Create input array for model
                input_array = np.array([[model_input_dict[feature] for feature in feature_names]])
                
                # Make prediction
                prediction = ann_model.predict(input_array)
                
                # Display results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.subheader("🎯 Prediction Result")
                
                if ann_model.problem_type == "binary_classification":
                    st.success(f"**Predicted Class:** {prediction[0]}")
                    
                    # Show probability if available
                    probabilities = ann_model.predict_proba(input_array)
                    if probabilities is not None:
                        prob_class_1 = probabilities[0]
                        prob_class_0 = 1 - prob_class_1
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Probability (Class 1)", f"{prob_class_1:.4f}")
                        with col2:
                            st.metric("Probability (Class 0)", f"{prob_class_0:.4f}")
                        
                        # Show probability bar
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.barh(['Probability'], [prob_class_1], color='lightcoral', alpha=0.7, label='Class 1')
                        ax.barh(['Probability'], [prob_class_0], left=[prob_class_1], color='skyblue', alpha=0.7, label='Class 0')
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Probability')
                        ax.legend()
                        st.pyplot(fig)
                
                elif ann_model.problem_type == "multiclass_classification":
                    st.success(f"**Predicted Class:** {prediction[0]}")
                    
                    # Show all class probabilities
                    probabilities = ann_model.predict_proba(input_array)
                    if probabilities is not None:
                        classes = ann_model.label_encoder_target.classes_
                        prob_dict = dict(zip(classes, probabilities[0]))
                        
                        st.write("**Class Probabilities:**")
                        for class_name, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"{class_name}")
                            with col2:
                                st.write(f"{prob:.4f}")
                
                else:  # regression
                    st.success(f"**Predicted Value:** {prediction[0]:.4f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show input summary
                st.subheader("📋 Input Summary")
                input_summary_df = pd.DataFrame({
                    'Feature': list(input_data.keys()),
                    'Value': list(input_data.values())
                })
                st.dataframe(input_summary_df)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

def handle_load_model():
    st.header("📁 Load Saved Model")
    
    st.info("Upload all the model files that were saved during training:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_file = st.file_uploader("Upload Model (.h5)", type=['h5'])
        scaler_file = st.file_uploader("Upload Scaler (.pkl)", type=['pkl'])
        encoders_file = st.file_uploader("Upload Encoders (.pkl)", type=['pkl'])
    
    with col2:
        encoding_types_file = st.file_uploader("Upload Encoding Types (.pkl)", type=['pkl'])
        feature_mappings_file = st.file_uploader("Upload Feature Mappings (.pkl)", type=['pkl'])
        original_columns_info_file = st.file_uploader("Upload Original Columns Info (.pkl)", type=['pkl'])
    
    problem_type = st.selectbox(
        "Select Problem Type",
        ["binary_classification", "multiclass_classification", "regression"]
    )
    
    if st.button("🔧 Load Model", type="primary"):
        if all([model_file, scaler_file, encoders_file, encoding_types_file, feature_mappings_file, original_columns_info_file]):
            try:
                # Save uploaded files temporarily
                temp_dir = "temp_model"
                os.makedirs(temp_dir, exist_ok=True)
                
                model_path = os.path.join(temp_dir, "model.h5")
                scaler_path = os.path.join(temp_dir, "scaler.pkl")
                encoders_path = os.path.join(temp_dir, "encoders.pkl")
                encoding_types_path = os.path.join(temp_dir, "encoding_types.pkl")
                feature_mappings_path = os.path.join(temp_dir, "feature_mappings.pkl")
                original_columns_info_path = os.path.join(temp_dir, "original_columns_info.pkl")
                
                with open(model_path, "wb") as f:
                    f.write(model_file.getvalue())
                with open(scaler_path, "wb") as f:
                    f.write(scaler_file.getvalue())
                with open(encoders_path, "wb") as f:
                    f.write(encoders_file.getvalue())
                with open(encoding_types_path, "wb") as f:
                    f.write(encoding_types_file.getvalue())
                with open(feature_mappings_path, "wb") as f:
                    f.write(feature_mappings_file.getvalue())
                with open(original_columns_info_path, "wb") as f:
                    f.write(original_columns_info_file.getvalue())
                
                # Load target encoder if it exists (for classification)
                label_encoder_target_file = st.file_uploader("Upload Target Label Encoder (.pkl) - Optional for Classification", type=['pkl'])
                if label_encoder_target_file:
                    label_encoder_target_path = os.path.join(temp_dir, "label_encoder_target.pkl")
                    with open(label_encoder_target_path, "wb") as f:
                        f.write(label_encoder_target_file.getvalue())
                
                # Load columns to drop if available
                columns_to_drop_file = st.file_uploader("Upload Columns to Drop (.pkl) - Optional", type=['pkl'])
                if columns_to_drop_file:
                    columns_to_drop_path = os.path.join(temp_dir, "columns_to_drop.pkl")
                    with open(columns_to_drop_path, "wb") as f:
                        f.write(columns_to_drop_file.getvalue())
                
                # Initialize and load model
                ann_model = AutoANN()
                ann_model.load_model(temp_dir, problem_type)
                
                # Store in session state
                st.session_state.ann_model = ann_model
                st.session_state.model_trained = True
                st.session_state.problem_type = problem_type
                
                # Get feature names from the model
                if hasattr(ann_model, 'feature_mappings'):
                    feature_names = []
                    for mapping in ann_model.feature_mappings.values():
                        if mapping['type'] == 'onehot':
                            feature_names.extend(mapping['feature_names'])
                        else:
                            # For label encoded and numerical features, we need to infer the feature name
                            # This is a simplified approach - you might need to adjust based on your actual feature names
                            pass
                    st.session_state.feature_names = feature_names
                
                st.success("✅ Model loaded successfully! You can now make predictions.")
                
                # Clean up temporary files
                import shutil
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
        else:
            st.warning("⚠️ Please upload all required model files")

if __name__ == "__main__":
    main()
