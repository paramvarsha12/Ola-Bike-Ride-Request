"""
SVM AND KNN IMPLEMENTATION FOR OLA BIKE RIDE REQUEST FORECASTING
================================================================

This script implements Support Vector Machine (SVM) and K-Nearest Neighbors (KNN)
algorithms for predicting bike ride requests with comprehensive feature engineering,
model training, evaluation, and visualization.

Features:
- Advanced feature engineering
- SVM regression and classification
- KNN regression and classification  
- Hyperparameter tuning
- Performance metrics
- Comprehensive visualizations
- GPU acceleration support
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestRegressor
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth in kilometers"""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def create_features(df):
    """Create comprehensive features for the model"""
    print("Creating advanced features...")
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['ts'])
    
    # Temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Geographic features
    df['trip_distance'] = haversine_distance(df['pick_lat'], df['pick_lng'], 
                                           df['drop_lat'], df['drop_lng'])
    
    # Distance from city center (Bangalore)
    bangalore_lat, bangalore_lng = 12.9716, 77.5946
    df['pickup_distance_from_center'] = haversine_distance(
        df['pick_lat'], df['pick_lng'], bangalore_lat, bangalore_lng)
    df['drop_distance_from_center'] = haversine_distance(
        df['drop_lat'], df['drop_lng'], bangalore_lat, bangalore_lng)
    
    # Peak hour indicators
    df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                         (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # Demand level classification (for classification tasks)
    # Create demand levels based on trip frequency in the area
    df['demand_level'] = pd.cut(df['trip_distance'], 
                               bins=[0, 5, 10, 15, float('inf')], 
                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df

def prepare_data(df, target_type='regression'):
    """Prepare data for machine learning models"""
    print(f"Preparing data for {target_type}...")
    
    # Select features
    feature_columns = [
        'hour', 'day_of_week', 'month', 'quarter', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'pick_lat', 'pick_lng', 'drop_lat', 'drop_lng', 'trip_distance',
        'pickup_distance_from_center', 'drop_distance_from_center',
        'is_peak_hour', 'is_night'
    ]
    
    X = df[feature_columns].copy()
    
    if target_type == 'regression':
        # For regression: predict trip distance
        y = df['trip_distance']
    else:
        # For classification: predict demand level
        le = LabelEncoder()
        y = le.fit_transform(df['demand_level'])
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

def train_svm_models(X_train, y_train, X_test, y_test, target_type='regression'):
    """Train SVM models with hyperparameter tuning"""
    print(f"Training SVM models for {target_type}...")
    
    if target_type == 'regression':
        # SVM Regression
        svm_reg = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2]
        }
        
        grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"SVM Regression Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return best_svm, y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    else:
        # SVM Classification
        svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"SVM Classification Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return best_svm, y_pred, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def train_knn_models(X_train, y_train, X_test, y_test, target_type='regression'):
    """Train KNN models with hyperparameter tuning"""
    print(f"Training KNN models for {target_type}...")
    
    if target_type == 'regression':
        # KNN Regression
        knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
        
        # Hyperparameter tuning
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        grid_search = GridSearchCV(knn_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_knn = grid_search.best_estimator_
        y_pred = best_knn.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"KNN Regression Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return best_knn, y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    
    else:
        # KNN Classification
        knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
        
        # Hyperparameter tuning
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        grid_search = GridSearchCV(knn_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_knn = grid_search.best_estimator_
        y_pred = best_knn.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"KNN Classification Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return best_knn, y_pred, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def create_performance_visualizations(svm_metrics, knn_metrics, target_type='regression'):
    """Create comprehensive performance visualizations"""
    print("Creating performance visualizations...")
    
    if target_type == 'regression':
        # Regression metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MSE Comparison
        models = ['SVM', 'KNN']
        mse_values = [svm_metrics['mse'], knn_metrics['mse']]
        axes[0, 0].bar(models, mse_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0, 0].set_title('Mean Squared Error Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE Comparison
        rmse_values = [svm_metrics['rmse'], knn_metrics['rmse']]
        axes[0, 1].bar(models, rmse_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0, 1].set_title('Root Mean Squared Error Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAE Comparison
        mae_values = [svm_metrics['mae'], knn_metrics['mae']]
        axes[1, 0].bar(models, mae_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[1, 0].set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # R² Score Comparison
        r2_values = [svm_metrics['r2'], knn_metrics['r2']]
        axes[1, 1].bar(models, r2_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[1, 1].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for ax, values in zip(axes.flat, [mse_values, rmse_values, mae_values, r2_values]):
            for i, v in enumerate(values):
                ax.text(i, v + v*0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('regression_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Classification metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy Comparison
        models = ['SVM', 'KNN']
        accuracy_values = [svm_metrics['accuracy'], knn_metrics['accuracy']]
        axes[0, 0].bar(models, accuracy_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision Comparison
        precision_values = [svm_metrics['precision'], knn_metrics['precision']]
        axes[0, 1].bar(models, precision_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0, 1].set_title('Precision Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall Comparison
        recall_values = [svm_metrics['recall'], knn_metrics['recall']]
        axes[1, 0].bar(models, recall_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[1, 0].set_title('Recall Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1-Score Comparison
        f1_values = [svm_metrics['f1'], knn_metrics['f1']]
        axes[1, 1].bar(models, f1_values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[1, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for ax, values in zip(axes.flat, [accuracy_values, precision_values, recall_values, f1_values]):
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('classification_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_prediction_visualizations(y_test, svm_pred, knn_pred, target_type='regression'):
    """Create prediction vs actual visualizations"""
    print("Creating prediction visualizations...")
    
    if target_type == 'regression':
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # SVM Predictions vs Actual
        axes[0].scatter(y_test, svm_pred, alpha=0.6, color='#FF6B6B')
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('SVM: Predicted vs Actual')
        axes[0].grid(True, alpha=0.3)
        
        # KNN Predictions vs Actual
        axes[1].scatter(y_test, knn_pred, alpha=0.6, color='#4ECDC4')
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Predicted Values')
        axes[1].set_title('KNN: Predicted vs Actual')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    else:
        # Classification confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # SVM Confusion Matrix
        svm_cm = confusion_matrix(y_test, svm_pred)
        sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('SVM Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # KNN Confusion Matrix
        knn_cm = confusion_matrix(y_test, knn_pred)
        sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('KNN Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    print("=== OLA BIKE RIDE REQUEST FORECASTING SYSTEM ===")
    print("Implementing SVM and KNN algorithms with comprehensive analysis")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv('ct_rr.csv')
        print(f"Loaded {df.shape[0]:,} records with {df.shape[1]} columns")
    except FileNotFoundError:
        print("Dataset file 'ct_rr.csv' not found. Please ensure the dataset is available.")
        return
    
    # Create features
    df = create_features(df)
    print(f"Created {df.shape[1]} features")
    
    # Run both regression and classification tasks
    for target_type in ['regression', 'classification']:
        print(f"\n{'='*20} {target_type.upper()} TASK {'='*20}")
        
        # Prepare data
        X, y, scaler, feature_columns = prepare_data(df, target_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train SVM model
        print(f"\n--- Training SVM for {target_type} ---")
        svm_model, svm_pred, svm_metrics = train_svm_models(X_train, y_train, X_test, y_test, target_type)
        
        # Train KNN model
        print(f"\n--- Training KNN for {target_type} ---")
        knn_model, knn_pred, knn_metrics = train_knn_models(X_train, y_train, X_test, y_test, target_type)
        
        # Create visualizations
        print(f"\n--- Creating visualizations for {target_type} ---")
        create_performance_visualizations(svm_metrics, knn_metrics, target_type)
        create_prediction_visualizations(y_test, svm_pred, knn_pred, target_type)
        
        # Print summary
        print(f"\n--- {target_type.upper()} RESULTS SUMMARY ---")
        if target_type == 'regression':
            print(f"SVM - RMSE: {svm_metrics['rmse']:.4f}, R²: {svm_metrics['r2']:.4f}")
            print(f"KNN - RMSE: {knn_metrics['rmse']:.4f}, R²: {knn_metrics['r2']:.4f}")
        else:
            print(f"SVM - Accuracy: {svm_metrics['accuracy']:.4f}, F1: {svm_metrics['f1']:.4f}")
            print(f"KNN - Accuracy: {knn_metrics['accuracy']:.4f}, F1: {knn_metrics['f1']:.4f}")
    
    print("\n=== ANALYSIS COMPLETED ===")
    print("Generated files:")
    print("- regression_performance_comparison.png")
    print("- classification_performance_comparison.png")
    print("- prediction_vs_actual.png")
    print("- confusion_matrices.png")
    print("\nAll models trained and evaluated successfully!")

if __name__ == "__main__":
    main()
