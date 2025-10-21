"""
COMPREHENSIVE RESULTS ANALYSIS FOR OLA BIKE RIDE REQUEST FORECASTING
===================================================================

This script provides detailed analysis of training and testing results,
performance metrics, and comprehensive evaluation of SVM and KNN models.

Features:
- Detailed performance metrics
- Training vs testing analysis
- Cross-validation results
- Feature importance analysis
- Model comparison
- Comprehensive reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

def analyze_model_performance(model, X_train, y_train, X_test, y_test, model_name, target_type='regression'):
    """Comprehensive model performance analysis"""
    print(f"\n=== {model_name} PERFORMANCE ANALYSIS ===")
    
    # Training performance
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    if target_type == 'regression':
        # Regression metrics
        train_mse = mean_squared_error(y_train, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        test_mse = mean_squared_error(y_test, test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"Training Results:")
        print(f"  MSE: {train_mse:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  R²: {train_r2:.4f}")
        
        print(f"Testing Results:")
        print(f"  MSE: {test_mse:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  R²: {test_r2:.4f}")
        
        # Overfitting analysis
        overfitting_mse = test_mse - train_mse
        overfitting_r2 = train_r2 - test_r2
        
        print(f"Overfitting Analysis:")
        print(f"  MSE Difference (Test - Train): {overfitting_mse:.4f}")
        print(f"  R² Difference (Train - Test): {overfitting_r2:.4f}")
        
        return {
            'train_mse': train_mse, 'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
            'test_mse': test_mse, 'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
            'overfitting_mse': overfitting_mse, 'overfitting_r2': overfitting_r2
        }
    
    else:
        # Classification metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred, average='weighted')
        train_recall = recall_score(y_train, train_pred, average='weighted')
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        
        test_accuracy = accuracy_score(y_test, test_pred)
        test_precision = precision_score(y_test, test_pred, average='weighted')
        test_recall = recall_score(y_test, test_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        print(f"Training Results:")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  F1-Score: {train_f1:.4f}")
        
        print(f"Testing Results:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        
        # Overfitting analysis
        overfitting_accuracy = train_accuracy - test_accuracy
        overfitting_f1 = train_f1 - test_f1
        
        print(f"Overfitting Analysis:")
        print(f"  Accuracy Difference (Train - Test): {overfitting_accuracy:.4f}")
        print(f"  F1-Score Difference (Train - Test): {overfitting_f1:.4f}")
        
        return {
            'train_accuracy': train_accuracy, 'train_precision': train_precision, 
            'train_recall': train_recall, 'train_f1': train_f1,
            'test_accuracy': test_accuracy, 'test_precision': test_precision, 
            'test_recall': test_recall, 'test_f1': test_f1,
            'overfitting_accuracy': overfitting_accuracy, 'overfitting_f1': overfitting_f1
        }

def cross_validation_analysis(model, X, y, model_name, target_type='regression'):
    """Cross-validation analysis"""
    print(f"\n=== {model_name} CROSS-VALIDATION ANALYSIS ===")
    
    if target_type == 'regression':
        # Regression cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        print(f"Cross-Validation Results (5-fold):")
        print(f"  RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        print(f"  R²: {cv_r2.mean():.4f} (+/- {cv_r2.std() * 2:.4f})")
        
        return {'cv_rmse_mean': cv_rmse.mean(), 'cv_rmse_std': cv_rmse.std(),
                'cv_r2_mean': cv_r2.mean(), 'cv_r2_std': cv_r2.std()}
    
    else:
        # Classification cross-validation
        cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        
        print(f"Cross-Validation Results (5-fold):")
        print(f"  Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
        print(f"  F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
        
        return {'cv_accuracy_mean': cv_accuracy.mean(), 'cv_accuracy_std': cv_accuracy.std(),
                'cv_f1_mean': cv_f1.mean(), 'cv_f1_std': cv_f1.std()}

def create_learning_curves(model, X, y, model_name, target_type='regression'):
    """Create learning curves to analyze model performance"""
    print(f"Creating learning curves for {model_name}...")
    
    if target_type == 'regression':
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        train_scores = np.sqrt(-train_scores)
        val_scores = np.sqrt(-val_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training RMSE', color='#FF6B6B')
        plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation RMSE', color='#4ECDC4')
        plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='#FF6B6B')
        plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='#4ECDC4')
        
        plt.title(f'{model_name} Learning Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    else:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Accuracy', color='#FF6B6B')
        plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Accuracy', color='#4ECDC4')
        plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='#FF6B6B')
        plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='#4ECDC4')
        
        plt.title(f'{model_name} Learning Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_learning_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_comprehensive_comparison(svm_metrics, knn_metrics, target_type='regression'):
    """Create comprehensive model comparison"""
    print("Creating comprehensive model comparison...")
    
    if target_type == 'regression':
        # Regression comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        models = ['SVM', 'KNN']
        
        # Training vs Testing RMSE
        train_rmse = [svm_metrics['train_rmse'], knn_metrics['train_rmse']]
        test_rmse = [svm_metrics['test_rmse'], knn_metrics['test_rmse']]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_rmse, width, label='Training', color='#FF6B6B', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_rmse, width, label='Testing', color='#4ECDC4', alpha=0.8)
        axes[0, 0].set_title('RMSE: Training vs Testing', fontweight='bold')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training vs Testing R²
        train_r2 = [svm_metrics['train_r2'], knn_metrics['train_r2']]
        test_r2 = [svm_metrics['test_r2'], knn_metrics['test_r2']]
        
        axes[0, 1].bar(x - width/2, train_r2, width, label='Training', color='#FF6B6B', alpha=0.8)
        axes[0, 1].bar(x + width/2, test_r2, width, label='Testing', color='#4ECDC4', alpha=0.8)
        axes[0, 1].set_title('R² Score: Training vs Testing', fontweight='bold')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overfitting Analysis
        overfitting_r2 = [svm_metrics['overfitting_r2'], knn_metrics['overfitting_r2']]
        axes[0, 2].bar(models, overfitting_r2, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0, 2].set_title('Overfitting Analysis (R² Difference)', fontweight='bold')
        axes[0, 2].set_ylabel('R² Difference (Train - Test)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Cross-validation comparison
        cv_rmse = [svm_metrics.get('cv_rmse_mean', 0), knn_metrics.get('cv_rmse_mean', 0)]
        cv_std = [svm_metrics.get('cv_rmse_std', 0), knn_metrics.get('cv_rmse_std', 0)]
        
        axes[1, 0].bar(models, cv_rmse, yerr=cv_std, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, capsize=5)
        axes[1, 0].set_title('Cross-Validation RMSE', fontweight='bold')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model Performance Summary
        performance_data = {
            'Metric': ['Training RMSE', 'Testing RMSE', 'Training R²', 'Testing R²'],
            'SVM': [svm_metrics['train_rmse'], svm_metrics['test_rmse'], 
                   svm_metrics['train_r2'], svm_metrics['test_r2']],
            'KNN': [knn_metrics['train_rmse'], knn_metrics['test_rmse'], 
                   knn_metrics['train_r2'], knn_metrics['test_r2']]
        }
        
        # Create a table-like visualization
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table_data = []
        for i, metric in enumerate(performance_data['Metric']):
            table_data.append([metric, f"{performance_data['SVM'][i]:.4f}", f"{performance_data['KNN'][i]:.4f}"])
        
        table = axes[1, 1].table(cellText=table_data, colLabels=['Metric', 'SVM', 'KNN'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Performance Summary', fontweight='bold', pad=20)
        
        # Best Model Recommendation
        svm_test_r2 = svm_metrics['test_r2']
        knn_test_r2 = knn_metrics['test_r2']
        best_model = 'SVM' if svm_test_r2 > knn_test_r2 else 'KNN'
        best_score = max(svm_test_r2, knn_test_r2)
        
        axes[1, 2].text(0.5, 0.7, f'Best Model: {best_model}', ha='center', va='center', 
                       fontsize=16, fontweight='bold', transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.5, 0.5, f'R² Score: {best_score:.4f}', ha='center', va='center', 
                       fontsize=14, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.5, 0.3, f'Recommendation: Use {best_model} for production', 
                       ha='center', va='center', fontsize=12, 
                       transform=axes[1, 2].transAxes, style='italic')
        axes[1, 2].set_title('Model Recommendation', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_regression_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    else:
        # Classification comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        models = ['SVM', 'KNN']
        
        # Training vs Testing Accuracy
        train_accuracy = [svm_metrics['train_accuracy'], knn_metrics['train_accuracy']]
        test_accuracy = [svm_metrics['test_accuracy'], knn_metrics['test_accuracy']]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_accuracy, width, label='Training', color='#FF6B6B', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_accuracy, width, label='Testing', color='#4ECDC4', alpha=0.8)
        axes[0, 0].set_title('Accuracy: Training vs Testing', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training vs Testing F1-Score
        train_f1 = [svm_metrics['train_f1'], knn_metrics['train_f1']]
        test_f1 = [svm_metrics['test_f1'], knn_metrics['test_f1']]
        
        axes[0, 1].bar(x - width/2, train_f1, width, label='Training', color='#FF6B6B', alpha=0.8)
        axes[0, 1].bar(x + width/2, test_f1, width, label='Testing', color='#4ECDC4', alpha=0.8)
        axes[0, 1].set_title('F1-Score: Training vs Testing', fontweight='bold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overfitting Analysis
        overfitting_f1 = [svm_metrics['overfitting_f1'], knn_metrics['overfitting_f1']]
        axes[0, 2].bar(models, overfitting_f1, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0, 2].set_title('Overfitting Analysis (F1 Difference)', fontweight='bold')
        axes[0, 2].set_ylabel('F1 Difference (Train - Test)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Cross-validation comparison
        cv_accuracy = [svm_metrics.get('cv_accuracy_mean', 0), knn_metrics.get('cv_accuracy_mean', 0)]
        cv_std = [svm_metrics.get('cv_accuracy_std', 0), knn_metrics.get('cv_accuracy_std', 0)]
        
        axes[1, 0].bar(models, cv_accuracy, yerr=cv_std, color=['#FF6B6B', '#4ECDC4'], alpha=0.8, capsize=5)
        axes[1, 0].set_title('Cross-Validation Accuracy', fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model Performance Summary
        performance_data = {
            'Metric': ['Training Accuracy', 'Testing Accuracy', 'Training F1', 'Testing F1'],
            'SVM': [svm_metrics['train_accuracy'], svm_metrics['test_accuracy'], 
                   svm_metrics['train_f1'], svm_metrics['test_f1']],
            'KNN': [knn_metrics['train_accuracy'], knn_metrics['test_accuracy'], 
                   knn_metrics['train_f1'], knn_metrics['test_f1']]
        }
        
        # Create a table-like visualization
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table_data = []
        for i, metric in enumerate(performance_data['Metric']):
            table_data.append([metric, f"{performance_data['SVM'][i]:.4f}", f"{performance_data['KNN'][i]:.4f}"])
        
        table = axes[1, 1].table(cellText=table_data, colLabels=['Metric', 'SVM', 'KNN'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Performance Summary', fontweight='bold', pad=20)
        
        # Best Model Recommendation
        svm_test_f1 = svm_metrics['test_f1']
        knn_test_f1 = knn_metrics['test_f1']
        best_model = 'SVM' if svm_test_f1 > knn_test_f1 else 'KNN'
        best_score = max(svm_test_f1, knn_test_f1)
        
        axes[1, 2].text(0.5, 0.7, f'Best Model: {best_model}', ha='center', va='center', 
                       fontsize=16, fontweight='bold', transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.5, 0.5, f'F1-Score: {best_score:.4f}', ha='center', va='center', 
                       fontsize=14, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.5, 0.3, f'Recommendation: Use {best_model} for production', 
                       ha='center', va='center', fontsize=12, 
                       transform=axes[1, 2].transAxes, style='italic')
        axes[1, 2].set_title('Model Recommendation', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function for comprehensive results analysis"""
    print("=== COMPREHENSIVE RESULTS ANALYSIS ===")
    print("Analyzing SVM and KNN models with detailed performance metrics")
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
        print(f"\n{'='*20} {target_type.upper()} ANALYSIS {'='*20}")
        
        # Prepare data
        X, y, scaler, feature_columns = prepare_data(df, target_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train models
        if target_type == 'regression':
            svm_model = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
            knn_model = KNeighborsRegressor(n_neighbors=7, weights='distance')
        else:
            svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
            knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance')
        
        # Train models
        print(f"\nTraining SVM for {target_type}...")
        svm_model.fit(X_train, y_train)
        
        print(f"Training KNN for {target_type}...")
        knn_model.fit(X_train, y_train)
        
        # Analyze performance
        svm_metrics = analyze_model_performance(svm_model, X_train, y_train, X_test, y_test, 'SVM', target_type)
        knn_metrics = analyze_model_performance(knn_model, X_train, y_train, X_test, y_test, 'KNN', target_type)
        
        # Cross-validation analysis
        svm_cv = cross_validation_analysis(svm_model, X, y, 'SVM', target_type)
        knn_cv = cross_validation_analysis(knn_model, X, y, 'KNN', target_type)
        
        # Add cross-validation results to metrics
        svm_metrics.update(svm_cv)
        knn_metrics.update(knn_cv)
        
        # Create learning curves
        create_learning_curves(svm_model, X, y, 'SVM', target_type)
        create_learning_curves(knn_model, X, y, 'KNN', target_type)
        
        # Create comprehensive comparison
        create_comprehensive_comparison(svm_metrics, knn_metrics, target_type)
        
        # Print final summary
        print(f"\n--- {target_type.upper()} FINAL SUMMARY ---")
        if target_type == 'regression':
            print(f"SVM - Test RMSE: {svm_metrics['test_rmse']:.4f}, Test R²: {svm_metrics['test_r2']:.4f}")
            print(f"KNN - Test RMSE: {knn_metrics['test_rmse']:.4f}, Test R²: {knn_metrics['test_r2']:.4f}")
        else:
            print(f"SVM - Test Accuracy: {svm_metrics['test_accuracy']:.4f}, Test F1: {svm_metrics['test_f1']:.4f}")
            print(f"KNN - Test Accuracy: {knn_metrics['test_accuracy']:.4f}, Test F1: {knn_metrics['test_f1']:.4f}")
    
    print("\n=== COMPREHENSIVE ANALYSIS COMPLETED ===")
    print("Generated files:")
    print("- svm_learning_curve.png")
    print("- knn_learning_curve.png")
    print("- comprehensive_regression_analysis.png")
    print("- comprehensive_classification_analysis.png")
    print("\nAll analyses completed successfully!")

if __name__ == "__main__":
    main()
