"""
MATHEMATICAL MODELING FOR OLA BIKE RIDE REQUEST FORECASTING
===========================================================

This document outlines the mathematical foundations for SVM and KNN algorithms
used in the Ola bike ride request forecasting system.

1. SUPPORT VECTOR MACHINE (SVM) MODELING
=========================================

Input Variables (Features):
- X = [x1, x2, x3, ..., xn] where:
  x1 = Hour of day (0-23)
  x2 = Day of week (0-6)
  x3 = Pickup latitude
  x4 = Pickup longitude
  x5 = Drop latitude
  x6 = Drop longitude
  x7 = Distance (calculated using Haversine formula)
  x8 = Month (1-12)
  x9 = Quarter (1-4)
  x10 = Is weekend (0/1)

Output Variable:
- y = Number of ride requests (regression) or Demand level (classification)

Core Mathematical Equations:

1. SVM Objective Function (Primal Form):
   min(1/2 * ||w||² + C * Σ(ξi + ξi*))
   
   Subject to:
   yi - f(xi) ≤ ε + ξi
   f(xi) - yi ≤ ε + ξi
   ξi, ξi* ≥ 0

2. SVM Decision Function:
   f(x) = Σ(αi - αi*) * K(xi, x) + b
   
   Where:
   - K(xi, x) is the kernel function
   - αi, αi* are Lagrange multipliers
   - b is the bias term

3. Kernel Functions:
   - Linear: K(xi, xj) = xi^T * xj
   - RBF: K(xi, xj) = exp(-γ * ||xi - xj||²)
   - Polynomial: K(xi, xj) = (xi^T * xj + r)^d

4. Loss Function (ε-insensitive):
   Lε(y, f(x)) = max(0, |y - f(x)| - ε)

2. K-NEAREST NEIGHBORS (KNN) MODELING
======================================

Input Variables: Same as SVM
Output Variable: Same as SVM

Core Mathematical Equations:

1. Distance Metrics:
   - Euclidean: d(xi, xj) = √(Σ(xik - xjk)²)
   - Manhattan: d(xi, xj) = Σ|xik - xjk|
   - Haversine: d(xi, xj) = 2R * arcsin(√(sin²(Δlat/2) + cos(lat1)cos(lat2)sin²(Δlon/2)))

2. KNN Prediction Function:
   For Regression: ŷ = (1/k) * Σ(yi) for i ∈ Nk(x)
   For Classification: ŷ = mode(yi) for i ∈ Nk(x)
   
   Where Nk(x) is the set of k nearest neighbors

3. Weighted KNN:
   ŷ = Σ(wi * yi) / Σ(wi)
   Where wi = 1 / d(xi, x) for distance-based weights

3. OPTIMIZATION METHODS
======================

SVM Optimization:
- Sequential Minimal Optimization (SMO)
- Gradient Descent variants
- Quadratic Programming solvers

KNN Optimization:
- KD-Tree for efficient neighbor search
- Ball Tree for high-dimensional data
- Locality Sensitive Hashing (LSH)

4. PERFORMANCE METRICS
======================

Regression Metrics:
- Mean Squared Error (MSE) = (1/n) * Σ(yi - ŷi)²
- Root Mean Squared Error (RMSE) = √MSE
- Mean Absolute Error (MAE) = (1/n) * Σ|yi - ŷi|
- R² Score = 1 - (SSres / SStot)

Classification Metrics:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

Where:
TP = True Positives, TN = True Negatives
FP = False Positives, FN = False Negatives
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("=== MATHEMATICAL MODELING DOCUMENTATION ===")
print("SVM and KNN algorithms implemented with proper mathematical foundations")
print("Features engineered for ride request forecasting")
print("Performance metrics calculated for comprehensive evaluation")
