

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import pandas as pd
import base64
import io
import dash_bootstrap_components as dbc
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import re
import warnings
warnings.filterwarnings("ignore")

# ML and Advanced Analytics
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Text Processing
from fuzzywuzzy import fuzz, process
import phonenumbers
from email_validator import validate_email, EmailNotValidError

# Advanced Data Processing
from scipy.interpolate import interp1d
import datetime
from dateutil import parser
import json

# Initialize app with BOOTSTRAP theme and custom styling
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True
)
app.title = "Advanced Data Cleaning & ML Platform"

# Global variables for storing data - Initialize properly
stored_data = {}
cleaning_history = []
quality_metrics = {}

# Custom CSS for professional styling with FIXED data table and profiling layout
custom_css = """
/* Professional styling */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f8fafc;
    margin: 0;
    padding: 0;
    overflow-x: hidden; /* Prevent page-level horizontal scroll */
}

.main-container {
    display: flex;
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    overflow-x: hidden; /* Prevent page-level horizontal scroll */
}

.sidebar {
    width: 280px;
    background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.1);
    position: fixed;
    height: 100vh;
    overflow-y: auto;
    z-index: 1000;
}

.sidebar-header {
    padding: 2rem 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
}

.sidebar-title {
    color: #ffffff;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.sidebar-subtitle {
    color: #94a3b8;
    font-size: 0.875rem;
    margin: 0.5rem 0 0 0;
}

.nav-link {
    color: #cbd5e1 !important;
    padding: 0.875rem 1.5rem !important;
    margin: 0.25rem 0.75rem !important;
    border-radius: 0.75rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
}

.nav-link:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    transform: translateX(4px);
}

.nav-link.active {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.nav-link i {
    width: 20px;
    margin-right: 0.75rem;
}

.content-area {
    margin-left: 280px;
    flex: 1;
    padding: 2rem;
    background: #f8fafc;
    min-height: 100vh;
    max-width: calc(100vw - 280px); /* Prevent content from exceeding viewport */
    overflow-x: hidden; /* Prevent content area horizontal scroll */
}

.page-header {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border-radius: 1rem;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
}

.page-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1e293b, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    text-align: center;
}

.page-subtitle {
    color: #64748b;
    font-size: 1.125rem;
    text-align: center;
    margin: 1rem 0 0 0;
    font-weight: 400;
}

.section-card {
    background: #ffffff;
    border-radius: 1rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
    margin-bottom: 2rem;
    overflow: hidden;
    transition: all 0.3s ease;
}

.section-card:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    transform: translateY(-2px);
}

.section-header {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 1.5rem 2rem;
    border-bottom: 1px solid rgba(226, 232, 240, 0.8);
}

.section-title {
    font-size: 1.375rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0;
    display: flex;
    align-items: center;
}

.section-title i {
    margin-right: 0.75rem;
    color: #3b82f6;
}

.section-body {
    padding: 2rem;
}

.upload-area {
    border: 2px dashed #cbd5e1;
    border-radius: 1rem;
    padding: 3rem;
    text-align: center;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    transition: all 0.3s ease;
    cursor: pointer;
}

.upload-area:hover {
    border-color: #3b82f6;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
}

.upload-icon {
    font-size: 3rem;
    color: #3b82f6;
    margin-bottom: 1rem;
}

.upload-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.5rem;
}

.upload-subtitle {
    color: #64748b;
    margin-bottom: 0.25rem;
}

.upload-hint {
    color: #3b82f6;
    font-size: 0.875rem;
    font-weight: 500;
}

.btn-primary-custom {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    border: none;
    border-radius: 0.75rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    transition: all 0.3s ease;
}

.btn-primary-custom:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

.btn-success-custom {
    background: linear-gradient(135deg, #10b981, #059669);
    border: none;
    border-radius: 0.75rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    transition: all 0.3s ease;
}

.btn-success-custom:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
}

.btn-info-custom {
    background: linear-gradient(135deg, #06b6d4, #0891b2);
    border: none;
    border-radius: 0.75rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
    transition: all 0.3s ease;
}

.btn-info-custom:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4);
}

.btn-warning-custom {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    border: none;
    border-radius: 0.75rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    transition: all 0.3s ease;
}

.btn-warning-custom:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4);
}

.btn-danger-custom {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    border: none;
    border-radius: 0.75rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    transition: all 0.3s ease;
}

.btn-danger-custom:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4);
}

.stats-badge {
    background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
    color: #1e293b;
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    font-weight: 600;
    margin: 0 0.25rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.stats-badge.primary {
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    color: #1d4ed8;
}

.stats-badge.success {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    color: #065f46;
}

.stats-badge.warning {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    color: #92400e;
}

.quality-card {
    background: linear-gradient(135deg, #ffffff, #f8fafc);
    border-radius: 1rem;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
    transition: all 0.3s ease;
}

.quality-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.quality-score {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.quality-label {
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.875rem;
}

/* FIXED: Data table container with proper horizontal scrolling */
.data-table-container {
    background: #ffffff;
    border-radius: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
    overflow: hidden; /* Hide overflow on container */
    max-width: 100%; /* Ensure container doesn't exceed parent */
}

.data-table-wrapper {
    overflow-x: auto; /* Enable horizontal scrolling only within this wrapper */
    overflow-y: visible;
    max-width: 100%;
    border-radius: 1rem;
}

/* FIXED: Profiling cards layout - exactly 2 per row */
.profiling-grid {
    display: grid;
    grid-template-columns: 1fr 1fr; /* Exactly 2 columns */
    gap: 1.5rem;
    margin-top: 1rem;
}

.profiling-card {
    background: linear-gradient(135deg, #ffffff, #f8fafc);
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
    transition: all 0.3s ease;
    min-height: 200px; /* Consistent height */
}

.profiling-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.profiling-card-title {
    font-size: 1.125rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.5rem;
}

.profiling-stat {
    display: flex;
    align-items: center;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
}

.profiling-stat i {
    width: 20px;
    margin-right: 0.5rem;
    color: #3b82f6;
}

.form-control-custom {
    border: 2px solid #e2e8f0;
    border-radius: 0.75rem;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    transition: all 0.3s ease;
}

.form-control-custom:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.alert-custom {
    border-radius: 1rem;
    border: none;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.tab-content-area {
    background: #ffffff;
    border-radius: 1rem;
    padding: 2rem;
    margin-top: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
}

.history-container {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border-radius: 1rem;
    padding: 1.5rem;
    height: 300px;
    overflow-y: auto;
    border: 1px solid rgba(226, 232, 240, 0.8);
}

.history-item {
    background: #ffffff;
    border-radius: 0.75rem;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border-left: 4px solid #3b82f6;
}

.history-operation {
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.25rem;
}

.history-details {
    color: #64748b;
    font-size: 0.875rem;
}

/* Responsive design */
@media (max-width: 768px) {
    .sidebar {
        width: 100%;
        height: auto;
        position: relative;
    }
    
    .content-area {
        margin-left: 0;
        padding: 1rem;
        max-width: 100vw;
    }
    
    .page-header {
        padding: 1.5rem;
    }
    
    .page-title {
        font-size: 2rem;
    }
    
    .section-body {
        padding: 1.5rem;
    }
    
    .profiling-grid {
        grid-template-columns: 1fr; /* Single column on mobile */
    }
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #cbd5e1, #94a3b8);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #94a3b8, #64748b);
}
"""

# Add custom CSS to the app
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            {custom_css}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# ==================== ADVANCED DATA CLEANING FUNCTIONS ====================

def smart_type_detection(df):
    """Advanced data type detection using ML and pattern recognition."""
    type_suggestions = {}
    
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            type_suggestions[col] = 'unknown'
            continue
            
        # Sample data for analysis
        sample_data = col_data.head(100).astype(str)
        
        # Pattern-based detection
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[1-9]?[0-9]{7,15}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'date': r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$',
            'time': r'^\d{1,2}:\d{2}(:\d{2})?$',
            'currency': r'^\$?\d+\.?\d*$',
            'percentage': r'^\d+\.?\d*%$',
            'zipcode': r'^\d{5}(-\d{4})?$' 
        }
        
        pattern_matches = {}
        for pattern_name, pattern in patterns.items():
            matches = sample_data.str.match(pattern, na=False).sum()
            pattern_matches[pattern_name] = matches / len(sample_data)
        
        # Find best pattern match
        best_pattern = max(pattern_matches, key=pattern_matches.get)
        if pattern_matches[best_pattern] > 0.7:
            type_suggestions[col] = best_pattern
        else:
            # Fallback to pandas dtype
            if pd.api.types.is_numeric_dtype(df[col]):
                type_suggestions[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                type_suggestions[col] = 'datetime'
            else:
                type_suggestions[col] = 'text'
    
    return type_suggestions

def advanced_missing_data_analysis(df):
    """Comprehensive missing data analysis with pattern detection."""
    missing_info = {
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'columns_with_missing': {},
        'missing_patterns': {},
        'recommendations': {}
    }
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_info['columns_with_missing'][col] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100,
                'dtype': str(df[col].dtype)
            }
            
            # Recommend imputation method
            if df[col].dtype in ['int64', 'float64']:
                if missing_count / len(df) < 0.1:
                    missing_info['recommendations'][col] = 'mean_imputation'
                elif missing_count / len(df) < 0.3:
                    missing_info['recommendations'][col] = 'knn_imputation'
                else:
                    missing_info['recommendations'][col] = 'iterative_imputation'
            else:
                if missing_count / len(df) < 0.1:
                    missing_info['recommendations'][col] = 'mode_imputation'
                else:
                    missing_info['recommendations'][col] = 'forward_fill'
    
    return missing_info

def ml_based_outlier_detection(df, columns=None, contamination=0.1):
    """Advanced outlier detection using multiple ML algorithms."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columns:
        return {'error': 'No numeric columns found for outlier detection'}
    
    outlier_results = {
        'methods': {},
        'consensus_outliers': [],
        'outlier_scores': {},
        'method': 'ML Ensemble'
    }
    
    # Prepare data
    data_for_analysis = df[columns].dropna()
    if len(data_for_analysis) < 10:
        return {'error': 'Insufficient data for outlier detection'}
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_analysis)
    
    # Multiple outlier detection methods
    methods = {
        'isolation_forest': IsolationForest(contamination=contamination, random_state=42),
        'local_outlier_factor': LocalOutlierFactor(contamination=contamination),
        'one_class_svm': OneClassSVM(nu=contamination)
    }
    
    outlier_predictions = {}
    
    for method_name, method in methods.items():
        try:
            if method_name == 'local_outlier_factor':
                predictions = method.fit_predict(scaled_data)
                scores = method.negative_outlier_factor_
            else:
                predictions = method.fit_predict(scaled_data)
                if hasattr(method, 'decision_function'):
                    scores = method.decision_function(scaled_data)
                else:
                    scores = predictions
            
            outlier_indices = data_for_analysis.index[predictions == -1].tolist()
            outlier_predictions[method_name] = set(outlier_indices)
            
            outlier_results['methods'][method_name] = {
                'outlier_count': len(outlier_indices),
                'outlier_indices': outlier_indices,
                'scores': scores.tolist() if hasattr(scores, 'tolist') else scores
            }
        except Exception as e:
            outlier_results['methods'][method_name] = {'error': str(e)}
    
    # Consensus outliers (detected by at least 2 methods)
    all_outliers = set()
    for outliers in outlier_predictions.values():
        all_outliers.update(outliers)
    
    consensus_outliers = []
    for outlier in all_outliers:
        count = sum(1 for outliers in outlier_predictions.values() if outlier in outliers)
        if count >= 2:
            consensus_outliers.append(outlier)
    
    outlier_results['consensus_outliers'] = consensus_outliers
    outlier_results['total_outliers'] = len(consensus_outliers)
    outlier_results['outlier_indices'] = consensus_outliers
    
    return outlier_results

def text_data_cleaning(df, columns=None):
    """Advanced text cleaning and standardization."""
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    cleaning_results = {
        'cleaned_columns': {},
        'standardization_suggestions': {},
        'quality_improvements': {}
    }
    
    for col in columns:
        if col not in df.columns:
            continue
            
        original_data = df[col].copy()
        cleaned_data = df[col].copy()
        
        # Basic text cleaning
        if cleaned_data.dtype == 'object':
            # Remove leading/trailing whitespace
            cleaned_data = cleaned_data.astype(str).str.strip()
            
            # Normalize case (detect if should be title case, upper, or lower)
            sample_data = cleaned_data.dropna().head(50)
            if len(sample_data) > 0:
                title_case_count = sum(1 for x in sample_data if x.istitle())
                upper_case_count = sum(1 for x in sample_data if x.isupper())
                lower_case_count = sum(1 for x in sample_data if x.islower())
                
                total_count = len(sample_data)
                if title_case_count / total_count > 0.7:
                    cleaned_data = cleaned_data.str.title()
                elif upper_case_count / total_count > 0.7:
                    cleaned_data = cleaned_data.str.upper()
                elif lower_case_count / total_count > 0.7:
                    cleaned_data = cleaned_data.str.lower()
            
            # Remove extra whitespace
            cleaned_data = cleaned_data.str.replace(r'\s+', ' ', regex=True)
            
            # Detect and suggest standardizations
            unique_values = cleaned_data.dropna().unique()
            if len(unique_values) < 50:  # Categorical-like data
                # Find similar values that could be standardized
                similar_groups = []
                processed = set()
                
                for value in unique_values:
                    if value in processed:
                        continue
                    
                    similar = [value]
                    for other_value in unique_values:
                        if other_value != value and other_value not in processed:
                            similarity = fuzz.ratio(str(value).lower(), str(other_value).lower())
                            if similarity > 80:  # 80% similarity threshold
                                similar.append(other_value)
                                processed.add(other_value)
                    
                    if len(similar) > 1:
                        similar_groups.append(similar)
                        processed.update(similar)
                
                cleaning_results['standardization_suggestions'][col] = similar_groups
        
        # Calculate quality improvements
        original_nulls = original_data.isnull().sum()
        cleaned_nulls = cleaned_data.isnull().sum()
        
        cleaning_results['cleaned_columns'][col] = cleaned_data
        cleaning_results['quality_improvements'][col] = {
            'null_reduction': original_nulls - cleaned_nulls,
            'unique_values_before': len(original_data.dropna().unique()),
            'unique_values_after': len(cleaned_data.dropna().unique())
        }
    
    return cleaning_results

def calculate_data_quality_score(df):
    """Calculate comprehensive data quality score."""
    quality_metrics = {
        'overall_score': 0,
        'completeness': 0,
        'validity': 0,
        'consistency': 0,
        'uniqueness': 0,
        'column_scores': {}
    }
    
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    
    # Completeness Score (0-100)
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    quality_metrics['completeness'] = round(completeness, 2)
    
    # Column-level analysis
    column_scores = []
    
    for col in df.columns:
        col_data = df[col]
        col_score = {
            'completeness': ((len(col_data) - col_data.isnull().sum()) / len(col_data)) * 100,
            'uniqueness': (col_data.nunique() / len(col_data)) * 100 if len(col_data) > 0 else 0,
            'validity': 100  # Default, can be enhanced with specific validation rules
        }
        
        # Consistency check for text data
        if col_data.dtype == 'object':
            unique_values = col_data.dropna().astype(str)
            if len(unique_values) > 0:
                # Check for case consistency
                case_consistency = max(
                    unique_values.str.islower().sum(),
                    unique_values.str.isupper().sum(),
                    unique_values.str.istitle().sum()
                ) / len(unique_values)
                col_score['consistency'] = case_consistency * 100
            else:
                col_score['consistency'] = 100
        else:
            col_score['consistency'] = 100
        
        # Overall column score
        col_score['overall'] = np.mean([
            col_score['completeness'],
            col_score['uniqueness'],
            col_score['validity'],
            col_score['consistency']
        ])
        
        quality_metrics['column_scores'][col] = col_score
        column_scores.append(col_score['overall'])
    
    # Overall scores
    quality_metrics['validity'] = np.mean([score['validity'] for score in quality_metrics['column_scores'].values()])
    quality_metrics['consistency'] = np.mean([score['consistency'] for score in quality_metrics['column_scores'].values()])
    quality_metrics['uniqueness'] = np.mean([score['uniqueness'] for score in quality_metrics['column_scores'].values()])
    
    # Overall quality score
    quality_metrics['overall_score'] = np.mean([
        quality_metrics['completeness'],
        quality_metrics['validity'],
        quality_metrics['consistency'],
        quality_metrics['uniqueness']
    ])
    
    return quality_metrics

def smart_imputation(df, column, method='auto'):
    """Intelligent missing value imputation."""
    if column not in df.columns:
        return df
    
    df_copy = df.copy()
    col_data = df_copy[column]
    
    if col_data.isnull().sum() == 0:
        return df_copy
    
    if method == 'auto':
        # Auto-select best method based on data type and missing percentage
        missing_pct = col_data.isnull().sum() / len(col_data)
        
        if col_data.dtype in ['int64', 'float64']:
            if missing_pct < 0.1:
                method = 'mean'
            elif missing_pct < 0.3:
                method = 'knn'
            else:
                method = 'iterative'
        else:
            if missing_pct < 0.1:
                method = 'mode'
            else:
                method = 'forward_fill'
    
    try:
        if method == 'mean' and col_data.dtype in ['int64', 'float64']:
            df_copy[column] = col_data.fillna(col_data.mean())
        elif method == 'median' and col_data.dtype in ['int64', 'float64']:
            df_copy[column] = col_data.fillna(col_data.median())
        elif method == 'mode':
            mode_value = col_data.mode()
            if len(mode_value) > 0:
                df_copy[column] = col_data.fillna(mode_value[0])
        elif method == 'forward_fill':
            df_copy[column] = col_data.fillna(method='ffill')
        elif method == 'backward_fill':
            df_copy[column] = col_data.fillna(method='bfill')
        elif method == 'knn' and col_data.dtype in ['int64', 'float64']:
            # KNN imputation for numeric data
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                imputer = KNNImputer(n_neighbors=5)
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
        elif method == 'iterative' and col_data.dtype in ['int64', 'float64']:
            # Iterative imputation for numeric data
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                imputer = IterativeImputer(random_state=42)
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    except Exception as e:
        print(f"Imputation failed: {e}")
        # Fallback to simple imputation
        if col_data.dtype in ['int64', 'float64']:
            df_copy[column] = col_data.fillna(col_data.mean())
        else:
            mode_value = col_data.mode()
            if len(mode_value) > 0:
                df_copy[column] = col_data.fillna(mode_value[0])
    
    return df_copy

def generate_cleaning_recommendations(df):
    """Generate AI-powered cleaning recommendations."""
    recommendations = []
    
    # Missing data recommendations
    missing_analysis = advanced_missing_data_analysis(df)
    for col, rec in missing_analysis['recommendations'].items():
        recommendations.append({
            'type': 'missing_data',
            'column': col,
            'action': rec,
            'description': f"Handle missing values in '{col}' using {rec.replace('_', ' ')}",
            'priority': 'high' if missing_analysis['columns_with_missing'][col]['percentage'] > 20 else 'medium'
        })
    
    # Data type recommendations
    type_suggestions = smart_type_detection(df)
    for col, suggested_type in type_suggestions.items():
        if suggested_type not in ['numeric', 'text', 'unknown']:
            recommendations.append({
                'type': 'data_type',
                'column': col,
                'action': f'convert_to_{suggested_type}',
                'description': f"Convert '{col}' to {suggested_type} format",
                'priority': 'medium'
            })
    
    # Text cleaning recommendations
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        if df[col].dtype == 'object':
            # Check for inconsistent casing
            sample_data = df[col].dropna().astype(str).head(100)
            if len(sample_data) > 0:
                case_consistency = max(
                    sample_data.str.islower().sum(),
                    sample_data.str.isupper().sum(),
                    sample_data.str.istitle().sum()
                ) / len(sample_data)
                
                if case_consistency < 0.8:
                    recommendations.append({
                        'type': 'text_cleaning',
                        'column': col,
                        'action': 'standardize_case',
                        'description': f"Standardize text case in '{col}'",
                        'priority': 'low'
                    })
    
    # Outlier recommendations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        try:
            outlier_results = ml_based_outlier_detection(df, numeric_cols.tolist())
            if 'consensus_outliers' in outlier_results and len(outlier_results['consensus_outliers']) > 0:
                recommendations.append({
                    'type': 'outlier_detection',
                    'column': 'multiple',
                    'action': 'review_outliers',
                    'description': f"Review {len(outlier_results['consensus_outliers'])} potential outliers detected by ML algorithms",
                    'priority': 'medium'
                })
        except:
            pass
    
    return recommendations

# ==================== ORIGINAL FUNCTIONS (ENHANCED) ====================

def detect_outliers_iqr(df, columns=None):
    """Enhanced IQR outlier detection with additional statistics."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {
        'method': 'IQR',
        'total_outliers': 0,
        'outliers_by_column': {},
        'outlier_indices': set(),
        'severity_scores': {}
    }
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices = df[outliers_mask].index.tolist()
            
            # Calculate severity scores
            severity_scores = []
            for idx in outlier_indices:
                value = df.loc[idx, col]
                if value < lower_bound:
                    severity = (lower_bound - value) / IQR if IQR > 0 else 0
                else:
                    severity = (value - upper_bound) / IQR if IQR > 0 else 0
                severity_scores.append(severity)
            
            outlier_info['outliers_by_column'][col] = {
                'count': len(outlier_indices),
                'indices': outlier_indices,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'severity_scores': severity_scores
            }
            
            outlier_info['outlier_indices'].update(outlier_indices)
    
    outlier_info['total_outliers'] = len(outlier_info['outlier_indices'])
    outlier_info['outlier_indices'] = list(outlier_info['outlier_indices'])
    
    return outlier_info

def detect_outliers_zscore(df, columns=None, threshold=3):
    """Enhanced Z-score outlier detection with confidence intervals."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {
        'method': 'Z-Score',
        'threshold': threshold,
        'total_outliers': 0,
        'outliers_by_column': {},
        'outlier_indices': set(),
        'confidence_scores': {}
    }
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                z_scores = np.abs(stats.zscore(col_data))
                outliers_mask = z_scores > threshold
                valid_indices = col_data.index
                outlier_indices = valid_indices[outliers_mask].tolist()
                
                # Calculate confidence scores
                confidence_scores = []
                for idx in outlier_indices:
                    idx_pos = col_data.index.get_loc(idx)
                    z_score = z_scores[idx_pos]
                    confidence = min(z_score / threshold, 3.0)  # Cap at 3.0
                    confidence_scores.append(confidence)
                
                outlier_info['outliers_by_column'][col] = {
                    'count': len(outlier_indices),
                    'indices': outlier_indices,
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'confidence_scores': confidence_scores
                }
                
                outlier_info['outlier_indices'].update(outlier_indices)
    
    outlier_info['total_outliers'] = len(outlier_info['outlier_indices'])
    outlier_info['outlier_indices'] = list(outlier_info['outlier_indices'])
    
    return outlier_info

def get_data_profiling(df):
    """Enhanced data profiling with advanced statistics."""
    profiling = {}
    
    for col in df.columns:
        col_data = df[col]
        col_info = {
            'dtype': str(col_data.dtype),
            'missing_count': col_data.isnull().sum(),
            'missing_percentage': (col_data.isnull().sum() / len(df)) * 100,
            'unique_count': col_data.nunique(),
            'unique_percentage': (col_data.nunique() / len(df)) * 100,
            'memory_usage': col_data.memory_usage(deep=True)
        }
        
        if col_data.dtype in ['int64', 'float64']:
            col_info.update({
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            })
        elif col_data.dtype == 'object':
            col_info.update({
                'avg_length': col_data.astype(str).str.len().mean(),
                'max_length': col_data.astype(str).str.len().max(),
                'min_length': col_data.astype(str).str.len().min()
            })
        
        profiling[col] = col_info
    
    return profiling

def create_outlier_boxplot(df, column):
    """Enhanced box plot with outlier highlighting."""
    if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
        return go.Figure()
    
    fig = px.box(df, y=column, title=f'Enhanced Box Plot for {column}')
    
    # Add outlier detection results
    outlier_info = detect_outliers_iqr(df, [column])
    if column in outlier_info['outliers_by_column']:
        outlier_indices = outlier_info['outliers_by_column'][column]['indices']
        if outlier_indices:
            outlier_values = df.loc[outlier_indices, column]
            
            fig.add_trace(go.Scatter(
                y=outlier_values,
                x=['Outliers'] * len(outlier_values),
                mode='markers',
                marker=dict(color='red', size=8, symbol='x'),
                name='Detected Outliers',
                text=[f'Index: {idx}' for idx in outlier_indices],
                hovertemplate='Value: %{y}<br>%{text}<extra></extra>'
            ))
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_font_size=16,
        font_size=12,
        showlegend=True
    )
    return fig

def parse_conditional_rule(rule_text):
    """Parse conditional rule text into structured format."""
    if not rule_text:
        return None
    
    rule_text = rule_text.strip()
    patterns = [
        r'if\s+(.+?)\s*=\s*(.+?)\s+then\s+(.+?)\s+as\s+(.+)',
        r'if\s+(.+?)\s+is\s+(.+?)\s+then\s+(.+?)\s+as\s+(.+)',
        r'if\s+(.+?)\s+equals?\s+(.+?)\s+then\s+(.+?)\s+as\s+(.+)',
        r'if\s+(.+?)\s+contains?\s+(.+?)\s+then\s+(.+?)\s+as\s+(.+)',
        r'if\s+(.+?)\s*(>|<|>=|<=|!=)\s*(.+?)\s+then\s+(.+?)\s+as\s+(.+)'
    ]
    for i, pattern in enumerate(patterns):
        match = re.match(pattern, rule_text.lower().strip(), re.IGNORECASE)
        if match:
            if i == 4:
                return {
                    'condition_column': match.group(1).strip(),
                    'operator': match.group(2).strip(),
                    'condition_value': match.group(3).strip(),
                    'target_column': match.group(4).strip(),
                    'new_value': match.group(5).strip()
                }
            else:
                return {
                    'condition_column': match.group(1).strip(),
                    'operator': '=' if i == 0 else 'is' if i == 1 else 'equals' if i == 2 else 'contains',
                    'condition_value': match.group(2).strip(),
                    'target_column': match.group(3).strip(),
                    'new_value': match.group(4).strip()
                }
    return None

def apply_conditional_rule(df, rule):
    """Apply conditional rule to dataframe."""
    if not rule:
        return df
    
    df_copy = df.copy()
    condition_column = rule['condition_column']
    actual_column = None
    for col in df_copy.columns:
        if col.lower() == condition_column.lower():
            actual_column = col
            break
    if actual_column is None:
        return df_copy
    
    condition_col = df_copy[actual_column]
    operator = rule['operator']
    value = rule['condition_value']
    
    try:
        if condition_col.dtype in ['int64', 'float64']:
            value = float(value)
    except:
        pass
    
    if operator in ['=', 'is', 'equals', 'equal', '==']:
        if condition_col.dtype == 'object':
            mask = condition_col.astype(str).str.lower() == str(value).lower()
        else:
            mask = condition_col == value
    elif operator == 'contains':
        mask = condition_col.astype(str).str.contains(str(value), case=False, na=False)
    elif operator == '>':
        mask = condition_col > value
    elif operator == '<':
        mask = condition_col < value
    elif operator == '>=':
        mask = condition_col >= value
    elif operator == '<=':
        mask = condition_col <= value
    elif operator == '!=':
        mask = condition_col != value
    else:
        mask = pd.Series([False] * len(df_copy))
    
    target_column = rule['target_column']
    actual_target_column = None
    for col in df_copy.columns:
        if col.lower() == target_column.lower():
            actual_target_column = col
            break
    if actual_target_column is None:
        actual_target_column = target_column
        df_copy[actual_target_column] = None
    
    df_copy.loc[mask, actual_target_column] = rule['new_value']
    return df_copy

# ==================== PROFESSIONAL UI COMPONENTS ====================

# Professional sidebar with enhanced styling
sidebar = html.Div([
    html.Div([
        html.H4("DataClean Pro", className="sidebar-title"),
        html.P("AI-Powered Analytics", className="sidebar-subtitle")
    ], className="sidebar-header"),
    
    dbc.Nav([
        dbc.NavLink([
            html.I(className="fas fa-cloud-upload-alt"),
            "Data Upload"
        ], href="#upload-section", active="exact", className="nav-link"),
        
        dbc.NavLink([
            html.I(className="fas fa-table"),
            "Data Overview"
        ], href="#overview-section", active="exact", className="nav-link"),
        
        dbc.NavLink([
            html.I(className="fas fa-chart-line"),
            "Data Profiling"
        ], href="#profiling-section", active="exact", className="nav-link"),
        
        dbc.NavLink([
            html.I(className="fas fa-robot"),
            "Smart Analysis"
        ], href="#smart-section", active="exact", className="nav-link"),
        
        dbc.NavLink([
            html.I(className="fas fa-search"),
            "Outlier Detection"
        ], href="#outlier-section", active="exact", className="nav-link"),
        
        dbc.NavLink([
            html.I(className="fas fa-magic"),
            "ML Cleaning"
        ], href="#ml-section", active="exact", className="nav-link"),
        
        dbc.NavLink([
            html.I(className="fas fa-broom"),
            "Advanced Cleaning"
        ], href="#cleaning-section", active="exact", className="nav-link"),
        
        dbc.NavLink([
            html.I(className="fas fa-award"),
            "Quality Dashboard"
        ], href="#quality-section", active="exact", className="nav-link"),
        
        dbc.NavLink([
            html.I(className="fas fa-download"),
            "Export Results"
        ], href="#export-section", active="exact", className="nav-link"),
    ], vertical=True, pills=True)
], className="sidebar")

from dash.dependencies import ALL

# --- Automatic Pattern Learning and Null Value Handling Block ---
automatic_pattern_learning = html.Div([
    html.H5([
        html.I(className="fas fa-brain", style={'marginRight': '0.5rem', 'color': '#667eea'}),
        "Automatic Pattern Learning"
    ], style={'marginBottom': '1.5rem', 'color': '#2c3e50'}),
    html.Div([
        dcc.Checklist(
            id='enable-pattern-learning',
            options=[
                {'label': [
                    html.I(className="fas fa-robot", style={'marginRight': '0.5rem', 'color': '#17a2b8'}),
                    html.Span("Enable Logical NULL Learning from Clean Sample")
                ], 'value': 'enable'}
            ],
            value=[],
            style={'marginBottom': '1rem'},
            labelStyle={'display': 'flex', 'alignItems': 'center', 'padding': '0.5rem', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'border': '1px solid #e9ecef'}
        ),
        html.Label([
            html.I(className="fas fa-sliders-h", style={'marginRight': '0.5rem', 'color': '#6f42c1'}),
            "NULL Threshold (0-1):"
        ], style={'fontWeight': 'bold', 'marginBottom': '0.5rem', 'display': 'flex', 'alignItems': 'center'}),
        dcc.Input(
            id='null-threshold',
            type='number',
            value=0.95,
            min=0,
            max=1,
            step=0.01,
            style={'width': '100%', 'padding': '0.75rem', 'marginBottom': '1rem', 'borderRadius': '8px', 'border': '2px solid #e9ecef'},
            className='form-control'
        ),
        html.Label([
            html.I(className="fas fa-sliders-h", style={'marginRight': '0.5rem', 'color': '#6f42c1'}),
            "Minimum Support (0-1):"
        ], style={'fontWeight': 'bold', 'marginBottom': '0.5rem', 'display': 'flex', 'alignItems': 'center'}),
        dcc.Input(
            id='min-support',
            type='number',
            value=0.1,
            min=0,
            max=1,
            step=0.01,
            style={'width': '100%', 'padding': '0.75rem', 'marginBottom': '1rem', 'borderRadius': '8px', 'border': '2px solid #e9ecef'},
            className='form-control'
        )
    ])
], style={'marginBottom': '2rem'})

null_value_handling = html.Div([
    html.H5([
        html.I(className="fas fa-question-circle", style={'marginRight': '0.5rem', 'color': '#667eea'}),
        "Null Value Handling"
    ], style={'marginBottom': '1.5rem', 'color': '#2c3e50'}),
    html.Div([
        dcc.Dropdown(
            id='null-method',
            options=[
                {'label': '🚫 No Action', 'value': 'none'},
                {'label': '🗑️ Remove Rows with Nulls', 'value': 'remove_rows'},
                {'label': '📊 Fill with Mean (Numerical)', 'value': 'fill_mean'},
                {'label': '📈 Fill with Median (Numerical)', 'value': 'fill_median'},
                {'label': '🎯 Fill with Mode', 'value': 'fill_mode'},
                {'label': '📉 Interpolate (Numerical)', 'value': 'interpolate'},
                {'label': '🤖 KNN Imputation', 'value': 'knn'}
            ],
            value='none',
            style={'marginBottom': '1rem'},
            className='enhanced-dropdown'
        ),
        html.Div([
            html.Span("💡 ", style={'fontSize': '1.2rem'}),
            html.Span("Choose how to handle missing values in your dataset. KNN imputation uses neighboring data points for more accurate predictions.", 
                     style={'color': '#6c757d', 'fontSize': '0.9rem'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '0.75rem', 'borderRadius': '8px', 'border': '1px solid #e9ecef'})
    ])
], style={'marginBottom': '2rem'})

# Professional main content area
content = html.Div([
    # Page Header
    html.Div([
        html.H1("Advanced Data Cleaning & ML Platform", className="page-title"),
        html.P("Transform your data with intelligent cleaning, machine learning automation, and professional analytics", 
               className="page-subtitle")
    ], className="page-header"),
    
    # Data Upload Section
    html.Div([
        html.Div([
            html.Div([
                html.H4([html.I(className="fas fa-cloud-upload-alt"), "Smart Data Import"], className="section-title")
            ], className="section-header"),
            html.Div([
                dcc.Upload(
                    id="upload-data",
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt upload-icon"),
                        html.H5("Drag & Drop or Click to Upload", className="upload-title"),
                        html.P("Supports CSV, Excel (.xlsx, .xls), JSON files", className="upload-subtitle"),
                        html.P("Auto-detects data types and suggests cleaning actions", className="upload-hint")
                    ]),
                    className="upload-area",
                    multiple=False
                ),
                html.Div(id="file-info", className="mt-4")
            ], className="section-body")
        ], className="section-card")
    ], id="upload-section"),
    
    # Data Overview Section
    html.Div([
        html.Div([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H4([html.I(className="fas fa-table"), "Enhanced Data Overview"], className="section-title")
                    ], width=8),
                    dbc.Col([
                        html.Div(id="data-stats", className="text-end")
                    ], width=4)
                ])
            ], className="section-header"),
            html.Div([
                html.Div(id="data-table-container", children=[
                    html.Div([
                        html.I(className="fas fa-table fa-3x text-muted mb-3"),
                        html.H5("No data uploaded yet", className="text-muted"),
                        html.P("Upload your dataset to begin analysis", className="text-muted")
                    ], className="text-center py-5")
                ], className="data-table-container")
            ], className="section-body")
        ], className="section-card")
    ], id="overview-section"),
    
    # Data Profiling Section
    html.Div([
        html.Div([
            html.Div([
                html.H4([html.I(className="fas fa-chart-line"), "Advanced Data Profiling & Statistics"], className="section-title")
            ], className="section-header"),
            html.Div([
                html.Div(id="profiling-content", children=[
                    html.Div([
                        html.I(className="fas fa-chart-bar fa-3x text-muted mb-3"),
                        html.H5("Upload data to view comprehensive profiling", className="text-muted"),
                        html.P("Get detailed statistics and insights about your dataset", className="text-muted")
                    ], className="text-center py-5")
                ])
            ], className="section-body")
        ], className="section-card")
    ], id="profiling-section"),
    
    # Smart Analysis Section
    html.Div([
        html.Div([
            html.Div([
                html.H4([html.I(className="fas fa-robot"), "AI-Powered Smart Analysis"], className="section-title")
            ], className="section-header"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Button([html.I(className="fas fa-search me-2"), "Analyze Data Quality"], 
                                 id="analyze-quality-btn", className="btn-primary-custom me-3 mb-3"),
                        dbc.Button([html.I(className="fas fa-brain me-2"), "Generate Recommendations"], 
                                 id="generate-recommendations-btn", className="btn-info-custom me-3 mb-3"),
                        dbc.Button([html.I(className="fas fa-cog me-2"), "Auto-Detect Data Types"], 
                                 id="detect-types-btn", className="btn-success-custom mb-3"),
                    ], width=12),
                    dbc.Col([
                        html.Div(id="smart-analysis-results", className="mt-4")
                    ], width=12)
                ])
            ], className="section-body")
        ], className="section-card")
    ], id="smart-section"),
    
    # Outlier Detection Section
    html.Div([
        html.Div([
            html.Div([
                html.H4([html.I(className="fas fa-search"), "Advanced Outlier Detection & Treatment"], className="section-title")
            ], className="section-header"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Detection Method:", className="fw-bold mb-3"),
                        dbc.RadioItems(
                            id="outlier-method",
                            options=[
                                {"label": "Interquartile Range (IQR)", "value": "iqr"},
                                {"label": "Z-Score", "value": "zscore"},
                                {"label": "🤖 ML Ensemble (Recommended)", "value": "ml_ensemble"},
                                {"label": "Isolation Forest", "value": "isolation_forest"},
                                {"label": "Local Outlier Factor", "value": "lof"}
                            ],
                            value="ml_ensemble",
                            className="mb-4"
                        ),
                        html.Label("Columns to Analyze:", className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id="outlier-columns",
                            placeholder="Select columns (leave empty for all numeric columns)",
                            multi=True,
                            className="mb-4 form-control-custom"
                        ),
                        dbc.Button([html.I(className="fas fa-search me-2"), "Detect Outliers"], 
                                 id="detect-outliers-btn", className="btn-primary-custom"),
                    ], width=4),
                    dbc.Col([
                        html.Div(id="outlier-results", className="mt-3")
                    ], width=8)
                ])
            ], className="section-body")
        ], className="section-card")
    ], id="outlier-section"),
    
    # ML Cleaning Section
    html.Div([
        html.Div([
            html.Div([
                html.H4([html.I(className="fas fa-magic"), "Machine Learning Data Cleaning"], className="section-title")
            ], className="section-header"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H6([html.I(className="fas fa-cog me-2"), "Smart Imputation"], className="text-primary mb-3"),
                        html.Label("Select Column:", className="fw-bold mb-2"),
                        dcc.Dropdown(id="imputation-column", placeholder="Choose column with missing values", 
                                   className="mb-3 form-control-custom"),
                        html.Label("Imputation Method:", className="fw-bold mb-2"),
                        dbc.RadioItems(
                            id="imputation-method",
                            options=[
                                {"label": "🤖 Auto-Select Best Method", "value": "auto"},
                                {"label": "Mean/Mode", "value": "mean"},
                                {"label": "Median", "value": "median"},
                                {"label": "KNN Imputation", "value": "knn"},
                                {"label": "Iterative Imputation (MICE)", "value": "iterative"},
                                {"label": "Forward Fill", "value": "forward_fill"},
                                {"label": "Backward Fill", "value": "backward_fill"}
                            ],
                            value="auto",
                            className="mb-4"
                        ),
                        dbc.Button([html.I(className="fas fa-magic me-2"), "Apply Imputation"], 
                                 id="apply-imputation-btn", className="btn-success-custom"),
                    ], width=6),
                    dbc.Col([
                        html.H6([html.I(className="fas fa-edit me-2"), "Text Standardization"], className="text-primary mb-3"),
                        html.Label("Select Text Column:", className="fw-bold mb-2"),
                        dcc.Dropdown(id="text-column", placeholder="Choose text column", 
                                   className="mb-3 form-control-custom"),
                        dbc.Checklist(
                            id="text-cleaning-options",
                            options=[
                                {"label": "Normalize Case", "value": "normalize_case"},
                                {"label": "Remove Extra Whitespace", "value": "remove_whitespace"},
                                {"label": "Standardize Similar Values", "value": "standardize_similar"}
                            ],
                            value=["normalize_case", "remove_whitespace"],
                            className="mb-4"
                        ),
                        dbc.Button([html.I(className="fas fa-edit me-2"), "Clean Text"], 
                                 id="clean-text-btn", className="btn-info-custom"),
                    ], width=6)
                ]),
                html.Hr(className="my-4"),
                html.Div(id="ml-cleaning-results", className="mt-4")
            ], className="section-body")
        ], className="section-card")
    ], id="ml-section"),
    
    # Advanced Cleaning Section
    html.Div([
        html.Div([
            html.Div([
                html.H4([html.I(className="fas fa-broom"), "Advanced Data Cleaning Operations"], className="section-title")
            ], className="section-header"),
            html.Div([
                # --- Inserted Pattern Learning and Null Handling Block ---
                automatic_pattern_learning,
                null_value_handling,
                dbc.Tabs([
                    dbc.Tab(label="🔧 Basic Operations", tab_id="basic-cleaning"),
                    dbc.Tab(label="📋 Conditional Rules", tab_id="conditional-cleaning"),
                    dbc.Tab(label="🗑️ Remove Data", tab_id="remove-data"),
                    dbc.Tab(label="📊 Transform Data", tab_id="transform-data")
                ], id="cleaning-tabs", active_tab="basic-cleaning"),
                html.Div(id="cleaning-tab-content", className="tab-content-area")
            ], className="section-body")
        ], className="section-card")
    ], id="cleaning-section"),
    
    # Quality Dashboard Section
    html.Div([
        html.Div([
            html.Div([
                html.H4([html.I(className="fas fa-award"), "Data Quality Dashboard"], className="section-title")
            ], className="section-header"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Button([html.I(className="fas fa-calculator me-2"), "Calculate Quality Score"], 
                                 id="calculate-quality-btn", className="btn-primary-custom mb-4"),
                    ], width=12),
                    dbc.Col([
                        html.Div(id="quality-dashboard", className="mt-3")
                    ], width=12)
                ])
            ], className="section-body")
        ], className="section-card")
    ], id="quality-section"),
    
    # Export Section
    html.Div([
        html.Div([
            html.Div([
                html.H4([html.I(className="fas fa-download"), "Export Cleaned Data & Reports"], className="section-title")
            ], className="section-header"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Export Format:", className="fw-bold mb-3"),
                        dbc.RadioItems(
                            id="export-format",
                            options=[
                                {"label": "CSV", "value": "csv"},
                                {"label": "Excel (.xlsx)", "value": "xlsx"},
                                {"label": "JSON", "value": "json"}
                            ],
                            value="csv",
                            className="mb-4"
                        ),
                        dbc.ButtonGroup([
                            dbc.Button([html.I(className="fas fa-download me-2"), "Download Data"], 
                                     id="download-btn", className="btn-success-custom"),
                            dbc.Button([html.I(className="fas fa-file-alt me-2"), "Generate Report"], 
                                     id="report-btn", className="btn-info-custom")
                        ], className="mb-3"),
                        html.Div(id="export-status", className="mt-3")
                    ], width=6),
                    dbc.Col([
                        html.H6([html.I(className="fas fa-history me-2"), "Cleaning History"], className="text-primary mb-3"),
                        html.Div(id="cleaning-history", className="history-container")
                    ], width=6)
                ])
            ], className="section-body")
        ], className="section-card")
    ], id="export-section")
], className="content-area")

# Main app layout with professional structure
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="stored-data"),
    dcc.Store(id="cleaning-operations"),
    dcc.Store(id="quality-metrics-store"),
    sidebar,
    content
], className="main-container")

# --- Smooth scroll to section when sidebar link is clicked ---
app.clientside_callback(
    """
    function(pathname) {
        if (!pathname) return window.dash_clientside.no_update;
        var hash = pathname.split('#')[1];
        if (hash) {
            var el = document.getElementById(hash);
            if (el) {
                el.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('content-area-scroll-anchor', 'children'),
    Input('url', 'href')
)

# Add a hidden anchor for scroll callback
if not hasattr(app, '_content_area_scroll_anchor_added'):
    content.children = [
        html.Div(id='content-area-scroll-anchor', style={'display': 'none'}),
        *content.children
    ]
    app._content_area_scroll_anchor_added = True

# ==================== CALLBACKS ====================

# File upload callback (enhanced with FIXED data table)
@app.callback(
    [Output("stored-data", "data"),
     Output("file-info", "children"),
     Output("data-table-container", "children"),
     Output("data-stats", "children"),
     Output("outlier-columns", "options"),
     Output("imputation-column", "options"),
     Output("text-column", "options")],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")]
)
def update_output(contents, filename):
    if contents is None:
        return None, "", html.Div([
            html.I(className="fas fa-table fa-3x text-muted mb-3"),
            html.H5("No data uploaded yet", className="text-muted"),
            html.P("Upload your dataset to begin analysis", className="text-muted")
        ], className="text-center py-5"), "", [], [], []
    
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xlsx' in filename or 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'json' in filename:
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, dbc.Alert("Unsupported file format!", color="danger", className="alert-custom"), "", "", [], [], []
        
        # Store original data - Use global variables properly
        global stored_data
        stored_data = {
            'original': df.copy(),
            'current': df.copy()
        }
        
        # File info
        file_info = dbc.Alert([
            html.H6([html.I(className="fas fa-check-circle me-2"), f"Successfully loaded: {filename}"], className="mb-2"),
            html.P([html.I(className="fas fa-chart-bar me-2"), f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"], className="mb-1"),
            html.P([html.I(className="fas fa-memory me-2"), f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB"], className="mb-0")
        ], color="success", className="alert-custom")
        
        # FIXED: Data table with proper horizontal scrolling wrapper
        data_table = html.Div([
            dash_table.DataTable(
                data=df.head(100).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'width': '100%'}, # Remove overflowX from here
                style_cell={
                    'textAlign': 'left', 
                    'padding': '12px',
                    'fontFamily': 'Inter, sans-serif',
                    'fontSize': '14px',
                    'minWidth': '120px', 
                    'width': '150px', 
                    'maxWidth': '200px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_header={
                    'backgroundColor': '#f8fafc',
                    'fontWeight': '600',
                    'color': '#1e293b',
                    'border': '1px solid #e2e8f0'
                },
                style_data={
                    'backgroundColor': '#ffffff',
                    'color': '#374151',
                    'border': '1px solid #f3f4f6'
                },
                page_size=10,
                sort_action="native",
                filter_action="native"
            )
        ], className="data-table-wrapper") # This wrapper has the horizontal scroll
        
        # Data stats
        stats = html.Div([
            html.Span([html.I(className="fas fa-table me-1"), f"Rows: {df.shape[0]}"], className="stats-badge primary"),
            html.Span([html.I(className="fas fa-columns me-1"), f"Columns: {df.shape[1]}"], className="stats-badge primary"),
            html.Span([html.I(className="fas fa-exclamation-triangle me-1"), f"Missing: {df.isnull().sum().sum()}"], 
                     className="stats-badge warning" if df.isnull().sum().sum() > 0 else "stats-badge success")
        ])
        
        # Dropdown options
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        
        outlier_options = [{"label": col, "value": col} for col in numeric_cols]
        imputation_options = [{"label": f"{col} ({df[col].isnull().sum()} missing)", "value": col} for col in missing_cols]
        text_options = [{"label": col, "value": col} for col in text_cols]
        
        return df.to_dict('records'), file_info, data_table, stats, outlier_options, imputation_options, text_options
        
    except Exception as e:
        return None, dbc.Alert(f"Error processing file: {str(e)}", color="danger", className="alert-custom"), "", "", [], [], []

# Enhanced profiling callback with FIXED layout
@app.callback(
    Output("profiling-content", "children"),
    [Input("stored-data", "data")]
)
def update_profiling_content(data):
    if not data:
        return html.Div([
            html.I(className="fas fa-chart-bar fa-3x text-muted mb-3"),
            html.H5("Upload data to view comprehensive profiling", className="text-muted"),
            html.P("Get detailed statistics and insights about your dataset", className="text-muted")
        ], className="text-center py-5")
    
    df = pd.DataFrame(data)
    profiling = get_data_profiling(df)
    
    # FIXED: Create profiling cards using CSS Grid for exactly 2 per row
    profiling_cards = []
    
    for col, info in profiling.items():
        # Determine card color based on data quality
        missing_pct = info['missing_percentage']
        if missing_pct == 0:
            border_color = "#10b981"
        elif missing_pct < 10:
            border_color = "#f59e0b"
        else:
            border_color = "#ef4444"
        
        card_content = [
            html.H6(col, className="profiling-card-title"),
            html.Div([html.I(className="fas fa-tag"), f"Type: {info['dtype']}"], className="profiling-stat"),
            html.Div([html.I(className="fas fa-exclamation-triangle"), f"Missing: {info['missing_count']} ({missing_pct:.1f}%)"], className="profiling-stat"),
            html.Div([html.I(className="fas fa-fingerprint"), f"Unique: {info['unique_count']} ({info['unique_percentage']:.1f}%)"], className="profiling-stat")
        ]
        
        # Add numeric statistics if available
        if 'mean' in info:
            card_content.extend([
                html.Hr(style={"margin": "1rem 0"}),
                html.Div([html.I(className="fas fa-calculator"), f"Mean: {info['mean']:.2f}"], className="profiling-stat"),
                html.Div([html.I(className="fas fa-chart-line"), f"Median: {info['median']:.2f}"], className="profiling-stat"),
                html.Div([html.I(className="fas fa-arrows-alt-h"), f"Range: {info['min']:.2f} - {info['max']:.2f}"], className="profiling-stat")
            ])
        
        # Add text statistics if available
        if 'avg_length' in info:
            card_content.extend([
                html.Hr(style={"margin": "1rem 0"}),
                html.Div([html.I(className="fas fa-ruler"), f"Avg Length: {info['avg_length']:.1f}"], className="profiling-stat"),
                html.Div([html.I(className="fas fa-arrows-alt-h"), f"Length Range: {info['min_length']} - {info['max_length']}"], className="profiling-stat")
            ])
        
        profiling_cards.append(
            html.Div(card_content, className="profiling-card", style={"border-left": f"4px solid {border_color}"})
        )
    
    return html.Div(profiling_cards, className="profiling-grid")

# Smart Analysis callbacks
@app.callback(
    Output("smart-analysis-results", "children"),
    [Input("analyze-quality-btn", "n_clicks"),
     Input("generate-recommendations-btn", "n_clicks"),
     Input("detect-types-btn", "n_clicks")],
    [State("stored-data", "data")]
)
def smart_analysis(quality_clicks, rec_clicks, types_clicks, data):
    if not data:
        return html.Div([
            html.I(className="fas fa-robot fa-3x text-muted mb-3"),
            html.H5("Please upload data first", className="text-muted"),
            html.P("Upload your dataset to begin AI-powered analysis", className="text-muted")
        ], className="text-center py-4")
    
    ctx = callback_context
    if not ctx.triggered:
        return html.Div([
            html.I(className="fas fa-lightbulb fa-2x text-primary mb-3"),
            html.H6("Click any button above to start smart analysis", className="text-muted")
        ], className="text-center py-4")
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = pd.DataFrame(data)
    
    if button_id == "analyze-quality-btn":
        missing_analysis = advanced_missing_data_analysis(df)
        return html.Div([
            html.Div([
                html.H6([html.I(className="fas fa-chart-pie me-2"), "Missing Data Analysis"], className="section-title"),
                html.P(f"Total missing values: {missing_analysis['total_missing']} ({missing_analysis['missing_percentage']:.1f}%)"),
                html.H6("Columns with missing data:", className="mt-3"),
                html.Ul([
                    html.Li(f"{col}: {info['count']} missing ({info['percentage']:.1f}%)")
                    for col, info in missing_analysis['columns_with_missing'].items()
                ]) if missing_analysis['columns_with_missing'] else html.P([
                    html.I(className="fas fa-check-circle text-success me-2"),
                    "No missing data found! 🎉"
                ])
            ], className="section-card")
        ])
    
    elif button_id == "generate-recommendations-btn":
        recommendations = generate_cleaning_recommendations(df)
        if not recommendations:
            return dbc.Alert([
                html.I(className="fas fa-thumbs-up me-2"),
                "Great! No major data quality issues detected."
            ], color="success", className="alert-custom")
        
        rec_cards = []
        for rec in recommendations[:5]:  # Show top 5 recommendations
            color = {"high": "danger", "medium": "warning", "low": "info"}[rec['priority']]
            icon = {"high": "fas fa-exclamation-triangle", "medium": "fas fa-exclamation-circle", "low": "fas fa-info-circle"}[rec['priority']]
            rec_cards.append(
                dbc.Alert([
                    html.H6([html.I(className=f"{icon} me-2"), rec['description']], className="mb-1"),
                    html.Small(f"Priority: {rec['priority'].title()} | Type: {rec['type'].replace('_', ' ').title()}")
                ], color=color, className="alert-custom mb-3")
            )
        
        return html.Div([
            html.H6([html.I(className="fas fa-brain me-2"), "AI Recommendations:"], className="mb-3"),
            html.Div(rec_cards)
        ])
    
    elif button_id == "detect-types-btn":
        type_suggestions = smart_type_detection(df)
        type_cards = []
        for col, suggested_type in type_suggestions.items():
            current_type = str(df[col].dtype)
            if suggested_type not in ['numeric', 'text', 'unknown']:
                type_cards.append(
                    dbc.Alert([
                        html.Strong(f"{col}: "),
                        f"Current: {current_type} → Suggested: {suggested_type}"
                    ], color="info", className="alert-custom mb-2")
                )
        
        if not type_cards:
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "All data types look correct!"
            ], color="success", className="alert-custom")
        
        return html.Div([
            html.H6([html.I(className="fas fa-search me-2"), "Data Type Suggestions:"], className="mb-3"),
            html.Div(type_cards)
        ])
    
    return html.Div()

# Outlier detection callback (enhanced)
@app.callback(
    Output("outlier-results", "children"),
    [Input("detect-outliers-btn", "n_clicks")],
    [State("stored-data", "data"),
     State("outlier-method", "value"),
     State("outlier-columns", "value")]
)
def detect_outliers_callback(n_clicks, data, method, selected_columns):
    if not n_clicks or not data:
        return html.Div([
            html.I(className="fas fa-search fa-3x text-muted mb-3"),
            html.H6("Select method and click 'Detect Outliers' to analyze", className="text-muted")
        ], className="text-center py-4")
    
    df = pd.DataFrame(data)
    columns = selected_columns if selected_columns else df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columns:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "No numeric columns available for outlier detection."
        ], color="warning", className="alert-custom")
    
    try:
        if method == "iqr":
            outlier_info = detect_outliers_iqr(df, columns)
        elif method == "zscore":
            outlier_info = detect_outliers_zscore(df, columns)
        elif method == "ml_ensemble":
            outlier_info = ml_based_outlier_detection(df, columns)
        elif method == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            numeric_data = df[columns].dropna()
            if len(numeric_data) < 10:
                return dbc.Alert("Insufficient data for outlier detection.", color="warning", className="alert-custom")
            predictions = iso_forest.fit_predict(numeric_data)
            outlier_indices = numeric_data.index[predictions == -1].tolist()
            outlier_info = {
                'method': 'Isolation Forest',
                'total_outliers': len(outlier_indices),
                'outlier_indices': outlier_indices
            }
        elif method == "lof":
            lof = LocalOutlierFactor(contamination=0.1)
            numeric_data = df[columns].dropna()
            if len(numeric_data) < 10:
                return dbc.Alert("Insufficient data for outlier detection.", color="warning", className="alert-custom")
            predictions = lof.fit_predict(numeric_data)
            outlier_indices = numeric_data.index[predictions == -1].tolist()
            outlier_info = {
                'method': 'Local Outlier Factor',
                'total_outliers': len(outlier_indices),
                'outlier_indices': outlier_indices
            }
        
        # Create results display
        results = [
            dbc.Alert([
                html.H6([html.I(className="fas fa-search me-2"), f"{outlier_info.get('method', method.upper())} Results"]),
                html.P(f"Total outliers detected: {outlier_info.get('total_outliers', 0)}")
            ], color="info", className="alert-custom")
        ]
        
        # Add detailed results for each column
        if 'outliers_by_column' in outlier_info:
            for col, col_info in outlier_info['outliers_by_column'].items():
                if col_info['count'] > 0:
                    results.append(
                        html.Div([
                            html.Div([
                                html.H6([html.I(className="fas fa-chart-bar me-2"), col], className="section-title")
                            ], className="section-header"),
                            html.Div([
                                html.P(f"Outliers: {col_info['count']}"),
                                html.P(f"Indices: {col_info['indices'][:10]}{'...' if len(col_info['indices']) > 10 else ''}"),
                                dcc.Graph(figure=create_outlier_boxplot(df, col))
                            ], className="section-body")
                        ], className="section-card mb-3")
                    )
        
        # Add action buttons
        if outlier_info.get('total_outliers', 0) > 0:
            results.append(
                dbc.ButtonGroup([
                    dbc.Button([html.I(className="fas fa-trash me-2"), "Remove Outliers"], 
                             id="remove-outliers-btn", className="btn-danger-custom"),
                    dbc.Button([html.I(className="fas fa-cut me-2"), "Cap Outliers"], 
                             id="cap-outliers-btn", className="btn-warning-custom"),
                    dbc.Button([html.I(className="fas fa-tag me-2"), "Mark as Anomalies"], 
                             id="mark-outliers-btn", className="btn-info-custom")
                ], className="mt-3")
            )
        
        return html.Div(results)
        
    except Exception as e:
        return dbc.Alert(f"Error in outlier detection: {str(e)}", color="danger", className="alert-custom")

# ML Cleaning callbacks
@app.callback(
    Output("ml-cleaning-results", "children"),
    [Input("apply-imputation-btn", "n_clicks"),
     Input("clean-text-btn", "n_clicks")],
    [State("stored-data", "data"),
     State("imputation-column", "value"),
     State("imputation-method", "value"),
     State("text-column", "value"),
     State("text-cleaning-options", "value")]
)
def ml_cleaning_operations(impute_clicks, text_clicks, data, impute_col, impute_method, text_col, text_options):
    global stored_data, cleaning_history
    
    if not data:
        return html.Div([
            html.I(className="fas fa-upload fa-3x text-muted mb-3"),
            html.H6("Please upload data first", className="text-muted")
        ], className="text-center py-4")
    
    ctx = callback_context
    if not ctx.triggered:
        return html.Div([
            html.I(className="fas fa-magic fa-2x text-primary mb-3"),
            html.H6("Select options and click buttons above to perform ML cleaning", className="text-muted")
        ], className="text-center py-4")
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = pd.DataFrame(data)
    
    if button_id == "apply-imputation-btn" and impute_col:
        try:
            original_missing = df[impute_col].isnull().sum()
            df_imputed = smart_imputation(df, impute_col, impute_method)
            new_missing = df_imputed[impute_col].isnull().sum()
            
            # Update stored data
            stored_data['current'] = df_imputed
            
            # Add to cleaning history
            cleaning_history.append({
                'operation': f'Imputation ({impute_method})',
                'column': impute_col,
                'details': f'Filled {original_missing - new_missing} missing values',
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
            })
            
            return dbc.Alert([
                html.H6([html.I(className="fas fa-check-circle me-2"), "Imputation Completed"]),
                html.P(f"Column: {impute_col}"),
                html.P(f"Method: {impute_method}"),
                html.P(f"Missing values filled: {original_missing - new_missing}")
            ], color="success", className="alert-custom")
            
        except Exception as e:
            return dbc.Alert(f"Error in imputation: {str(e)}", color="danger", className="alert-custom")
    
    elif button_id == "clean-text-btn" and text_col:
        try:
            cleaning_results = text_data_cleaning(df, [text_col])
            
            if text_col in cleaning_results['cleaned_columns']:
                df[text_col] = cleaning_results['cleaned_columns'][text_col]
                
                # Update stored data
                stored_data['current'] = df
                
                # Add to cleaning history
                cleaning_history.append({
                    'operation': 'Text Cleaning',
                    'column': text_col,
                    'details': f'Applied: {", ".join(text_options)}',
                    'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
                })
                
                improvements = cleaning_results['quality_improvements'][text_col]
                suggestions = cleaning_results['standardization_suggestions'].get(text_col, [])
                
                result_content = [
                    html.H6([html.I(className="fas fa-check-circle me-2"), "Text Cleaning Completed"]),
                    html.P(f"Column: {text_col}"),
                    html.P(f"Unique values before: {improvements['unique_values_before']}"),
                    html.P(f"Unique values after: {improvements['unique_values_after']}")
                ]
                
                if suggestions:
                    result_content.append(html.H6([html.I(className="fas fa-lightbulb me-2"), "Standardization Suggestions:"]))
                    for group in suggestions[:3]:  # Show first 3 groups
                        result_content.append(html.P(f"Similar values: {', '.join(group)}"))
                
                return dbc.Alert(result_content, color="success", className="alert-custom")
            
        except Exception as e:
            return dbc.Alert(f"Error in text cleaning: {str(e)}", color="danger", className="alert-custom")
    
    return html.Div()

# Quality Dashboard callback
@app.callback(
    Output("quality-dashboard", "children"),
    [Input("calculate-quality-btn", "n_clicks")],
    [State("stored-data", "data")]
)
def update_quality_dashboard(n_clicks, data):
    if not n_clicks or not data:
        return html.Div([
            html.I(className="fas fa-calculator fa-3x text-muted mb-3"),
            html.H6("Click 'Calculate Quality Score' to analyze data quality", className="text-muted")
        ], className="text-center py-4")
    
    df = pd.DataFrame(data)
    quality_metrics = calculate_data_quality_score(df)
    
    # Overall score card
    score_color = "#10b981" if quality_metrics['overall_score'] >= 80 else "#f59e0b" if quality_metrics['overall_score'] >= 60 else "#ef4444"
    
    overall_card = html.Div([
        html.Div(f"{quality_metrics['overall_score']:.1f}", className="quality-score", style={"color": score_color}),
        html.Div("Overall Quality Score", className="quality-label")
    ], className="quality-card")
    
    # Metric cards
    metrics_row = dbc.Row([
        dbc.Col([
            html.Div([
                html.Div(f"{quality_metrics['completeness']:.1f}%", className="quality-score", style={"color": "#06b6d4"}),
                html.Div("Completeness", className="quality-label")
            ], className="quality-card")
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div(f"{quality_metrics['validity']:.1f}%", className="quality-score", style={"color": "#3b82f6"}),
                html.Div("Validity", className="quality-label")
            ], className="quality-card")
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div(f"{quality_metrics['consistency']:.1f}%", className="quality-score", style={"color": "#f59e0b"}),
                html.Div("Consistency", className="quality-label")
            ], className="quality-card")
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div(f"{quality_metrics['uniqueness']:.1f}%", className="quality-score", style={"color": "#10b981"}),
                html.Div("Uniqueness", className="quality-label")
            ], className="quality-card")
        ], width=3)
    ], className="mb-4")
    
    # Column-level quality table
    column_data = []
    for col, scores in quality_metrics['column_scores'].items():
        column_data.append({
            'Column': col,
            'Overall': f"{scores['overall']:.1f}%",
            'Completeness': f"{scores['completeness']:.1f}%",
            'Validity': f"{scores['validity']:.1f}%",
            'Consistency': f"{scores['consistency']:.1f}%",
            'Uniqueness': f"{scores['uniqueness']:.1f}%"
        })
    
    quality_table = dash_table.DataTable(
        data=column_data,
        columns=[{"name": col, "id": col} for col in column_data[0].keys()],
        style_cell={
            'textAlign': 'center', 
            'padding': '12px',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#f8fafc',
            'fontWeight': '600',
            'color': '#1e293b',
            'border': '1px solid #e2e8f0'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{Overall} >= 80'},
                'backgroundColor': '#d1fae5',
                'color': '#065f46',
            },
            {
                'if': {'filter_query': '{Overall} < 60'},
                'backgroundColor': '#fecaca',
                'color': '#991b1b',
            }
        ]
    )
    
    return html.Div([
        dbc.Row([
            dbc.Col([overall_card], width=4),
            dbc.Col([
                html.H6([html.I(className="fas fa-chart-pie me-2"), "Quality Breakdown"], className="mb-3"),
                metrics_row
            ], width=8)
        ]),
        html.H6([html.I(className="fas fa-table me-2"), "Column-Level Quality Analysis"], className="mb-3 mt-4"),
        html.Div([quality_table], className="data-table-container")
    ])

# Cleaning tabs callback
@app.callback(
    Output("cleaning-tab-content", "children"),
    [Input("cleaning-tabs", "active_tab")],
    [State("stored-data", "data")]
)
def update_cleaning_tab_content(active_tab, data):
    if not data:
        return html.Div([
            html.I(className="fas fa-upload fa-3x text-muted mb-3"),
            html.H6("Please upload data first", className="text-muted")
        ], className="text-center py-4")
    
    df = pd.DataFrame(data)
    
    if active_tab == "basic-cleaning":
        return dbc.Row([
            dbc.Col([
                html.H6([html.I(className="fas fa-cog me-2"), "Basic Operations"], className="mb-4"),
                html.Label("Select Column:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id="basic-cleaning-column",
                    options=[{"label": col, "value": col} for col in df.columns],
                    placeholder="Choose column",
                    className="mb-3 form-control-custom"
                ),
                html.Label("Operation:", className="fw-bold mb-2"),
                dbc.RadioItems(
                    id="basic-operation",
                    options=[
                        {"label": "Remove Duplicates", "value": "remove_duplicates"},
                        {"label": "Fill Missing Values", "value": "fill_missing"},
                        {"label": "Convert Data Type", "value": "convert_type"},
                        {"label": "Rename Column", "value": "rename_column"}
                    ],
                    className="mb-4"
                ),
                dbc.Button([html.I(className="fas fa-play me-2"), "Apply Operation"], 
                         id="apply-basic-btn", className="btn-primary-custom"),
                html.Div(id="basic-cleaning-result", className="mt-4")
            ], width=12)
        ])
    
    elif active_tab == "conditional-cleaning":
        return dbc.Row([
            dbc.Col([
                html.H6([html.I(className="fas fa-code me-2"), "Conditional Data Cleaning"], className="mb-4"),
                html.Label("Conditional Rule:", className="fw-bold mb-2"),
                dbc.Textarea(
                    id="conditional-rule",
                    placeholder="Example: if age > 65 then category as senior",
                    rows=3,
                    className="mb-3 form-control-custom"
                ),
                html.Small("Supported formats:", className="text-muted fw-bold"),
                html.Ul([
                    html.Li("if [column] = [value] then [target_column] as [new_value]"),
                    html.Li("if [column] > [value] then [target_column] as [new_value]"),
                    html.Li("if [column] contains [text] then [target_column] as [new_value]")
                ], className="small text-muted mb-4"),
                dbc.Button([html.I(className="fas fa-magic me-2"), "Apply Rule"], 
                         id="apply-rule-btn", className="btn-success-custom"),
                html.Div(id="rule-result", className="mt-4")
            ], width=12)
        ])
    
    elif active_tab == "remove-data":
        return dbc.Row([
            dbc.Col([
                html.H6([html.I(className="fas fa-trash me-2"), "Remove Data"], className="mb-4"),
                dbc.Checklist(
                    id="remove-options",
                    options=[
                        {"label": "Remove rows with any missing values", "value": "remove_missing_rows"},
                        {"label": "Remove duplicate rows", "value": "remove_duplicates"},
                        {"label": "Remove outliers (detected)", "value": "remove_outliers"},
                        {"label": "Remove empty columns", "value": "remove_empty_cols"}
                    ],
                    className="mb-4"
                ),
                dbc.Button([html.I(className="fas fa-trash me-2"), "Remove Selected"], 
                         id="remove-data-btn", className="btn-danger-custom"),
                html.Div(id="remove-result", className="mt-4")
            ], width=12)
        ])
    
    elif active_tab == "transform-data":
        return dbc.Row([
            dbc.Col([
                html.H6([html.I(className="fas fa-exchange-alt me-2"), "Transform Data"], className="mb-4"),
                html.Label("Select Column:", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id="transform-column",
                    options=[{"label": col, "value": col} for col in df.select_dtypes(include=[np.number]).columns],
                    placeholder="Choose numeric column",
                    className="mb-3 form-control-custom"
                ),
                html.Label("Transformation:", className="fw-bold mb-2"),
                dbc.RadioItems(
                    id="transform-operation",
                    options=[
                        {"label": "Log Transform", "value": "log"},
                        {"label": "Square Root", "value": "sqrt"},
                        {"label": "Standardize (Z-score)", "value": "standardize"},
                        {"label": "Normalize (0-1)", "value": "normalize"}
                    ],
                    className="mb-4"
                ),
                dbc.Button([html.I(className="fas fa-calculator me-2"), "Apply Transformation"], 
                         id="apply-transform-btn", className="btn-info-custom"),
                html.Div(id="transform-result", className="mt-4")
            ], width=12)
        ])
    
    return html.Div()

# Conditional rule callback
@app.callback(
    Output("rule-result", "children"),
    [Input("apply-rule-btn", "n_clicks")],
    [State("stored-data", "data"),
     State("conditional-rule", "value")]
)
def apply_conditional_rule_callback(n_clicks, data, rule_text):
    global stored_data, cleaning_history
    
    if not n_clicks or not data or not rule_text:
        return html.Div()
    
    df = pd.DataFrame(data)
    rule = parse_conditional_rule(rule_text)
    
    if not rule:
        return dbc.Alert("Invalid rule format. Please check the syntax.", color="danger", className="alert-custom")
    
    try:
        df_modified = apply_conditional_rule(df, rule)
        
        # Update stored data
        stored_data['current'] = df_modified
        
        # Add to cleaning history
        cleaning_history.append({
            'operation': 'Conditional Rule',
            'column': rule['target_column'],
            'details': f"Applied rule: {rule_text}",
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
        })
        
        return dbc.Alert([
            html.H6([html.I(className="fas fa-check-circle me-2"), "Conditional Rule Applied"]),
            html.P(f"Rule: {rule_text}"),
            html.P(f"Target column: {rule['target_column']}")
        ], color="success", className="alert-custom")
        
    except Exception as e:
        return dbc.Alert(f"Error applying rule: {str(e)}", color="danger", className="alert-custom")

# Cleaning history callback
@app.callback(
    Output("cleaning-history", "children"),
    [Input("stored-data", "data")]
)
def update_cleaning_history(data):
    global cleaning_history
    if not cleaning_history:
        return html.Div([
            html.I(className="fas fa-history fa-2x text-muted mb-2"),
            html.P("No cleaning operations performed yet.", className="text-muted small")
        ], className="text-center py-4")
    
    history_items = []
    for i, operation in enumerate(reversed(cleaning_history[-10:])):  # Show last 10 operations
        history_items.append(
            html.Div([
                html.Div(operation['operation'], className="history-operation"),
                html.Div(f"Column: {operation['column']}", className="history-details"),
                html.Div(f"{operation['details']} at {operation['timestamp']}", className="history-details")
            ], className="history-item")
        )
    
    return html.Div(history_items)

# Export callbacks
@app.callback(
    Output("export-status", "children"),
    [Input("download-btn", "n_clicks"),
     Input("report-btn", "n_clicks")],
    [State("stored-data", "data"),
     State("export-format", "value")]
)
def handle_export(download_clicks, report_clicks, data, export_format):
    if not data:
        return html.Div()
    
    ctx = callback_context
    if not ctx.triggered:
        return html.Div()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "download-btn":
        return dbc.Alert([
            html.H6([html.I(className="fas fa-download me-2"), "Download Ready"]),
            html.P(f"Your cleaned data is ready for download in {export_format.upper()} format."),
            html.P("Note: In a production environment, this would trigger an actual file download.")
        ], color="success", className="alert-custom")
    
    elif button_id == "report-btn":
        global cleaning_history
        report_content = [
            html.H6([html.I(className="fas fa-file-alt me-2"), "Data Cleaning Report"]),
            html.P(f"Total operations performed: {len(cleaning_history)}"),
            html.P("Operations summary:"),
            html.Ul([
                html.Li(f"{op['operation']} on {op['column']}: {op['details']}")
                for op in cleaning_history[-5:]  # Last 5 operations
            ])
        ]
        
        return dbc.Alert(report_content, color="info", className="alert-custom")
    
    return html.Div()

if __name__ == '__main__':
 import os

 port = int(os.environ.get("PORT", 8050))
 app.run_server(debug=False, host="0.0.0.0", port=port)

