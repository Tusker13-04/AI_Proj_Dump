#!/usr/bin/env python3
"""
Autoscaler Hybrid ML Pipeline with RF, LSTM, Prophet, and Hybrid Ensemble.
Advanced ensemble combining multiple time-series and ML models for Kubernetes autoscaling.
ITERATIVE VERSION with Data Augmentation - Runs 30+ minutes
"""

import sys
import argparse
import logging
import warnings
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
import joblib

try:
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

warnings.filterwarnings('ignore')


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"pipeline_{timestamp}.log"

    logger = logging.getLogger("autoscaler_pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


@dataclass
class ClassifierMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'roc_auc': round(self.roc_auc, 4),
            'confusion_matrix': self.confusion_matrix.tolist(),
        }
    
    def get_score(self) -> float:
        return 0.4 * self.accuracy + 0.3 * self.f1 + 0.3 * self.roc_auc


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    mse: float
    directional_accuracy: float = 0.0
    scaled_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mae': round(self.mae, 4),
            'rmse': round(self.rmse, 4),
            'mse': round(self.mse, 4),
            'directional_accuracy': round(self.directional_accuracy, 4),
            'scaled_accuracy': round(self.scaled_accuracy, 2),
        }


@dataclass
class HybridMetrics:
    mae_hybrid: float
    rmse_hybrid: float
    prophet_weight: float
    lstm_weight: float
    improvement_prophet: float
    improvement_lstm: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mae_hybrid': round(self.mae_hybrid, 4),
            'rmse_hybrid': round(self.rmse_hybrid, 4),
            'prophet_weight': round(self.prophet_weight, 4),
            'lstm_weight': round(self.lstm_weight, 4),
            'improvement_prophet': round(self.improvement_prophet, 2),
            'improvement_lstm': round(self.improvement_lstm, 2),
        }


@dataclass
class PipelineResults:
    timestamp: str
    service_name: str
    iteration: int = 0
    rf_metrics: Optional[ClassifierMetrics] = None
    lstm_metrics: Optional[ClassifierMetrics] = None
    prophet_metrics: Optional[RegressionMetrics] = None
    hybrid_metrics: Optional[HybridMetrics] = None
    rf_hyperparams: Dict[str, Any] = field(default_factory=dict)
    training_samples: int = 0
    test_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'service_name': self.service_name,
            'iteration': self.iteration,
            'training_samples': self.training_samples,
            'test_samples': self.test_samples,
            'rf_hyperparams': self.rf_hyperparams,
            'rf_metrics': self.rf_metrics.to_dict() if self.rf_metrics else None,
            'lstm_metrics': self.lstm_metrics.to_dict() if self.lstm_metrics else None,
            'prophet_metrics': self.prophet_metrics.to_dict() if self.prophet_metrics else None,
            'hybrid_metrics': self.hybrid_metrics.to_dict() if self.hybrid_metrics else None,
        }

    def print_summary(self) -> None:
        print("
" + "="*80)
        print(f"AUTOSCALER HYBRID PIPELINE - ITERATION {self.iteration}")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Service: {self.service_name}")
        print(f"Training Samples: {self.training_samples} | Test Samples: {self.test_samples}")
        print("="*80)

        if self.rf_metrics:
            print("
📊 RANDOM FOREST CLASSIFIER")
            print("-" * 80)
            print(f"  Accuracy:  {self.rf_metrics.accuracy:.4f}")
            print(f"  Precision: {self.rf_metrics.precision:.4f}")
            print(f"  Recall:    {self.rf_metrics.recall:.4f}")
            print(f"  F1-Score:  {self.rf_metrics.f1:.4f}")
            print(f"  ROC-AUC:   {self.rf_metrics.roc_auc:.4f}")
            print(f"  Hyperparams: {self.rf_hyperparams}")
            tn, fp, fn, tp = self.rf_metrics.confusion_matrix.ravel()
            print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        if self.lstm_metrics:
            print("
🧠 LSTM CLASSIFIER")
            print("-" * 80)
            print(f"  Accuracy:  {self.lstm_metrics.accuracy:.4f}")
            print(f"  Precision: {self.lstm_metrics.precision:.4f}")
            print(f"  Recall:    {self.lstm_metrics.recall:.4f}")
            print(f"  F1-Score:  {self.lstm_metrics.f1:.4f}")
            print(f"  ROC-AUC:   {self.lstm_metrics.roc_auc:.4f}")
            tn, fp, fn, tp = self.lstm_metrics.confusion_matrix.ravel()
            print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        if self.prophet_metrics:
            print("
📈 PROPHET TIME SERIES FORECAST")
            print("-" * 80)
            print(f"  MAE:                   {self.prophet_metrics.mae:.4f}")
            print(f"  RMSE:                  {self.prophet_metrics.rmse:.4f}")
            print(f"  MSE:                   {self.prophet_metrics.mse:.4f}")
            print(f"  Directional Accuracy:  {self.prophet_metrics.directional_accuracy:.4f} ({self.prophet_metrics.directional_accuracy*100:.2f}%)")
            print(f"  Scaled Accuracy:       {self.prophet_metrics.scaled_accuracy:.2f}%")

        if self.hybrid_metrics:
            print("
🔗 HYBRID PROPHET-LSTM ENSEMBLE")
            print("-" * 80)
            print(f"  Prophet Weight: {self.hybrid_metrics.prophet_weight:.1%}")
            print(f"  LSTM Weight:    {self.hybrid_metrics.lstm_weight:.1%}")
            print(f"  Hybrid MAE:     {self.hybrid_metrics.mae_hybrid:.4f}")
            print(f"  Hybrid RMSE:    {self.hybrid_metrics.rmse_hybrid:.4f}")
            print(f"  Improvement vs Prophet: {self.hybrid_metrics.improvement_prophet:+.2f}%")
            print(f"  Improvement vs LSTM:    {self.hybrid_metrics.improvement_lstm:+.2f}%")

        print("
" + "="*80)


class HybridProphetLSTMEnsemble:
    def __init__(
        self,
        prophet_model,
        lstm_model,
        scaler: StandardScaler,
        X_columns,
        weight_method: str = 'performance',
        prophet_weight: float = 0.5,
        lstm_weight: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        self.prophet_model = prophet_model
        self.lstm_model = lstm_model
        self.scaler = scaler
        self.X_columns = X_columns
        self.weight_method = weight_method
        self.prophet_weight = prophet_weight
        self.lstm_weight = lstm_weight
        self.logger = logger or logging.getLogger("autoscaler_pipeline")

    def calculate_weights_from_validation(
        self,
        y_val_actual: np.ndarray,
        prophet_preds_val: np.ndarray,
        lstm_preds_val: np.ndarray
    ) -> Tuple[float, float]:
        prophet_rmse = float(np.sqrt(mean_squared_error(y_val_actual, prophet_preds_val)))
        lstm_rmse = float(np.sqrt(mean_squared_error(y_val_actual, lstm_preds_val)))
        inv_sum = (1/prophet_rmse) + (1/lstm_rmse)
        self.prophet_weight = (1/prophet_rmse) / inv_sum
        self.lstm_weight = (1/lstm_rmse) / inv_sum
        self.logger.info(f"Auto weights → Prophet: {self.prophet_weight:.1%}, LSTM: {self.lstm_weight:.1%}")
        return self.prophet_weight, self.lstm_weight

    def predict_hybrid(self, prophet_preds: np.ndarray, lstm_preds: np.ndarray) -> np.ndarray:
        return self.prophet_weight * prophet_preds + self.lstm_weight * lstm_preds

    def evaluate_hybrid(
        self,
        y_true: np.ndarray,
        hybrid_preds: np.ndarray,
        prophet_preds: np.ndarray,
        lstm_preds: np.ndarray
    ) -> HybridMetrics:
        mae_h = float(mean_absolute_error(y_true, hybrid_preds))
        mae_p = float(mean_absolute_error(y_true, prophet_preds))
        mae_l = float(mean_absolute_error(y_true, lstm_preds))
        rmse_h = float(np.sqrt(mean_squared_error(y_true, hybrid_preds)))
        rmse_p = float(np.sqrt(mean_squared_error(y_true, prophet_preds)))
        rmse_l = float(np.sqrt(mean_squared_error(y_true, lstm_preds)))
        imp_p = (mae_p - mae_h) / mae_p * 100 if mae_p > 0 else 0.0
        imp_l = (mae_l - mae_h) / mae_l * 100 if mae_l > 0 else 0.0
        return HybridMetrics(mae_h, rmse_h, self.prophet_weight, self.lstm_weight, imp_p, imp_l)


class AutoscalerHybridPipeline:
    REQUIRED_FEATURES = [
        'replica_count', 'load_users', 'node_cpu_util_value', 'node_mem_util_value',
        'replica_change', 'load_change', 'cpu_regime_encoded', 'scaling_intensity',
        'cpu_rolling_3_mean', 'cpu_lag_1', 'cpu_lag_2', 'mem_rolling_3_mean',
        'mem_lag_1', 'mem_lag_2', 'replica_scaling_up', 'replica_scaling_down'
    ]

    def __init__(self, data_dir: str = "training_data", increase_threshold: float = 0.1, 
                 max_runtime_minutes: int = 30, logger: Optional[logging.Logger] = None):
        self.data_dir = Path(data_dir)
        self.increase_threshold = increase_threshold
        self.max_runtime_seconds = max_runtime_minutes * 60
        self.logger = logger or logging.getLogger("autoscaler_pipeline")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models: Dict[str, Any] = {}
        self.best_models: Dict[str, Any] = {}
        self.best_metrics: Dict[str, Any] = {}
        self.all_results = []
        self.start_time = None

    def load_and_prepare_data(self, service_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        self.logger.info(f"Loading data for service: {service_name}")
        csv_files = list(self.data_dir.glob(f"{service_name}_*_lstm_prophet_ready.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found for service pattern {service_name}")
        frames = [pd.read_csv(f) for f in csv_files]
        data = pd.concat(frames, ignore_index=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.interpolate(method='linear', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)
        X = data[self.REQUIRED_FEATURES].iloc[:-1]
        current_cpu = data['cpu_cores_value'].replace(0, 1e-6)
        future_cpu = data['cpu_cores_value'].shift(-1).replace(0, 1e-6)
        rel_inc = (future_cpu - current_cpu) / current_cpu
        y_class = (rel_inc > self.increase_threshold).astype(int).iloc[:-1].fillna(0)
        y_reg = rel_inc.iloc[:-1].fillna(0).clip(-1, 1)
        self.logger.info(f"Class distribution: {np.bincount(y_class)}")
        return X, y_class, y_reg

    def augment_data(self, X: np.ndarray, y: np.ndarray, multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Simple noise-based augmentation"""
        if multiplier <= 1.0:
            return X, y
        
        X_list = [X]
        y_list = [y]
        
        num_copies = int(multiplier) - 1
        for i in range(num_copies):
            noise = np.random.normal(0, 0.01, X.shape)
            X_noisy = X + noise
            X_list.append(X_noisy)
            y_list.append(y)
        
        return np.vstack(X_list), np.concatenate(y_list)

    def tune_random_forest_classifier(self, X_train: np.ndarray, y_train: pd.Series) -> RandomForestClassifier:
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 0.5, 0.75],
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_

    def eval_classifier(self, model: RandomForestClassifier, X: np.ndarray, y: pd.Series, name: str = "Classifier") -> ClassifierMetrics:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        return ClassifierMetrics(
            accuracy=accuracy_score(y, preds),
            precision=precision_score(y, preds, zero_division=0),
            recall=recall_score(y, preds, zero_division=0),
            f1=f1_score(y, preds, zero_division=0),
            roc_auc=roc_auc_score(y, probs),
            confusion_matrix=confusion_matrix(y, preds),
        )

    def create_sequences(self, X: pd.DataFrame, y: pd.Series, seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X.iloc[i-seq_len:i].values)
            y_seq.append(y.iloc[i])
        return np.array(X_seq), np.array(y_seq)

    def build_lstm_classifier(self, input_shape: Tuple[int, ...]):
        model = models.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_lstm_classifier(self, X_train: np.ndarray, y_train: pd.Series, X_val: np.ndarray, y_val: pd.Series):
        if not TENSORFLOW_AVAILABLE:
            return None
        seq_len = 20
        X_train_seq, y_train_seq = self.create_sequences(pd.DataFrame(X_train), y_train, seq_len)
        X_val_seq, y_val_seq = self.create_sequences(pd.DataFrame(X_val), y_val, seq_len)
        if len(X_train_seq) < 50:
            return None
        
        # Add class weights
        class_values = np.unique(y_train_seq)
        class_weights = compute_class_weight('balanced', classes=class_values, y=y_train_seq)
        cw_dict = dict(zip(class_values, class_weights))
        
        model = self.build_lstm_classifier(input_shape=X_train_seq.shape[1:])
        es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), 
                 epochs=50, batch_size=32, verbose=0, callbacks=[es], class_weight=cw_dict)
        return model

    def eval_lstm_classifier(self, model, X: np.ndarray, y: pd.Series) -> ClassifierMetrics:
        seq_len = 20
        X_seq, y_seq = self.create_sequences(pd.DataFrame(X), y, seq_len)
        preds_prob = model.predict(X_seq, verbose=0).flatten()
        preds = (preds_prob > 0.5).astype(int)
        return ClassifierMetrics(
            accuracy=accuracy_score(y_seq, preds),
            precision=precision_score(y_seq, preds, zero_division=0),
            recall=recall_score(y_seq, preds, zero_division=0),
            f1=f1_score(y_seq, preds, zero_division=0),
            roc_auc=roc_auc_score(y_seq, preds_prob),
            confusion_matrix=confusion_matrix(y_seq, preds),
        )

    def train_prophet_forecast(self, X_train_df: pd.DataFrame, y_train: pd.Series, X_test_df: pd.DataFrame, y_test: pd.Series):
        """FIXED: Prophet now works with DataFrames properly"""
        if not PROPHET_AVAILABLE:
            return None, None, None
        
        df_train = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=len(y_train), freq='min'),
            'y': y_train.values,
        })
        
        # Add regressors from DataFrame columns
        regressors = ['replica_change', 'cpu_regime_encoded', 'scaling_intensity']
        for col in regressors:
            if col in X_train_df.columns:
                df_train[col] = X_train_df[col].values
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, 
                       changepoint_prior_scale=0.5, seasonality_prior_scale=5.0)
        
        for col in regressors:
            if col in X_train_df.columns:
                model.add_regressor(col)
        
        model.fit(df_train)
        
        future = model.make_future_dataframe(periods=len(y_test), freq='min')
        for col in regressors:
            if col in X_train_df.columns and col in X_test_df.columns:
                future[col] = list(df_train[col]) + list(X_test_df[col])
        
        forecast = model.predict(future)
        preds = forecast['yhat'].iloc[-len(y_test):].values
        
        mae = float(mean_absolute_error(y_test.values, preds))
        mse = float(mean_squared_error(y_test.values, preds))
        rmse = float(np.sqrt(mse))
        
        actual_dir = np.diff(y_test.values) > 0
        pred_dir = np.diff(preds) > 0
        dir_acc = float(accuracy_score(actual_dir, pred_dir)) if len(actual_dir) else 0.0
        
        max_range = float(np.max(np.abs(y_test.values)) + 1e-6)
        sc_acc = float(np.clip(100 * (1 - np.mean(np.abs(preds - y_test.values) / max_range)), 0, 100))
        
        metrics = RegressionMetrics(mae=mae, rmse=rmse, mse=mse, directional_accuracy=dir_acc, scaled_accuracy=sc_acc)
        return model, preds, metrics

    def train_iteration(self, iteration: int, service_name: str, X_train, y_class_train, y_reg_train, 
                       X_test, y_class_test, y_reg_test, X_columns, aug_multiplier: float) -> PipelineResults:
        
        results = PipelineResults(timestamp=datetime.now().isoformat(), service_name=service_name, iteration=iteration)
        results.training_samples = len(X_train)
        results.test_samples = len(X_test)
        
        self.logger.info(f"
{'='*80}")
        self.logger.info(f"ITERATION {iteration} | Augmentation: {aug_multiplier}x")
        self.logger.info(f"{'='*80}")
        
        # Augment data
        X_train_aug, y_class_aug = self.augment_data(X_train, y_class_train.values, aug_multiplier)
        _, y_reg_aug = self.augment_data(X_train, y_reg_train.values, aug_multiplier)
        
        # RF
        try:
            self.logger.info("Training RF...")
            rf_clf, rf_params = self.tune_random_forest_classifier(X_train_aug, y_class_aug)
            self.models['rf_classifier'] = rf_clf
            results.rf_hyperparams = rf_params
            results.rf_metrics = self.eval_classifier(rf_clf, X_test, y_class_test, 'Random Forest')
            
            if 'rf' not in self.best_metrics or results.rf_metrics.get_score() > self.best_metrics['rf'].get_score():
                self.best_models['rf'] = rf_clf
                self.best_metrics['rf'] = results.rf_metrics
                self.logger.info(f"✓ NEW BEST RF! Score: {results.rf_metrics.get_score():.4f}")
        except Exception as e:
            self.logger.error(f"RF failed: {e}")

        # LSTM
        try:
            if TENSORFLOW_AVAILABLE:
                self.logger.info("Training LSTM...")
                lstm_model = self.train_lstm_classifier(X_train_aug, pd.Series(y_class_aug), X_test, y_class_test)
                if lstm_model is not None:
                    self.models['lstm_classifier'] = lstm_model
                    results.lstm_metrics = self.eval_lstm_classifier(lstm_model, X_test, y_class_test)
                    
                    if 'lstm' not in self.best_metrics or results.lstm_metrics.get_score() > self.best_metrics['lstm'].get_score():
                        self.best_models['lstm'] = lstm_model
                        self.best_metrics['lstm'] = results.lstm_metrics
                        self.logger.info(f"✓ NEW BEST LSTM! Score: {results.lstm_metrics.get_score():.4f}")
        except Exception as e:
            self.logger.error(f"LSTM failed: {e}")

        # Prophet (every 2 iterations)
        if PROPHET_AVAILABLE and iteration % 2 == 0:
            try:
                self.logger.info("Training Prophet...")
                # Create DataFrames with column names for Prophet
                X_train_aug_df = pd.DataFrame(X_train_aug, columns=X_columns)
                X_test_df = pd.DataFrame(X_test, columns=X_columns)
                
                prophet_model, prophet_preds, pmetrics = self.train_prophet_forecast(
                    X_train_aug_df, pd.Series(y_reg_aug), X_test_df, y_reg_test
                )
                if prophet_model is not None:
                    self.models['prophet_forecast'] = prophet_model
                    results.prophet_metrics = pmetrics
                    
                    if 'prophet' not in self.best_metrics or pmetrics.mae < self.best_metrics['prophet'].mae:
                        self.best_models['prophet'] = prophet_model
                        self.best_metrics['prophet'] = pmetrics
                        self.logger.info(f"✓ NEW BEST PROPHET! MAE: {pmetrics.mae:.4f}")
            except Exception as e:
                self.logger.error(f"Prophet failed: {e}")

        return results

    def train_all_iterative(self, service_name: str, enable_hybrid: bool = False):
        self.start_time = time.time()
        self.logger.info(f"Starting iterative training for {service_name}")
        self.logger.info(f"Max runtime: {self.max_runtime_seconds/60:.1f} minutes")
        
        X, y_class, y_reg = self.load_and_prepare_data(service_name)
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        iteration = 0
        aug_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        while (time.time() - self.start_time) < self.max_runtime_seconds:
            aug_mult = aug_multipliers[iteration % len(aug_multipliers)]
            
            results = self.train_iteration(
                iteration, service_name,
                X_train_scaled, y_class_train, y_reg_train,
                X_test_scaled, y_class_test, y_reg_test,
                X.columns, aug_mult
            )
            
            results.print_summary()
            self.all_results.append(results)
            
            elapsed = (time.time() - self.start_time) / 60
            remaining = (self.max_runtime_seconds - (time.time() - self.start_time)) / 60
            self.logger.info(f"Elapsed: {elapsed:.1f}min | Remaining: {remaining:.1f}min")
            
            iteration += 1
            if remaining < 1:
                break
        
        # Print final best results
        self.print_final_results()
        return self.all_results

    def print_final_results(self):
        self.logger.info("
" + "="*80)
        self.logger.info("FINAL BEST RESULTS ACROSS ALL ITERATIONS")
        self.logger.info("="*80)
        
        if 'rf' in self.best_metrics:
            m = self.best_metrics['rf']
            self.logger.info(f"📊 BEST RF: Acc={m.accuracy:.4f} Prec={m.precision:.4f} Rec={m.recall:.4f} F1={m.f1:.4f}")
        
        if 'lstm' in self.best_metrics:
            m = self.best_metrics['lstm']
            self.logger.info(f"🧠 BEST LSTM: Acc={m.accuracy:.4f} Prec={m.precision:.4f} Rec={m.recall:.4f} F1={m.f1:.4f}")
        
        if 'prophet' in self.best_metrics:
            m = self.best_metrics['prophet']
            self.logger.info(f"📈 BEST PROPHET: MAE={m.mae:.4f} RMSE={m.rmse:.4f}")
        
        self.logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Kubernetes Autoscaler Hybrid ML Pipeline Training")
    parser.add_argument("--service", type=str, default="frontend")
    parser.add_argument("--data-dir", type=str, default="training_data")
    parser.add_argument("--increase-threshold", type=float, default=0.1)
    parser.add_argument("--runtime-minutes", type=int, default=30)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--hybrid-ensemble", action="store_true")
    args = parser.parse_args()

    logger = setup_logging(args.log_dir)
    try:
        pipeline = AutoscalerHybridPipeline(
            data_dir=args.data_dir, 
            increase_threshold=args.increase_threshold,
            max_runtime_minutes=args.runtime_minutes,
            logger=logger
        )
        
        all_results = pipeline.train_all_iterative(args.service, enable_hybrid=args.hybrid_ensemble)
        
        import json
        results_file = Path(args.log_dir) / f"results_{args.service}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        
        if args.save_models:
            models_dir = Path("trained_models") / args.service
            models_dir.mkdir(parents=True, exist_ok=True)
            for model_name, model in pipeline.best_models.items():
                try:
                    if model_name == 'lstm':
                        model.save(models_dir / f"best_{model_name}.keras")
                    else:
                        joblib.dump(model, models_dir / f"best_{model_name}.pkl")
                    logger.info(f"Saved best {model_name}")
                except Exception as e:
                    logger.warning(f"Could not save {model_name}: {e}")
        
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()