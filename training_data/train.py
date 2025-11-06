#!/usr/bin/env python3
"""
Hybrid ML Training Pipeline: RandomForest + LSTM + Prophet
==========================================================
Trains three models and creates a 40% LSTM + 60% Prophet hybrid
"""

import os
import sys
import argparse
import logging
import warnings
import json
import time
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import traceback
import glob

# CPU optimization
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(mp.cpu_count())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Core ML Stack
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import joblib

# Check library availability
TENSORFLOW_AVAILABLE = False
PROPHET_AVAILABLE = False

# TensorFlow/LSTM
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    
    # Configure GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU configured for TensorFlow")
        except RuntimeError:
            pass
    
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow/LSTM available")
except ImportError:
    print("❌ TensorFlow/LSTM not available")

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    print("✅ Prophet available")
except ImportError:
    print("❌ Prophet not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class HybridMLPipeline:
    """Hybrid ML Pipeline: RandomForest + LSTM + Prophet + 40/60 Hybrid"""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = {}
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self, service_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training data"""
        print(f"🔄 Loading data for {service_name}...")
        
        # Find all CSV files for the service
        csv_files = list(self.data_dir.glob(f"*{service_name}*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found for service: {service_name}")
        
        # Load and combine all files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if len(df) > 0 and 'cpu_cores_value' in df.columns:
                    dataframes.append(df)
                    print(f"  📁 Loaded: {csv_file.name} ({len(df)} rows)")
            except Exception as e:
                print(f"  ⚠️  Skipped {csv_file.name}: {e}")
        
        if not dataframes:
            raise ValueError(f"No valid data files found for {service_name}")
        
        # Combine all data
        combined_data = pd.concat(dataframes, ignore_index=True)
        print(f"✅ Combined data: {len(combined_data)} rows, {len(combined_data.columns)} columns")
        
        # Create features and target
        X, y = self._prepare_features_and_target(combined_data)
        
        print(f"✅ Prepared: Features={X.shape}, Target={y.shape}")
        return X, y
    
    def _prepare_features_and_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
        
        # Create contextual features if not present
        if 'replica_change' not in data.columns and 'replica_count' in data.columns:
            data['replica_change'] = data['replica_count'].diff().fillna(0)
        
        if 'load_change' not in data.columns and 'load_users' in data.columns:
            data['load_change'] = data['load_users'].diff().fillna(0)
        
        if 'cpu_regime_encoded' not in data.columns and 'cpu_cores_value' in data.columns:
            # Create CPU regime
            cpu_low = data['cpu_cores_value'].quantile(0.33)
            cpu_high = data['cpu_cores_value'].quantile(0.67)
            
            cpu_regime = ['medium'] * len(data)
            cpu_regime = ['low' if cpu <= cpu_low else 'high' if cpu >= cpu_high else 'medium' 
                         for cpu in data['cpu_cores_value']]
            
            data['cpu_regime'] = cpu_regime
            data['cpu_regime_encoded'] = self.label_encoder.fit_transform(cpu_regime)
        
        # Define target (predict future CPU usage)
        if 'cpu_cores_value' in data.columns:
            # Use relative change (better for regime transitions)
            current_cpu = data['cpu_cores_value']
            future_cpu = data['cpu_cores_value'].shift(-1)
            
            # Predict log ratio to handle large changes
            target = np.log1p(future_cpu / (current_cpu + 1e-10))
            target = target.dropna()  # Remove last row
        else:
            raise ValueError("No cpu_cores_value column found")
        
        # Select features (exclude target and identifiers)
        exclude_columns = [
            'cpu_cores_value',  # Don't leak target
            'timestamp', 'experiment_id', 'service', 'scenario',
            'cpu_regime'  # Keep encoded version only
        ]
        
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        if not feature_columns:
            # Create basic features
            features = pd.DataFrame({
                'replica_count': data.get('replica_count', 4),
                'load_users': data.get('load_users', 100),
                'time_trend': np.linspace(0, 1, len(data))
            })
        else:
            features = data[feature_columns].copy()
        
        # Align features with target (remove last row)
        features = features.iloc[:-1]
        
        # Clean data
        features = features.fillna(0)
        target = target.fillna(0)
        
        return features, target
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        print("🌲 Training Random Forest...")
        
        try:
            # Scale features
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.7,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            rf_predictions = rf_model.predict(X_test_scaled)
            
            # Calculate metrics
            rf_metrics = self._calculate_metrics(y_test, rf_predictions, "Random Forest")
            
            self.models['random_forest'] = rf_model
            self.results['random_forest'] = {
                'model': rf_model,
                'predictions': rf_predictions,
                'metrics': rf_metrics
            }
            
            print(f"✅ Random Forest - R²: {rf_metrics['r2']:.4f}, MAE: {rf_metrics['mae']:.6f}")
            return rf_metrics
            
        except Exception as e:
            print(f"❌ Random Forest training failed: {e}")
            return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'mape': float('inf')}
    
    def train_lstm(self, X_train, X_test, y_train, y_test):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("⚠️  LSTM skipped - TensorFlow not available")
            return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'mape': float('inf')}
        
        print("🧠 Training LSTM...")
        
        try:
            # Create sequences for LSTM
            sequence_length = min(10, len(X_train) // 3)
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
            
            if len(X_train_seq) < 10:
                print("⚠️  LSTM skipped - insufficient sequence data")
                return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'mape': float('inf')}
            
            # Build LSTM model
            lstm_model = models.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2])),
                layers.Dropout(0.2),
                layers.LSTM(32, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            
            lstm_model.compile(
                optimizer='adam',
                loss='huber',
                metrics=['mae']
            )
            
            # Train LSTM
            early_stop = callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            
            lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Make predictions
            lstm_predictions = lstm_model.predict(X_test_seq, verbose=0).flatten()
            
            # Calculate metrics
            lstm_metrics = self._calculate_metrics(y_test_seq, lstm_predictions, "LSTM")
            
            self.models['lstm'] = lstm_model
            self.results['lstm'] = {
                'model': lstm_model,
                'predictions': lstm_predictions,
                'test_target': y_test_seq,
                'metrics': lstm_metrics,
                'sequence_length': sequence_length
            }
            
            print(f"✅ LSTM - R²: {lstm_metrics['r2']:.4f}, MAE: {lstm_metrics['mae']:.6f}")
            return lstm_metrics
            
        except Exception as e:
            print(f"❌ LSTM training failed: {e}")
            return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'mape': float('inf')}
    
    def train_prophet(self, X_train, X_test, y_train, y_test):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            print("⚠️  Prophet skipped - Prophet not available")
            return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'mape': float('inf')}
        
        print("📈 Training Prophet...")
        
        try:
            # Prepare Prophet data
            prophet_data = pd.DataFrame({
                'ds': pd.date_range(start='2024-01-01', periods=len(y_train), freq='min'),
                'y': y_train.values
            })
            
            # Initialize Prophet
            prophet_model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            # Add external regressors (contextual features)
            contextual_features = [col for col in X_train.columns 
                                 if any(keyword in col.lower() for keyword in ['regime', 'change', 'scaling', 'pattern'])]
            
            for feature in contextual_features[:5]:  # Limit to 5 best features
                try:
                    prophet_model.add_regressor(feature)
                    prophet_data[feature] = X_train[feature].values
                except Exception:
                    continue
            
            # Train Prophet
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prophet_model.fit(prophet_data)
            
            # Make future dataframe for predictions
            future = prophet_model.make_future_dataframe(periods=len(y_test), freq='min')
            
            # Add regressor values for prediction period
            for feature in contextual_features[:5]:
                if feature in prophet_data.columns:
                    try:
                        # Combine train and test regressor values
                        train_values = prophet_data[feature].tolist()
                        test_values = X_test[feature].tolist() if len(X_test) > 0 else [0] * len(y_test)
                        future[feature] = train_values + test_values[:len(y_test)]
                    except Exception:
                        future[feature] = 0
            
            # Make predictions
            forecast = prophet_model.predict(future)
            prophet_predictions = forecast['yhat'].iloc[-len(y_test):].values
            
            # Calculate metrics
            prophet_metrics = self._calculate_metrics(y_test, prophet_predictions, "Prophet")
            
            self.models['prophet'] = prophet_model
            self.results['prophet'] = {
                'model': prophet_model,
                'predictions': prophet_predictions,
                'metrics': prophet_metrics,
                'regressors': contextual_features[:5]
            }
            
            print(f"✅ Prophet - R²: {prophet_metrics['r2']:.4f}, MAE: {prophet_metrics['mae']:.6f}")
            return prophet_metrics
            
        except Exception as e:
            print(f"❌ Prophet training failed: {e}")
            return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'mape': float('inf')}
    
    def create_hybrid_model(self, X_test, y_test, lstm_weight=0.4, prophet_weight=0.6):
        """Create hybrid model: 40% LSTM + 60% Prophet"""
        print(f"🎭 Creating Hybrid Model ({lstm_weight*100:.0f}% LSTM + {prophet_weight*100:.0f}% Prophet)...")
        
        try:
            # Get predictions from both models
            lstm_result = self.results.get('lstm')
            prophet_result = self.results.get('prophet')
            
            if not lstm_result or not prophet_result:
                print("❌ Hybrid model requires both LSTM and Prophet to be trained")
                return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'mape': float('inf')}
            
            lstm_predictions = lstm_result['predictions']
            prophet_predictions = prophet_result['predictions']
            
            # Align predictions (use minimum length)
            min_length = min(len(lstm_predictions), len(prophet_predictions), len(y_test))
            
            lstm_pred_aligned = lstm_predictions[:min_length]
            prophet_pred_aligned = prophet_predictions[:min_length]
            y_test_aligned = y_test.iloc[:min_length] if hasattr(y_test, 'iloc') else y_test[:min_length]
            
            # Create hybrid predictions
            hybrid_predictions = (
                lstm_weight * lstm_pred_aligned + 
                prophet_weight * prophet_pred_aligned
            )
            
            # Calculate hybrid metrics
            hybrid_metrics = self._calculate_metrics(y_test_aligned, hybrid_predictions, "Hybrid (40% LSTM + 60% Prophet)")
            
            self.results['hybrid'] = {
                'predictions': hybrid_predictions,
                'metrics': hybrid_metrics,
                'lstm_weight': lstm_weight,
                'prophet_weight': prophet_weight,
                'lstm_r2': lstm_result['metrics']['r2'],
                'prophet_r2': prophet_result['metrics']['r2']
            }
            
            print(f"✅ Hybrid Model - R²: {hybrid_metrics['r2']:.4f}, MAE: {hybrid_metrics['mae']:.6f}")
            return hybrid_metrics
            
        except Exception as e:
            print(f"❌ Hybrid model creation failed: {e}")
            return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'mape': float('inf')}
    
    def _create_sequences(self, X, y, sequence_length):
        """Create sequences for LSTM training"""
        if len(X) <= sequence_length:
            return np.array([]), np.array([])
        
        # Scale features for LSTM
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i] if hasattr(y, 'iloc') else y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics"""
        try:
            # Ensure arrays are proper format
            if hasattr(y_true, 'values'):
                y_true = y_true.values
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            
            # Handle different lengths
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # MAPE (handle division by zero)
            try:
                mape = mean_absolute_percentage_error(y_true, y_pred) if np.any(y_true != 0) else float('inf')
            except:
                mape = float('inf')
            
            # RMSE
            rmse = np.sqrt(mse)
            
            return {
                'r2': r2,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'samples': len(y_true)
            }
            
        except Exception as e:
            print(f"❌ Metrics calculation failed for {model_name}: {e}")
            return {'r2': 0, 'mae': float('inf'), 'mse': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'samples': 0}
    
    def train_all_models(self, service_name: str):
        """Train all models and create hybrid"""
        print(f"🚀 Starting training pipeline for {service_name}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load and prepare data
            X, y = self.load_and_prepare_data(service_name)
            
            # Split data (chronological split)
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            print(f"📊 Data Split: Train={len(X_train)}, Test={len(X_test)}")
            print("=" * 60)
            
            # Train individual models
            rf_metrics = self.train_random_forest(X_train, X_test, y_train, y_test)
            lstm_metrics = self.train_lstm(X_train, X_test, y_train, y_test)
            prophet_metrics = self.train_prophet(X_train, X_test, y_train, y_test)
            
            print("\n" + "=" * 60)
            
            # Create hybrid model
            hybrid_metrics = self.create_hybrid_model(X_test, y_test, lstm_weight=0.4, prophet_weight=0.6)
            
            training_time = time.time() - start_time
            
            # Display comprehensive results
            self._display_results(rf_metrics, lstm_metrics, prophet_metrics, hybrid_metrics, training_time)
            
            return {
                'random_forest': rf_metrics,
                'lstm': lstm_metrics,
                'prophet': prophet_metrics,
                'hybrid': hybrid_metrics,
                'training_time': training_time
            }
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            return None
    
    def _display_results(self, rf_metrics, lstm_metrics, prophet_metrics, hybrid_metrics, training_time):
        """Display comprehensive training results"""
        
        print("\n🎯 HYBRID ML TRAINING RESULTS")
        print("=" * 80)
        
        # Model comparison table
        print(f"{'Model':<25} {'R²':<12} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'MAPE':<12}")
        print("-" * 80)
        
        models_data = [
            ("Random Forest", rf_metrics),
            ("LSTM", lstm_metrics), 
            ("Prophet", prophet_metrics),
            ("🎭 Hybrid (40%+60%)", hybrid_metrics)
        ]
        
        for model_name, metrics in models_data:
            r2 = metrics['r2']
            mae = metrics['mae'] if metrics['mae'] != float('inf') else 'N/A'
            mse = metrics['mse'] if metrics['mse'] != float('inf') else 'N/A'
            rmse = metrics['rmse'] if metrics['rmse'] != float('inf') else 'N/A'
            mape = f"{metrics['mape']:.4f}" if metrics['mape'] != float('inf') else 'N/A'
            
            # Color coding based on R²
            if r2 > 0.6:
                status = "🟢 EXCELLENT"
            elif r2 > 0.3:
                status = "🟡 GOOD"
            elif r2 > 0.0:
                status = "🔵 ACCEPTABLE"
            else:
                status = "🔴 POOR"
            
            print(f"{model_name:<25} {r2:<12.4f} {mae:<12} {mse:<12} {rmse:<12} {mape:<12} {status}")
        
        print("=" * 80)
        
        # Best model identification
        best_r2 = max([rf_metrics['r2'], lstm_metrics['r2'], prophet_metrics['r2'], hybrid_metrics['r2']])
        
        if hybrid_metrics['r2'] == best_r2:
            best_model = "🎭 Hybrid Model"
        elif lstm_metrics['r2'] == best_r2:
            best_model = "🧠 LSTM"
        elif prophet_metrics['r2'] == best_r2:
            best_model = "📈 Prophet"
        else:
            best_model = "🌲 Random Forest"
        
        print(f"🏆 BEST MODEL: {best_model} (R² = {best_r2:.4f})")
        
        # Hybrid model details
        if 'hybrid' in self.results:
            hybrid_result = self.results['hybrid']
            print(f"\n🎭 HYBRID MODEL BREAKDOWN:")
            print(f"  • LSTM Weight: {hybrid_result['lstm_weight']*100:.0f}% (R² = {hybrid_result['lstm_r2']:.4f})")
            print(f"  • Prophet Weight: {hybrid_result['prophet_weight']*100:.0f}% (R² = {hybrid_result['prophet_r2']:.4f})")
            print(f"  • Combined R²: {hybrid_result['metrics']['r2']:.4f}")
            
            # Performance improvement
            individual_best = max(hybrid_result['lstm_r2'], hybrid_result['prophet_r2'])
            improvement = ((hybrid_result['metrics']['r2'] - individual_best) / max(0.001, abs(individual_best))) * 100
            
            if improvement > 0:
                print(f"  • 📈 Hybrid Improvement: +{improvement:.1f}% over best individual model")
            else:
                print(f"  • 📉 Hybrid Performance: {improvement:.1f}% vs best individual model")
        
        print(f"\n⏱️  Total Training Time: {training_time:.2f} seconds")
        print("=" * 80)
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if best_r2 > 0.5:
            print("🎉 Excellent performance! Model ready for production.")
        elif best_r2 > 0.2:
            print("✅ Good performance! Consider additional feature engineering.")
        elif best_r2 > 0.0:
            print("⚠️  Acceptable performance. More training data recommended.")
        else:
            print("❌ Poor performance. Check data quality and feature engineering.")
        
        # Feature importance (Random Forest)
        if 'random_forest' in self.results and hasattr(self.results['random_forest']['model'], 'feature_importances_'):
            rf_model = self.results['random_forest']['model']
            importances = rf_model.feature_importances_
            feature_names = X_train.columns if 'X_train' in locals() else ['feature_' + str(i) for i in range(len(importances))]
            
            top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n🎯 TOP 5 FEATURES (Random Forest):")
            for feature, importance in top_features:
                print(f"  • {feature}: {importance:.4f}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Hybrid ML Training: RandomForest + LSTM + Prophet")
    parser.add_argument("--service", type=str, default="frontend", help="Service name to train")
    parser.add_argument("--data-dir", type=str, default="training_data", help="Data directory")
    parser.add_argument("--save-models", action="store_true", help="Save trained models")
    parser.add_argument("--lstm-weight", type=float, default=0.4, help="LSTM weight in hybrid (default: 0.4)")
    parser.add_argument("--prophet-weight", type=float, default=0.6, help="Prophet weight in hybrid (default: 0.6)")
    
    args = parser.parse_args()
    
    # Validate weights
    if abs(args.lstm_weight + args.prophet_weight - 1.0) > 0.01:
        print(f"❌ Error: Weights must sum to 1.0 (got {args.lstm_weight + args.prophet_weight})")
        return
    
    try:
        print("🚀 HYBRID ML TRAINING PIPELINE")
        print("=" * 80)
        print(f"Service: {args.service}")
        print(f"Data Directory: {args.data_dir}")
        print(f"Hybrid Weights: {args.lstm_weight*100:.0f}% LSTM + {args.prophet_weight*100:.0f}% Prophet")
        print(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
        print(f"Prophet Available: {PROPHET_AVAILABLE}")
        print("=" * 80)
        
        # Initialize pipeline
        pipeline = HybridMLPipeline(data_dir=args.data_dir)
        
        # Train all models
        results = pipeline.train_all_models(args.service)
        
        if results:
            # Save models if requested
            if args.save_models:
                model_dir = Path("models") / args.service
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Save models
                if 'random_forest' in pipeline.models:
                    joblib.dump(pipeline.models['random_forest'], model_dir / "random_forest.joblib")
                    print(f"💾 Saved Random Forest model")
                
                if 'lstm' in pipeline.models and TENSORFLOW_AVAILABLE:
                    pipeline.models['lstm'].save(str(model_dir / "lstm_model"))
                    print(f"💾 Saved LSTM model")
                
                # Save hybrid configuration
                hybrid_config = {
                    'lstm_weight': args.lstm_weight,
                    'prophet_weight': args.prophet_weight,
                    'results': {k: {metric: float(v) if v != float('inf') else None 
                               for metric, v in result['metrics'].items()} 
                              for k, result in pipeline.results.items() if 'metrics' in result}
                }
                
                with open(model_dir / "hybrid_config.json", 'w') as f:
                    json.dump(hybrid_config, f, indent=2)
                print(f"💾 Saved hybrid configuration")
            
            print(f"\n🎉 Training pipeline completed successfully!")
        else:
            print(f"\n❌ Training pipeline failed!")
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
