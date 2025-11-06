#!/usr/bin/env python3
"""
Quick fix for DataFrame enhancement errors
Adds LSTM + Prophet friendly features to existing CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path

def quick_enhance_csv_files():
    """Fix and enhance all collected CSV files for LSTM + Prophet"""
    
    data_dir = Path('training_data')
    csv_files = list(data_dir.glob('frontend_*.csv'))
    
    print(f"🔧 Found {len(csv_files)} CSV files to enhance...")
    
    enhanced_count = 0
    
    for csv_file in csv_files:
        try:
            print(f"Processing: {csv_file.name}")
            
            # Load original data
            df = pd.read_csv(csv_file)
            
            if len(df) < 5:
                print(f"  ⚠️  Skipping {csv_file.name}: Too few rows ({len(df)})")
                continue
            
            # Add basic LSTM + Prophet friendly features
            
            # 1. Add scaling context features
            if 'replica_count' in df.columns:
                df['replica_change'] = df['replica_count'].diff().fillna(0)
                df['replica_scaling_up'] = (df['replica_change'] > 0).astype(int)
                df['replica_scaling_down'] = (df['replica_change'] < 0).astype(int)
            else:
                df['replica_change'] = 0
                df['replica_scaling_up'] = 0
                df['replica_scaling_down'] = 0
            
            if 'load_users' in df.columns:
                df['load_change'] = df['load_users'].diff().fillna(0)
                df['load_increasing'] = (df['load_change'] > 0).astype(int)
                df['load_decreasing'] = (df['load_change'] < 0).astype(int)
            else:
                df['load_change'] = 0
                df['load_increasing'] = 0
                df['load_decreasing'] = 0
            
            # 2. Add CPU regime classification
            if 'cpu_cores_value' in df.columns:
                # Use quantiles for dynamic regime boundaries
                cpu_low = df['cpu_cores_value'].quantile(0.33)
                cpu_high = df['cpu_cores_value'].quantile(0.67)
                
                df['cpu_regime'] = 'medium'  # Default
                df.loc[df['cpu_cores_value'] <= cpu_low, 'cpu_regime'] = 'low'
                df.loc[df['cpu_cores_value'] >= cpu_high, 'cpu_regime'] = 'high'
                
                # Numeric encoding for ML
                regime_map = {'low': 0, 'medium': 1, 'high': 2}
                df['cpu_regime_encoded'] = df['cpu_regime'].map(regime_map)
                
                # Regime changes (LSTM loves these transitions!)
                df['cpu_regime_change'] = (df['cpu_regime_encoded'].diff() != 0).astype(int)
            else:
                df['cpu_regime'] = 'medium'
                df['cpu_regime_encoded'] = 1
                df['cpu_regime_change'] = 0
            
            # 3. Add time-based features (Prophet seasonality)
            df['time_idx'] = range(len(df))
            if len(df) > 1:
                # Normalize time for patterns
                normalized_time = df['time_idx'] / (len(df) - 1)
                
                # Realistic hourly pattern (24-hour cycle simulation)
                hour_sim = (normalized_time * 24) % 24
                df['hour_pattern'] = 0.5 + 0.3 * np.sin(2 * np.pi * (hour_sim - 8) / 24)
                
                # Day of week pattern (7-day cycle simulation)
                day_sim = (normalized_time * 7) % 7
                df['day_pattern'] = 1.0 + 0.2 * np.sin(2 * np.pi * day_sim / 7)
                
                # Growth trend
                df['growth_trend'] = 1.0 + 0.3 * normalized_time
            else:
                df['hour_pattern'] = 0.5
                df['day_pattern'] = 1.0
                df['growth_trend'] = 1.0
            
            # 4. Add interaction features (LSTM pattern recognition)
            df['replica_load_interaction'] = df.get('replica_count', 4) * df.get('load_users', 100) / 100
            df['scaling_intensity'] = np.abs(df['replica_change']) + np.abs(df['load_change']) / 100
            
            # 5. Add rolling features for LSTM sequences
            if len(df) >= 3:
                if 'cpu_cores_value' in df.columns:
                    df['cpu_rolling_3_mean'] = df['cpu_cores_value'].rolling(3, min_periods=1).mean()
                    df['cpu_rolling_3_std'] = df['cpu_cores_value'].rolling(3, min_periods=1).std().fillna(0)
                
                if 'mem_bytes_value' in df.columns:
                    df['mem_rolling_3_mean'] = df['mem_bytes_value'].rolling(3, min_periods=1).mean()
            
            # 6. Add lag features (LSTM loves these!)
            for lag in [1, 2]:
                if len(df) > lag:
                    if 'cpu_cores_value' in df.columns:
                        df[f'cpu_lag_{lag}'] = df['cpu_cores_value'].shift(lag).fillna(df['cpu_cores_value'].iloc[0])
                    if 'mem_bytes_value' in df.columns:
                        df[f'mem_lag_{lag}'] = df['mem_bytes_value'].shift(lag).fillna(df['mem_bytes_value'].iloc[0])
            
            # 7. Clean any remaining NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Save enhanced version
            enhanced_filename = csv_file.stem + "_lstm_prophet_ready.csv"
            enhanced_path = data_dir / enhanced_filename
            df.to_csv(enhanced_path, index=False)
            
            enhanced_count += 1
            print(f"  ✅ Enhanced: {enhanced_filename} ({len(df)} rows, {len(df.columns)} features)")
            
        except Exception as e:
            print(f"  ❌ Failed to enhance {csv_file.name}: {e}")
    
    print(f"\n🎉 Enhancement Complete: {enhanced_count} files enhanced!")
    print(f"📊 Ready for LSTM + Prophet training with contextual features!")
    return enhanced_count

if __name__ == "__main__":
    quick_enhance_csv_files()
