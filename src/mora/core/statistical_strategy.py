"""
Statistical rightsizing strategy implementation
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class StatisticalRightsizer:
    """Statistical rightsizing strategy based on historical usage patterns"""
    
    def __init__(self, cpu_percentile: float = 95.0, memory_buffer_percentage: float = 15.0):
        """
        Initialize statistical rightsizer
        
        Args:
            cpu_percentile: Percentile to use for CPU requests (default: 95th)
            memory_buffer_percentage: Buffer percentage for memory requests (default: 15%)
        """
        self.cpu_percentile = cpu_percentile
        self.memory_buffer_percentage = memory_buffer_percentage
        
        logger.info(f"Initialized statistical rightsizer with {cpu_percentile}th percentile CPU and {memory_buffer_percentage}% memory buffer")
    
    def parse_cpu_value(self, cpu_str: str) -> float:
        """
        Parse CPU value from Kubernetes format to cores
        
        Args:
            cpu_str: CPU value in Kubernetes format (e.g., "100m", "1", "0.5")
            
        Returns:
            CPU value in cores (float)
        """
        if not cpu_str or cpu_str == "Unknown":
            return 0.0
            
        cpu_str = str(cpu_str).strip()
        
        if cpu_str.endswith('m'):
            # Convert millicores to cores
            return float(cpu_str[:-1]) / 1000.0
        elif cpu_str.endswith('n'):
            # Convert nancores to cores (rare case)
            return float(cpu_str[:-1]) / 1000000000.0
        else:
            # Assume it's already in cores
            try:
                return float(cpu_str)
            except ValueError:
                logger.warning(f"Could not parse CPU value: {cpu_str}")
                return 0.0
    
    def parse_memory_value(self, memory_str: str) -> int:
        """
        Parse memory value from Kubernetes format to bytes
        
        Args:
            memory_str: Memory value in Kubernetes format (e.g., "128Mi", "1Gi", "1073741824")
            
        Returns:
            Memory value in bytes (int)
        """
        if not memory_str or memory_str == "Unknown":
            return 0
            
        memory_str = str(memory_str).strip()
        
        if memory_str.endswith('Ki'):
            return int(float(memory_str[:-2]) * 1024)
        elif memory_str.endswith('Mi'):
            return int(float(memory_str[:-2]) * 1024 * 1024)
        elif memory_str.endswith('Gi'):
            return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
        elif memory_str.endswith('K'):
            return int(float(memory_str[:-1]) * 1000)
        elif memory_str.endswith('M'):
            return int(float(memory_str[:-1]) * 1000 * 1000)
        elif memory_str.endswith('G'):
            return int(float(memory_str[:-1]) * 1000 * 1000 * 1000)
        else:
            try:
                return int(float(memory_str))
            except ValueError:
                logger.warning(f"Could not parse memory value: {memory_str}")
                return 0
    
    def format_cpu_value(self, cores: float) -> str:
        """
        Format CPU value from cores to Kubernetes format
        
        Args:
            cores: CPU value in cores
            
        Returns:
            Formatted CPU string
        """
        if cores < 1.0:
            # Convert to millicores for small values
            millicores = int(cores * 1000)
            return f"{millicores}m"
        else:
            return f"{cores:0.3f}".rstrip('0').rstrip('.')
    
    def format_memory_value(self, bytes_val: int) -> str:
        """
        Format memory value from bytes to Kubernetes format
        
        Args:
            bytes_val: Memory value in bytes
            
        Returns:
            Formatted memory string
        """
        if bytes_val >= 1024**3:  # >= 1 GiB
            return f"{bytes_val / (1024**3):0.1f}Gi"
        elif bytes_val >= 1024**2:  # >= 1 MiB
            return f"{bytes_val / (1024**2):0.0f}Mi"
        elif bytes_val >= 1024:  # >= 1 KiB
            return f"{bytes_val / 1024:0.0f}Ki"
        else:
            return str(bytes_val)
    
    def analyze_cpu_usage(self, cpu_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze CPU usage data and calculate recommendations
        
        Args:
            cpu_df: DataFrame with CPU usage metrics
            
        Returns:
            Dictionary with CPU analysis results
        """
        if cpu_df.empty:
            logger.warning("No CPU data available for analysis")
            return {
                'percentile_value': 0.0,
                'recommended_requests': self.format_cpu_value(0.1),  # Default minimum
                'current_usage_stats': {},
                'has_data': False
            }
        
        try:
            # Find the value column(s) - Prometheus data structure
            value_columns = []
            for col in cpu_df.columns:
                if 'value' in col.lower() or col.startswith('__value'):
                    value_columns.append(col)
            
            if not value_columns:
                # Try to find numeric columns
                numeric_cols = cpu_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    value_columns = [numeric_cols[0]]  # Use first numeric column
            
            if not value_columns:
                logger.warning("Could not find CPU value columns in DataFrame")
                return {
                    'percentile_value': 0.0,
                    'recommended_requests': self.format_cpu_value(0.1),
                    'current_usage_stats': {},
                    'has_data': False
                }
            
            # Combine all value columns and calculate statistics
            cpu_values = []
            for col in value_columns:
                cpu_values.extend(cpu_df[col].dropna().tolist())
            
            if not cpu_values:
                return {
                    'percentile_value': 0.0,
                    'recommended_requests': self.format_cpu_value(0.1),
                    'current_usage_stats': {},
                    'has_data': False
                }
            
            cpu_series = pd.Series(cpu_values)
            
            # Calculate percentile (95th by default)
            percentile_value = cpu_series.quantile(self.cpu_percentile / 100.0)
            
            # Calculate other statistics
            stats = {
                'mean': cpu_series.mean(),
                'median': cpu_series.median(),
                'std': cpu_series.std(),
                'min': cpu_series.min(),
                'max': cpu_series.max(),
                'percentile_value': percentile_value,
                'percentile': self.cpu_percentile,
                'count': len(cpu_values)
            }
            
            # Use percentile value as recommendation, with minimum of 100m
            recommended_cores = max(percentile_value, 0.1)
            
            return {
                'percentile_value': percentile_value,
                'recommended_requests': self.format_cpu_value(recommended_cores),
                'current_usage_stats': stats,
                'has_data': True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CPU usage: {e}")
            return {
                'percentile_value': 0.0,
                'recommended_requests': self.format_cpu_value(0.1),
                'current_usage_stats': {},
                'has_data': False,
                'error': str(e)
            }
    
    def analyze_memory_usage(self, memory_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze memory usage data and calculate recommendations
        
        Args:
            memory_df: DataFrame with memory usage metrics
            
        Returns:
            Dictionary with memory analysis results
        """
        if memory_df.empty:
            logger.warning("No memory data available for analysis")
            return {
                'max_usage_bytes': 0,
                'recommended_requests': self.format_memory_value(128 * 1024 * 1024),  # 128Mi default
                'current_usage_stats': {},
                'has_data': False
            }
        
        try:
            # Find the value column(s) - similar to CPU analysis
            value_columns = []
            for col in memory_df.columns:
                if 'value' in col.lower() or col.startswith('__value'):
                    value_columns.append(col)
            
            if not value_columns:
                # Try to find numeric columns
                numeric_cols = memory_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    value_columns = [numeric_cols[0]]
            
            if not value_columns:
                logger.warning("Could not find memory value columns in DataFrame")
                return {
                    'max_usage_bytes': 0,
                    'recommended_requests': self.format_memory_value(128 * 1024 * 1024),
                    'current_usage_stats': {},
                    'has_data': False
                }
            
            # Combine all value columns
            memory_values = []
            for col in value_columns:
                memory_values.extend(memory_df[col].dropna().tolist())
            
            if not memory_values:
                return {
                    'max_usage_bytes': 0,
                    'recommended_requests': self.format_memory_value(128 * 1024 * 1024),
                    'current_usage_stats': {},
                    'has_data': False
                }
            
            memory_series = pd.Series(memory_values)
            
            # Find maximum usage
            max_usage_bytes = memory_series.max()
            
            # Calculate statistics
            stats = {
                'mean': memory_series.mean(),
                'median': memory_series.median(),
                'std': memory_series.std(),
                'min': memory_series.min(),
                'max': max_usage_bytes,
                'count': len(memory_values)
            }
            
            # Add buffer to maximum usage
            buffer_factor = 1 + (self.memory_buffer_percentage / 100.0)
            recommended_bytes = int(max_usage_bytes * buffer_factor)
            
            # Ensure minimum of 128Mi
            min_bytes = 128 * 1024 * 1024
            recommended_bytes = max(recommended_bytes, min_bytes)
            
            return {
                'max_usage_bytes': max_usage_bytes,
                'recommended_requests': self.format_memory_value(recommended_bytes),
                'buffer_percentage': self.memory_buffer_percentage,
                'current_usage_stats': stats,
                'has_data': True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing memory usage: {e}")
            return {
                'max_usage_bytes': 0,
                'recommended_requests': self.format_memory_value(128 * 1024 * 1024),
                'current_usage_stats': {},
                'has_data': False,
                'error': str(e)
            }
    
    def generate_recommendations(
        self, 
        service_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate rightsizing recommendations for a service
        
        Args:
            service_data: Service data from data pipeline
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            deployment = service_data.get('deployment', {})
            metrics = service_data.get('metrics', {})
            service_name = service_data.get('service_name', 'unknown')
            
            if not deployment:
                logger.warning(f"No deployment data available for {service_name}")
                return recommendations
            
            containers = deployment.get('containers', [])
            cpu_df = metrics.get('cpu', pd.DataFrame())
            memory_df = metrics.get('memory', pd.DataFrame())
            
            # Analyze CPU and memory usage
            cpu_analysis = self.analyze_cpu_usage(cpu_df)
            memory_analysis = self.analyze_memory_usage(memory_df)
            
            for container in containers:
                container_name = container['name']
                current_resources = container.get('resources', {}).get('requests', {})
                
                current_cpu = current_resources.get('cpu', 'Unknown')
                current_memory = current_resources.get('memory', 'Unknown')
                
                # Generate recommendations
                recommendation = {
                    'service_name': service_name,
                    'container_name': container_name,
                    'namespace': service_data.get('namespace', 'default'),
                    'timestamp': datetime.now().isoformat(),
                    'strategy': 'statistical',
                    'parameters': {
                        'cpu_percentile': self.cpu_percentile,
                        'memory_buffer_percentage': self.memory_buffer_percentage
                    },
                    'current_requests': {
                        'cpu': current_cpu,
                        'memory': current_memory
                    },
                    'recommended_requests': {
                        'cpu': cpu_analysis['recommended_requests'],
                        'memory': memory_analysis['recommended_requests']
                    },
                    'analysis': {
                        'cpu': cpu_analysis,
                        'memory': memory_analysis
                    }
                }
                
                recommendations.append(recommendation)
            
            logger.info(f"Generated {len(recommendations)} recommendations for {service_name}")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def validate_recommendations(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate recommendations for reasonableness
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        for rec in recommendations:
            try:
                # Parse recommended values
                recommended_cpu = self.parse_cpu_value(rec['recommended_requests']['cpu'])
                recommended_memory = self.parse_memory_value(rec['recommended_requests']['memory'])
                
                # Validate CPU recommendations
                if recommended_cpu < 0.01:  # Less than 10m
                    validation['warnings'].append(f"Very low CPU recommendation for {rec['container_name']}: {recommended_cpu}")
                elif recommended_cpu > 8.0:  # More than 8 cores
                    validation['warnings'].append(f"High CPU recommendation for {rec['container_name']}: {recommended_cpu} cores")
                
                # Validate memory recommendations
                min_memory = 32 * 1024 * 1024  # 32Mi
                max_memory = 32 * 1024 * 1024 * 1024  # 32Gi
                
                if recommended_memory < min_memory:
                    validation['warnings'].append(f"Very low memory recommendation for {rec['container_name']}: {rec['recommended_requests']['memory']}")
                elif recommended_memory > max_memory:
                    validation['warnings'].append(f"Very high memory recommendation for {rec['container_name']}: {rec['recommended_requests']['memory']}")
                
            except Exception as e:
                validation['errors'].append(f"Error validating recommendation for {rec.get('container_name', 'unknown')}: {e}")
                validation['is_valid'] = False
        
        return validation
