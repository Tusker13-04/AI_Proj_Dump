"""
Data pipeline for collecting and processing metrics data
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from ..k8s.client import KubernetesClient
from ..k8s.discovery import ServiceDiscovery
from ..monitoring.prometheus_client import PrometheusClient
from ..monitoring.grafana_client import GrafanaClient


logger = logging.getLogger(__name__)


class DataPipeline:
    """Pipeline for collecting and processing metrics data for rightsizing analysis"""
    
    def __init__(
        self, 
        namespace: str = "hipster-shop",
        prometheus_url: str = "http://localhost:9090",
        grafana_url: str = "http://localhost:4000"
    ):
        """
        Initialize data pipeline
        
        Args:
            namespace: Default Kubernetes namespace
            prometheus_url: Prometheus server URL
            grafana_url: Grafana server URL
        """
        self.namespace = namespace
        
        # Initialize clients
        self.k8s_client = KubernetesClient(namespace=namespace)
        self.service_discovery = ServiceDiscovery(self.k8s_client)
        self.prometheus_client = PrometheusClient(prometheus_url)
        self.grafana_client = GrafanaClient(grafana_url)
        
        logger.info(f"Initialized data pipeline for namespace {namespace}")
    
    def test_connections(self) -> Dict[str, bool]:
        """
        Test connections to Kubernetes, Prometheus, and Grafana
        
        Returns:
            Dictionary with connection status for each service
        """
        results = {}
        
        try:
            results['kubernetes'] = self.k8s_client.test_connection()
        except Exception as e:
            logger.error(f"Kubernetes connection test failed: {e}")
            results['kubernetes'] = False
        
        try:
            results['prometheus'] = self.prometheus_client.test_connection()
        except Exception as e:
            logger.error(f"Prometheus connection test failed: {e}")
            results['prometheus'] = False
        
        try:
            results['grafana'] = self.grafana_client.test_connection()
        except Exception as e:
            logger.error(f"Grafana connection test failed: {e}")
            results['grafana'] = False
        
        return results
    
    def collect_service_data(
        self, 
        service_name: str, 
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Collect comprehensive data for a specific service
        
        Args:
            service_name: Name of the service to collect data for
            duration_minutes: Duration to collect metrics for
            
        Returns:
            Dictionary containing all collected data
        """
        logger.info(f"Collecting data for service {service_name}")
        
        # Validate service exists
        if not self.service_discovery.validate_service_exists(service_name, self.namespace):
            raise ValueError(f"Service {service_name} not found in namespace {self.namespace}")
        
        data = {
            'service_name': service_name,
            'namespace': self.namespace,
            'collection_timestamp': datetime.now(),
            'duration_minutes': duration_minutes
        }
        
        try:
            # Get Kubernetes deployment information
            deployment_info = self.k8s_client.get_deployment(service_name, self.namespace)
            data['deployment'] = deployment_info
            
            # Get pod information
            pods = self.k8s_client.get_pods(
                namespace=self.namespace,
                label_selector=f"app={service_name}"  # Common label pattern
            )
            data['pods'] = pods
            
            # Get Prometheus metrics
            metrics = self.prometheus_client.get_pod_metrics(
                namespace=self.namespace,
                service_name=service_name,
                duration_minutes=duration_minutes
            )
            data['metrics'] = metrics
            
            logger.info(f"Successfully collected data for service {service_name}")
            
        except Exception as e:
            logger.error(f"Error collecting data for service {service_name}: {e}")
            data['error'] = str(e)
        
        return data
    
    def collect_multiple_services_data(
        self, 
        service_names: List[str], 
        duration_minutes: int = 60
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect data for multiple services
        
        Args:
            service_names: List of service names
            duration_minutes: Duration to collect metrics for
            
        Returns:
            Dictionary mapping service names to their data
        """
        results = {}
        
        for service_name in service_names:
            try:
                results[service_name] = self.collect_service_data(service_name, duration_minutes)
            except Exception as e:
                logger.error(f"Failed to collect data for {service_name}: {e}")
                results[service_name] = {
                    'service_name': service_name,
                    'error': str(e),
                    'collection_timestamp': datetime.now()
                }
        
        return results
    
    def get_historical_cpu_usage(self, service_name: str, duration_hours: int = 24) -> pd.DataFrame:
        """
        Get historical CPU usage for statistical analysis
        
        Args:
            service_name: Name of the service
            duration_hours: Duration to query in hours
            
        Returns:
            DataFrame with CPU usage data
        """
        return self.prometheus_client.get_cpu_metrics(
            namespace=self.namespace,
            service_name=service_name,
            duration_minutes=duration_hours * 60
        )
    
    def get_historical_memory_usage(self, service_name: str, duration_hours: int = 24) -> pd.DataFrame:
        """
        Get historical memory usage for statistical analysis
        
        Args:
            service_name: Name of the service
            duration_hours: Duration to query in hours
            
        Returns:
            DataFrame with memory usage data
        """
        return self.prometheus_client.get_memory_metrics(
            namespace=self.namespace,
            service_name=service_name,
            duration_minutes=duration_hours * 60
        )
    
    def validate_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the quality of collected data
        
        Args:
            data: Collected data dictionary
            
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check if service exists
        if 'deployment' not in data or data['deployment'] is None:
            validation['is_valid'] = False
            validation['issues'].append("No deployment information found")
        
        # Check if metrics are available
        metrics = data.get('metrics', {})
        if not metrics:
            validation['warnings'].append("No metrics data available")
        else:
            # Check CPU metrics
            cpu_df = metrics.get('cpu')
            if cpu_df is None or cpu_df.empty:
                validation['warnings'].append("No CPU metrics data")
            
            # Check memory metrics
            memory_df = metrics.get('memory')
            if memory_df is None or memory_df.empty:
                validation['warnings'].append("No memory metrics data")
        
        # Check data recency
        if 'collection_timestamp' in data:
            collection_time = data['collection_timestamp']
            age = datetime.now() - collection_time
            if age > timedelta(hours=1):
                validation['warnings'].append(f"Data is {age} old")
        
        return validation
    
    def discover_all_services(self) -> List[str]:
        """
        Discover all services in the namespace
        
        Returns:
            List of service names
        """
        return self.service_discovery.get_microservices(self.namespace)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entire system
        
        Returns:
            System summary information
        """
        try:
            services = self.discover_all_services()
            
            summary = {
                'namespace': self.namespace,
                'total_services': len(services),
                'services': services,
                'connections': self.test_connections(),
                'timestamp': datetime.now()
            }
            
            # Get basic stats for each service
            service_stats = {}
            for service_name in services[:5]:  # Limit to first 5 for performance
                try:
                    deployment = self.k8s_client.get_deployment(service_name, self.namespace)
                    if deployment:
                        service_stats[service_name] = {
                            'replicas': deployment['replicas'],
                            'ready_replicas': deployment['ready_replicas'],
                            'containers': len(deployment['containers'])
                        }
                except Exception as e:
                    logger.warning(f"Could not get stats for {service_name}: {e}")
            
            summary['service_stats'] = service_stats
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting system summary: {e}")
            return {
                'namespace': self.namespace,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def setup_grafana_integration(self) -> Dict[str, Any]:
        """
        Set up Grafana dashboard for MOrA monitoring
        
        Returns:
            Dictionary with setup results and dashboard information
        """
        try:
            # Test Grafana connection first
            if not self.grafana_client.test_connection():
                logger.error("Cannot connect to Grafana")
                return {
                    'success': False,
                    'error': 'Grafana connection failed'
                }
            
            # Verify Prometheus data source is configured
            if not self.grafana_client.verify_prometheus_datasource():
                logger.warning("Prometheus data source may not be properly configured")
            
            # Create or update MOrA dashboard
            dashboard_uid = self.grafana_client.create_mora_dashboard(self.namespace)
            
            if dashboard_uid:
                dashboard_url = self.grafana_client.get_dashboard_url(dashboard_uid)
                logger.info(f"MOrA Grafana dashboard created/updated: {dashboard_url}")
                
                return {
                    'success': True,
                    'dashboard_uid': dashboard_uid,
                    'dashboard_url': dashboard_url,
                    'namespace': self.namespace
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create dashboard'
                }
                
        except Exception as e:
            logger.error(f"Grafana integration setup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
