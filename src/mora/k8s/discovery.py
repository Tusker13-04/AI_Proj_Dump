"""
Service discovery mechanism for MOrA
"""
import logging
from typing import List, Dict, Optional, Set
from .client import KubernetesClient


logger = logging.getLogger(__name__)


class ServiceDiscovery:
    """Service discovery for microservices in Kubernetes"""
    
    def __init__(self, k8s_client: KubernetesClient):
        """
        Initialize service discovery
        
        Args:
            k8s_client: Kubernetes client instance
        """
        self.k8s_client = k8s_client
    
    def discover_services(self, namespace: str) -> List[Dict[str, str]]:
        """
        Discover all services/deployments in a namespace
        
        Args:
            namespace: Namespace to discover services in
            
        Returns:
            List of service information
        """
        try:
            deployments = self.k8s_client.get_deployments(namespace=namespace)
            services = []
            
            for deployment in deployments:
                service_info = {
                    'name': deployment['name'],
                    'namespace': deployment['namespace'],
                    'type': 'deployment',
                    'replicas': deployment['replicas'],
                    'ready_replicas': deployment['ready_replicas'],
                    'containers': [container['name'] for container in deployment['containers']],
                    'labels': deployment.get('labels', {}),
                    'status': 'ready' if deployment['ready_replicas'] == deployment['replicas'] else 'not_ready'
                }
                services.append(service_info)
            
            logger.info(f"Discovered {len(services)} services in namespace {namespace}")
            return services
            
        except Exception as e:
            logger.error(f"Error discovering services in namespace {namespace}: {e}")
            return []
    
    def get_microservices(self, namespace: str, filter_patterns: Optional[List[str]] = None) -> List[str]:
        """
        Get list of microservice names, optionally filtered
        
        Args:
            namespace: Namespace to search in
            filter_patterns: List of patterns to filter service names (e.g., ['frontend', 'api'])
            
        Returns:
            List of microservice names
        """
        services = self.discover_services(namespace)
        service_names = [service['name'] for service in services]
        
        if filter_patterns:
            filtered_services = []
            for pattern in filter_patterns:
                for service_name in service_names:
                    if pattern.lower() in service_name.lower():
                        filtered_services.append(service_name)
            return list(set(filtered_services))  # Remove duplicates
        
        return service_names
    
    def get_service_labels(self, namespace: str) -> Dict[str, Dict[str, str]]:
        """
        Get labels for all services
        
        Args:
            namespace: Namespace to query
            
        Returns:
            Dictionary mapping service names to their labels
        """
        services = self.discover_services(namespace)
        labels_map = {}
        
        for service in services:
            labels_map[service['name']] = service['labels']
        
        return labels_map
    
    def get_services_by_label(self, namespace: str, label_key: str, label_value: str = None) -> List[Dict[str, str]]:
        """
        Get services filtered by label
        
        Args:
            namespace: Namespace to search in
            label_key: Label key to filter by
            label_value: Label value to match (if None, returns services with the key)
            
        Returns:
            List of matching services
        """
        services = self.discover_services(namespace)
        filtered_services = []
        
        for service in services:
            labels = service.get('labels', {})
            if label_key in labels:
                if label_value is None or labels[label_key] == label_value:
                    filtered_services.append(service)
        
        return filtered_services
    
    def validate_service_exists(self, service_name: str, namespace: str) -> bool:
        """
        Check if a service exists
        
        Args:
            service_name: Name of the service to check
            namespace: Namespace to check in
            
        Returns:
            True if service exists, False otherwise
        """
        try:
            deployment = self.k8s_client.get_deployment(service_name, namespace)
            return deployment is not None
        except Exception as e:
            logger.error(f"Error validating service {service_name}: {e}")
            return False
