"""
Kubernetes API client for MOrA
"""
import os
import logging
from typing import List, Dict, Optional, Any
from kubernetes import client, config
from kubernetes.client.rest import ApiException


logger = logging.getLogger(__name__)


class KubernetesClient:
    """Client for interacting with Kubernetes API"""
    
    def __init__(self, namespace: str = "default"):
        """
        Initialize Kubernetes client
        
        Args:
            namespace: Default namespace for operations
        """
        self.namespace = namespace
        self.apps_v1 = None
        self.core_v1 = None
        self.metrics_v1 = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Kubernetes API clients"""
        try:
            # Try to load from cluster if running inside Kubernetes
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            try:
                # Load from kubeconfig file (for local development)
                config.load_kube_config()
                logger.info("Loaded kubeconfig from default location")
            except config.ConfigException as e:
                logger.error(f"Could not load Kubernetes config: {e}")
                raise
        
        # Initialize API clients
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        try:
            self.metrics_v1 = client.CustomObjectsApi()
        except Exception:
            logger.warning("Metrics API not available")
    
    def get_deployments(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all deployments in the specified namespace
        
        Args:
            namespace: Namespace to query (defaults to instance namespace)
            
        Returns:
            List of deployment information
        """
        namespace = namespace or self.namespace
        
        try:
            response = self.apps_v1.list_namespaced_deployment(namespace=namespace)
            deployments = []
            
            for deployment in response.items:
                deployment_info = {
                    'name': deployment.metadata.name,
                    'namespace': deployment.metadata.namespace,
                    'replicas': deployment.spec.replicas,
                    'ready_replicas': deployment.status.ready_replicas or 0,
                    'labels': deployment.metadata.labels or {},
                    'creation_timestamp': deployment.metadata.creation_timestamp,
                    'containers': []
                }
                
                for container in deployment.spec.template.spec.containers:
                    container_info = {
                        'name': container.name,
                        'image': container.image,
                        'resources': {
                            'requests': container.resources.requests or {},
                            'limits': container.resources.limits or {}
                        }
                    }
                    deployment_info['containers'].append(container_info)
                
                deployments.append(deployment_info)
            
            logger.info(f"Retrieved {len(deployments)} deployments from namespace {namespace}")
            return deployments
            
        except ApiException as e:
            logger.error(f"Error getting deployments: {e}")
            raise
    
    def get_deployment(self, name: str, namespace: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get specific deployment information
        
        Args:
            name: Deployment name
            namespace: Namespace to query (defaults to instance namespace)
            
        Returns:
            Deployment information or None if not found
        """
        namespace = namespace or self.namespace
        
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
            
            deployment_info = {
                'name': deployment.metadata.name,
                'namespace': deployment.metadata.namespace,
                'replicas': deployment.spec.replicas,
                'ready_replicas': deployment.status.ready_replicas or 0,
                'labels': deployment.metadata.labels or {},
                'creation_timestamp': deployment.metadata.creation_timestamp,
                'containers': []
            }
            
            for container in deployment.spec.template.spec.containers:
                container_info = {
                    'name': container.name,
                    'image': container.image,
                    'resources': {
                        'requests': container.resources.requests or {},
                        'limits': container.resources.limits or {}
                    }
                }
                deployment_info['containers'].append(container_info)
            
            return deployment_info
            
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Deployment {name} not found in namespace {namespace}")
                return None
            logger.error(f"Error getting deployment {name}: {e}")
            raise
    
    def get_pods(self, namespace: Optional[str] = None, label_selector: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get pods in the specified namespace
        
        Args:
            namespace: Namespace to query (defaults to instance namespace)
            label_selector: Label selector for filtering pods
            
        Returns:
            List of pod information
        """
        namespace = namespace or self.namespace
        
        try:
            response = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector
            )
            
            pods = []
            for pod in response.items:
                pod_info = {
                    'name': pod.metadata.name,
                    'namespace': pod.metadata.namespace,
                    'status': pod.status.phase,
                    'node_name': pod.spec.node_name,
                    'labels': pod.metadata.labels or {},
                    'creation_timestamp': pod.metadata.creation_timestamp,
                    'containers': []
                }
                
                for container in pod.spec.containers:
                    container_info = {
                        'name': container.name,
                        'image': container.image,
                        'resources': {
                            'requests': container.resources.requests or {},
                            'limits': container.resources.limits or {}
                        }
                    }
                    
                    # Get container status
                    for status in pod.status.container_statuses or []:
                        if status.name == container.name:
                            container_info['status'] = {
                                'ready': status.ready,
                                'restart_count': status.restart_count,
                                'state': str(status.state) if status.state else None
                            }
                            break
                    
                    pod_info['containers'].append(container_info)
                
                pods.append(pod_info)
            
            logger.info(f"Retrieved {len(pods)} pods from namespace {namespace}")
            return pods
            
        except ApiException as e:
            logger.error(f"Error getting pods: {e}")
            raise
    
    def get_service_metrics(self, service_name: str, namespace: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific service (pods)
        
        Args:
            service_name: Name of the service/deployment
            namespace: Namespace to query (defaults to instance namespace)
            
        Returns:
            Metrics information or None if not available
        """
        namespace = namespace or self.namespace
        
        try:
            if not self.metrics_v1:
                logger.warning("Metrics API not available")
                return None
            
            # Get pod metrics using CustomObjectsApi
            # This requires metrics-server to be running
            group = "metrics.k8s.io"
            version = "v1beta1"
            plural = "pods"
            
            response = self.metrics_v1.list_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural
            )
            
            service_metrics = []
            for item in response.get('items', []):
                pod_name = item['metadata']['name']
                if service_name in pod_name:  # Simple name matching
                    for container in item.get('containers', []):
                        container_metrics = {
                            'pod_name': pod_name,
                            'container_name': container['name'],
                            'cpu_usage': container.get('usage', {}).get('cpu', '0'),
                            'memory_usage': container.get('usage', {}).get('memory', '0')
                        }
                        service_metrics.append(container_metrics)
            
            return {
                'service_name': service_name,
                'namespace': namespace,
                'metrics': service_metrics,
                'timestamp': response.get('kind', 'Unknown')
            }
            
        except Exception as e:
            logger.warning(f"Could not get metrics for service {service_name}: {e}")
            return None
    
    def scale_deployment(self, name: str, namespace: Optional[str] = None, replicas: int = 1) -> bool:
        """
        Scale a deployment to the specified number of replicas
        
        Args:
            name: Deployment name
            namespace: Namespace (defaults to instance namespace)
            replicas: Number of replicas
            
        Returns:
            True if scaling was successful, False otherwise
        """
        namespace = namespace or self.namespace
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
            
            # Update replica count
            deployment.spec.replicas = replicas
            
            # Apply the update
            self.apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body={'spec': {'replicas': replicas}}
            )
            
            logger.info(f"Scaled deployment {name} to {replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to scale deployment {name}: {e}")
            return False

    def list_deployments(self, namespace: Optional[str] = None) -> List:
        """
        List deployments - compatibility method for existing code
        
        Args:
            namespace: Namespace to query
            
        Returns:
            List of deployment objects
        """
        namespace = namespace or self.namespace
        
        try:
            response = self.apps_v1.list_namespaced_deployment(namespace=namespace)
            return response.items
        except ApiException as e:
            logger.error(f"Error listing deployments: {e}")
            return []

    def test_connection(self) -> bool:
        """
        Test Kubernetes API connection
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to list nodes to test connection
            self.core_v1.list_node()
            logger.info("Kubernetes API connection successful")
            return True
        except Exception as e:
            logger.error(f"Kubernetes API connection failed: {e}")
            return False
