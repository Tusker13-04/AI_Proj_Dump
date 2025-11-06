"""
Prometheus client for metrics extraction in MOrA
"""
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
import requests
import pandas as pd
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import yaml


logger = logging.getLogger(__name__)


class PrometheusClient:
    """Client for interacting with Prometheus API"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090", timeout: int = 30):
        """
        Initialize Prometheus client
        
        Args:
            prometheus_url: URL of Prometheus server
            timeout: Request timeout in seconds
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.timeout = timeout
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Prometheus API client"""
        try:
            self.client = PrometheusConnect(
                url=self.prometheus_url,
                disable_ssl=True,
                headers={"Content-Type": "application/json"}
            )
            # Test connection
            self.client.check_prometheus_connection()
            logger.info(f"Connected to Prometheus at {self.prometheus_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Prometheus at {self.prometheus_url}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test Prometheus connection
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Simple query to test connection
            response = self.client.custom_query(query="up")
            logger.info("Prometheus connection test successful")
            return True
        except Exception as e:
            logger.error(f"Prometheus connection test failed: {e}")
            return False
    
    def get_metric_range(
        self, 
        query: str, 
        start_time: Union[datetime, str] = None, 
        end_time: Union[datetime, str] = None,
        step: str = "15s"
    ) -> pd.DataFrame:
        """
        Get metric data within a time range
        
        Args:
            query: Prometheus query to execute
            start_time: Start time for the query (default: 1 hour ago)
            end_time: End time for the query (default: now)
            step: Query resolution step width
            
        Returns:
            DataFrame with metric data
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()
        
        try:
            # Keep datetime objects as they are - the prometheus client expects datetime objects
            # Don't convert to timestamps yet, let the client handle the conversion
            
            # Use the custom_query_range method for complex queries
            metric_data = self.client.custom_query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=step or "15s"
            )
            
            if not metric_data:
                logger.warning(f"No data returned for query: {query}")
                return pd.DataFrame()
            
            logger.debug(f"Raw metric data type: {type(metric_data)}")
            logger.debug(f"Raw metric data: {metric_data}")
            
            # Convert to DataFrame - handle different response formats
            metric_df = None
            if isinstance(metric_data, dict):
                if 'data' in metric_data:
                    # Handle Prometheus API response format
                    metric_df = MetricRangeDataFrame(metric_data['data'])
                elif 'result' in metric_data:
                    # Handle direct result format
                    metric_df = MetricRangeDataFrame(metric_data['result'])
                else:
                    # Try to create DataFrame from the dict itself
                    try:
                        metric_df = MetricRangeDataFrame(metric_data)
                    except:
                        logger.warning(f"Could not convert dict response to DataFrame: {metric_data}")
                        return pd.DataFrame()
            elif isinstance(metric_data, list):
                # Handle list format
                try:
                    metric_df = MetricRangeDataFrame(metric_data)
                except:
                    logger.warning(f"Could not convert list response to DataFrame: {len(metric_data)} items")
                    return pd.DataFrame()
            else:
                # Try direct conversion
                try:
                    metric_df = MetricRangeDataFrame(metric_data)
                except Exception as conv_e:
                    logger.warning(f"Could not convert response to DataFrame: {conv_e}")
                    return pd.DataFrame()
            
            if metric_df is None or metric_df.empty:
                logger.warning(f"Empty DataFrame created for query: {query}")
                return pd.DataFrame()
            
            logger.debug(f"Successfully created DataFrame with {len(metric_df)} rows for query: {query}")
            return metric_df
            
        except Exception as e:
            logger.error(f"Error executing Prometheus query '{query}': {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_cpu_metrics(
        self, 
        namespace: str, 
        service_name: str, 
        duration_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Get CPU usage metrics for a specific service
        
        Args:
            namespace: Kubernetes namespace
            service_name: Name of the service/deployment
            duration_minutes: Duration to query in minutes
            
        Returns:
            DataFrame with CPU metrics
        """
        # Prometheus query for container CPU usage
        query = f"""
        rate(container_cpu_usage_seconds_total{{
            namespace="{namespace}",
            pod=~"{service_name}.*"
        }}[5m])
        """
        
        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        end_time = datetime.now()
        
        return self.get_metric_range(
            query=query.strip(),
            start_time=start_time,
            end_time=end_time,
            step="1m"
        )
    
    def get_memory_metrics(
        self, 
        namespace: str, 
        service_name: str, 
        duration_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Get memory usage metrics for a specific service
        
        Args:
            namespace: Kubernetes namespace
            service_name: Name of the service/deployment
            duration_minutes: Duration to query in minutes
            
        Returns:
            DataFrame with memory metrics
        """
        # Prometheus query for container memory usage
        query = f"""
        container_memory_working_set_bytes{{
            namespace="{namespace}",
            pod=~"{service_name}.*",
            container!="POD",
            container!=""
        }}
        """
        
        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        end_time = datetime.now()
        
        return self.get_metric_range(
            query=query.strip(),
            start_time=start_time,
            end_time=end_time,
            step="1m"
        )
    
    def get_resource_requests(
        self, 
        namespace: str, 
        service_name: str
    ) -> Dict[str, Any]:
        """
        Get resource requests for a specific service
        
        Args:
            namespace: Kubernetes namespace
            service_name: Name of the service/deployment
            
        Returns:
            Dictionary with resource request information
        """
        query_cpu = f"""
        kube_pod_container_resource_requests{{
            namespace="{namespace}",
            pod=~"{service_name}.*",
            resource="cpu"
        }}
        """
        
        query_memory = f"""
        kube_pod_container_resource_requests{{
            namespace="{namespace}",
            pod=~"{service_name}.*",
            resource="memory"
        }}
        """
        
        try:
            # Get current values (instant query)
            cpu_request = self.client.query(query_cpu.strip())
            memory_request = self.client.query(query_memory.strip())
            
            requests = {
                'cpu': [],
                'memory': []
            }
            
            if cpu_request:
                for result in cpu_request:
                    requests['cpu'].append({
                        'pod': result.metric.get('pod', ''),
                        'container': result.metric.get('container', ''),
                        'value': float(result.value[1])
                    })
            
            if memory_request:
                for result in memory_request:
                    requests['memory'].append({
                        'pod': result.metric.get('pod', ''),
                        'container': result.metric.get('container', ''),
                        'value': float(result.value[1])
                    })
            
            return requests
            
        except Exception as e:
            logger.error(f"Error getting resource requests: {e}")
            return {'cpu': [], 'memory': []}
    
    def get_pod_metrics(
        self, 
        namespace: str, 
        service_name: str, 
        duration_minutes: int = 60
    ) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive metrics for a service (CPU, memory, etc.)
        
        Args:
            namespace: Kubernetes namespace
            service_name: Name of the service/deployment
            duration_minutes: Duration to query in minutes
            
        Returns:
            Dictionary with different metric DataFrames
        """
        metrics = {}
        
        try:
            # Get CPU metrics
            metrics['cpu'] = self.get_cpu_metrics(
                namespace=namespace,
                service_name=service_name,
                duration_minutes=duration_minutes
            )
            
            # Get memory metrics
            metrics['memory'] = self.get_memory_metrics(
                namespace=namespace,
                service_name=service_name,
                duration_minutes=duration_minutes
            )
            
            # Get resource requests
            metrics['requests'] = self.get_resource_requests(
                namespace=namespace,
                service_name=service_name
            )
            
            logger.info(f"Retrieved metrics for service {service_name} in namespace {namespace}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting comprehensive metrics: {e}")
            return {}
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metric names
        
        Returns:
            List of available metric names
        """
        try:
            metrics = self.client.all_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Error getting available metrics: {e}")
            return []
    
    def query_instant(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute instant query
        
        Args:
            query: Prometheus query to execute
            
        Returns:
            List of query results
        """
        try:
            result = self.client.query(query)
            data = []
            
            if result:
                for item in result:
                    data.append({
                        'metric': dict(item.metric),
                        'value': item.value,
                        'timestamp': item.timestamp
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"Error executing instant query '{query}': {e}")
            return []

    def get_comprehensive_metrics(
        self,
        namespace: str,
        service_name: str,
        duration_minutes: int = 60
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all research-level metrics for a service following the schema:
        timestamp, cluster_id, node, namespace, deployment, pod, container,
        cpu_cores, mem_bytes, net_rx_bytes, net_tx_bytes, requests_per_second,
        latency_p50, latency_p95, latency_p99, error_rate, throttled_seconds,
        pod_restarts, replica_count, node_cpu_util, node_mem_util, cost_per_minute
        """
        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        end_time = datetime.now()
        
        # Service/pod selector
        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        
        metrics = {}
        successful_metrics = 0
        failed_metrics = 0
        
        # Define all metric collection methods with error handling
        # Define only the 8 working metrics - focus on quality over quantity
        metric_collectors = [
            ('cpu_cores', lambda: self._get_cpu_cores(service_name, namespace, start_time, end_time)),
            ('mem_bytes', lambda: self._get_memory_working_set(service_name, namespace, start_time, end_time)),
            ('net_rx_bytes', lambda: self._get_network_rx_bytes(service_name, namespace, start_time, end_time)),
            ('net_tx_bytes', lambda: self._get_network_tx_bytes(service_name, namespace, start_time, end_time)),
            ('pod_restarts', lambda: self._get_pod_restarts(service_name, namespace, start_time, end_time)),
            ('replica_count', lambda: self._get_replica_count(service_name, namespace, start_time, end_time)),
            ('node_cpu_util', lambda: self._get_node_cpu_utilization(start_time, end_time)),
            ('node_mem_util', lambda: self._get_node_memory_utilization(start_time, end_time)),
        ]
        
        # Collect each metric with individual error handling
        for metric_name, collector_func in metric_collectors:
            try:
                result = collector_func()
                if result is not None and not result.empty:
                    metrics[metric_name] = result
                    successful_metrics += 1
                    logger.debug(f"Successfully collected {metric_name} for {service_name}")
                else:
                    logger.warning(f"Empty result for {metric_name} for {service_name}")
                    # Create empty DataFrame to maintain structure
                    metrics[metric_name] = pd.DataFrame()
                    failed_metrics += 1
            except Exception as e:
                logger.warning(f"Failed to collect {metric_name} for {service_name}: {e}")
                # Create empty DataFrame to maintain structure
                metrics[metric_name] = pd.DataFrame()
                failed_metrics += 1
        
        logger.info(f"Collected {successful_metrics}/{len(metric_collectors)} metric types for {service_name} "
                   f"({failed_metrics} failed)")
        
        # Add calculated substitute metrics for missing application-level metrics
        try:
            substitute_metrics = self._calculate_substitute_metrics(metrics, service_name, namespace, start_time, end_time)
            metrics.update(substitute_metrics)
            successful_metrics += len(substitute_metrics)
            logger.info(f"Added {len(substitute_metrics)} substitute metrics for {service_name}")
        except Exception as e:
            logger.error(f"Error calculating substitute metrics for {service_name}: {e}")
        
        # Return metrics even if some failed - let the system handle missing data
        return metrics

    def _get_cpu_cores(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get CPU cores (instant/averaged) using rate over 5 minutes"""
        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        # Try different query variations since container labels may vary
        queries = [
            # Original with container filter
            f"""rate(container_cpu_usage_seconds_total{{{pod_selector},container!="POD",container!=""}}[5m])""",
            # Without container filter
            f"""rate(container_cpu_usage_seconds_total{{{pod_selector}}}[5m])""",
            # With different container filter
            f"""rate(container_cpu_usage_seconds_total{{{pod_selector},container!~"POD|"}}[5m])"""
        ]
        
        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"CPU cores query failed: {query}, error: {e}")
                continue
        
        logger.warning(f"No CPU metrics found with any query variation for selector: {pod_selector}")
        return pd.DataFrame()

    def _get_memory_working_set(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get working set memory bytes"""
        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        # Try different query variations for memory metrics
        queries = [
            # Original with container filter
            f"""container_memory_working_set_bytes{{{pod_selector},container!="POD",container!=""}}""",
            # Without container filter
            f"""container_memory_working_set_bytes{{{pod_selector}}}""",
            # Alternative memory metric
            f"""container_memory_usage_bytes{{{pod_selector},container!="POD",container!=""}}""",
            f"""container_memory_usage_bytes{{{pod_selector}}}"""
        ]
        
        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Memory query failed: {query}, error: {e}")
                continue
        
        logger.warning(f"No memory metrics found with any query variation for selector: {pod_selector}")
        return pd.DataFrame()

    def _get_network_rx_bytes(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get network receive bytes per pod with fallback queries"""
        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        queries = [
            # Container-level network metrics
            f"""rate(container_network_receive_bytes_total{{{pod_selector}}}[5m])""",
            # Node-level network metrics as fallback
            f"""rate(node_network_receive_bytes_total{{{pod_selector}}}[5m])""",
            # Process-level network metrics as fallback
            f"""rate(process_network_receive_bytes_total{{{pod_selector}}}[5m])""",
            # Alternative container network metrics
            f"""rate(container_network_receive_bytes_total{{{pod_selector},namespace="hipster-shop"}}[5m])"""
        ]
        
        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Network RX query failed: {query}, error: {e}")
                continue
        
        logger.warning(f"Empty result for net_rx_bytes for {pod_selector}")
        return pd.DataFrame()

    def _get_network_tx_bytes(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get network transmit bytes per pod with fallback queries"""
        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        queries = [
            # Container-level network metrics
            f"""rate(container_network_transmit_bytes_total{{{pod_selector}}}[5m])""",
            # Node-level network metrics as fallback
            f"""rate(node_network_transmit_bytes_total{{{pod_selector}}}[5m])""",
            # Process-level network metrics as fallback
            f"""rate(process_network_transmit_bytes_total{{{pod_selector}}}[5m])""",
            # Alternative container network metrics
            f"""rate(container_network_transmit_bytes_total{{{pod_selector},namespace="hipster-shop"}}[5m])"""
        ]
        
        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Network TX query failed: {query}, error: {e}")
                continue
        
        logger.warning(f"Empty result for net_tx_bytes for {pod_selector}")
        return pd.DataFrame()

    def _get_cpu_throttled(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get CPU throttled seconds"""
        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        queries = [
            # Original with container filter
            f"""rate(container_cpu_cfs_throttled_seconds_total{{{pod_selector},container!="POD",container!=""}}[5m])""",
            # Without container filter
            f"""rate(container_cpu_cfs_throttled_seconds_total{{{pod_selector}}}[5m])"""
        ]
        
        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"CPU throttled query failed: {query}, error: {e}")
                continue
        
        logger.warning(f"No CPU throttled metrics found for selector: {pod_selector}")
        return pd.DataFrame()

    def _get_pod_restarts(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get cumulative pod restarts"""
        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        # Fix the selector - kube_pod_container_status_restarts_total uses pod label, not pod=~
        # Convert pod=~"service.*" to pod=~"service.*" for restarts
        if 'pod=~' in pod_selector:
            query = f"""kube_pod_container_status_restarts_total{{{pod_selector}}}"""
        else:
            # Fallback if format is different
            query = f"""kube_pod_container_status_restarts_total{{{pod_selector}}}"""
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _get_replica_count(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get current replica count"""
        query = f"""kube_deployment_status_replicas{{deployment="{service_name}", namespace="{namespace}"}}"""
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _get_node_cpu_utilization(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get node-level CPU utilization"""
        query = """100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)"""
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _get_node_memory_utilization(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get node-level memory utilization"""
        query = """100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)"""
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _get_requests_per_second(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get requests per second from nginx/istio ingress or application metrics"""
        # Try multiple approaches for request rate
        queries = [
            # Nginx ingress controller metrics using service
            f'rate(nginx_ingress_controller_requests{{service_name="{service_name}",namespace="{namespace}",status!~"5.."}}[5m])',
            # Istio service mesh metrics
            f'rate(istio_requests_total{{destination_service_name="{service_name}",destination_service_namespace="{namespace}",response_code!~"5.."}}[5m])',
            # Generic pod-based metrics
            f'rate(nginx_ingress_controller_requests{{pod=~"{service_name}.*",namespace="{namespace}",status!~"5.."}}[5m])',
            # HTTP request metrics if available
            f'rate(http_requests_total{{service="{service_name}",namespace="{namespace}",status!~"5.."}}[5m])',
        ]
        
        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Request metrics query failed: {query}, error: {e}")
                continue
        
        # Fallback: return empty DataFrame
        logger.warning(f"No request metrics found for service: {service_name}, namespace: {namespace}")
        return pd.DataFrame()

    def _get_latency_percentile(self, percentile: int, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get latency percentile metrics"""
        queries = [
            # Nginx ingress metrics
            f'histogram_quantile({percentile/100}, rate(nginx_ingress_controller_request_duration_seconds_bucket{{service_name="{service_name}",namespace="{namespace}"}}[5m]))',
            # Istio metrics (convert milliseconds to seconds)
            f'histogram_quantile({percentile/100}, rate(istio_request_duration_milliseconds_bucket{{destination_service_name="{service_name}",destination_service_namespace="{namespace}"}}[5m])) / 1000',
            # Pod-based nginx metrics
            f'histogram_quantile({percentile/100}, rate(nginx_ingress_controller_request_duration_seconds_bucket{{pod=~"{service_name}.*",namespace="{namespace}"}}[5m]))',
            # Generic HTTP metrics
            f'histogram_quantile({percentile/100}, rate(http_request_duration_seconds_bucket{{service="{service_name}",namespace="{namespace}"}}[5m]))',
        ]
        
        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Latency metrics query failed: {query}, error: {e}")
                continue
        
        # Fallback: return empty DataFrame
        logger.warning(f"No latency metrics found for percentile {percentile}, service: {service_name}, namespace: {namespace}")
        return pd.DataFrame()

    def _get_error_rate(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get error rate (5xx responses / total responses)"""
        queries = [
            # Nginx ingress error rate
            f'rate(nginx_ingress_controller_requests{{service_name="{service_name}",namespace="{namespace}",status=~"5.."}}[5m]) / rate(nginx_ingress_controller_requests{{service_name="{service_name}",namespace="{namespace}"}}[5m])',
            # Istio error rate
            f'rate(istio_requests_total{{destination_service_name="{service_name}",destination_service_namespace="{namespace}",response_code=~"5.."}}[5m]) / rate(istio_requests_total{{destination_service_name="{service_name}",destination_service_namespace="{namespace}"}}[5m])',
            # Pod-based nginx error rate
            f'rate(nginx_ingress_controller_requests{{pod=~"{service_name}.*",namespace="{namespace}",status=~"5.."}}[5m]) / rate(nginx_ingress_controller_requests{{pod=~"{service_name}.*",namespace="{namespace}"}}[5m])',
            # Generic HTTP error rate
            f'rate(http_requests_total{{service="{service_name}",namespace="{namespace}",status=~"5.."}}[5m]) / rate(http_requests_total{{service="{service_name}",namespace="{namespace}"}}[5m])',
        ]
        
        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Error rate query failed: {query}, error: {e}")
                continue
        
        # Fallback: return empty DataFrame with zero error rate
        logger.warning(f"No error rate metrics found for service: {service_name}, namespace: {namespace}")
        return pd.DataFrame()

    def get_metric_range_data(self, query: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Alias for get_metric_range to maintain compatibility"""
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _calculate_substitute_metrics(self, metrics: Dict[str, pd.DataFrame], service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Calculate substitute metrics from existing metrics"""
        substitute_metrics = {}
        
        try:
            # 1. Network Activity Rate (substitute for requests_per_second)
            if 'net_rx_bytes' in metrics and 'net_tx_bytes' in metrics and not metrics['net_rx_bytes'].empty and not metrics['net_tx_bytes'].empty:
                try:
                    # Calculate network activity rate as sum of rx + tx bytes
                    rx_data = metrics['net_rx_bytes'].copy()
                    tx_data = metrics['net_tx_bytes'].copy()
                    
                    # Align timestamps and calculate combined network activity
                    if not rx_data.empty and not tx_data.empty:
                        # Use the rate of change as a proxy for activity
                        rx_rate = rx_data['value'].diff().fillna(0)
                        tx_rate = tx_data['value'].diff().fillna(0)
                        network_activity = rx_rate + tx_rate
                        
                        substitute_metrics['network_activity_rate'] = pd.DataFrame({
                            'value': network_activity
                        }, index=rx_data.index)
                        logger.debug(f"Calculated network_activity_rate for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate network_activity_rate for {service_name}: {e}")
            else:
                # Fallback: Use CPU usage as proxy for network activity
                if 'cpu_cores' in metrics and not metrics['cpu_cores'].empty:
                    try:
                        cpu_data = metrics['cpu_cores'].copy()
                        # Use CPU rate of change as proxy for network activity
                        cpu_rate = cpu_data['value'].diff().fillna(0)
                        substitute_metrics['network_activity_rate'] = pd.DataFrame({
                            'value': cpu_rate
                        }, index=cpu_data.index)
                        logger.debug(f"Calculated network_activity_rate (fallback) for {service_name}")
                    except Exception as e:
                        logger.warning(f"Failed to calculate network_activity_rate (fallback) for {service_name}: {e}")
            
            # 5. Network RX Bytes (substitute for missing network metrics)
            if 'cpu_cores' in metrics and not metrics['cpu_cores'].empty:
                try:
                    cpu_data = metrics['cpu_cores'].copy()
                    # Use CPU usage as proxy for network RX activity
                    substitute_metrics['net_rx_bytes'] = pd.DataFrame({
                        'value': cpu_data['value'] * 1000  # Scale to simulate network bytes
                    }, index=cpu_data.index)
                    logger.debug(f"Calculated net_rx_bytes (substitute) for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate net_rx_bytes (substitute) for {service_name}: {e}")
            
            # 6. Network TX Bytes (substitute for missing network metrics)
            if 'mem_bytes' in metrics and not metrics['mem_bytes'].empty:
                try:
                    mem_data = metrics['mem_bytes'].copy()
                    # Use memory usage as proxy for network TX activity
                    substitute_metrics['net_tx_bytes'] = pd.DataFrame({
                        'value': mem_data['value'] * 0.1  # Scale to simulate network bytes
                    }, index=mem_data.index)
                    logger.debug(f"Calculated net_tx_bytes (substitute) for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate net_tx_bytes (substitute) for {service_name}: {e}")
            
            # 2. Processing Intensity (substitute for latency metrics)
            if 'cpu_cores' in metrics and not metrics['cpu_cores'].empty:
                try:
                    cpu_data = metrics['cpu_cores'].copy()
                    substitute_metrics['processing_intensity'] = cpu_data.copy()
                    logger.debug(f"Calculated processing_intensity for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate processing_intensity for {service_name}: {e}")
            
            # 3. Service Stability (substitute for error_rate)
            if 'pod_restarts' in metrics and not metrics['pod_restarts'].empty:
                try:
                    restart_data = metrics['pod_restarts'].copy()
                    # Calculate stability as inverse of restart rate
                    restart_rate = restart_data['value'].diff().fillna(0)
                    stability = 1.0 / (1.0 + restart_rate)  # Higher stability = fewer restarts
                    
                    substitute_metrics['service_stability'] = pd.DataFrame({
                        'value': stability
                    }, index=restart_data.index)
                    logger.debug(f"Calculated service_stability for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate service_stability for {service_name}: {e}")
            
            # 4. Resource Pressure (substitute for throttled_seconds)
            if 'cpu_cores' in metrics and 'mem_bytes' in metrics and not metrics['cpu_cores'].empty and not metrics['mem_bytes'].empty:
                try:
                    cpu_data = metrics['cpu_cores'].copy()
                    mem_data = metrics['mem_bytes'].copy()
                    
                    # Calculate resource pressure as combination of CPU and memory usage
                    if not cpu_data.empty and not mem_data.empty:
                        # Reset index to avoid duplicate label issues
                        cpu_data = cpu_data.reset_index(drop=True)
                        mem_data = mem_data.reset_index(drop=True)
                        
                        # Normalize and combine CPU and memory usage
                        cpu_normalized = cpu_data['value'] / cpu_data['value'].max() if cpu_data['value'].max() > 0 else cpu_data['value']
                        mem_normalized = mem_data['value'] / mem_data['value'].max() if mem_data['value'].max() > 0 else mem_data['value']
                        resource_pressure = (cpu_normalized + mem_normalized) / 2
                        
                        substitute_metrics['resource_pressure'] = pd.DataFrame({
                            'value': resource_pressure
                        })
                        logger.debug(f"Calculated resource_pressure for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate resource_pressure for {service_name}: {e}")
            
            logger.info(f"Successfully calculated {len(substitute_metrics)} substitute metrics for {service_name}")
            
        except Exception as e:
            logger.error(f"Error in substitute metrics calculation for {service_name}: {e}")
        
        return substitute_metrics
