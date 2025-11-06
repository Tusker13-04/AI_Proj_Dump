"""
Grafana client for dashboard integration and visualization in MOrA
"""
import logging
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GrafanaClient:
    """Client for interacting with Grafana API"""
    
    def __init__(self, grafana_url: str = "http://localhost:4000", 
                 admin_user: str = "admin", admin_password: str = "admin",
                 timeout: int = 30):
        """
        Initialize Grafana client
        
        Args:
            grafana_url: URL of Grafana server
            admin_user: Grafana admin username
            admin_password: Grafana admin password
            timeout: Request timeout in seconds
        """
        self.grafana_url = grafana_url.rstrip('/')
        self.admin_user = admin_user
        self.admin_password = admin_password
        self.timeout = timeout
        self.session = requests.Session()
        self.session.auth = (admin_user, admin_password)
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info(f"GrafanaClient initialized for {grafana_url}")
    
    def test_connection(self) -> bool:
        """
        Test Grafana connection and authentication
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.grafana_url}/api/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("Successfully connected to Grafana")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Grafana: {e}")
            return False
    
    def get_dashboard(self, dashboard_uid: str) -> Optional[Dict[str, Any]]:
        """
        Get dashboard by UID
        
        Args:
            dashboard_uid: Dashboard UID
            
        Returns:
            Dashboard data or None if not found
        """
        try:
            response = self.session.get(
                f"{self.grafana_url}/api/dashboards/uid/{dashboard_uid}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get dashboard {dashboard_uid}: {e}")
            return None
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """
        List all dashboards
        
        Returns:
            List of dashboard summaries
        """
        try:
            response = self.session.get(
                f"{self.grafana_url}/api/search?type=dash-db",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list dashboards: {e}")
            return []
    
    def get_data_source(self, name: str = "Prometheus") -> Optional[Dict[str, Any]]:
        """
        Get data source configuration
        
        Args:
            name: Data source name (default: Prometheus)
            
        Returns:
            Data source configuration or None
        """
        try:
            response = self.session.get(
                f"{self.grafana_url}/api/datasources/name/{name}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get data source {name}: {e}")
            return None
    
    def create_mora_dashboard(self, namespace: str = "hipster-shop") -> Optional[str]:
        """
        Create MOrA-specific dashboard for monitoring microservices
        
        Args:
            namespace: Kubernetes namespace to monitor
            
        Returns:
            Dashboard UID if successful, None otherwise
        """
        dashboard_config = {
            "dashboard": {
                "id": None,
                "uid": None,
                "title": f"MOrA - {namespace} Monitoring",
                "tags": ["mora", "microservices", "kubernetes", namespace],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU Usage by Service",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": f'avg(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", container!="POD", container!=""}}[5m])) by (pod)',
                                "datasource": "Prometheus",
                                "legendFormat": "{{pod}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "short",
                                "color": {"mode": "palette-classic"}
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Memory Usage by Service",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": f'avg(container_memory_working_set_bytes{{namespace="{namespace}", container!="POD", container!=""}}) by (pod)',
                                "datasource": "Prometheus",
                                "legendFormat": "{{pod}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "bytes",
                                "color": {"mode": "palette-classic"}
                            }
                        }
                    },
                    {
                        "id": 3,
                        "title": "Network I/O by Pod",
                        "type": "timeseries",
                        "targets": [
                            {
                                "expr": f'rate(container_network_receive_bytes_total{{namespace="{namespace}"}}[5m])',
                                "datasource": "Prometheus",
                                "legendFormat": "RX {{pod}}"
                            },
                            {
                                "expr": f'rate(container_network_transmit_bytes_total{{namespace="{namespace}"}}[5m])',
                                "datasource": "Prometheus",
                                "legendFormat": "TX {{pod}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "binBps",
                                "color": {"mode": "palette-classic"}
                            }
                        }
                    },
                    {
                        "id": 4,
                        "title": "Pod Replicas by Deployment",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": f'kube_deployment_status_replicas{{namespace="{namespace}"}}',
                                "datasource": "Prometheus",
                                "legendFormat": "{{deployment}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "short",
                                "color": {"mode": "thresholds"}
                            }
                        }
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            },
            "overwrite": True
        }
        
        try:
            response = self.session.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=dashboard_config,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Created MOrA dashboard: {result.get('uid', 'unknown')}")
            return result.get('uid')
        except Exception as e:
            logger.error(f"Failed to create MOrA dashboard: {e}")
            return None
    
    def verify_prometheus_datasource(self) -> bool:
        """
        Verify that Prometheus data source is configured and accessible
        
        Returns:
            True if Prometheus data source is working, False otherwise
        """
        try:
            # Test Prometheus data source connection
            test_query = {"query": "up"}
            response = self.session.post(
                f"{self.grafana_url}/api/ds/query",
                json={
                    "queries": [
                        {
                            "datasource": {"type": "prometheus", "uid": "prometheus"},
                            "expr": "up",
                            "refId": "A"
                        }
                    ]
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info("Prometheus data source is working in Grafana")
                return True
            else:
                logger.warning(f"Prometheus data source test returned: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify Prometheus data source: {e}")
            return False
    
    def get_dashboard_url(self, dashboard_uid: str) -> str:
        """
        Get the URL for accessing a dashboard
        
        Args:
            dashboard_uid: Dashboard UID
            
        Returns:
            Dashboard URL
        """
        return f"{self.grafana_url}/d/{dashboard_uid}"
