"""
Isolated data acquisition pipeline for MOrA training
"""
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import concurrent.futures
import threading

from ..k8s.client import KubernetesClient
from ..monitoring.prometheus_client import PrometheusClient
from .load_generator import LoadGenerator

logger = logging.getLogger(__name__)


class DataAcquisitionPipeline:
    """
    Implements the "train in isolation" methodology for collecting clean training data.
    """

    def __init__(
        self,
        namespace: str = "hipster-shop",
        prometheus_url: str = "http://localhost:9090",
        data_dir: str = "training_data",
        k8s_client: Optional[KubernetesClient] = None,
        prom_client: Optional[PrometheusClient] = None,
        load_generator: Optional[LoadGenerator] = None
    ):
        self.namespace = namespace
        self.prometheus_url = prometheus_url
        self.data_dir = data_dir
        self.k8s_client = k8s_client or KubernetesClient()
        self.prom_client = prom_client or PrometheusClient(prometheus_url)
        self.load_generator = load_generator or LoadGenerator(
            namespace=namespace,
            prometheus_url=prometheus_url,
            k8s_client=self.k8s_client,
            prom_client=self.prom_client
        )
        
        # Data collection configuration
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"DataAcquisitionPipeline initialized for namespace: {namespace}")

    def _get_experiment_id(self, target_service: str, scenario: str, replica_count: int, load_users: int) -> str:
        """Generate consistent experiment ID."""
        return f"{target_service}_{scenario}_replicas_{replica_count}_users_{load_users}"

    def _is_experiment_completed(self, experiment_id: str) -> bool:
        """Check if an experiment has already been completed."""
        try:
            # Check for JSON result file
            json_file = os.path.join(self.data_dir, f"{experiment_id}.json")
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    return data.get("status") in ["completed", "completed_with_warnings"]
            return False
        except Exception:
            return False

    def _get_completed_experiments(self, target_service: str) -> set:
        """Get set of completed experiment IDs for a service."""
        completed = set()
        try:
            files = os.listdir(self.data_dir)
            for file in files:
                if file.startswith(f"{target_service}_") and file.endswith(".json"):
                    experiment_id = file.replace(".json", "")
                    if self._is_experiment_completed(experiment_id):
                        completed.add(experiment_id)
        except Exception as e:
            logger.warning(f"Could not check completed experiments: {e}")
        return completed

    def run_isolated_training_experiment(
        self,
        target_service: str,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run clean steady-state training experiments for isolated data collection.
        Cycles through ALL scenarios, and for EACH scenario, runs the
        full double-loop of replicas and load levels.
        """
        # Load training config (steady-state experiments)
        if config is None:
            # Use resource-optimized defaults
            config = {
                "experiment_duration_minutes": 15,
                "sample_interval": "30s",
                "replica_counts": [1, 2, 4],
                "load_levels_users": [5, 10, 20, 30, 50, 75],
                "test_scenarios": ["browsing", "checkout"],
                "stabilization_wait_seconds": 180
            }
        
        replica_counts = config["replica_counts"]
        load_levels = config["load_levels_users"]
        experiment_duration = config["experiment_duration_minutes"]
        
        # Get the list of scenarios to test
        scenarios_to_run = config.get("test_scenarios", ["browsing", "checkout"])
        
        logger.info(f"Starting CLEAN training experiments for {target_service}")
        logger.info(f"Scenarios: {scenarios_to_run}")
        logger.info(f"Replica counts: {replica_counts}")
        logger.info(f"Load levels: {load_levels} users")
        logger.info(f"Experiment duration: {experiment_duration} minutes each")
        
        experiment_results = {
            "target_service": target_service,
            "experiment_type": "steady_state_training",
            "start_time": datetime.now().isoformat(),
            "experiments": [],
            "total_combinations": len(replica_counts) * len(load_levels) * len(scenarios_to_run)
        }
        
        try:
            # Step 1: Over-provision all non-target services
            logger.info("Step 1: Over-provisioning non-target services")
            if not self.load_generator.overprovision_non_target_services(target_service):
                raise RuntimeError("Failed to over-provision non-target services")
            
            # Check for completed experiments before starting
            completed_experiments = self._get_completed_experiments(target_service)
            logger.info(f"Found {len(completed_experiments)} previously completed experiments")

            # NEW TRIPLE LOOP with resume capability
            experiment_count = 0
            skipped_count = 0

            # 1. Outer loop: Scenarios
            for scenario in scenarios_to_run:
                logger.info(f"--- Starting Scenario: {scenario} ---")
                
                # 2. Middle loop: Replicas
                for replica_count in replica_counts:
                    
                    # 3. Inner loop: Load levels
                    for load_users in load_levels:
                        experiment_id = self._get_experiment_id(target_service, scenario, replica_count, load_users)
                        
                        # Skip if already completed
                        if experiment_id in completed_experiments:
                            skipped_count += 1
                            logger.info(f"⏭️  Skipping completed experiment: {experiment_id}")
                            continue
                        
                        experiment_count += 1
                        
                        logger.info(f"🔄 Experiment {experiment_count}/{experiment_results['total_combinations']}: "
                                  f"Scenario={scenario}, Replicas={replica_count}, Users={load_users}")
                        logger.info(f"📊 Progress: {skipped_count} completed, {experiment_count} running, "
                                  f"{experiment_results['total_combinations'] - skipped_count - experiment_count} remaining")
                        
                        # Run experiment with progress tracking
                        experiment_result = self._run_steady_state_experiment(
                            target_service=target_service,
                            replica_count=replica_count,
                            load_users=load_users,
                            duration_minutes=experiment_duration,
                            experiment_id=experiment_id,
                            test_scenario=scenario,
                            config=config
                        )
                        
                        # Save immediately after each experiment
                        self._save_experiment_data(experiment_id, experiment_result)
                        experiment_results["experiments"].append(experiment_result)
                        
                        # Brief pause between experiments
                        logger.info("⏳ Waiting 60 seconds between experiments...")
                        time.sleep(60)
            
            experiment_results["end_time"] = datetime.now().isoformat()
            experiment_results["status"] = "completed"
            
            # Save training results
            self._save_training_results(target_service, experiment_results)
            
            logger.info(f"Completed isolated training experiment for {target_service}")
            return experiment_results
            
        except Exception as e:
            logger.error(f"Isolated training experiment failed: {e}")
            experiment_results["status"] = "failed"
            experiment_results["error"] = str(e)
            return experiment_results

    def _run_single_experiment(
        self,
        target_service: str,
        load_scenario: Dict[str, Any],
        replica_count: int,
        collection_duration_minutes: int
    ) -> Dict[str, Any]:
        """
        Run a single experiment with specific load and replica configuration.
        """
        experiment_id = f"{target_service}_{load_scenario['name']}_replicas_{replica_count}"
        logger.info(f"Starting experiment: {experiment_id}")
        
        experiment_result = {
            "experiment_id": experiment_id,
            "target_service": target_service,
            "load_scenario": load_scenario,
            "replica_count": replica_count,
            "collection_start": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Scale target service to desired replica count
            logger.info(f"Scaling {target_service} to {replica_count} replicas")
            if not self.k8s_client.scale_deployment(target_service, self.namespace, replica_count):
                raise RuntimeError(f"Failed to scale {target_service} to {replica_count} replicas")
            
            # Wait for scaling to complete
            time.sleep(60)
            
            # Step 2: Start load test
            logger.info(f"Starting load test: {load_scenario['users']} users for {load_scenario['duration']} minutes")
            script_path = self.load_generator.create_jmeter_script(
                script_name=f"{experiment_id}_load",
                target_host="localhost",
                target_port=8080,
                test_scenario="browsing",  # Default to browsing, could be parameterized
                num_users=load_scenario['users']
            )
            
            load_test_result = self.load_generator.run_load_test(
                script_path=script_path,
                duration_minutes=load_scenario['duration']
            )
            
            experiment_result["load_test"] = load_test_result
            
            # Step 3: Collect metrics during the experiment
            logger.info("Collecting metrics for all containers in target service")
            metrics_data = {}
            
            # Get container names for the target service
            containers = self._get_service_containers(target_service)
            
            for container_name in containers:
                # Collect CPU metrics
                cpu_data = self._collect_metrics(
                    target_service, container_name, "cpu", 
                    collection_duration_minutes
                )
                if cpu_data and not cpu_data.get("error"):
                    metrics_data[f"{container_name}_cpu"] = cpu_data["metrics"]
                
                # Collect Memory metrics
                memory_data = self._collect_metrics(
                    target_service, container_name, "memory", 
                    collection_duration_minutes
                )
                if memory_data and not memory_data.get("error"):
                    metrics_data[f"{container_name}_memory"] = memory_data["metrics"]
            
            experiment_result["metrics"] = metrics_data
            experiment_result["collection_end"] = datetime.now().isoformat()
            experiment_result["status"] = "completed"
            
            # Save individual experiment data
            self._save_experiment_data(experiment_id, experiment_result)
            
            logger.info(f"Completed experiment: {experiment_id}")
            return experiment_result
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            experiment_result["status"] = "failed"
            experiment_result["error"] = str(e)
            return experiment_result

    def _run_steady_state_experiment(
        self,
        target_service: str,
        replica_count: int,
        load_users: int,
        duration_minutes: int,
        experiment_id: str,
        test_scenario: str = "browsing",
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a single steady-state experiment with fixed replicas and load.
        This ensures clean data for ML training without mixing load levels.
        """
        logger.info(f"Running steady-state experiment: {experiment_id}")
        
        experiment_result = {
            "experiment_id": experiment_id,
            "target_service": target_service,
            "replica_count": replica_count,
            "load_users": load_users,
            "duration_minutes": duration_minutes,
            "collection_start": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Set exact replica count
            logger.info(f"Setting {target_service} to {replica_count} replicas")
            if not self.k8s_client.scale_deployment(target_service, self.namespace, replica_count):
                raise RuntimeError(f"Failed to scale {target_service} to {replica_count} replicas")
            
            # Wait for scaling to stabilize (use config value if available)
            stabilization_wait = config.get("stabilization_wait_seconds", 90) if config else 90
            logger.info(f"Waiting {stabilization_wait} seconds for scaling to stabilize...")
            time.sleep(stabilization_wait)
            
            # Step 2: Run load test with FIXED user count
            if load_users > 0:
                logger.info(f"Starting load test: {load_users} users for {duration_minutes} minutes")
                
                script_path = self.load_generator.create_jmeter_script(
                    script_name=f"{experiment_id}_load",
                    target_host="localhost",
                    target_port=8080,
                    test_scenario=test_scenario,  # Use the scenario parameter
                    num_users=load_users  # Pass the actual load level
                )
                
                # Start load test and collect metrics simultaneously
                load_test_result = self.load_generator.run_load_test(
                    script_path=script_path,
                    duration_minutes=duration_minutes
                )
                experiment_result["load_test"] = load_test_result
            else:
                logger.info("No load test (idle scenario)")
                experiment_result["load_test"] = {"users": 0, "status": "idle"}
            
            # Step 3: Collect CLEAN metrics during the experiment using enhanced client
            logger.info("Collecting steady-state metrics...")
            metrics_data = {}
            
            # Use the enhanced comprehensive metrics collection
            sample_interval = config.get("sample_interval", "30s") if config else "30s"
            
            try:
                # Always use comprehensive metrics collection - it's more robust
                logger.info(f"Collecting comprehensive metrics for {target_service}...")
                comprehensive_metrics = self.prom_client.get_comprehensive_metrics(
                    namespace=self.namespace,
                    service_name=target_service,
                    duration_minutes=duration_minutes
                )
                
                if comprehensive_metrics and len(comprehensive_metrics) > 0:
                    metrics_data.update(comprehensive_metrics)
                    logger.info(f"✅ Comprehensive metrics collection successful for {target_service}: {len(comprehensive_metrics)} metrics collected")
                    logger.info(f"📊 Metrics collected: {list(comprehensive_metrics.keys())}")
                else:
                    logger.warning(f"⚠️ No comprehensive metrics returned for {target_service}")
                    # Don't fall back to basic metrics - comprehensive should always work
                    raise RuntimeError("No metrics returned from comprehensive collection")
                
            except Exception as e:
                logger.error(f"❌ Failed to collect comprehensive metrics for {target_service}: {e}")
                logger.error("🚨 CRITICAL: Comprehensive metrics collection failed - this should not happen!")
                # Don't fall back to basic metrics - this indicates a serious issue
                raise RuntimeError(f"Comprehensive metrics collection failed: {e}")
            
            experiment_result["metrics"] = metrics_data
            experiment_result["collection_end"] = datetime.now().isoformat()
            
            # Step 4: Validate data quality
            quality_validation = self._validate_experiment_data_quality(metrics_data, config)
            experiment_result["data_quality"] = quality_validation
            
            if quality_validation["status"] == "passed":
                experiment_result["status"] = "completed"
                logger.info(f"Completed steady-state experiment: {experiment_id}")
            else:
                experiment_result["status"] = "completed_with_warnings"
                logger.warning(f"Experiment {experiment_id} completed with quality warnings: {quality_validation['warnings']}")
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"Steady-state experiment {experiment_id} failed: {e}")
            experiment_result["status"] = "failed"
            experiment_result["error"] = str(e)
            return experiment_result

    def _validate_experiment_data_quality(self, metrics_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate the quality of collected experiment data based on configured quality checks.
        """
        
        quality_checks = config.get("data_quality_checks", {}) if config else {}
        
        # Default quality check values
        min_completeness = quality_checks.get("min_data_completeness_percent", 90)
        max_nan_percent = quality_checks.get("max_metric_nan_percent", 5)
        max_std_dev_percent = quality_checks.get("max_std_dev_percent", 10)
        
        validation_result = {
            "status": "passed",
            "warnings": [],
            "checks": {
                "completeness": {"passed": True, "details": ""},
                "nan_values": {"passed": True, "details": ""},
                "stability": {"passed": True, "details": ""}
            }
        }
        
        try:
            # Check 1: Data Completeness
            total_metrics = len(metrics_data)
            if total_metrics == 0:
                validation_result["status"] = "failed"
                validation_result["warnings"].append("No metrics data collected")
                validation_result["checks"]["completeness"] = {
                    "passed": False, 
                    "details": "No metrics collected"
                }
                return validation_result
            
            # Expected metrics from config
            required_metrics = config.get("required_metrics", []) if config else []
            if required_metrics:
                collected_metric_types = set()
                for metric_key in metrics_data.keys():
                    # Extract metric type from key (e.g., "cpu_cores", "mem_bytes")
                    for req_metric in required_metrics:
                        if req_metric in metric_key:
                            collected_metric_types.add(req_metric)
                
                completeness_percent = (len(collected_metric_types) / len(required_metrics)) * 100
                if completeness_percent < min_completeness:
                    validation_result["warnings"].append(
                        f"Low data completeness: {completeness_percent:.1f}% (required: {min_completeness}%)"
                    )
                    validation_result["checks"]["completeness"] = {
                        "passed": completeness_percent >= min_completeness,
                        "details": f"Collected {len(collected_metric_types)}/{len(required_metrics)} required metrics"
                    }
            
            # Check 2: NaN Values in Critical Metrics
            critical_metrics = ["cpu_cores", "mem_bytes", "requests_per_second"]
            for metric_key, metric_value in metrics_data.items():
                if any(critical in metric_key.lower() for critical in critical_metrics):
                    if isinstance(metric_value, pd.DataFrame):
                        # Check for NaN values in DataFrame
                        nan_percent = (metric_value.isnull().sum().sum() / metric_value.size) * 100
                        if nan_percent > max_nan_percent:
                            validation_result["warnings"].append(
                                f"High NaN values in {metric_key}: {nan_percent:.1f}% (limit: {max_nan_percent}%)"
                            )
                            validation_result["checks"]["nan_values"] = {
                                "passed": nan_percent <= max_nan_percent,
                                "details": f"{metric_key}: {nan_percent:.1f}% NaN values"
                            }
            
            # Check 3: Stability Check (last 5 minutes for CPU/Memory)
            for metric_key, metric_value in metrics_data.items():
                if isinstance(metric_value, pd.DataFrame) and any(stability_metric in metric_key.lower() for stability_metric in ["cpu_cores", "mem_bytes"]):
                    if not metric_value.empty and len(metric_value) >= 10:  # Need enough data points
                        # Take last 5 minutes of data (assuming data points every few seconds)
                        last_data = metric_value.tail(20)  # Approximate last 5 minutes
                        
                        if 'value' in last_data.columns:
                            values = last_data['value'].dropna()
                            if len(values) >= 5:
                                mean_val = values.mean()
                                std_val = values.std()
                                if mean_val > 0:  # Avoid division by zero
                                    cv = (std_val / mean_val) * 100  # Coefficient of variation
                                    if cv > max_std_dev_percent:
                                        validation_result["warnings"].append(
                                            f"High variability in {metric_key}: CV={cv:.1f}% (limit: {max_std_dev_percent}%)"
                                        )
                                        validation_result["checks"]["stability"] = {
                                            "passed": cv <= max_std_dev_percent,
                                            "details": f"{metric_key}: CV={cv:.1f}% (unstable)"
                                        }
            
            # Overall status
            if validation_result["warnings"]:
                validation_result["status"] = "warnings"
            
            return validation_result
            
        except Exception as e:
            logger.warning(f"Data quality validation failed: {e}")
            validation_result["status"] = "error"
            validation_result["warnings"].append(f"Quality validation error: {str(e)}")
            return validation_result

    def _collect_steady_state_metrics(
        self,
        service_name: str,
        container_name: str,
        resource_type: str,
        duration_minutes: int
    ) -> Dict[str, Any]:
        """Collect steady-state metrics for a specific container and resource type."""
        try:
            start_time = datetime.now() - timedelta(minutes=duration_minutes)
            end_time = datetime.now()
            
            if resource_type == "cpu_cores":
                metrics_df = self.prom_client.get_cpu_metrics(
                    namespace=self.namespace,
                    container_name=container_name,
                    start_time=start_time,
                    end_time=end_time
                )
            elif resource_type == "mem_bytes":
                metrics_df = self.prom_client.get_memory_metrics(
                    namespace=self.namespace,
                    container_name=container_name,
                    start_time=start_time,
                    end_time=end_time
                )
            else:
                return {"error": f"Unsupported resource type: {resource_type}"}
            
            return {
                "metrics": metrics_df,
                "resource_type": resource_type,
                "container": container_name
            }
            
        except Exception as e:
            logger.error(f"Error collecting {resource_type} metrics: {e}")
            return {"error": str(e)}

    def _save_training_results(self, target_service: str, results: Dict[str, Any]) -> str:
        """Save training experiment results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{target_service}_training_results_{timestamp}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Training results saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")
            return ""

    def _get_service_containers(self, service_name: str) -> List[str]:
        """Get container names for a service deployment."""
        try:
            deployment = self.k8s_client.get_deployment(service_name, self.namespace)
            if not deployment:
                return []
            
            containers = []
            for container in deployment.spec.template.spec.containers:
                containers.append(container.name)
            
            return containers
            
        except Exception as e:
            logger.error(f"Failed to get containers for {service_name}: {e}")
            return []

    def _collect_metrics(
        self,
        target_service: str,
        container_name: str,
        resource_type: str,
        duration_minutes: int
    ) -> Dict[str, Any]:
        """Collect metrics for a specific service, container, and resource type."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=duration_minutes)
            
            # Construct Prometheus query
            if resource_type == "cpu":
                query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{self.namespace}", pod=~"{target_service}-.*", container="{container_name}"}}[1m])) by (container)'
            elif resource_type == "memory":
                query = f'sum(container_memory_working_set_bytes{{namespace="{self.namespace}", pod=~"{target_service}-.*", container="{container_name}"}}) by (container)'
            else:
                return {"error": f"Unsupported resource type: {resource_type}"}
            
            metrics_df = self.prom_client.get_metric_range(query, start_time, end_time)
            
            if metrics_df.empty:
                logger.warning(f"No {resource_type} metrics for {target_service}/{container_name}")
                return {"error": "No metrics data"}
            
            return {
                "service_name": target_service,
                "container_name": container_name,
                "resource_type": resource_type,
                "data_points": len(metrics_df),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "metrics": metrics_df
            }
            
        except Exception as e:
            logger.error(f"Failed to collect {resource_type} metrics for {target_service}/{container_name}: {e}")
            return {"error": str(e)}

    def _save_experiment_results(self, target_service: str, results: Dict[str, Any]):
        """Save complete experiment results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{target_service}_experiment_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            # Convert DataFrames to serializable format
            serializable_results = self._make_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Saved experiment results to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")

    def _save_experiment_data(self, experiment_id: str, data: Dict[str, Any]):
        """Save individual experiment data."""
        try:
            filepath = os.path.join(self.data_dir, f"{experiment_id}.json")
            
            # Create unified CSV with all metrics in one file
            processed_data = data.copy()
            if "metrics" in processed_data:
                # Combine all metrics into one unified DataFrame
                unified_data = self._create_unified_metrics_dataframe(experiment_id, processed_data)
                
                if unified_data is not None and not unified_data.empty:
                    # Save unified CSV file
                    unified_csv_path = os.path.join(self.data_dir, f"{experiment_id}.csv")
                    unified_data.to_csv(unified_csv_path, index=False)
                    processed_data["unified_csv_path"] = unified_csv_path
                    logger.info(f"Saved unified metrics to {unified_csv_path} with {len(unified_data)} rows")
                else:
                    logger.warning(f"No unified data created for {experiment_id}")
            
            with open(filepath, 'w') as f:
                json.dump(self._make_serializable(processed_data), f, indent=2, default=str)
            
            logger.info(f"Saved experiment data to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment data: {e}")

    def _create_unified_metrics_dataframe(self, experiment_id: str, data: Dict[str, Any]) -> pd.DataFrame:
        """Create a unified DataFrame with all metrics for one experiment."""
        try:
            # Extract experiment context
            service = data.get("target_service", "")
            replica_count = data.get("replica_count", 0)
            load_users = data.get("load_users", 0)
            
            # Extract scenario from experiment_id
            scenario = "browsing"  # default
            if "_browsing_" in experiment_id:
                scenario = "browsing"
            elif "_checkout_" in experiment_id:
                scenario = "checkout"
            
            # Get metrics data
            metrics_data = data.get("metrics", {})
            if not metrics_data:
                logger.warning(f"No metrics data found for {experiment_id}")
                return pd.DataFrame()
            
            # Find the maximum length across all metrics
            max_length = 0
            for metric_name, metric_df in metrics_data.items():
                if isinstance(metric_df, pd.DataFrame) and not metric_df.empty:
                    max_length = max(max_length, len(metric_df))
            
            if max_length == 0:
                logger.warning(f"No valid metrics data found for {experiment_id}")
                return pd.DataFrame()
            
            # Create unified DataFrame
            unified_rows = []
            for i in range(max_length):
                row = {
                    "timestamp": f"t_{i}",
                    "experiment_id": experiment_id,
                    "service": service,
                    "scenario": scenario,
                    "replica_count": replica_count,
                    "load_users": load_users
                }
                
                # Add all metric values for this time point
                for metric_name, metric_df in metrics_data.items():
                    if isinstance(metric_df, pd.DataFrame) and not metric_df.empty and i < len(metric_df):
                        # Extract the 'value' column if it exists, otherwise use the first numeric column
                        if 'value' in metric_df.columns:
                            row[f"{metric_name}_value"] = metric_df.iloc[i]['value']
                        else:
                            # Use the first numeric column
                            numeric_cols = metric_df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                row[f"{metric_name}_value"] = metric_df.iloc[i][numeric_cols[0]]
                    else:
                        # Handle missing data points
                        row[f"{metric_name}_value"] = None
                
                unified_rows.append(row)
            
            unified_df = pd.DataFrame(unified_rows)
            logger.info(f"Created unified DataFrame for {experiment_id} with {len(unified_df)} rows and {len(unified_df.columns)} columns")
            return unified_df
            
        except Exception as e:
            logger.error(f"Failed to create unified DataFrame for {experiment_id}: {e}")
            return pd.DataFrame()

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return {"_type": "DataFrame", "shape": obj.shape, "columns": obj.columns.tolist()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def load_experiment_data(self, target_service: str, experiment_id: str) -> Dict[str, Any]:
        """Load previously collected experiment data."""
        try:
            filepath = os.path.join(self.data_dir, f"{experiment_id}.json")
            
            if not os.path.exists(filepath):
                return {"error": f"Experiment data not found: {filepath}"}
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load CSV files if they exist
            if "metrics" in data:
                for key, value in data["metrics"].items():
                    if key.endswith("_csv_path") and os.path.exists(value):
                        csv_data = pd.read_csv(value)
                        metric_key = key.replace("_csv_path", "")
                        data["metrics"][metric_key] = csv_data
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load experiment data: {e}")
            return {"error": str(e)}

    def list_available_experiments(self, target_service: str = None) -> List[str]:
        """List all available experiment data files."""
        try:
            files = os.listdir(self.data_dir)
            experiment_files = [f for f in files if f.endswith('.json') and 'experiment' in f]
            
            if target_service:
                experiment_files = [f for f in experiment_files if target_service in f]
            
            return experiment_files
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []

    def run_dynamic_evaluation_experiment(
        self,
        target_service: str,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run dynamic evaluation experiment (for Phase 4).
        This uses the dynamic load profile where load changes every 15-30 minutes.
        """
        if config is None:
            config = {
                "collection_duration_minutes": 120,
                "sample_interval": "30s",
                "dynamic_load_scenarios": [
                    {"name": "idle", "users": 0, "duration_minutes": 15},
                    {"name": "low", "users": 10, "duration_minutes": 30},
                    {"name": "medium", "users": 50, "duration_minutes": 30},
                    {"name": "high", "users": 100, "duration_minutes": 30},
                    {"name": "peak", "users": 200, "duration_minutes": 15}
                ]
            }
        
        logger.info(f"Starting dynamic evaluation experiment for {target_service}")
        logger.info(f"Total duration: {config['collection_duration_minutes']} minutes")
        
        experiment_results = {
            "target_service": target_service,
            "experiment_type": "dynamic_evaluation",
            "start_time": datetime.now().isoformat(),
            "config": config,
            "scenarios": []
        }
        
        try:
            # Over-provision non-target services first
            logger.info("Over-provisioning non-target services for evaluation")
            if not self.load_generator.overprovision_non_target_services(target_service):
                raise RuntimeError("Failed to over-provision non-target services")
            
            # Run dynamic scenarios
            for scenario in config["dynamic_load_scenarios"]:
                logger.info(f"Running scenario: {scenario['name']} ({scenario['users']} users, {scenario['duration_minutes']} min)")
                
                scenario_result = self._run_dynamic_scenario(
                    target_service=target_service,
                    scenario=scenario,
                    config=config
                )
                
                experiment_results["scenarios"].append(scenario_result)
            
            experiment_results["end_time"] = datetime.now().isoformat()
            experiment_results["status"] = "completed"
            
            # Save evaluation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{target_service}_evaluation_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(experiment_results, f, indent=2, default=str)
            
            logger.info(f"Dynamic evaluation completed for {target_service}")
            return experiment_results
            
        except Exception as e:
            logger.error(f"Dynamic evaluation failed for {target_service}: {e}")
            experiment_results["status"] = "failed"
            experiment_results["error"] = str(e)
            return experiment_results

    def _run_dynamic_scenario(
        self,
        target_service: str,
        scenario: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single dynamic scenario."""
        scenario_result = {
            "scenario_name": scenario["name"],
            "users": scenario["users"],
            "duration_minutes": scenario["duration_minutes"],
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Run load test if users > 0
            if scenario["users"] > 0:
                script_path = self.load_generator.create_jmeter_script(
                    script_name=f"{target_service}_{scenario['name']}_dynamic",
                    target_host="localhost",
                    target_port=8080,
                    test_scenario="browsing",
                    num_users=scenario["users"]
                )
                
                load_result = self.load_generator.run_load_test(
                    script_path=script_path,
                    duration_minutes=scenario["duration_minutes"]
                )
                scenario_result["load_test"] = load_result
            
            # Collect metrics during the scenario
            metrics = self.prom_client.get_comprehensive_metrics(
                namespace=self.namespace,
                service_name=target_service,
                duration_minutes=scenario["duration_minutes"]
            )
            
            scenario_result["metrics"] = {k: v for k, v in metrics.items() if isinstance(v, dict)}
            scenario_result["end_time"] = datetime.now().isoformat()
            scenario_result["status"] = "completed"
            
            return scenario_result
            
        except Exception as e:
            logger.error(f"Dynamic scenario {scenario['name']} failed: {e}")
            scenario_result["status"] = "failed"
            scenario_result["error"] = str(e)
            return scenario_result

    def run_parallel_training_experiments(
        self,
        services: List[str],
        config: Dict[str, Any] = None,
        max_workers: int = 1
    ) -> Dict[str, Any]:
        """
        Run training experiments in parallel across multiple services.
        This dramatically reduces total collection time.
        """
        if config is None:
            config = {
                "training": {
                    "steady_state_config": {
                        "experiment_duration_minutes": 15,
                        "replica_counts": [1, 2, 4],
                        "load_levels_users": [5, 10, 20, 30, 50, 75],
                        "test_scenarios": ["browsing", "checkout"]
                    }
                }
            }
        
        # Handle both old and new config structures
        if "training" in config:
            steady_config = config.get("training", {}).get("steady_state_config", {})
        else:
            steady_config = config.get("steady_state_config", {})
            
        replica_counts = steady_config.get("replica_counts", [1, 2, 4])
        load_levels = steady_config.get("load_levels_users", [5, 10, 20, 30, 50, 75])
        scenarios_to_run = steady_config.get("test_scenarios", ["browsing", "checkout"])
        experiment_duration = steady_config.get("experiment_duration_minutes", 15)
        
        total_experiments = len(services) * len(scenarios_to_run) * len(replica_counts) * len(load_levels)
        
        logger.info(f"🚀 Starting PARALLEL training for {len(services)} services")
        logger.info(f"Services: {services}")
        logger.info(f"Total experiments: {total_experiments}")
        
        # Create experiment queue
        experiment_queue = []
        for service in services:
            for scenario in scenarios_to_run:
                for replica_count in replica_counts:
                    for load_users in load_levels:
                        experiment_id = self._get_experiment_id(service, scenario, replica_count, load_users)
                        experiment_queue.append({
                            "service": service,
                            "scenario": scenario,
                            "replica_count": replica_count,
                            "load_users": load_users,
                            "experiment_id": experiment_id,
                            "duration_minutes": experiment_duration
                        })
        
        # Track results
        results = {
            "start_time": datetime.now().isoformat(),
            "services": services,
            "total_experiments": total_experiments,
            "experiments": [],
            "status": "running"
        }
        
        # Thread-safe result collection and service-level locking
        results_lock = threading.Lock()
        service_locks = {service: threading.Lock() for service in services}
        completed_count = 0
        
        def run_experiment_worker(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
            """Worker function to run a single experiment with conflict prevention."""
            nonlocal completed_count
            
            try:
                # Check if experiment already completed before starting
                experiment_id = experiment_config["experiment_id"]
                if self._is_experiment_completed(experiment_id):
                    logger.info(f"⏭️  Skipping already completed: {experiment_id}")
                    with results_lock:
                        completed_count += 1
                    return {
                        "experiment_id": experiment_id,
                        "status": "skipped",
                        "reason": "already_completed"
                    }
                
                logger.info(f"🔄 Starting: {experiment_id}")
                
                # Use service-level locking to prevent scaling conflicts
                service = experiment_config["service"]
                with service_locks[service]:
                    result = self._run_steady_state_experiment(
                        target_service=service,
                        replica_count=experiment_config["replica_count"],
                        load_users=experiment_config["load_users"],
                        duration_minutes=experiment_config["duration_minutes"],
                        experiment_id=experiment_id,
                        test_scenario=experiment_config["scenario"],
                        config=config
                    )
                
                # Save data immediately to prevent loss on interruption
                self._save_experiment_data(experiment_id, result)
                
                # Thread-safe progress update
                with results_lock:
                    completed_count += 1
                    logger.info(f"✅ Completed: {experiment_id} ({completed_count}/{total_experiments})")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Failed: {experiment_config['experiment_id']} - {e}")
                return {
                    "experiment_id": experiment_config["experiment_id"],
                    "status": "failed",
                    "error": str(e)
                }
        
        # Determine optimal parallelization
        # We can run experiments in parallel if they use different services
        # or different replica counts (since we can scale independently)
        max_workers = min(max_workers, len(services), 8)  # Cap at 8 workers for safety
        
        logger.info(f"🔀 Running with {max_workers} parallel workers")
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_exp = {
                    executor.submit(run_experiment_worker, exp): exp 
                    for exp in experiment_queue
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_exp):
                    experiment_config = future_to_exp[future]
                    try:
                        result = future.result()
                        results["experiments"].append(result)
                    except Exception as e:
                        logger.error(f"Experiment {experiment_config['experiment_id']} generated exception: {e}")
                        results["experiments"].append({
                            "experiment_id": experiment_config["experiment_id"],
                            "status": "failed",
                            "error": str(e)
                        })
        
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        results["end_time"] = datetime.now().isoformat()
        results["status"] = "completed"
        
        logger.info(f"🎉 Parallel training completed for {len(services)} services")
        return results

    def process_collected_data_for_training(
        self, 
        target_services: List[str] = None,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Process all collected experiment data into a training dataset.
        Combines JSON metadata and CSV metrics into a unified DataFrame.
        """
        if output_file is None:
            output_file = os.path.join(self.data_dir, "training_data_master.csv")
        
        logger.info("Processing collected experiment data for training...")
        
        try:
            # Get all experiment files
            experiment_files = []
            for file in os.listdir(self.data_dir):
                if file.endswith('.json') and not file.startswith('.'):
                    experiment_files.append(file)
            
            logger.info(f"Found {len(experiment_files)} experiment files")
            
            if not experiment_files:
                return {"status": "no_data", "message": "No experiment data found"}
            
            # Filter by target services if specified
            if target_services:
                filtered_files = []
                for service in target_services:
                    for file in experiment_files:
                        if file.startswith(f"{service}_"):
                            filtered_files.append(file)
                experiment_files = filtered_files
            
            # Process each experiment file
            training_data = []
            processed_count = 0
            
            for exp_file in experiment_files:
                try:
                    # Load experiment metadata
                    exp_path = os.path.join(self.data_dir, exp_file)
                    with open(exp_path, 'r') as f:
                        exp_data = json.load(f)
                    
                    # Skip failed experiments
                    if exp_data.get("status") not in ["completed", "completed_with_warnings"]:
                        continue
                    
                    # Extract experiment parameters
                    experiment_id = exp_data.get("experiment_id", "")
                    service = exp_data.get("target_service", "")
                    replica_count = exp_data.get("replica_count", 0)
                    load_users = exp_data.get("load_users", 0)
                    
                    # Extract scenario from experiment_id
                    scenario = "browsing"  # default
                    if "_browsing_" in experiment_id:
                        scenario = "browsing"
                    elif "_checkout_" in experiment_id:
                        scenario = "checkout"
                    
                    # Load CSV metrics if available
                    metrics_data = {}
                    if "metrics" in exp_data:
                        for key, value in exp_data["metrics"].items():
                            if key.endswith("_csv_path") and os.path.exists(value):
                                try:
                                    csv_data = pd.read_csv(value)
                                    metric_name = key.replace("_csv_path", "")
                                    metrics_data[metric_name] = csv_data
                                except Exception as e:
                                    logger.warning(f"Failed to load CSV {value}: {e}")
                    
                    # Create training rows for each time point
                    if metrics_data:
                        # Get the longest time series as base
                        max_length = max(len(df) for df in metrics_data.values() if isinstance(df, pd.DataFrame))
                        
                        for i in range(max_length):
                            row = {
                                "experiment_id": experiment_id,
                                "service": service,
                                "scenario": scenario,
                                "replica_count": replica_count,
                                "load_users": load_users,
                                "timestamp": f"t_{i}"  # placeholder, will be replaced with actual timestamps
                            }
                            
                            # Add all metric values for this time point
                            for metric_name, metric_df in metrics_data.items():
                                if isinstance(metric_df, pd.DataFrame) and i < len(metric_df):
                                    for col in metric_df.columns:
                                        if col != "timestamp":  # avoid duplicate timestamp columns
                                            row[f"{metric_name}_{col}"] = metric_df.iloc[i][col]
                                else:
                                    # Handle missing data points
                                    for col in metric_df.columns if isinstance(metric_df, pd.DataFrame) else []:
                                        if col != "timestamp":
                                            row[f"{metric_name}_{col}"] = None
                            
                            training_data.append(row)
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process {exp_file}: {e}")
                    continue
            
            if not training_data:
                return {"status": "no_valid_data", "message": "No valid training data found"}
            
            # Create DataFrame
            training_df = pd.DataFrame(training_data)
            
            # Add required regressor columns for Prophet
            if "requests_per_second" not in training_df.columns:
                # Try to estimate from load_users if not available
                training_df["requests_per_second"] = training_df["load_users"] * 2.0  # rough estimate
            
            # Ensure replica_count is available as regressor
            training_df["replica_count"] = training_df["replica_count"].fillna(0)
            
            # Save training dataset
            training_df.to_csv(output_file, index=False)
            
            result = {
                "status": "completed",
                "output_file": output_file,
                "experiments_processed": processed_count,
                "total_rows": len(training_df),
                "columns": training_df.columns.tolist(),
                "shape": training_df.shape
            }
            
            logger.info(f"Successfully processed training data: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process training data: {e}")
            return {"status": "failed", "error": str(e)}
