"""
Load generation and data acquisition for MOrA training
"""
import logging
import time
import subprocess
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

from ..k8s.client import KubernetesClient
from ..monitoring.prometheus_client import PrometheusClient

logger = logging.getLogger(__name__)


class LoadGenerator:
    """
    Manages load generation using JMeter and coordinates data collection.
    """

    def __init__(
        self,
        namespace: str = "hipster-shop",
        prometheus_url: str = "http://localhost:9090",
        k8s_client: Optional[KubernetesClient] = None,
        prom_client: Optional[PrometheusClient] = None
    ):
        self.namespace = namespace
        self.prometheus_url = prometheus_url
        self.k8s_client = k8s_client or KubernetesClient()
        self.prom_client = prom_client or PrometheusClient(prometheus_url)
        
        # JMeter configuration
        self.jmeter_scripts_dir = "jmeter_scripts"
        os.makedirs(self.jmeter_scripts_dir, exist_ok=True)
        
        logger.info(f"LoadGenerator initialized for namespace: {namespace}")

    def create_jmeter_script(
        self, 
        script_name: str, 
        target_host: str, 
        target_port: int = 80,
        test_scenario: str = "browsing",
        num_users: int = None
    ) -> str:
        """
        Create JMeter test script for different user scenarios.
        """
        script_path = os.path.join(self.jmeter_scripts_dir, f"{script_name}.jmx")
        
        # Define test scenarios
        scenarios = {
            "browsing": {
                "threads": 25,
                "ramp_time": 30,
                "duration": 900,  # 15 minutes
                "requests": [
                    {"path": "/", "weight": 40},
                    {"path": "/product", "weight": 30},
                    {"path": "/category", "weight": 20},
                    {"path": "/search", "weight": 10}
                ]
            },
            "checkout": {
                "threads": 25,
                "ramp_time": 30,
                "duration": 900,  # 15 minutes
                "requests": [
                    {"path": "/", "weight": 20},
                    {"path": "/cart", "weight": 30},
                    {"path": "/checkout", "weight": 40},
                    {"path": "/order", "weight": 10}
                ]
            },
            "registration": {
                "threads": 10,
                "ramp_time": 30,
                "duration": 600,  # 10 minutes
                "requests": [
                    {"path": "/", "weight": 30},
                    {"path": "/register", "weight": 40},
                    {"path": "/login", "weight": 30}
                ]
            },
            "search": {
                "threads": 30,
                "ramp_time": 45,
                "duration": 1200,  # 20 minutes
                "requests": [
                    {"path": "/", "weight": 20},
                    {"path": "/search?q=electronics", "weight": 40},
                    {"path": "/search?q=clothing", "weight": 30},
                    {"path": "/search?q=books", "weight": 10}
                ]
            }
        }
        
        if test_scenario not in scenarios:
            raise ValueError(f"Unknown test scenario: {test_scenario}")
        
        scenario = scenarios[test_scenario].copy()
        
        # Override threads with num_users if provided
        if num_users is not None:
            scenario['threads'] = num_users
        
        # Generate JMeter XML script
        jmeter_script = self._generate_jmeter_xml(
            script_name, target_host, target_port, scenario
        )
        
        with open(script_path, 'w') as f:
            f.write(jmeter_script)
        
        logger.info(f"Created JMeter script: {script_path}")
        return script_path

    def _generate_jmeter_xml(
        self, 
        script_name: str, 
        target_host: str, 
        target_port: int, 
        scenario: Dict[str, Any]
    ) -> str:
        """Generate JMeter test plan XML."""
        
        # Base XML template
        xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.4.1">
  <hashTree>
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="{script_name}" enabled="true">
      <stringProp name="TestPlan.comments">MOrA load test for {script_name}</stringProp>
      <boolProp name="TestPlan.functional_mode">false</boolProp>
      <boolProp name="TestPlan.tearDown_on_shutdown">true</boolProp>
      <boolProp name="TestPlan.serialize_threadgroups">false</boolProp>
      <elementProp name="TestPlan.arguments" elementType="Arguments" guiclass="ArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
        <collectionProp name="Arguments.arguments"/>
      </elementProp>
      <stringProp name="TestPlan.user_define_classpath"></stringProp>
    </TestPlan>
    <hashTree>
      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Thread Group" enabled="true">
        <stringProp name="ThreadGroup.on_sample_error">continue</stringProp>
        <elementProp name="ThreadGroup.main_controller" elementType="LoopController" guiclass="LoopControllerGui" testclass="LoopController" testname="Loop Controller" enabled="true">
          <boolProp name="LoopController.continue_forever">false</boolProp>
          <intProp name="LoopController.loops">1</intProp>
        </elementProp>
        <stringProp name="ThreadGroup.num_threads">{scenario['threads']}</stringProp>
        <stringProp name="ThreadGroup.ramp_time">{scenario['ramp_time']}</stringProp>
        <boolProp name="ThreadGroup.scheduler">true</boolProp>
        <stringProp name="ThreadGroup.duration">${{__P(duration,{scenario['duration']})}}</stringProp>
        <stringProp name="ThreadGroup.delay"></stringProp>
        <boolProp name="ThreadGroup.same_user_on_next_iteration">true</boolProp>
      </ThreadGroup>
      <hashTree>"""
        
        # Add HTTP requests for each scenario
        for request in scenario['requests']:
            xml_template += f"""
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="HTTP Request {request['path']}" enabled="true">
          <elementProp name="HTTPsampler.Arguments" elementType="Arguments" guiclass="HTTPArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
            <collectionProp name="Arguments.arguments"/>
          </elementProp>
          <stringProp name="HTTPSampler.domain">{target_host}</stringProp>
          <stringProp name="HTTPSampler.port">{target_port}</stringProp>
          <stringProp name="HTTPSampler.protocol">http</stringProp>
          <stringProp name="HTTPSampler.contentEncoding"></stringProp>
          <stringProp name="HTTPSampler.path">{request['path']}</stringProp>
          <stringProp name="HTTPSampler.method">GET</stringProp>
          <boolProp name="HTTPSampler.follow_redirects">true</boolProp>
          <boolProp name="HTTPSampler.auto_redirects">false</boolProp>
          <boolProp name="HTTPSampler.use_keepalive">true</boolProp>
          <boolProp name="HTTPSampler.DO_MULTIPART_POST">false</boolProp>
          <stringProp name="HTTPSampler.embedded_url_re"></stringProp>
          <stringProp name="HTTPSampler.connect_timeout"></stringProp>
          <stringProp name="HTTPSampler.response_timeout"></stringProp>
        </HTTPSamplerProxy>
        <hashTree/>"""
        
        xml_template += """
      </hashTree>
      <ConfigTestElement guiclass="HttpDefaultsGui" testclass="ConfigTestElement" testname="HTTP Request Defaults" enabled="true">
        <elementProp name="HTTPsampler.Arguments" elementType="Arguments" guiclass="HTTPArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
          <collectionProp name="Arguments.arguments"/>
        </elementProp>
        <stringProp name="HTTPSampler.domain"></stringProp>
        <stringProp name="HTTPSampler.port"></stringProp>
        <stringProp name="HTTPSampler.protocol">http</stringProp>
        <stringProp name="HTTPSampler.contentEncoding"></stringProp>
        <stringProp name="HTTPSampler.path"></stringProp>
        <stringProp name="HTTPSampler.concurrentPool">6</stringProp>
        <stringProp name="HTTPSampler.connect_timeout"></stringProp>
        <stringProp name="HTTPSampler.response_timeout"></stringProp>
      </ConfigTestElement>
    </hashTree>
  </hashTree>
</jmeterTestPlan>"""
        
        return xml_template

    def run_load_test(
        self, 
        script_path: str, 
        target_host: str = "localhost",
        target_port: int = 8080,
        duration_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run JMeter load test and return results.
        """
        try:
            # Check if JMeter is available
            jmeter_cmd = "jmeter"
            jmeter_paths = ["jmeter", "/opt/jmeter/bin/jmeter", "/usr/local/bin/jmeter", "/usr/bin/jmeter"]
            
            for path in jmeter_paths:
                try:
                    # Add the JMeter bin directory to PATH temporarily for subprocess calls
                    env = os.environ.copy()
                    if path != "jmeter":
                        jmeter_dir = os.path.dirname(path)
                        env["PATH"] = jmeter_dir + ":" + env.get("PATH", "")
                    
                    subprocess.run([path, "--version"], capture_output=True, check=True, env=env)
                    jmeter_cmd = path
                    logger.info(f"Found JMeter at: {path}")
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            else:
                logger.error("JMeter not found. Please install Apache JMeter.")
                raise RuntimeError("JMeter is required but not installed")
            
            # Generate results file
            results_dir = tempfile.mkdtemp(prefix="jmeter_results_")
            results_file = os.path.join(results_dir, "results.jtl")
            
            # Build JMeter command
            cmd = [
                jmeter_cmd,
                "-n",  # non-GUI mode
                "-t", script_path,  # test plan
                "-l", results_file,  # results file
                "-H", target_host,  # proxy host
                "-P", str(target_port),  # proxy port
            ]
            
            if duration_minutes:
                cmd.extend(["-J", f"duration={duration_minutes * 60}"])
            
            logger.info(f"Running JMeter test: {' '.join(cmd)}")
            
            # Run the test with proper environment for JMeter
            start_time = datetime.now()
            env = os.environ.copy()
            if jmeter_cmd != "jmeter":
                jmeter_dir = os.path.dirname(jmeter_cmd)
                env["PATH"] = jmeter_dir + ":" + env.get("PATH", "")
            
            # Run JMeter with strict timeout to prevent infinite loops
            timeout_seconds = (duration_minutes * 60 + 60) if duration_minutes else 900  # Add 1 minute buffer
            logger.info(f"Running JMeter with {timeout_seconds} second timeout")
            
            # Start JMeter process
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
            
            try:
                # Wait for completion with timeout
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                logger.warning(f"JMeter test timed out after {timeout_seconds} seconds, killing process")
                process.kill()
                stdout, stderr = process.communicate()
                result = subprocess.CompletedProcess(cmd, -1, stdout, stderr)
                logger.error("JMeter test was killed due to timeout - possible infinite loop")
            end_time = datetime.now()
            
            if result.returncode != 0:
                logger.error(f"JMeter test failed: {result.stderr}")
                return {"error": f"Test failed: {result.stderr}"}
            
            # Parse results
            test_results = self._parse_jmeter_results(results_file)
            test_results.update({
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "script_path": script_path
            })
            
            logger.info(f"Load test completed successfully: {test_results}")
            return test_results
            
        except Exception as e:
            logger.error(f"Load test execution failed: {e}")
            return {"error": str(e)}

    def _parse_jmeter_results(self, results_file: str) -> Dict[str, Any]:
        """Parse JMeter results file (JTL format)."""
        try:
            if not os.path.exists(results_file):
                return {"error": "Results file not found"}
            
            # Read JTL file (CSV format)
            df = pd.read_csv(results_file, sep=',')
            
            # Calculate metrics
            total_requests = len(df)
            successful_requests = len(df[df['success'] == 'true'])
            failed_requests = total_requests - successful_requests
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "avg_response_time": df['elapsed'].mean() if 'elapsed' in df.columns else 0,
                "max_response_time": df['elapsed'].max() if 'elapsed' in df.columns else 0,
                "min_response_time": df['elapsed'].min() if 'elapsed' in df.columns else 0,
                "throughput": total_requests / (df['timeStamp'].max() - df['timeStamp'].min()) * 1000 if len(df) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to parse JMeter results: {e}")
            return {"error": f"Failed to parse results: {e}"}

    def overprovision_non_target_services(self, target_service: str) -> bool:
        """
        Over-provision all services except the target to isolate its behavior.
        """
        try:
            logger.info(f"Over-provisioning all services except {target_service}")
            
            # Get all deployments in the namespace
            deployments = self.k8s_client.list_deployments(self.namespace)
            
            for deployment in deployments:
                service_name = deployment.metadata.name
                if service_name == target_service:
                    continue  # Skip the target service
                
                # Scale up non-target services
                target_replicas = 3  # Over-provision to 3 replicas
                if deployment.spec.replicas < target_replicas:
                    logger.info(f"Scaling {service_name} to {target_replicas} replicas")
                    self.k8s_client.scale_deployment(service_name, self.namespace, target_replicas)
            
            # Wait for scaling to complete
            time.sleep(30)
            return True
            
        except Exception as e:
            logger.error(f"Failed to over-provision services: {e}")
            return False

    def collect_training_data(
        self,
        target_service: str,
        container_name: str,
        resource_type: str,
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Collect training data for a specific service during load test.
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=duration_minutes)
            
            # Query metrics from Prometheus
            if resource_type == "cpu":
                query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{self.namespace}", pod=~"{target_service}-.*", container="{container_name}"}}[1m])) by (container)'
            elif resource_type == "memory":
                query = f'sum(container_memory_working_set_bytes{{namespace="{self.namespace}", pod=~"{target_service}-.*", container="{container_name}"}}) by (container)'
            else:
                raise ValueError(f"Unsupported resource type: {resource_type}")
            
            metrics_df = self.prom_client.get_metric_range(query, start_time, end_time)
            
            if metrics_df.empty:
                logger.warning(f"No metrics collected for {target_service}/{container_name}/{resource_type}")
                return {"error": "No metrics data available"}
            
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
            logger.error(f"Failed to collect training data: {e}")
            return {"error": str(e)}
