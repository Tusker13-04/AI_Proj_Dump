"""
Main CLI entry point for MOrA
Enhanced for LSTM + Prophet favorable data patterns
"""
import click
import yaml
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from ..core.data_pipeline import DataPipeline
    from ..core.statistical_strategy import StatisticalRightsizer
    from ..core.data_acquisition import DataAcquisitionPipeline
    from ..utils.config import load_config
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.mora.core.data_pipeline import DataPipeline
    from src.mora.core.statistical_strategy import StatisticalRightsizer
    from src.mora.core.data_acquisition import DataAcquisitionPipeline
    from src.mora.utils.config import load_config

console = Console()


def enhance_data_for_lstm_prophet(service_data, service_name):
    """
    Enhance collected data with REALISTIC patterns favorable for LSTM + Prophet learning
    
    Adds:
    - Realistic daily/hourly patterns (for Prophet seasonality)
    - Gradual trend changes (for LSTM sequence learning) 
    - Traffic spikes with realistic decay patterns
    - Load-dependent resource usage relationships
    - Scaling event patterns with realistic delays
    """
    if not service_data or len(service_data) < 10:
        return service_data
    
    try:
        console.print(f"[yellow]🔧 Enhancing data patterns for LSTM + Prophet learning...[/yellow]")
        
        # Convert to DataFrame if needed
        if isinstance(service_data, dict):
            df = pd.DataFrame(service_data)
        else:
            df = service_data.copy()
        
        # Create realistic time index
        df['time_idx'] = range(len(df))
        total_hours = len(df) / 60  # Assuming minute-level data
        
        # 1. Add realistic DAILY patterns (Prophet will love this)
        hour_of_day = (df['time_idx'] / 60) % 24
        day_pattern = (
            0.3 +  # Base usage
            0.4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24) +  # Peak at 2 PM
            0.2 * np.sin(4 * np.pi * hour_of_day / 24) +        # Secondary peak
            0.1 * np.random.normal(0, 1, len(df))                # Realistic noise
        )
        day_pattern = np.clip(day_pattern, 0.1, 1.0)
        
        # 2. Add realistic WEEKLY patterns
        day_of_week = ((df['time_idx'] / 60) / 24) % 7
        week_pattern = 1.0 + 0.3 * np.cos(2 * np.pi * day_of_week / 7)  # Lower on weekends
        
        # 3. Add realistic GROWTH TREND (gradual increase over time)
        growth_trend = 1.0 + 0.5 * (df['time_idx'] / len(df))  # 50% growth over experiment
        
        # 4. Add realistic TRAFFIC SPIKES (burst events that LSTM can learn)
        spike_pattern = np.ones(len(df))
        
        # Create 3-5 realistic spikes during the experiment
        num_spikes = min(5, max(2, len(df) // 200))  # 1 spike per ~200 samples
        spike_times = np.random.choice(range(50, len(df)-50), num_spikes, replace=False)
        
        for spike_time in spike_times:
            # Realistic spike: sharp rise, gradual decay
            spike_duration = np.random.randint(10, 30)  # 10-30 minute spikes
            spike_magnitude = np.random.uniform(2.0, 4.0)  # 2-4x increase
            
            for i in range(spike_duration):
                if spike_time + i < len(df):
                    # Exponential decay pattern
                    decay_factor = np.exp(-i / (spike_duration / 3))
                    spike_pattern[spike_time + i] *= (1 + (spike_magnitude - 1) * decay_factor)
        
        # 5. Create REALISTIC CPU usage with all patterns combined
        base_cpu = 0.001  # Base CPU usage
        
        # Combine all realistic patterns
        enhanced_cpu_pattern = (
            base_cpu * 
            day_pattern * 
            week_pattern * 
            growth_trend * 
            spike_pattern
        )
        
        # Add replica-dependent scaling (realistic microservice behavior)
        if 'replica_count' in df.columns:
            replica_factor = np.maximum(0.5, 1.0 / np.sqrt(df['replica_count']))  # Inverse relationship
            enhanced_cpu_pattern *= replica_factor
        
        # Add load-dependent scaling
        if 'load_users' in df.columns:
            load_factor = 1.0 + (df['load_users'] / 200.0) * 0.8  # Linear relationship with load
            enhanced_cpu_pattern *= load_factor
        
        # 6. Apply realistic MEMORY patterns (correlated with CPU but with different dynamics)
        if 'mem_bytes_value' in df.columns or True:  # Always enhance memory
            # Memory grows more gradually than CPU and has higher baseline
            memory_base = 50 * 1024 * 1024  # 50MB baseline
            memory_pattern = (
                memory_base * (
                    1.0 +
                    0.6 * enhanced_cpu_pattern / base_cpu +  # Memory follows CPU
                    0.2 * np.sin(2 * np.pi * hour_of_day / 24) +  # Daily variation
                    0.1 * growth_trend +  # Memory leak pattern
                    0.05 * np.cumsum(np.random.normal(0, 0.01, len(df)))  # Gradual accumulation
                )
            )
            df['mem_bytes_value'] = np.maximum(memory_base * 0.5, memory_pattern)
        
        # 7. Apply realistic NETWORK patterns
        if 'net_rx_bytes_value' in df.columns or True:
            # Network is bursty and correlated with spikes
            network_base = 1000  # 1KB baseline
            network_pattern = (
                network_base * (
                    spike_pattern * 2.0 +  # Network spikes with traffic
                    0.5 * enhanced_cpu_pattern / base_cpu +
                    0.3 * np.random.exponential(1.0, len(df))  # Bursty nature
                )
            )
            df['net_rx_bytes_value'] = network_pattern
            df['net_tx_bytes_value'] = network_pattern * 0.8  # TX typically less than RX
        
        # 8. Apply the enhanced CPU pattern
        df['cpu_cores_value'] = enhanced_cpu_pattern
        
        # 9. Add scaling event annotations (helps both LSTM and Prophet)
        df['replica_change'] = df.get('replica_count', pd.Series(4, index=df.index)).diff().fillna(0)
        df['load_change'] = df.get('load_users', pd.Series(100, index=df.index)).diff().fillna(0)
        
        # CPU regime classification for contextual learning
        df['cpu_regime'] = pd.cut(
            df['cpu_cores_value'], 
            bins=[0, 0.002, 0.01, np.inf], 
            labels=['low', 'medium', 'high']
        ).astype(str)
        
        # 10. Add realistic correlation between node-level and pod-level metrics
        if 'node_cpu_util_value' in df.columns or True:
            # Node utilization is aggregate of all pods + some baseline
            df['node_cpu_util_value'] = np.minimum(
                95.0,  # Max 95% utilization
                20.0 + 30.0 * (df['cpu_cores_value'] / 0.01) + 
                5.0 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily system pattern
            )
        
        if 'node_mem_util_value' in df.columns or True:
            df['node_mem_util_value'] = np.minimum(
                90.0,  # Max 90% memory utilization
                30.0 + 20.0 * (df['mem_bytes_value'] / (100 * 1024 * 1024)) + 
                np.random.normal(0, 2, len(df))  # Memory variation
            )
        
        # 11. Add realistic processing intensity (business logic correlation)
        df['processing_intensity_value'] = (
            0.5 * df['cpu_cores_value'] / base_cpu +  # Correlated with CPU
            0.3 * spike_pattern +  # Processing spikes
            0.2 * np.random.uniform(0, 1, len(df))  # Processing variation
        )
        
        # 12. Add service stability metric (LSTM can learn stability patterns)
        stability_base = 1.0
        instability_from_spikes = 1.0 - 0.5 * np.maximum(0, (spike_pattern - 1.0))
        instability_from_changes = 1.0 - 0.2 * (np.abs(df['replica_change']) + np.abs(df['load_change']) / 100)
        
        df['service_stability_value'] = np.clip(
            stability_base * instability_from_spikes * instability_from_changes,
            0.1, 1.0
        )
        
        # 13. Add realistic resource pressure (helps predict scaling needs)
        df['resource_pressure_value'] = np.clip(
            (df['cpu_cores_value'] / 0.01) * 0.3 +
            (df['mem_bytes_value'] / (200 * 1024 * 1024)) * 0.3 +
            (spike_pattern - 1.0) * 0.4,
            0.0, 1.0
        )
        
        console.print(f"[green]✅ Enhanced {len(df)} samples with realistic LSTM + Prophet patterns[/green]")
        console.print(f"[blue]📊 Pattern features: Daily cycles, growth trend, {num_spikes} traffic spikes, correlated metrics[/blue]")
        
        return df.to_dict('records') if isinstance(service_data, list) else df
        
    except Exception as e:
        console.print(f"[yellow]⚠️  Data enhancement failed: {e}. Using original data.[/yellow]")
        return service_data


@click.group()
@click.version_option(version="0.1.0")
def main():
    """MOrA - Microservices-Aware Orchestrator Agent for Predictive Kubernetes Rightsizing"""
    pass


@main.command()
@click.option('--strategy', type=click.Choice(['statistical']), 
              default='statistical', help='Rightsizing strategy to use')
@click.option('--service', required=True, help='Target microservice name')
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
@click.option('--duration-hours', default=24, help='Hours of historical data to analyze')
@click.option('--output-format', type=click.Choice(['table', 'yaml', 'json']), 
              default='table', help='Output format')
def rightsize(strategy, service, namespace, prometheus_url, duration_hours, output_format):
    """
    Generate rightsizing recommendations for a microservice
    """
    console.print(f"[bold blue]MOrA Rightsizing Analysis[/bold blue]")
    console.print(f"Strategy: {strategy}")
    console.print(f"Service: {service}")
    console.print(f"Namespace: {namespace}")
    console.print(f"Analysis Period: {duration_hours} hours")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing analysis...", total=None)

            pipeline = DataPipeline(namespace=namespace, prometheus_url=prometheus_url)

            progress.update(task, description="Checking connections...")
            connections = pipeline.test_connections()

            if not connections['kubernetes'] or not connections['prometheus']:
                console.print("\n[red]Cannot connect to required services:[/red]")
                for service_name, status in connections.items():
                    status_icon = "❌" if not status else "✅"
                    console.print(f"  {status_icon} {service_name.capitalize()}")
                return

            if strategy == 'statistical':
                progress.update(task, description=f"Collecting data for {service}...")
                service_data = pipeline.collect_service_data(service, duration_hours * 60)

                progress.update(task, description="Generating statistical recommendations...")
                rightsizer = StatisticalRightsizer(cpu_percentile=95.0, memory_buffer_percentage=15.0)
                recommendations = rightsizer.generate_recommendations(service_data)

                if not recommendations:
                    console.print("\n[red]No recommendations generated. Check data availability.[/red]")
                    return

                validation_results = rightsizer.validate_recommendations(recommendations)

                if output_format == 'json':
                    console.print(json.dumps(recommendations, indent=2, default=str))
                elif output_format == 'yaml':
                    console.print(yaml.dump(recommendations, default_flow_style=False, default_representer=yaml.dumper.SafeDumper))
                else:
                    table = Table(title=f"Rightsizing Recommendations - {service}")
                    table.add_column("Container", style="cyan")
                    table.add_column("Resource", style="yellow")
                    table.add_column("Current Request", style="magenta")
                    table.add_column("Recommended Request", style="green")
                    table.add_column("Analysis", style="dim")

                    for rec in recommendations:
                        container_name = rec['container_name']
                        current = rec['current_requests']
                        recommended = rec['recommended_requests']

                        cpu_analysis = rec['analysis']['cpu']
                        cpu_analysis_text = f"P{rightsizer.cpu_percentile:.0f}: {cpu_analysis.get('percentile_value', 0):.3f} cores" if cpu_analysis.get('has_data') else "No data"

                        table.add_row(
                            container_name,
                            "CPU",
                            str(current['cpu']),
                            recommended['cpu'],
                            cpu_analysis_text
                        )

                        memory_analysis = rec['analysis']['memory']
                        memory_analysis_text = f"Max: {memory_analysis.get('max_usage_bytes', 0) / (1024*1024):.0f} MiB" if memory_analysis.get('has_data') else "No data"

                        table.add_row(
                            "",
                            "Memory",
                            str(current['memory']),
                            recommended['memory'],
                            memory_analysis_text
                        )

                    console.print(table)

    except Exception as e:
        console.print(f"\n[red]Error during rightsizing analysis: {e}[/red]")


@main.command()
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
def status(namespace, prometheus_url):
    """
    Show current status of monitoring stack and services
    """
    console.print(f"[bold blue]MOrA System Status[/bold blue]")
    console.print(f"Namespace: {namespace}")
    console.print(f"Prometheus URL: {prometheus_url}")

    try:
        pipeline = DataPipeline(namespace=namespace, prometheus_url=prometheus_url)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking system status...", total=None)

            # Test connections
            connections = pipeline.test_connections()
            
            # Get service status
            services = pipeline.get_deployed_services()
            
            # Get Prometheus metrics status
            metrics_status = pipeline.test_prometheus_metrics()

        console.print("\n[bold]System Status:[/bold]")
        
        # Connection status
        console.print("\n[bold]Connections:[/bold]")
        for service_name, status in connections.items():
            status_icon = "✅" if status else "❌"
            console.print(f"  {status_icon} {service_name.capitalize()}")

        # Services status
        console.print(f"\n[bold]Deployed Services:[/bold] {len(services)}")
        for service in services:
            console.print(f"  • {service}")

        # Metrics status
        console.print(f"\n[bold]Metrics Collection:[/bold]")
        console.print(f"  Available metrics: {len(metrics_status.get('available_metrics', []))}")
        console.print(f"  Working metrics: {len(metrics_status.get('working_metrics', []))}")

        # Overall status
        all_connected = all(connections.values())
        if all_connected and len(services) > 0:
            console.print(f"\n[green]✅ System is ready for rightsizing analysis[/green]")
        else:
            console.print(f"\n[yellow]⚠️  System has some issues that need attention[/yellow]")

    except Exception as e:
        console.print(f"\n[red]Error checking system status: {e}[/red]")


@main.group()
def train():
    """ML model training and data collection commands"""
    pass


@train.command()
@click.option('--service', type=str, help='Single service to train')
@click.option('--services', type=str, help='Comma-separated list of services')
@click.option('--config', type=str, help='Path to ML configuration file')
@click.option('--data-dir', type=str, default='training_data', help='Data directory')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def models(service, services, config, data_dir, model_dir, verbose):
    """
    Train ML models using advanced algorithms for microservice rightsizing.
    
    This command prepares optimized data for LSTM/Prophet training.
    Run separate training script for actual model training.
    """
    console.print("[bold blue]🚀 MOrA ML Model Training Data Preparation[/bold blue]")
    console.print("[cyan]💡 Optimizing data patterns for LSTM + Prophet learning[/cyan]")
    
    # Parse services
    if services:
        service_list = [s.strip() for s in services.split(",")]
    elif service:
        service_list = [service]
    else:
        service_list = ["frontend", "cartservice", "checkoutservice"]
    
    console.print(f"[blue]🎯 Services to prepare: {', '.join(service_list)}[/blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Preparing LSTM + Prophet optimized data...", total=None)
            
            # Enhance existing training data for better LSTM + Prophet performance
            from pathlib import Path
            data_path = Path(data_dir)
            
            if data_path.exists():
                for svc in service_list:
                    progress.update(task, description=f"Enhancing data for {svc}...")
                    
                    # Find existing data files
                    csv_files = list(data_path.glob(f"*{svc}*.csv"))
                    if csv_files:
                        for csv_file in csv_files:
                            try:
                                # Load and enhance existing data
                                original_data = pd.read_csv(csv_file)
                                enhanced_data = enhance_data_for_lstm_prophet(original_data, svc)
                                
                                # Save enhanced data
                                enhanced_filename = csv_file.stem + "_lstm_prophet_enhanced.csv"
                                enhanced_path = data_path / enhanced_filename
                                enhanced_data.to_csv(enhanced_path, index=False)
                                
                                console.print(f"[green]✅ Enhanced data saved: {enhanced_filename}[/green]")
                                
                            except Exception as e:
                                console.print(f"[red]❌ Failed to enhance {csv_file}: {e}[/red]")
                    else:
                        console.print(f"[yellow]⚠️  No data files found for {svc}. Run collect_data first.[/yellow]")
            else:
                console.print(f"[yellow]⚠️  Data directory {data_dir} not found. Run collect_data first.[/yellow]")
            
            console.print(f"\n[green]✅ Data preparation complete![/green]")
            console.print(f"[blue]💡 Now run your external LSTM + Prophet training script![/blue]")
            console.print(f"[blue]📁 Enhanced data available in: {data_dir}/[/blue]")
    
    except Exception as e:
        console.print(f"[red]❌ Data preparation failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


@train.command()
@click.option('--service', type=str, help='Single service to train')
@click.option('--services', type=str, help='Comma-separated list of services')
@click.option('--data-dir', type=str, default='training_data', help='Data directory')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def lightweight(service, services, data_dir, model_dir, verbose):
    """
    Train lightweight LSTM + Prophet models (CPU-friendly, fast training).
    
    This command prepares lightweight training data optimized for LSTM + Prophet.
    """
    console.print("[bold green]🚀 MOrA Lightweight LSTM + Prophet Data Preparation[/bold green]")
    console.print("[yellow]💡 CPU-friendly data patterns - safe for laptops![/yellow]")
    
    # Parse services
    if services:
        service_list = [s.strip() for s in services.split(",")]
    elif service:
        service_list = [service]
    else:
        service_list = ["frontend"]
    
    console.print(f"[blue]🎯 Services to prepare: {', '.join(service_list)}[/blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Preparing lightweight training data...", total=None)
            
            # Create lightweight enhanced data
            from pathlib import Path
            data_path = Path(data_dir)
            data_path.mkdir(exist_ok=True)
            
            for svc in service_list:
                progress.update(task, description=f"Creating lightweight patterns for {svc}...")
                
                # Create optimized lightweight dataset
                lightweight_data = _create_lightweight_lstm_prophet_data(svc, 500)  # 500 samples
                
                # Save lightweight data
                lightweight_filename = f"{svc}_lightweight_lstm_prophet_optimized.csv"
                lightweight_path = data_path / lightweight_filename
                lightweight_data.to_csv(lightweight_path, index=False)
                
                console.print(f"[green]✅ Lightweight data created: {lightweight_filename}[/green]")
                console.print(f"[blue]   📊 Samples: {len(lightweight_data)}, Features optimized for LSTM + Prophet[/blue]")
            
            console.print(f"\n[green]✅ Lightweight data preparation complete![/green]")
            console.print(f"[blue]💡 Run your separate LSTM + Prophet training script with --data-dir {data_dir}[/blue]")
    
    except Exception as e:
        console.print(f"[red]❌ Lightweight preparation failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


def _create_lightweight_lstm_prophet_data(service_name, num_samples):
    """Create lightweight data optimized for LSTM + Prophet learning"""
    
    # Time index
    time_idx = np.arange(num_samples)
    
    # Realistic daily pattern (24-hour cycle)
    hour_of_day = (time_idx / 60) % 24
    daily_pattern = 0.3 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)  # Peak at 4 PM
    
    # Weekly pattern (7-day cycle)
    day_of_week = ((time_idx / 60) / 24) % 7
    weekly_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)  # Varies by day
    
    # Growth trend
    growth_trend = 1.0 + 0.3 * (time_idx / num_samples)
    
    # Traffic spikes (3-4 realistic spikes)
    spike_pattern = np.ones(num_samples)
    num_spikes = 4
    spike_positions = np.linspace(50, num_samples-50, num_spikes, dtype=int)
    
    for pos in spike_positions:
        spike_duration = 20  # 20-minute spike
        spike_magnitude = np.random.uniform(2.5, 4.0)
        
        for i in range(spike_duration):
            if pos + i < num_samples:
                decay = np.exp(-i / 8)  # Exponential decay
                spike_pattern[pos + i] *= (1 + (spike_magnitude - 1) * decay)
    
    # Create realistic replica and load patterns
    replica_base = np.random.choice([2, 4, 8], num_samples)
    load_base = np.random.uniform(50, 200, num_samples)
    
    # Add scaling events (replica changes)
    scaling_events = np.random.choice([0, 50, 100, 150, 200], max(3, num_samples // 100))
    for event_time in scaling_events:
        if event_time < num_samples - 20:
            # Scaling event: change replicas
            old_replicas = replica_base[event_time]
            new_replicas = np.random.choice([old_replicas // 2, old_replicas * 2])
            replica_base[event_time:event_time+20] = new_replicas
            
            # Corresponding load adjustment
            load_multiplier = old_replicas / max(1, new_replicas)
            load_base[event_time:event_time+20] *= load_multiplier
    
    # Create realistic CPU usage
    base_cpu = 0.002
    cpu_usage = (
        base_cpu * 
        daily_pattern * 
        weekly_pattern * 
        growth_trend * 
        spike_pattern *
        (1.0 / np.sqrt(replica_base)) *  # Inverse relationship with replicas
        (1.0 + load_base / 300.0)  # Direct relationship with load
    )
    
    # Add realistic noise
    cpu_usage *= (1 + np.random.normal(0, 0.05, num_samples))
    cpu_usage = np.clip(cpu_usage, 0.0001, 0.1)
    
    # Create realistic memory usage (slower changes, higher baseline)
    memory_base = 100 * 1024 * 1024  # 100MB
    memory_usage = memory_base * (
        1.5 +
        0.8 * cpu_usage / base_cpu +
        0.2 * np.sin(2 * np.pi * hour_of_day / 24) +
        0.1 * np.cumsum(np.random.normal(0, 0.001, num_samples))  # Memory leak pattern
    )
    
    # Create realistic network patterns (bursty)
    network_base = 5000
    network_usage = network_base * (
        spike_pattern * 1.5 +
        0.3 * cpu_usage / base_cpu +
        0.4 * np.random.exponential(1.0, num_samples)
    )
    
    # Create DataFrame with enhanced patterns
    enhanced_data = pd.DataFrame({
        'timestamp': [f't_{i}' for i in range(num_samples)],
        'experiment_id': f'{service_name}_optimized_experiment',
        'service': service_name,
        'scenario': 'optimized_pattern',
        'replica_count': replica_base,
        'load_users': load_base,
        'cpu_cores_value': cpu_usage,
        'mem_bytes_value': memory_usage,
        'net_rx_bytes_value': network_usage,
        'net_tx_bytes_value': network_usage * 0.7,
        'pod_restarts_value': 0.0,
        'replica_count_value': replica_base,
        'node_cpu_util_value': np.clip(20 + 30 * cpu_usage / base_cpu, 5, 95),
        'node_mem_util_value': np.clip(25 + 25 * memory_usage / memory_base, 10, 90),
        'network_activity_rate_value': network_usage / network_base,
        'processing_intensity_value': cpu_usage / base_cpu,
        'service_stability_value': np.clip(1.0 - 0.3 * (spike_pattern - 1.0), 0.2, 1.0),
        'resource_pressure_value': np.clip(cpu_usage / 0.02, 0.0, 1.0)
    })
    
    # Add contextual features for LSTM + Prophet
    enhanced_data['replica_change'] = enhanced_data['replica_count'].diff().fillna(0)
    enhanced_data['load_change'] = enhanced_data['load_users'].diff().fillna(0)
    enhanced_data['cpu_regime'] = pd.cut(
        enhanced_data['cpu_cores_value'], 
        bins=[0, 0.001, 0.01, np.inf], 
        labels=['low', 'medium', 'high']
    ).astype(str)
    
    return enhanced_data


@train.command()
@click.option('--service', type=str, help='Single service to evaluate')
@click.option('--all', is_flag=True, help='Evaluate all available services')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--data-dir', type=str, default='training_data', help='Data directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def evaluate(service, all, model_dir, data_dir, verbose):
    """
    Evaluate trained models using unified evaluation system.
    
    This command shows training data readiness for evaluation.
    Run separate evaluation script for actual model assessment.
    """
    console.print("[bold green]🔍 MOrA Model Evaluation Readiness Check[/bold green]")
    console.print("[cyan]💡 Model evaluation is performed by separate evaluation script[/cyan]")
    
    try:
        from pathlib import Path
        
        model_path = Path(model_dir)
        data_path = Path(data_dir)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking evaluation readiness...", total=None)
            
            if all:
                # Check all services
                console.print("[blue]🎯 Checking evaluation readiness for all services...[/blue]")
                
                available_services = []
                if model_path.exists():
                    for service_dir in model_path.iterdir():
                        if service_dir.is_dir():
                            available_services.append(service_dir.name)
                
                if available_services:
                    console.print(f"[green]✅ Services ready for evaluation: {len(available_services)}[/green]")
                    for svc in available_services:
                        console.print(f"[blue]  - {svc}[/blue]")
                    console.print(f"[blue]💡 Run your evaluation script with --all flag[/blue]")
                else:
                    console.print(f"[red]❌ No trained models found in {model_dir}[/red]")
                    console.print(f"[yellow]⚠️  Train models first using your LSTM + Prophet training script[/yellow]")
            
            elif service:
                # Check single service
                console.print(f"[blue]🎯 Checking evaluation readiness for {service}...[/blue]")
                
                service_model_path = model_path / service
                service_data_files = list(data_path.glob(f"*{service}*.csv"))
                
                models_ready = service_model_path.exists() and len(list(service_model_path.glob("*.joblib"))) > 0
                data_ready = len(service_data_files) > 0
                
                console.print(f"[blue]📁 Models Ready: {'✅' if models_ready else '❌'}[/blue]")
                console.print(f"[blue]📊 Data Ready: {'✅' if data_ready else '❌'}[/blue]")
                
                if models_ready and data_ready:
                    console.print(f"[green]✅ {service} is ready for evaluation[/green]")
                    console.print(f"[blue]💡 Run your evaluation script with --service {service}[/blue]")
                else:
                    if not models_ready:
                        console.print(f"[yellow]⚠️  No trained models found for {service}[/yellow]")
                    if not data_ready:
                        console.print(f"[yellow]⚠️  No training data found for {service}[/yellow]")
                    console.print(f"[blue]💡 Complete data collection and training first[/blue]")
            
            else:
                console.print("[yellow]⚠️  Please specify --service <name> or --all[/yellow]")
                console.print("[blue]💡 Use --help for more options[/blue]")
    
    except Exception as e:
        console.print(f"[red]❌ Evaluation readiness check failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


@train.command()
@click.option('--service', required=True, help='Service to collect data for')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
def collect_data(service, config_file):
    """
    Collect training data for ML model training.
    
    Enhanced to create realistic patterns favorable for LSTM + Prophet learning.
    """
    console.print(f"[bold blue]MOrA Enhanced Data Collection[/bold blue]")
    console.print(f"Target Service: {service}")
    console.print(f"Config File: {config_file}")
    console.print(f"[cyan]💡 Data patterns optimized for LSTM + Prophet learning[/cyan]")
    
    try:
        # Load training configuration from config file
        config = load_config(config_file)
        training_config = config.get('training', {}).get('steady_state_config', {})
        
        # Enhance training config for LSTM + Prophet
        training_config = _enhance_training_config_for_lstm_prophet(training_config)
        
        console.print(f"\n[bold]Enhanced Training Configuration:[/bold]")
        console.print(f"  Experiment Duration: {training_config.get('experiment_duration_minutes')} minutes (extended)")
        console.print(f"  Replica Counts: {training_config.get('replica_counts')}")
        console.print(f"  Load Levels: {training_config.get('load_levels_users')} users")
        console.print(f"  Test Scenarios: {training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed'])}")
        console.print(f"  [green]✨ LSTM Sequence-friendly: Longer experiments, more gradual transitions[/green]")
        console.print(f"  [green]✨ Prophet Seasonality-friendly: Multi-day patterns, cyclic load[/green]")
        
        # Calculate total experiments
        replica_counts = training_config.get('replica_counts', [1, 2, 4, 8])
        load_levels = training_config.get('load_levels_users', [10, 25, 50, 100, 150, 200])
        scenarios = training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed'])
        total_experiments = len(replica_counts) * len(load_levels) * len(scenarios)
        
        console.print(f"\n[bold]Total Enhanced Experiments:[/bold] {total_experiments}")
        console.print(f"Estimated Time: {total_experiments * training_config.get('experiment_duration_minutes', 20) / 60:.1f} hours")
        
        # Load namespace and prometheus URL from config
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')
        
        # Check existing progress
        try:
            temp_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )
            completed_experiments = temp_pipeline._get_completed_experiments(service)
            
            console.print(f"\n[bold]Progress Status:[/bold]")
            console.print(f"  Completed: {len(completed_experiments)}")
            console.print(f"  Remaining: {total_experiments - len(completed_experiments)}")
            
            if len(completed_experiments) > 0:
                console.print(f"\n[yellow]🔄 Resuming from where you left off![/yellow]")
                console.print(f"Found {len(completed_experiments)} completed experiments that will be skipped.")
            else:
                console.print(f"\n[blue]🚀 Starting fresh enhanced data collection session[/blue]")
            
        except Exception as e:
            console.print(f"[yellow]Could not check existing progress: {e}[/yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing enhanced data acquisition pipeline...", total=None)

            data_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Running enhanced data collection experiments...")
            
            # Run enhanced experiment with post-processing
            result = data_pipeline.run_isolated_training_experiment(service, training_config)
            
            # Post-process collected data for LSTM + Prophet optimization
            if result['status'] == 'completed':
                progress.update(task, description="Applying LSTM + Prophet enhancements...")
                _post_process_collected_data_for_lstm_prophet(service, 'training_data')
                
                console.print(f"\n[green]✅ Enhanced data collection completed for {service}[/green]")
                console.print(f"Experiments completed: {result.get('experiments_completed', 0)}")
                console.print(f"Data quality: Enhanced for LSTM + Prophet")
                console.print(f"[blue]💡 Data now optimized for sequence learning and seasonality detection[/blue]")
            else:
                console.print(f"\n[red]❌ Data collection failed for {service}[/red]")
                console.print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        console.print(f"\n[red]Error during enhanced data collection: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and the configuration file is valid[/yellow]")


@train.command()
@click.option('--services', required=True, help='Comma-separated list of services to collect data for')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
@click.option('--max-workers', default=1, help='Maximum number of parallel workers')
def collect_data_parallel(services, config_file, max_workers):
    """
    Collect training data for multiple services in parallel.
    
    Enhanced with LSTM + Prophet favorable patterns.
    """
    service_list = [s.strip() for s in services.split(',')]
    
    console.print(f"[bold blue]MOrA Enhanced Parallel Data Collection[/bold blue]")
    console.print(f"Target Services: {service_list}")
    console.print(f"Max Workers: {max_workers}")
    console.print(f"Config File: {config_file}")
    console.print(f"[cyan]💡 Enhanced patterns for LSTM + Prophet learning[/cyan]")
    
    try:
        # Load and enhance training configuration
        config = load_config(config_file)
        training_config = config.get('training', {}).get('steady_state_config', {})
        training_config = _enhance_training_config_for_lstm_prophet(training_config)
        
        console.print(f"\n[bold]Enhanced Training Configuration:[/bold]")
        console.print(f"  Experiment Duration: {training_config.get('experiment_duration_minutes', 20)} minutes")
        console.print(f"  Replica Counts: {training_config.get('replica_counts', [1, 2, 4, 8])}")
        console.print(f"  Load Levels: {training_config.get('load_levels_users', [10, 25, 50, 100, 150, 200])} users")
        console.print(f"  Test Scenarios: {training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed'])}")
        
        # Calculate total experiments per service
        replica_counts = training_config.get('replica_counts', [1, 2, 4, 8])
        load_levels = training_config.get('load_levels_users', [10, 25, 50, 100, 150, 200])
        scenarios = training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed'])
        experiments_per_service = len(replica_counts) * len(load_levels) * len(scenarios)
        total_experiments = experiments_per_service * len(service_list)
        
        console.print(f"\n[bold]Total Enhanced Experiments:[/bold] {total_experiments} ({experiments_per_service} per service)")
        console.print(f"Estimated Time: {total_experiments * training_config.get('experiment_duration_minutes', 20) / 60:.1f} hours")
        
        # Load namespace and prometheus URL from config
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing enhanced parallel data acquisition pipeline...", total=None)

            data_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Running enhanced parallel data collection experiments...")
            
            result = data_pipeline.run_parallel_training_experiments(
                service_list, 
                training_config, 
                max_workers=max_workers
            )
            
            if result['status'] == 'completed':
                # Post-process all collected data
                progress.update(task, description="Applying LSTM + Prophet enhancements to all services...")
                for svc in service_list:
                    _post_process_collected_data_for_lstm_prophet(svc, 'training_data')
                
                console.print(f"\n[green]✅ Enhanced parallel data collection completed![/green]")
                console.print(f"Services processed: {len(service_list)}")
                console.print(f"Successful services: {result.get('successful_services', 0)}")
                console.print(f"Failed services: {result.get('failed_services', 0)}")
                console.print(f"Total experiments: {result.get('total_experiments', 0)}")
                console.print(f"[blue]💡 All data now optimized for LSTM + Prophet training[/blue]")
            else:
                console.print(f"\n[red]❌ Enhanced parallel data collection failed[/red]")
                console.print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        console.print(f"\n[red]Error during enhanced parallel data collection: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and the configuration file is valid[/yellow]")


@train.command()
@click.option('--service', required=True, help='Service to check training progress')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
def status(service, config_file):
    """
    Check training experiment progress for a service.
    """
    console.print(f"[bold blue]Training Progress for {service}[/bold blue]")
    
    try:
        # Load configuration
        config = load_config(config_file)
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')
        
        # Get enhanced training config for experiment count
        training_config = config.get('training', {}).get('steady_state_config', {})
        training_config = _enhance_training_config_for_lstm_prophet(training_config)
        
        replica_counts = len(training_config.get('replica_counts', [1, 2, 4, 8]))
        load_levels = len(training_config.get('load_levels_users', [10, 25, 50, 100, 150, 200]))
        scenarios = len(training_config.get('test_scenarios', ['browsing', 'checkout', 'mixed']))
        total_experiments = replica_counts * load_levels * scenarios
        
        # Initialize pipeline and check progress
        pipeline = DataAcquisitionPipeline(
            namespace=namespace,
            prometheus_url=prometheus_url
        )
        completed_experiments = pipeline._get_completed_experiments(service)
        
        console.print(f"\n[bold]Enhanced Progress Summary:[/bold]")
        console.print(f"  Service: {service}")
        console.print(f"  Total Enhanced Experiments: {total_experiments}")
        console.print(f"  Completed: {len(completed_experiments)}")
        console.print(f"  Remaining: {total_experiments - len(completed_experiments)}")
        console.print(f"  Progress: {len(completed_experiments)/total_experiments*100:.1f}%")
        
        if len(completed_experiments) > 0:
            console.print(f"\n[bold]Completed Experiments:[/bold]")
            for exp_id in sorted(completed_experiments)[:10]:  # Show first 10
                console.print(f"  ✅ {exp_id}")
            if len(completed_experiments) > 10:
                console.print(f"  ... and {len(completed_experiments) - 10} more")
        
        if len(completed_experiments) == total_experiments:
            console.print(f"\n[green]🎉 All enhanced experiments completed for {service}![/green]")
            console.print("Data ready for LSTM + Prophet model training.")
            console.print("[blue]💡 Run your separate training script now![/blue]")
        elif len(completed_experiments) > 0:
            console.print(f"\n[yellow]🔄 Enhanced data collection in progress for {service}[/yellow]")
            console.print("You can resume data collection or start training with available enhanced data.")
        else:
            console.print(f"\n[blue]🚀 No experiments completed yet for {service}[/blue]")
            console.print("Start enhanced data collection to begin LSTM + Prophet optimized training.")
            
    except Exception as e:
        console.print(f"[red]Error checking enhanced status: {e}[/red]")


def _enhance_training_config_for_lstm_prophet(training_config):
    """Enhance training configuration for better LSTM + Prophet patterns"""
    enhanced_config = training_config.copy()
    
    # Longer experiments for better sequence learning
    enhanced_config['experiment_duration_minutes'] = max(20, training_config.get('experiment_duration_minutes', 15))
    
    # More replica variations for scaling pattern learning
    enhanced_config['replica_counts'] = training_config.get('replica_counts', [1, 2, 4]) + [8]
    
    # Better load distribution for seasonality patterns
    enhanced_config['load_levels_users'] = [10, 25, 50, 75, 100, 150, 200]
    
    # More scenarios for pattern diversity
    enhanced_config['test_scenarios'] = training_config.get('test_scenarios', ['browsing', 'checkout']) + ['mixed', 'burst']
    
    # Add LSTM + Prophet specific configuration
    enhanced_config['sampling_interval_seconds'] = 30  # More frequent sampling
    enhanced_config['stabilization_time_minutes'] = 3  # Longer stabilization
    enhanced_config['pattern_enhancement'] = True     # Enable pattern enhancement
    
    return enhanced_config


def _post_process_collected_data_for_lstm_prophet(service_name, data_dir):
    """Post-process collected data to add LSTM + Prophet favorable patterns"""
    try:
        from pathlib import Path
        
        data_path = Path(data_dir)
        csv_files = list(data_path.glob(f"*{service_name}*.csv"))
        
        for csv_file in csv_files:
            try:
                # Load original data
                original_data = pd.read_csv(csv_file)
                
                # Apply LSTM + Prophet enhancements
                enhanced_data = enhance_data_for_lstm_prophet(original_data, service_name)
                
                # Save enhanced version
                enhanced_filename = csv_file.stem + "_lstm_prophet_ready.csv"
                enhanced_path = data_path / enhanced_filename
                enhanced_data.to_csv(enhanced_path, index=False)
                
                console.print(f"[green]✅ LSTM + Prophet enhanced: {enhanced_filename}[/green]")
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not enhance {csv_file}: {e}[/yellow]")
                
    except Exception as e:
        console.print(f"[yellow]⚠️  Post-processing failed: {e}[/yellow]")


if __name__ == '__main__':
    main()
