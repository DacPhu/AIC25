#!/usr/bin/env python3
"""
Distributed deployment script for AIC25.
Helps deploy multiple backend instances across different devices.
"""
import argparse
import subprocess
import sys
import os
import yaml
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load distributed configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available on a host"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False

def start_backend_instance(
    host: str, 
    port: int, 
    registry_endpoints: List[str] = None,
    device_profile: str = "desktop",
    work_dir: str = None
) -> subprocess.Popen:
    """Start a backend instance"""
    
    env = os.environ.copy()
    env.update({
        "AIC25_HOST": host,
        "AIC25_PORT": str(port),
        "AIC25_DEVICE_PROFILE": device_profile
    })
    
    if registry_endpoints:
        env["AIC25_REGISTRY_ENDPOINTS"] = ",".join(registry_endpoints)
    
    if work_dir:
        env["AIC25_WORK_DIR"] = work_dir
    
    # Command to start the backend
    cmd = [
        sys.executable, "-m", "src.entry.web.startup"
    ]
    
    print(f"Starting backend on {host}:{port} with profile '{device_profile}'")
    
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    return process

def wait_for_service(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for a service to become healthy"""
    url = f"http://{host}:{port}/api/v1/system/health"
    
    for _ in range(timeout):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("status") == "healthy":
                    return True
        except requests.RequestException:
            pass
        time.sleep(1)
    
    return False

def get_cluster_status(registry_endpoint: str) -> Dict[str, Any]:
    """Get cluster status from a registry endpoint"""
    try:
        response = requests.get(f"{registry_endpoint}/api/v1/registry/services", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        print(f"Failed to get cluster status: {e}")
    
    return {"success": False, "services": []}

def deploy_single_device(
    config: Dict[str, Any],
    ports: List[int] = None,
    instances: int = 1
) -> List[subprocess.Popen]:
    """Deploy multiple instances on a single device"""
    
    if ports is None:
        base_port = config.get("network", {}).get("port", 5000)
        ports = [base_port + i for i in range(instances)]
    
    processes = []
    registry_endpoints = []
    
    # Start first instance (acts as initial registry)
    first_port = ports[0]
    if check_port_available("127.0.0.1", first_port):
        process = start_backend_instance(
            host="0.0.0.0",
            port=first_port,
            device_profile=config.get("backend", {}).get("device_profile", "desktop")
        )
        processes.append(process)
        
        # Wait for first instance to start
        if wait_for_service("127.0.0.1", first_port):
            registry_endpoints.append(f"http://127.0.0.1:{first_port}")
            print(f"✓ Backend started on port {first_port}")
        else:
            print(f"✗ Failed to start backend on port {first_port}")
    else:
        print(f"✗ Port {first_port} is already in use")
    
    # Start additional instances
    for port in ports[1:]:
        if check_port_available("127.0.0.1", port):
            process = start_backend_instance(
                host="0.0.0.0",
                port=port,
                registry_endpoints=registry_endpoints,
                device_profile=config.get("backend", {}).get("device_profile", "desktop")
            )
            processes.append(process)
            
            # Wait for instance to start
            if wait_for_service("127.0.0.1", port):
                print(f"✓ Backend started on port {port}")
            else:
                print(f"✗ Failed to start backend on port {port}")
        else:
            print(f"✗ Port {port} is already in use")
    
    return processes

def deploy_from_config(config_path: str, deployment_name: str):
    """Deploy using a predefined configuration"""
    config = load_config(config_path)
    
    deployment = config.get("deployment_examples", {}).get(deployment_name)
    if not deployment:
        print(f"Deployment '{deployment_name}' not found in config")
        return []
    
    print(f"Deploying: {deployment.get('description', deployment_name)}")
    
    processes = []
    for instance in deployment.get("instances", []):
        device = instance.get("device", "localhost")
        port = instance.get("port", 5000)
        profile = instance.get("profile", "desktop")
        
        if device in ["localhost", "127.0.0.1"] or device.endswith(".local"):
            # Local deployment
            if check_port_available("127.0.0.1", port):
                process = start_backend_instance(
                    host="0.0.0.0",
                    port=port,
                    device_profile=profile
                )
                processes.append(process)
                
                if wait_for_service("127.0.0.1", port):
                    print(f"✓ Backend started on {device}:{port} ({profile})")
                else:
                    print(f"✗ Failed to start backend on {device}:{port}")
            else:
                print(f"✗ Port {port} is already in use on {device}")
        else:
            print(f"⚠ Remote deployment to {device}:{port} requires manual setup")
    
    return processes

def show_cluster_status(registry_endpoint: str):
    """Show cluster status"""
    status = get_cluster_status(registry_endpoint)
    
    if status.get("success"):
        services = status.get("services", [])
        print(f"\n📊 Cluster Status ({len(services)} services)")
        print("=" * 60)
        
        for service in services:
            device_name = service.get("device_name", "unknown")
            host = service.get("host", "unknown")
            port = service.get("port", "unknown")
            status_val = service.get("status", "unknown")
            capabilities = ", ".join(service.get("capabilities", []))
            
            status_icon = "✓" if status_val == "healthy" else "✗"
            print(f"{status_icon} {device_name} ({host}:{port}) - {capabilities}")
        
        print("=" * 60)
    else:
        print("✗ Failed to get cluster status")

def main():
    parser = argparse.ArgumentParser(description="Deploy AIC25 in distributed mode")
    parser.add_argument(
        "--config",
        default="src/external/distributed/config.yaml",
        help="Path to distributed configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single device deployment
    single_parser = subparsers.add_parser("single", help="Deploy on single device")
    single_parser.add_argument("--instances", type=int, default=1, help="Number of instances")
    single_parser.add_argument("--ports", nargs="+", type=int, help="Specific ports to use")
    
    # Predefined deployment
    deploy_parser = subparsers.add_parser("deploy", help="Deploy using predefined config")
    deploy_parser.add_argument("name", help="Deployment name from config")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show cluster status")
    status_parser.add_argument("--registry", default="http://127.0.0.1:5000", help="Registry endpoint")
    
    # Start single instance
    start_parser = subparsers.add_parser("start", help="Start single instance")
    start_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    start_parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    start_parser.add_argument("--profile", default="desktop", help="Device profile")
    start_parser.add_argument("--registry", nargs="+", help="Registry endpoints")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        return
    
    config = load_config(str(config_path))
    
    if args.command == "single":
        processes = deploy_single_device(config, args.ports, args.instances)
        
        if processes:
            print(f"\n🚀 Started {len(processes)} backend instances")
            print("Press Ctrl+C to stop all instances")
            
            try:
                # Wait for all processes
                for process in processes:
                    process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping all instances...")
                for process in processes:
                    process.terminate()
                    process.wait()
    
    elif args.command == "deploy":
        processes = deploy_from_config(str(config_path), args.name)
        
        if processes:
            print(f"\n🚀 Deployment '{args.name}' started with {len(processes)} local instances")
            print("Press Ctrl+C to stop all instances")
            
            try:
                for process in processes:
                    process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping all instances...")
                for process in processes:
                    process.terminate()
                    process.wait()
    
    elif args.command == "status":
        show_cluster_status(args.registry)
    
    elif args.command == "start":
        process = start_backend_instance(
            host=args.host,
            port=args.port,
            registry_endpoints=args.registry,
            device_profile=args.profile
        )
        
        print(f"🚀 Started backend on {args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping instance...")
            process.terminate()
            process.wait()

if __name__ == "__main__":
    main()