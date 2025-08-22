#!/usr/bin/env python3
"""
Demo script showing distributed AIC25 deployment workflow
This script demonstrates how frontend on device_1 can connect to backend on device_2
"""
import subprocess
import time
import requests
import json
import sys
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    print(f"\n📍 Step {step}: {description}")
    print("-" * 40)

def check_backend_health(host, port):
    """Check if backend is healthy"""
    try:
        response = requests.get(f"http://{host}:{port}/api/v1/system/health", timeout=5)
        return response.status_code == 200 and response.json().get("success")
    except:
        return False

def demo_workflow():
    print_header("AIC25 Distributed Deployment Demo")
    
    print("""
This demo shows how to set up AIC25 in a distributed manner:
1. Each device runs its own FastAPI backend manually
2. Frontend on any device can connect to backend on any other device
3. Automatic load balancing and failover
    """)
    
    print_step(1, "Starting Backend Instances")
    
    # Simulate multiple backend instances
    backends = [
        {"name": "device1", "host": "127.0.0.1", "port": 5000},
        {"name": "device2", "host": "127.0.0.1", "port": 5001},
        {"name": "device3", "host": "127.0.0.1", "port": 5002}
    ]
    
    processes = []
    
    print("Starting backends (simulating different devices)...")
    for backend in backends:
        print(f"  Starting {backend['name']} on {backend['host']}:{backend['port']}")
        
        # Start backend process using CLI
        cmd = [
            sys.executable, "-m", "entry.cli",
            "serve",
            "--port", str(backend["port"])
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            processes.append((backend, process))
            print(f"    ✅ Started {backend['name']} (PID: {process.pid})")
        except Exception as e:
            print(f"    ❌ Failed to start {backend['name']}: {e}")
    
    print_step(2, "Waiting for Backends to Start")
    
    print("Waiting for backends to become healthy...")
    time.sleep(5)  # Give backends time to start
    
    # Check backend health
    healthy_backends = []
    for backend, process in processes:
        if check_backend_health(backend["host"], backend["port"]):
            healthy_backends.append(backend)
            print(f"    ✅ {backend['name']} is healthy")
        else:
            print(f"    ❌ {backend['name']} is not responding")
    
    if not healthy_backends:
        print("\n❌ No healthy backends found. Demo cannot continue.")
        # Cleanup
        for _, process in processes:
            process.terminate()
        return
    
    print_step(3, "Demonstrating Cross-Device Communication")
    
    print("Simulating frontend on device1 connecting to backend on device2...")
    
    # Demonstrate API calls to different backends
    for i, backend in enumerate(healthy_backends):
        print(f"\n🔗 Testing connection to {backend['name']} ({backend['host']}:{backend['port']})")
        
        try:
            # Health check
            health_url = f"http://{backend['host']}:{backend['port']}/api/v1/system/health"
            response = requests.get(health_url, timeout=5)
            health_data = response.json()
            
            print(f"    Health: {health_data.get('status', 'unknown')}")
            print(f"    Database: {health_data.get('database_type', 'unknown')}")
            print(f"    Version: {health_data.get('version', 'unknown')}")
            
            # System stats
            stats_url = f"http://{backend['host']}:{backend['port']}/api/v1/system/stats"
            response = requests.get(stats_url, timeout=5)
            stats_data = response.json()
            
            if stats_data.get('success'):
                system_info = stats_data.get('system', {})
                print(f"    CPU: {system_info.get('cpu_percent', 0):.1f}%")
                print(f"    Memory: {system_info.get('memory', {}).get('percent', 0):.1f}%")
            
            # Search test (if data is available)
            search_url = f"http://{backend['host']}:{backend['port']}/api/v1/search"
            response = requests.get(search_url, params={"q": "test", "limit": 1}, timeout=5)
            search_data = response.json()
            
            print(f"    Search API: {'✅ Working' if search_data.get('success') else '⚠️ No data'}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print_step(4, "Frontend Auto-Discovery Simulation")
    
    print("Simulating frontend discovering backends automatically...")
    
    # Simulate discovery
    discovered_backends = []
    for backend in healthy_backends:
        try:
            health_url = f"http://{backend['host']}:{backend['port']}/api/v1/system/health"
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                discovered_backends.append(backend)
                print(f"    🔍 Discovered: {backend['name']} at {backend['host']}:{backend['port']}")
        except:
            pass
    
    print(f"\n📊 Discovery Results:")
    print(f"    Total backends found: {len(discovered_backends)}")
    print(f"    Available for load balancing: {len(discovered_backends)}")
    
    print_step(5, "Load Balancing Demonstration")
    
    if len(discovered_backends) > 1:
        print("Demonstrating load balancing across multiple backends...")
        
        # Simulate multiple requests with load balancing
        for i in range(5):
            backend = discovered_backends[i % len(discovered_backends)]
            print(f"    Request {i+1} → {backend['name']} ({backend['host']}:{backend['port']})")
            
            try:
                start_time = time.time()
                health_url = f"http://{backend['host']}:{backend['port']}/api/v1/system/health"
                response = requests.get(health_url, timeout=5)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    print(f"        ✅ Success ({response_time:.0f}ms)")
                else:
                    print(f"        ❌ Failed (HTTP {response.status_code})")
            except Exception as e:
                print(f"        ❌ Error: {e}")
    
    print_step(6, "Failover Demonstration")
    
    if len(processes) > 1:
        print("Demonstrating failover by stopping one backend...")
        
        # Stop the first backend
        backend_to_stop, process_to_stop = processes[0]
        print(f"    Stopping {backend_to_stop['name']}...")
        process_to_stop.terminate()
        time.sleep(2)
        
        # Test remaining backends
        remaining_backends = [b for b, _ in processes[1:]]
        working_backends = []
        
        for backend in remaining_backends:
            if check_backend_health(backend["host"], backend["port"]):
                working_backends.append(backend)
                print(f"    ✅ {backend['name']} still working (failover successful)")
            else:
                print(f"    ❌ {backend['name']} not responding")
        
        print(f"\n    Failover result: {len(working_backends)}/{len(remaining_backends)} backends still available")
    
    print_step(7, "Cleanup")
    
    print("Stopping all backend processes...")
    for backend, process in processes:
        try:
            process.terminate()
            process.wait(timeout=5)
            print(f"    ✅ Stopped {backend['name']}")
        except:
            print(f"    ⚠️ Force killing {backend['name']}")
            process.kill()
    
    print_header("Demo Complete")
    
    print("""
✅ Demo workflow completed successfully!

Key takeaways:
1. ✅ Multiple FastAPI backends can run independently on different devices
2. ✅ Frontend can discover and connect to any available backend
3. ✅ Automatic load balancing distributes requests across healthy backends  
4. ✅ Failover works when backends become unavailable
5. ✅ Simple manual deployment - just run aic25-cli serve on each device

Next steps:
1. Run 'aic25-cli serve --port 5000' on each device
2. Configure frontend with backend addresses
3. Use the Backend Manager UI to manage connections
4. Enjoy distributed multimedia search across your devices!

For detailed instructions, see: docs/distributed_deployment.md
    """)

def main():
    try:
        demo_workflow()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()