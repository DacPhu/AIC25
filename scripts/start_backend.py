#!/usr/bin/env python3
"""
Simple FastAPI Backend Startup Script for Manual Distributed Deployment
Run this script on each device to start a backend instance
"""
import argparse
import os
import sys
import socket
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_local_ip():
    """Get local IP address"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def check_port_available(host, port):
    """Check if port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Start AIC25 FastAPI Backend")
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0 - all interfaces)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000, 
        help="Port to bind to (default: 5000)"
    )
    parser.add_argument(
        "--work-dir", 
        help="Working directory for AIC25 data"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level", 
        default="info", 
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.work_dir:
        os.environ["AIC25_WORK_DIR"] = args.work_dir
    
    # Get local IP for display
    local_ip = get_local_ip()
    
    # Check if port is available
    if not check_port_available(args.host, args.port):
        print(f"❌ Port {args.port} is already in use on {args.host}")
        print(f"Try using a different port with --port argument")
        return 1
    
    print("🚀 Starting AIC25 FastAPI Backend")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Local IP: {local_ip}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print(f"Work Dir: {args.work_dir or os.getcwd()}")
    print("=" * 50)
    print(f"🌐 Backend will be accessible at:")
    print(f"   Local:    http://127.0.0.1:{args.port}")
    print(f"   Network:  http://{local_ip}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"   All IPs:  http://0.0.0.0:{args.port}")
    print("=" * 50)
    print("📍 API Endpoints:")
    print(f"   Health:   http://{local_ip}:{args.port}/api/v1/system/health")
    print(f"   Search:   http://{local_ip}:{args.port}/api/v1/search")
    print(f"   Docs:     http://{local_ip}:{args.port}/docs")
    print("=" * 50)
    print("🔧 Frontend Configuration:")
    print(f"   Add this backend to frontend:")
    print(f"   Name: {socket.gethostname()}")
    print(f"   Host: {local_ip}")
    print(f"   Port: {args.port}")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Import the FastAPI app
        from src.entry.web.app import app
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
        
        # Run the server
        server = uvicorn.Server(config)
        server.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except ImportError as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        print("Make sure you're running from the project root directory")
        return 1
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())