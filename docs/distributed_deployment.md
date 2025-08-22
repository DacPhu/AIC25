# Simple Distributed Deployment Guide

This guide shows how to manually deploy AIC25 across multiple devices where each device runs its own FastAPI backend, and any frontend can connect to any backend.

## Architecture Overview

```
Device 1 (192.168.1.100)          Device 2 (192.168.1.101)          Device 3 (192.168.1.102)
┌─────────────────────┐           ┌─────────────────────┐           ┌─────────────────────┐
│  Frontend (React)   │           │  Frontend (React)   │           │  Frontend (React)   │
│  Backend (FastAPI)  │           │  Backend (FastAPI)  │           │  Backend (FastAPI)  │
│  Port: 5000         │           │  Port: 5000         │           │  Port: 5001         │
└─────────────────────┘           └─────────────────────┘           └─────────────────────┘
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                          │
                              Network Communication
                         (Any frontend can connect to any backend)
```

## Quick Start

### 1. Start Backend on Each Device

**Device 1:**
```bash
cd /path/to/AIC25
aic25-cli serve --port 5000
```

**Device 2:**
```bash
cd /path/to/AIC25  
aic25-cli serve --port 5000
```

**Device 3:**
```bash
cd /path/to/AIC25
aic25-cli serve --port 5001
```

### 2. Configure Frontend

Open any frontend and:
1. Go to the Backend Manager (in the UI)
2. Click "Auto-Discover" or manually add backends:
   - Device 1: `192.168.1.100:5000`
   - Device 2: `192.168.1.101:5000`
   - Device 3: `192.168.1.102:5001`

### 3. Use the System

- Frontend automatically load balances across healthy backends
- If one backend goes down, requests automatically failover to others
- Each backend serves its own video/frame data independently

## Detailed Setup

### Backend Startup Options

```bash
python scripts/start_backend.py [OPTIONS]

Options:
  --host TEXT        Host to bind to (default: 0.0.0.0)
  --port INTEGER     Port to bind to (default: 5000)
  --work-dir TEXT    Working directory for AIC25 data
  --reload           Enable auto-reload for development
  --workers INTEGER  Number of worker processes (default: 1)
  --log-level TEXT   Log level (default: info)
```

### Environment Variables

You can also configure via environment variables:

```bash
# Set working directory
export AIC25_WORK_DIR="/path/to/your/data"

# Set backend hosts for auto-discovery (comma-separated)
export VITE_BACKEND_HOSTS="192.168.1.100:5000,192.168.1.101:5000,192.168.1.102:5001"

# Start backend
python scripts/start_backend.py --port 5000
```

### Frontend Configuration

The frontend can be configured in multiple ways:

#### 1. Environment Variables (.env file):
```env
VITE_BACKEND_HOSTS=192.168.1.100:5000,192.168.1.101:5000,192.168.1.102:5001
```

#### 2. Programmatic Configuration:
```typescript
import { addBackend } from './services/distributed_search';

// Add backends manually
addBackend('device1', '192.168.1.100', 5000);
addBackend('device2', '192.168.1.101', 5000);
addBackend('device3', '192.168.1.102', 5001);
```

#### 3. UI Configuration:
Use the Backend Manager component in the React app.

### Cross-Device Example

**Scenario**: 3 devices in the same office

**Device 1 (Main Server) - 192.168.1.100:**
```bash
# Has the main video dataset
python scripts/start_backend.py --host 0.0.0.0 --port 5000 --work-dir /data/main_videos
```

**Device 2 (Workstation) - 192.168.1.101:**
```bash
# Has additional video dataset
python scripts/start_backend.py --host 0.0.0.0 --port 5000 --work-dir /data/extra_videos
```

**Device 3 (Laptop) - 192.168.1.102:**
```bash
# Development/testing environment
python scripts/start_backend.py --host 0.0.0.0 --port 5001 --work-dir /Users/dev/test_videos --reload
```

**Any frontend can now:**
- Search across all three datasets
- Automatically failover if one device is unavailable
- Load balance requests across available devices

## API Endpoints

Each backend exposes these key endpoints:

### Health Check
```
GET http://{host}:{port}/api/v1/system/health
```

### Search
```
GET http://{host}:{port}/api/v1/search?q=query&limit=50
```

### Videos
```
GET http://{host}:{port}/api/v1/videos
```

### API Documentation
```
GET http://{host}:{port}/docs
```

## Load Balancing

The frontend automatically:

1. **Health Checks**: Monitors all backends every 30 seconds
2. **Failover**: Retries failed requests on other healthy backends
3. **Load Balancing**: Selects fastest responding backend
4. **Auto-Discovery**: Finds backends on local network

### Load Balancing Strategies

The system uses "health-weighted" load balancing by default:
- Faster response time = higher priority
- Failed requests lower backend priority
- Automatic failover on backend failure

## Data Management

### Shared Data
If all devices should access the same videos:
```bash
# Use network share or synchronized directories
python scripts/start_backend.py --work-dir /shared/aic25_data
```

### Distributed Data
If each device has different video collections:
```bash
# Device 1: News videos
python scripts/start_backend.py --work-dir /data/news_videos

# Device 2: Sports videos  
python scripts/start_backend.py --work-dir /data/sports_videos

# Device 3: Movie videos
python scripts/start_backend.py --work-dir /data/movie_videos
```

Frontend will search across all collections automatically.

## Monitoring

### Backend Manager UI

The React app includes a Backend Manager component that shows:
- All configured backends
- Health status and response times
- Auto-discovery controls
- Manual backend management

### Command Line Monitoring

Check backend status:
```bash
curl http://192.168.1.100:5000/api/v1/system/health
curl http://192.168.1.101:5000/api/v1/system/health  
curl http://192.168.1.102:5001/api/v1/system/health
```

Get system stats:
```bash
curl http://192.168.1.100:5000/api/v1/system/stats
```

## Troubleshooting

### Backend Not Starting
1. Check if port is in use: `netstat -an | grep :5000`
2. Check firewall settings
3. Verify Python dependencies are installed

### Frontend Can't Connect
1. Ensure backend is running: `curl http://{host}:{port}/api/v1/system/health`
2. Check network connectivity between devices
3. Verify firewall allows connections on the port
4. Check CORS settings in backend

### Poor Performance
1. Check network latency between devices
2. Monitor backend response times in Backend Manager
3. Consider running backends on faster hardware
4. Use different ports to avoid conflicts

### Discovery Issues
1. Ensure all devices are on the same network
2. Check firewall settings for discovery ports
3. Manually add backends if auto-discovery fails

## Security Notes

### Network Security
- Backends bind to `0.0.0.0` (all interfaces) by default
- Consider using firewall rules to restrict access
- Use VPN for connections over internet

### Data Security
- No authentication enabled by default
- Add API keys or JWT tokens for production use
- Consider HTTPS for sensitive data

### Recommended Production Setup
```bash
# Use specific interface instead of 0.0.0.0
python scripts/start_backend.py --host 192.168.1.100 --port 5000

# Use reverse proxy with SSL
# nginx/apache in front of FastAPI backends
```

## Advanced Usage

### Custom Configuration

Create a custom config file:
```yaml
# custom_config.yaml
backends:
  - name: "server1"
    host: "192.168.1.100"
    port: 5000
    weight: 2.0  # Higher priority
  - name: "server2"  
    host: "192.168.1.101"
    port: 5000
    weight: 1.0
  - name: "edge_device"
    host: "192.168.1.102" 
    port: 5001
    weight: 0.5  # Lower priority
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "scripts/start_backend.py", "--host", "0.0.0.0", "--port", "5000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aic25-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aic25-backend
  template:
    metadata:
      labels:
        app: aic25-backend
    spec:
      containers:
      - name: aic25
        image: aic25:latest
        ports:
        - containerPort: 5000
        env:
        - name: AIC25_WORK_DIR
          value: "/data"
```

This simple approach allows for easy distributed deployment where each device runs independently, and frontends can connect to any available backend across the network.