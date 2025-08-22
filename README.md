# AIC25 Multimedia Retrieval System

<div align="center">

![AIC25 Logo](https://via.placeholder.com/200x100?text=AIC25)

**Advanced Multimedia Search & Retrieval System**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue.svg)](https://typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

</div>

## Team Past Beggar - HCMUS

- Tran Nam Khanh
- Phan Le Dac Phu  
- Duong Minh Loi
- Nguyen Ngoc Thien An
- Nguyen Truong Thinh

## 🚀 Overview

AIC25 is a state-of-the-art multimedia retrieval system designed for scalability, performance, and ease of use. It supports distributed deployment, multi-modal search (text, image, audio), cross-device communication, and features a modern web interface.

### ✨ Key Highlights

- **🔍 Multi-Modal Search**: Text, image, OCR, and audio-based content retrieval
- **🌐 Distributed Architecture**: Deploy across multiple devices with automatic service discovery  
- **⚡ High Performance**: GPU acceleration, efficient indexing, and optimized search
- **🎯 Modern UI/UX**: React-based frontend with real-time search and video streaming
- **🐳 Docker Ready**: Easy deployment with containerization support
- **📊 Production Ready**: Comprehensive monitoring, logging, and health checks

## Key Features

- **Multi-Modal Search**: Text, image, OCR, and audio-based content retrieval
- **Distributed Architecture**: Deploy backends on multiple devices with automatic service discovery
- **Enhanced Routing**: Comprehensive API with validation, error handling, and load balancing
- **Cross-Device Communication**: Frontend on one device can connect to backend on another
- **Real-time Video Streaming**: HTTP range request support for efficient video playback
- **System Monitoring**: Health checks, resource monitoring, and performance metrics

## 📁 Project Structure

```
AIC25/
├── 📄 README.md                   # Main documentation
├── ⚙️ pyproject.toml              # Python project configuration
├── 🐳 docker/                     # Docker configurations
│   ├── Dockerfile.backend         # Backend container
│   ├── Dockerfile.frontend        # Frontend container
│   └── docker-compose.yaml        # Multi-service deployment
├── ⚙️ config/                     # Configuration files
│   ├── default.yaml               # Default settings
│   ├── development.yaml           # Development overrides
│   └── production.yaml            # Production settings  
├── 📚 docs/                       # Documentation
│   ├── api/                       # API documentation
│   ├── development/              # Developer guides
│   ├── deployment/               # Deployment guides
│   └── user-guide/               # User documentation
├── 🐍 src/                       # Python backend
│   └── aic25/                    # Main package
│       ├── api/                  # FastAPI routes
│       ├── cli/                  # CLI commands
│       ├── core/                 # Business logic
│       └── services/             # Service layer
├── 🌐 web/                       # React frontend
│   ├── src/                      # Frontend source
│   │   ├── components/           # React components
│   │   ├── pages/                # Page components
│   │   ├── services/             # API clients
│   │   └── hooks/                # Custom hooks
│   └── public/                   # Static assets
└── 🛠️ scripts/                   # Build & deployment scripts
    ├── dev-setup.py              # Development environment setup
    └── build.py                  # Build automation
```

## 🚀 Quick Start

### Method 1: Quick Install (Recommended)
```bash
pip install git+https://github.com/DacPhu/AIC25.git
```

### Method 2: Development Setup
```bash
git clone https://github.com/DacPhu/AIC25.git
cd AIC25

# Automated development setup
python scripts/dev-setup.py

# Manual setup
pip install -e .
cd web && npm install
```

### Method 3: Docker Deployment
```bash
git clone https://github.com/DacPhu/AIC25.git
cd AIC25/docker
docker-compose up -d
```

### Dependencies
Ensure you have the following system dependencies:
- Python 3.12+
- FFmpeg (for video processing)
- FAISS or Milvus (for vector indexing)

## Quick Start

### 1. Initialize Workspace
```bash
aic25-cli init [workspace_path]
cd [workspace_path]
```

### 2. Add Videos
```bash
aic25-cli add [video_directory_or_file]
```

### 3. Process and Index
```bash
aic25-cli analyse
aic25-cli index
```

### 4. Start Backend Server
```bash
aic25-cli serve --host 0.0.0.0 --port 8000
```

Your multimedia search API is now running at `http://localhost:8000`

## Distributed Deployment Guide

### Single Device Setup
For basic usage on one machine:
```bash
aic25-cli serve --host 0.0.0.0 --port 8000
```

### Multi-Device Distributed Setup

#### Device 1 (Primary Backend)
```bash
export AIC25_WORK_DIR=/path/to/your/workspace
aic25-cli serve --host 0.0.0.0 --port 8000
```

#### Device 2 (Secondary Backend)
```bash
export AIC25_WORK_DIR=/path/to/shared/workspace  # Can be different content
aic25-cli serve --host 0.0.0.0 --port 8001
```

#### Frontend Configuration
The frontend automatically discovers available backends. You can also manually specify backend endpoints:

```javascript
const backends = [
  'http://device1:8000',
  'http://device2:8001',
  'http://device3:8002'
];
```

### Load Balancing Strategies

The system supports multiple load balancing strategies:
- `round_robin`: Distribute requests evenly
- `health_weighted`: Route based on backend health scores
- `random`: Random backend selection
- `least_connections`: Route to backend with fewest active connections

```python
from services.load_balancer import LoadBalancer

load_balancer = LoadBalancer(strategy="health_weighted")
```

## API Documentation

### Core Search Endpoints

#### Text Search
```bash
GET /api/v1/search?q=your_query&limit=50&offset=0
```

#### Similar Frame Search
```bash
GET /api/v1/search/similar?id=video_id#frame_id&limit=50
```

#### Audio Content Search
```bash
GET /api/v1/search/audio?q=music_playing&limit=50
```

#### Semantic Search (Sentence Transformers)
```bash
GET /api/v1/search/semantic?q=a person walking through a busy street&limit=50
```

#### Batch Search
```bash
POST /api/v1/search/batch
Content-Type: application/json

{
  "queries": ["person walking", "car driving", "music playing"],
  "params": {
    "limit": 20,
    "model": "clip"
  }
}
```

### Video & Frame Endpoints

#### Get Frame Information
```bash
GET /api/v1/frames/{video_id}/{frame_id}
```

#### Stream Video with Range Support
```bash
GET /api/v1/videos/{video_id}/stream
Range: bytes=0-1048576
```

#### Get Video Thumbnail
```bash
GET /api/v1/videos/{video_id}/thumbnail
```

### System Monitoring

#### Health Check
```bash
GET /api/v1/system/health
```

#### System Statistics
```bash
GET /api/v1/system/stats
```

#### Available Models
```bash
GET /api/v1/system/models
```

## 🔧 Configuration

### Environment Variables

```bash
# Core configuration
export AIC25_WORK_DIR=/path/to/workspace
export AIC25_HOST=0.0.0.0
export AIC25_PORT=8000

# Database selection
export AIC25_DATABASE=faiss  # or milvus

# Service discovery
export AIC25_DISCOVERY_PORT=8080
export AIC25_REGISTRY_HOST=localhost
```

### Configuration File

The system automatically creates `layout/config.yaml` with comprehensive settings:

```yaml
webui:
  database: faiss
  features:
    - name: clip
      pretrained_model: "openai/clip-vit-base-patch16"
      batch_size: 16
    - name: sentence_transformer
      pretrained_model: "all-MiniLM-L6-v2"  # or "all-mpnet-base-v2" for better accuracy
      device: "auto"  # "cpu", "cuda", or "auto"
      description: "Semantic text search with sentence transformers"
  search:
    default_nprobe: 16
    max_results: 10000
    rerank:
      enabled: false
      top_k: 100

hardware:
  gpu:
    enabled: true
    mixed_precision: true
  cpu:
    num_threads: -1
```

## Advanced Usage

### Custom Search Parameters

```bash
# Advanced text search with OCR weighting
curl "http://localhost:8000/api/v1/search?q=person&ocr_weight=1.5&ocr_threshold=40&temporal_k=10000&nprobe=16"

# Video-specific search
curl "http://localhost:8000/api/v1/search?q=video:video_001&limit=100"

# OCR-based search
curl "http://localhost:8000/api/v1/search?q=OCR:'text_to_find'&limit=50"
```

### Integration Examples

#### Python Client
```python
import requests

response = requests.get('http://localhost:8000/api/v1/search', params={
    'q': 'person walking',
    'limit': 20,
    'model': 'clip'
})

results = response.json()
for frame in results['frames']:
    print(f"Video: {frame['video_id']}, Frame: {frame['frame_id']}")
    print(f"Image: {frame['frame_uri']}")
```

#### JavaScript/React
```javascript
const searchFrames = async (query) => {
  const response = await fetch(`/api/v1/search?q=${encodeURIComponent(query)}`);
  const data = await response.json();
  return data.frames;
};

const frames = await searchFrames('person walking');
```

### Semantic Search with Sentence Transformers

The system supports advanced semantic search using sentence transformers, which provide better natural language understanding compared to traditional keyword matching.

#### Available Models

**General Purpose Models:**
- `all-MiniLM-L6-v2`: Fast, lightweight, good general performance (default)
- `all-mpnet-base-v2`: Best overall performance, slower but more accurate
- `all-MiniLM-L12-v2`: Good balance of speed and performance

**Multilingual Models:**
- `paraphrase-multilingual-MiniLM-L12-v2`: Supports 50+ languages
- `paraphrase-multilingual-mpnet-base-v2`: Best multilingual performance

**Specialized Models:**
- `msmarco-distilbert-base-v4`: Optimized for question-answering
- `gtr-t5-base`: General text representation model

#### Configuration Example

```yaml
webui:
  features:
    - name: sentence_transformer
      pretrained_model: "all-mpnet-base-v2"  # For best accuracy
      device: "cuda"  # Use GPU acceleration
      description: "High-accuracy semantic search"
```

#### Usage Examples

```bash
# Natural language queries
curl "http://localhost:8000/api/v1/search/semantic?q=a person walking down a busy street at night"

# Complex descriptions
curl "http://localhost:8000/api/v1/search/semantic?q=someone cooking food in a modern kitchen with stainless steel appliances"

# Conceptual searches
curl "http://localhost:8000/api/v1/search/semantic?q=emotional moment between two people"
```

#### JavaScript Integration

```javascript
// Add semantic search option to frontend
const searchTypes = [
  { value: 'text', label: 'Text Search (CLIP)' },
  { value: 'semantic', label: 'Semantic Search (Sentence Transformers)' },
  { value: 'audio', label: 'Audio Content Search' },
  { value: 'similar', label: 'Similar Frame Search' }
];

const performSemanticSearch = async (query) => {
  const response = await fetch(`/api/v1/search/semantic?q=${encodeURIComponent(query)}`);
  const data = await response.json();
  return data.frames;
};
```

## Development

### Running Tests
```bash
pip install -e ".[dev]"

pytest tests/

pytest --cov=src tests/
```

### Code Quality
```bash
black src/
isort src/

flake8 src/
mypy src/
```

### Building Documentation
```bash
cd docs/
make html
```

## Troubleshooting

### Common Issues

**Backend fails to start:**
```bash
aic25-cli init .

pip install --upgrade aic25

cat layout/config.yaml
```

**Search returns no results:**
```bash
aic25-cli index --verify

curl http://localhost:8000/api/v1/system/models

curl http://localhost:8000/api/v1/system/health
```

**Cross-device connection issues:**
```bash
sudo ufw allow 8000

ping device_ip
telnet device_ip 8000

curl http://device_ip:8080/api/v1/registry/services
```

### Performance Optimization

**For large datasets:**
- Use GPU acceleration: Set `hardware.gpu.enabled: true`
- Increase batch sizes: Adjust `batch_size` in config
- Use Milvus for >1M frames: Set `database: milvus`

**For distributed deployment:**
- Use SSD storage for indices
- Ensure high-bandwidth network connection
- Monitor system resources with `/api/v1/system/stats`

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Frontend      │    │   Frontend      │
│   (Device 1)    │    │   (Device 2)    │    │   (Device 3)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────┬─────────────────────┬────────┘
                         │                     │
                ┌────────▼────────┐     ┌──────▼──────────┐
                │  Load Balancer  │     │ Service Registry│
                └────────┬────────┘     └─────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐ ┌───────▼──────┐ ┌───────▼──────┐
│   Backend    │ │   Backend    │ │   Backend    │
│  (Device A)  │ │  (Device B)  │ │  (Device C)  │
└──────────────┘ └──────────────┘ └──────────────┘
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Implementation Status

- [x] CLI Tools
  - [x] Video import and processing
  - [x] Feature extraction and indexing
  - [x] Multi-modal search capabilities
  - [x] Web server with enhanced routing
- [x] Enhanced API
  - [x] Comprehensive request/response validation
  - [x] Advanced search endpoints with pagination
  - [x] Video streaming with range support
  - [x] System monitoring and health checks
- [x] Distributed System
  - [x] Service discovery and registration
  - [x] Load balancing with multiple strategies
  - [x] Cross-device communication
  - [x] Automatic failover and health monitoring
- [x] Web Interface
  - [x] React-based frontend
  - [x] Real-time search and preview
  - [x] Multi-backend support
