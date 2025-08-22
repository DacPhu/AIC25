# Installation Guide

## System Requirements

- Python 3.12 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU support (optional, for better performance)
- FFmpeg (for video processing)

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
pip install git+https://github.com/DacPhu/AIC25.git
```

### Method 2: Development Install

```bash
git clone https://github.com/DacPhu/AIC25.git
cd AIC25
pip install -e .
```

## Post-Installation Setup

### 1. Initialize Workspace

```bash
# Create a new workspace
aic25-cli init my-workspace
cd my-workspace
```

### 2. Add Videos

```bash
# Add a single video
aic25-cli add /path/to/video.mp4

# Add a directory of videos
aic25-cli add /path/to/video-directory/
```

### 3. Process and Index

```bash
# Extract features from videos
aic25-cli analyse

# Build search index
aic25-cli index
```

### 4. Start the Server

```bash
# Start the web server
aic25-cli serve --host 0.0.0.0 --port 8000
```

## Verify Installation

Open your browser to `http://localhost:8000` and verify:

1. Web interface loads correctly
2. Search functionality works
3. Video playback functions properly

## Configuration

### Environment Variables

```bash
# Set workspace directory
export AIC25_WORK_DIR=/path/to/workspace

# Set configuration environment
export AIC25_CONFIG_ENV=production

# Set host and port
export AIC25_HOST=0.0.0.0
export AIC25_PORT=8000
```

### Configuration Files

Edit configuration files in the `config/` directory:

- `config/default.yaml` - Base settings
- `config/production.yaml` - Production overrides
- `config/development.yaml` - Development overrides

## Database Options

### FAISS (Default)
- Faster setup
- Good for smaller datasets (< 1M frames)
- No external dependencies

```yaml
webui:
  database: "faiss"
```

### Milvus
- Better for larger datasets (> 1M frames)
- Requires Milvus server setup
- Better scalability

```yaml
webui:
  database: "milvus"
```

## GPU Support

### NVIDIA GPU
```bash
# Install FAISS with GPU support
pip uninstall faiss-cpu
pip install faiss-gpu

# Enable GPU in configuration
webui:
  hardware:
    gpu:
      enabled: true
```

## Troubleshooting

### Common Issues

**Import Error: No module named 'aic25'**
- Ensure proper installation: `pip install -e .`

**Server fails to start**
- Check if port is available: `lsof -i :8000`
- Verify workspace initialization: `aic25-cli init`

**No search results**
- Ensure videos are processed: `aic25-cli analyse`
- Build search index: `aic25-cli index`
- Check logs for errors

**Video playback issues**
- Ensure FFmpeg is installed
- Check video file format compatibility

### Getting Help

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Review logs for error messages
3. Open an issue on GitHub if problems persist