# CLI Usage Guide - Database Selection

This guide explains how to use the redesigned CLI with support for both FAISS and Milvus databases.

## Overview

The CLI now supports both FAISS and Milvus databases through a unified interface. You can:
- Set a default database in the configuration file
- Override the database type via CLI arguments
- Use different databases for indexing and serving

## Commands

### Index Command

Index video features into a database.

```bash
# Use default database from config
aic25-cli index

# Use FAISS database explicitly
aic25-cli index --database faiss --collection my_faiss_collection

# Use Milvus database explicitly  
aic25-cli index --database milvus --collection my_milvus_collection

# Overwrite existing collection
aic25-cli index --database faiss --overwrite --collection new_collection

# Update existing records
aic25-cli index --database milvus --update --collection existing_collection
```

**Options:**
- `-c, --collection`: Name of collection (default: "default_collection")
- `-d, --database`: Database type - "faiss" or "milvus" (default: from config)
- `-o, --overwrite`: Overwrite existing collection
- `-u, --update`: Update existing records

### Serve Command

Start the web server and API.

```bash
# Use default database from config
aic25-cli serve

# Use FAISS database for serving
aic25-cli serve --database faiss

# Use Milvus database for serving
aic25-cli serve --database milvus --port 8080

# Development mode with FAISS
aic25-cli serve --database faiss --dev --port 3000
```

**Options:**
- `-p, --port`: Port number (default: 5100)
- `-d, --dev`: Development mode
- `-w, --workers`: Number of workers (default: 1)
- `--database`: Database type - "faiss" or "milvus" (default: from config)

### Other Commands

The `add` and `analyse` commands remain unchanged and work with both databases.

```bash
# Add videos (works with any database)
aic25-cli add /path/to/videos --directory

# Analyse keyframes (works with any database)
aic25-cli analyse --no-gpu
```

## Configuration

Set the default database type in `aic25_workspace/layout/config.yaml`:

```yaml
webui:
  database: "faiss"  # or "milvus"
  # ... other settings
```

### FAISS Configuration

```yaml
faiss:
  index_type: "IVF"  # "Flat", "IVF", "HNSW", "PQ"
  nlist: 128
  nprobe: 8
  # ... other FAISS settings
```

### Milvus Configuration

```yaml
milvus:
  fields:
    - field_name: "frame_id"
      datatype: "VARCHAR"
      max_length: 32
      is_primary: true
    # ... other fields
```

## Database Selection Priority

The system determines which database to use in this order:

1. CLI argument (`--database faiss` or `--database milvus`)
2. Configuration file setting (`webui.database`)
3. Default fallback ("faiss")

## Migration Between Databases

To migrate from Milvus to FAISS:

```bash
# 1. Re-index with FAISS
aic25-cli index --database faiss --collection migrated_collection

# 2. Update config to use FAISS by default
# Edit config.yaml: webui.database: "faiss"

# 3. Serve with FAISS
aic25-cli serve --database faiss
```

To migrate from FAISS to Milvus:

```bash
# 1. Start Milvus and re-index
aic25-cli index --database milvus --collection migrated_collection

# 2. Update config to use Milvus by default
# Edit config.yaml: webui.database: "milvus"

# 3. Serve with Milvus
aic25-cli serve --database milvus
```

## Performance Considerations

### FAISS
- **Best for**: Local deployments, fast startup, no external dependencies
- **Index types**:
  - `Flat`: Exact search, < 10k vectors
  - `IVF`: Good balance, 10k-1M vectors  
  - `HNSW`: Fastest search, > 100k vectors
  - `PQ`: Memory efficient, very large datasets

### Milvus
- **Best for**: Distributed deployments, advanced features, production scale
- **Features**: 
  - Distributed storage and computation
  - Advanced indexing algorithms
  - Built-in monitoring and management
  - ACID transactions

## Examples

### Complete Workflow with FAISS

```bash
# Add videos
aic25-cli add /path/to/video/dataset --directory

# Extract features
aic25-cli analyse

# Index with FAISS
aic25-cli index --database faiss --collection video_search

# Serve with FAISS
aic25-cli serve --database faiss --port 8080
```

### Complete Workflow with Milvus

```bash
# Add videos
aic25-cli add /path/to/video/dataset --directory

# Extract features
aic25-cli analyse

# Index with Milvus
aic25-cli index --database milvus --collection video_search

# Serve with Milvus
aic25-cli serve --database milvus --port 8080
```

### Mixed Usage

```bash
# Index in both databases for comparison
aic25-cli index --database faiss --collection faiss_test
aic25-cli index --database milvus --collection milvus_test

# Test serving with different databases
aic25-cli serve --database faiss --port 5100 &
aic25-cli serve --database milvus --port 5101 &
```

## Troubleshooting

### FAISS Issues
- **Memory errors**: Reduce batch size or use PQ index type
- **Slow indexing**: Try IVF or HNSW index types
- **Poor accuracy**: Increase nprobe parameter

### Milvus Issues
- **Connection failed**: Check if Milvus server is running
- **Collection not found**: Verify collection name and existence
- **Performance issues**: Check Milvus logs and resource usage

### General Issues
- **Features not found**: Run `analyse` command first
- **Import errors**: Check database dependencies are installed
- **Config errors**: Verify YAML syntax in config file