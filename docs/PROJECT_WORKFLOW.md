# AIC25 Project Workflow & Architecture Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Overall Workflow](#overall-workflow)
3. [Component Deep Dive](#component-deep-dive)
4. [Database Layer Architecture](#database-layer-architecture)
5. [Search Layer Architecture](#search-layer-architecture)
6. [Configuration System](#configuration-system)
7. [Performance Considerations](#performance-considerations)

---

## Project Overview

**AIC25** is a multimedia retrieval system designed for AI Challenge 2025. It processes video content, extracts multimodal features, and provides semantic search capabilities through both FAISS and Milvus databases.

### Key Features
- **Video Processing**: Keyframe extraction with adaptive algorithms
- **Multimodal Analysis**: CLIP visual embeddings + OCR text extraction
- **Dual Database Support**: FAISS (local) and Milvus (distributed)
- **Web Interface**: React-based search UI with real-time results
- **Scalable Architecture**: Parallel processing and configurable performance

---

## Overall Workflow

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────┐
│   Videos    │───▶│     ADD      │───▶│   ANALYSE     │───▶│   INDEX    │
│ (MP4 files) │    │  (Keyframes) │    │  (Features)   │    │ (Database) │
└─────────────┘    └──────────────┘    └───────────────┘    └────────────┘
                            │                    │                    │
                            ▼                    ▼                    ▼
                   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
                   │  aic25_workspace │  │  aic25_workspace │  │ FAISS/Milvus   │
                   │  └─ keyframes/   │  │  └─ features/    │  │ Collections     │
                   │     └─ video1/   │  │     └─ video1/   │  │                 │
                   │        └─ *.jpg  │  │        └─ 001/   │  │                 │
                   └─────────────────┘  │           └─*.npy │  └─────────────────┘
                                        └─────────────────┘           │
                                                                      ▼
┌──────────────┐   ┌─────────────┐   ┌──────────────┐    ┌─────────────────┐
│   Search     │◀──│  Web UI     │◀──│    SERVE     │◀───│   Indexing      │
│   Results    │   │  (React)    │   │  (FastAPI)   │    │   Complete      │
└──────────────┘   └─────────────┘   └──────────────┘    └─────────────────┘
```

### Workflow Steps

1. **ADD**: Import videos → Extract keyframes → Store in workspace
2. **ANALYSE**: Process keyframes → Extract CLIP + OCR features → Save features
3. **INDEX**: Load features → Build vector database → Create search indexes
4. **SERVE**: Start web server → Load search models → Provide search API

---

## Component Deep Dive

### 1. ADD Command - Video Ingestion

**Location**: `src/entry/cli/commands/add.py`

**Purpose**: Imports video files and extracts keyframes for analysis.

#### Under the Hood:

```python
# 1. Video Loading
def _load_video(self, video_path, do_move, do_overwrite, update_progress):
    video_id = video_path.stem  # Extract unique ID
    output_path = self._work_dir / "videos" / f"{video_id}.mp4"
    
    # Copy/Move video to workspace
    if do_move:
        shutil.move(video_path, output_path)
    else:
        shutil.copy(video_path, output_path)
```

```python
# 2. Keyframe Extraction
def _extract_keyframes(self, video_path, update_progress):
    # Get I-frames using FFprobe
    keyframes_list = get_keyframes_list(video_path)  
    max_scene_length = GlobalConfig.get("add", "max_scene_length", 25)
    
    # Extract frames using OpenCV
    cap = cv2.VideoCapture(str(video_path))
    frame_counter = 0
    scene_length = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Save frame if it's a keyframe OR scene is too long
        if scene_length >= max_scene_length or frame_counter in keyframes_list:
            cv2.imwrite(str(keyframe_dir / f"{frame_counter:06d}.jpg"), 
                       frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            scene_length = 0
        
        scene_length += 1
        frame_counter += 1
```

#### Key Algorithms:
- **Adaptive Keyframe Extraction**: Uses FFprobe to detect I-frames (scene changes)
- **Scene Length Limiting**: Ensures minimum frame density even in static scenes
- **Parallel Processing**: Multiple videos processed concurrently

#### Output Structure:
```
aic25_workspace/
├── videos/
│   ├── video1.mp4
│   └── video2.mp4
└── keyframes/
    ├── video1/
    │   ├── 000001.jpg
    │   ├── 000015.jpg
    │   └── 000032.jpg
    └── video2/
        ├── 000001.jpg
        └── 000028.jpg
```

---

### 2. ANALYSE Command - Feature Extraction

**Location**: `src/entry/cli/commands/analyse.py`

**Purpose**: Extracts multimodal features (visual + textual) from keyframes.

#### Under the Hood:

```python
# 1. Model Loading
def __call__(self, gpu, do_overwrite, verbose, *args, **kwargs):
    models = GlobalConfig.get("analyse", "features")
    
    for model_info in models:
        model_name = model_info["name"].lower()
        if model_name == "clip":
            model = CLIP(model_info["pretrained_model"])
        elif model_name == "ocr":
            model = TrOCR()  # EasyOCR wrapper
```

```python
# 2. Batch Processing
def _extract_features(self, model_name, model, video_id, batch_size, do_overwrite, update_progress):
    keyframe_files = self._get_keyframes_list(model_name, video_id, do_overwrite)
    
    # Extract features in batches
    features = model.get_image_features(
        keyframe_files, 
        batch_size,
        progress_callback
    )
    
    # Save features per frame
    for i, path in enumerate(keyframe_files):
        save_dir = features_dir / path.stem
        
        if isinstance(features[i], torch.Tensor):
            np.save(save_dir / f"{model_name}.npy", features[i])
        elif isinstance(features[i], str):
            with open(save_dir / f"{model_name}.txt", "w") as f:
                f.write(features[i])
        else:  # JSON data (OCR results)
            with open(save_dir / f"{model_name}.json", "w") as f:
                json.dump(features[i], f)
```

#### CLIP Feature Extraction:
```python
# src/services/analyse/features/clip.py
class CLIP:
    def get_image_features(self, image_paths, batch_size, progress_callback):
        features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess images
            images = [self.preprocess(Image.open(path)) for path in batch_paths]
            images = torch.stack(images).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model.encode_image(images)
                batch_features = F.normalize(batch_features, dim=-1)
            
            features.extend(batch_features.cpu())
            progress_callback(i // batch_size + 1, len(image_paths) // batch_size + 1)
        
        return features
```

#### OCR Feature Extraction:
```python
# src/services/analyse/features/trorc.py  
class TrOCR:
    def get_image_features(self, image_paths, batch_size, progress_callback):
        import easyocr
        reader = easyocr.Reader(['en', 'vi'])
        
        results = []
        for i, image_path in enumerate(image_paths):
            # Extract text with bounding boxes
            ocr_results = reader.readtext(str(image_path))
            
            # Format: [[bbox, text, confidence], ...]
            formatted_results = []
            for (bbox, text, conf) in ocr_results:
                if conf > self.confidence_threshold:
                    formatted_results.append([bbox, text, conf])
            
            results.append(formatted_results)
            progress_callback(i + 1, len(image_paths))
        
        return results
```

#### Output Structure:
```
aic25_workspace/
└── features/
    └── video1/
        ├── 000001/
        │   ├── clip.npy    # [512,] float32 vector
        │   └── ocr.json    # [[[bbox], text, conf], ...]
        ├── 000015/
        │   ├── clip.npy
        │   └── ocr.json
        └── 000032/
            ├── clip.npy
            └── ocr.json
```

---

### 3. INDEX Command - Database Population

**Location**: `src/entry/cli/commands/index.py`

**Purpose**: Loads extracted features and builds searchable vector database.

#### Under the Hood:

```python
# 1. Database Selection
def __call__(self, collection_name, database_type, do_overwrite, do_update, verbose, *args, **kwargs):
    # Use factory pattern to create database
    database = DatabaseFactory.create_database(
        database_type=database_type,  # "faiss" or "milvus"
        collection_name=collection_name,
        do_overwrite=do_overwrite,
        work_dir=str(self._work_dir)
    )
```

```python
# 2. Feature Loading & Processing
def _index_features(self, database, video_id, do_update, update_progress):
    features_dir = self._work_dir / "features" / video_id
    data_list = []
    
    # Process each frame's features
    for frame_path in features_dir.glob("*/"):
        frame_id = frame_path.stem
        data = {"frame_id": f"{video_id}#{frame_id}"}
        
        # Load all feature types for this frame
        for feature_path in frame_path.glob("*"):
            if feature_path.suffix == ".npy":
                feature = np.load(feature_path)
                data[feature_path.stem] = feature.tolist()  # CLIP embeddings
            elif feature_path.suffix == ".txt":
                with open(feature_path, "r") as f:
                    data[feature_path.stem] = f.read().lower()
            elif feature_path.suffix == ".json":
                with open(feature_path, "r") as f:
                    data[feature_path.stem] = json.load(f)  # OCR results
        
        data_list.append(data)
    
    # Batch insert into database
    database.insert(data_list, do_update)
```

#### FAISS Database Implementation:
```python
# src/services/index/faiss.py
class FAISSDatabase:
    def insert(self, data, do_update=False):
        vectors = []
        new_metadata = {}
        
        for item in data:
            frame_id = item["frame_id"]
            
            # Extract CLIP vector
            vector = np.array(item["clip"], dtype=np.float32)
            vectors.append(vector)
            new_metadata[frame_id] = item
        
        vectors = np.array(vectors, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Train index if needed
        if not self.index.is_trained:
            self.index.train(vectors)
        
        # Add to index
        start_id = self.index.ntotal
        self.index.add(vectors)
        
        # Update metadata mappings
        for i, frame_id in enumerate(new_metadata.keys()):
            faiss_id = start_id + i
            self.metadata[frame_id] = new_metadata[frame_id]
            self.id_to_frame[faiss_id] = frame_id
            self.frame_to_id[frame_id] = faiss_id
        
        self._save_database()
```

#### Milvus Database Implementation:
```python
# src/services/index/milvus.py
class MilvusDatabase:
    def insert(self, data, do_update=False):
        # Convert data to Milvus format
        entities = []
        for item in data:
            entities.append({
                "frame_id": item["frame_id"],
                "clip": item["clip"],  # Vector field
                "ocr": item["ocr"]     # JSON field
            })
        
        # Insert with automatic ID assignment
        insert_result = self._client.insert(
            collection_name=self._collection_name,
            data=entities
        )
        
        return insert_result
```

---

### 4. SERVE Command - Web Application

**Location**: `src/entry/cli/commands/serve.py`

**Purpose**: Starts the web server with search API and React UI.

#### Under the Hood:

```python
# 1. Database Configuration
def __call__(self, port, dev_mode, workers, database_type=None, *args, **kwargs):
    db_type = database_type or GlobalConfig.get("webui", "database") or "faiss"
    
    # Start Milvus server only if using Milvus
    if db_type.lower() == "milvus":
        MilvusDatabase.start_server()
    
    # Set database type for web app
    if database_type:
        GlobalConfig.set(database_type, "webui", "database")
```

```python
# 2. Frontend Building
def _build_frontend(self, port):
    # Build React application
    build_env = os.environ.copy()
    build_env["VITE_PORT"] = str(port)
    
    subprocess.run([
        "npm", "run", "build"
    ], env=build_env, cwd=frontend_dir)
    
    # Move built files to workspace
    web_dir = self._work_dir / ".web"
    built_dir.rename(web_dir / "dist")
```

#### FastAPI Backend:
```python
# src/entry/web/app.py
from services.search import SearchFactory

app = FastAPI()

@app.get("/search")
async def search(
    q: str,
    offset: int = 0,
    limit: int = 50,
    collection_name: str = "default_collection"
):
    # Create searcher based on configured database
    searcher = SearchFactory.create_searcher(collection_name)
    
    # Perform search
    results = searcher.search(
        q=q,
        offset=offset,
        limit=limit,
        nprobe=GlobalConfig.get("webui", "search", "default_nprobe", 16)
    )
    
    return results
```

---

## Database Layer Architecture

### Factory Pattern Implementation

```python
# src/services/index/database_factory.py
class DatabaseFactory:
    @staticmethod
    def create_database(database_type=None, collection_name="default", do_overwrite=False, work_dir=None):
        if database_type is None:
            database_type = GlobalConfig.get("webui", "database") or "faiss"
        
        if database_type.lower() == "faiss":
            return FAISSDatabase(collection_name, do_overwrite, work_dir)
        elif database_type.lower() == "milvus":
            return MilvusDatabase(collection_name, do_overwrite)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
```

### FAISS Implementation Details

```python
class FAISSDatabase:
    def _create_index(self, index_type: str):
        if index_type == "Flat":
            # Exact search - best for < 10k vectors
            self.index = faiss.IndexFlatIP(self.dimension)
            
        elif index_type == "IVF":
            # Inverted File - good balance for 10k-1M vectors
            nlist = GlobalConfig.get("faiss", "nlist") or 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        elif index_type == "HNSW":
            # Hierarchical NSW - fastest for > 100k vectors
            M = GlobalConfig.get("faiss", "M") or 16
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            
        elif index_type == "PQ":
            # Product Quantization - memory efficient
            m = GlobalConfig.get("faiss", "m") or 8
            self.index = faiss.IndexPQ(self.dimension, m, 8)
    
    def search(self, query_vectors, filter_func=None, offset=0, limit=50, nprobe=8):
        # Normalize query vectors
        query_vectors = np.array(query_vectors, dtype=np.float32)
        faiss.normalize_L2(query_vectors)
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        # Perform search
        k = min(offset + limit, self.index.ntotal)
        distances, indices = self.index.search(query_vectors, k)
        
        # Convert to SearchResult objects
        results = []
        for i in range(len(query_vectors)):
            query_results = []
            for j in range(len(indices[i])):
                if indices[i][j] == -1: break
                
                faiss_id = indices[i][j]
                frame_id = self.id_to_frame.get(faiss_id)
                entity = self.metadata[frame_id]
                
                # Apply filter if provided
                if filter_func and not filter_func(entity):
                    continue
                
                result = SearchResult(
                    frame_id=frame_id,
                    distance=float(distances[i][j]),
                    entity=entity
                )
                query_results.append(result)
            
            results.append(query_results[offset:offset + limit])
        
        return results
```

---

## Search Layer Architecture

### Unified Search Interface

```python
# src/services/search/search_factory.py
class SearchFactory:
    @staticmethod
    def create_searcher(collection_name, database_type=None):
        if database_type is None:
            database_type = GlobalConfig.get("webui", "database") or "faiss"
        
        if database_type.lower() == "faiss":
            return FAISSSearcher(collection_name)
        elif database_type.lower() == "milvus":
            return MilvusSearcher(collection_name)
```

### Advanced Search Features

```python
# src/services/search/faiss_searcher.py
class Searcher(BaseSearch):
    def search(self, q, param_filter="", offset=0, limit=50, nprobe=8, model="clip", 
               temporal_k=10000, ocr_weight=1.0, ocr_threshold=40, max_interval=250):
        
        # 1. Query Processing
        processed = self._process_query(q)  # Parse video: and OCR: filters
        
        # 2. Search Strategy Selection
        no_query = all([len(x) == 0 for x in processed["queries"]])
        no_advance = all([len(x) == 0 for x in processed["advance"]])
        
        if no_query and no_advance:
            # Video browsing mode
            return self._get_videos(processed["video_ids"], offset, limit, selected)
            
        elif len(processed["queries"]) == 1 and no_advance:
            # Simple text search
            return self._simple_search(processed, param_filter, offset, limit, nprobe, model)
            
        else:
            # Complex multimodal search with temporal consistency
            return self._complex_search(
                processed, param_filter, offset, limit, nprobe, model,
                temporal_k, ocr_weight, ocr_threshold, max_interval
            )
    
    def _complex_search(self, processed, filter_expr, offset, limit, nprobe, model, 
                       temporal_k, ocr_weight, ocr_threshold, max_interval):
        # 1. CLIP-based vector search
        text_features = self._models[model].get_text_features(processed["queries"]).tolist()
        results = self._database.search(text_features, filter_func, 0, temporal_k, nprobe)
        
        # 2. OCR post-processing
        for i in range(len(processed["queries"])):
            results[i] = self._process_advance(
                processed["advance"][i], results[i], ocr_weight, ocr_threshold
            )
        
        # 3. Temporal consistency fusion
        combined_results = self._combine_temporal_results(results, temporal_k, max_interval)
        
        return combined_results[offset:offset + limit]
```

#### OCR Processing Algorithm:
```python
def _process_advance(self, advance_query, result, ocr_weight, ocr_threshold):
    if "ocr" not in advance_query:
        return result
    
    query_ocr = advance_query["ocr"]
    res = deepcopy(result)
    
    for i, record in enumerate(result):
        ocr_distance = 0
        
        # Process each OCR query term
        for query_text in query_ocr:
            ocr = record.entity.get("ocr", [])
            ocr_text_distance = 0
            cnt = 0
            
            # Check against detected text in frame
            for text_item in ocr:
                text_content = text_item[-2] if isinstance(text_item, list) else str(text_item)
                
                # Fuzzy string matching
                partial_ratio = fuzz.partial_ratio(query_text.lower(), text_content.lower())
                
                if partial_ratio > ocr_threshold:
                    cnt += 1
                    ocr_text_distance += partial_ratio / 100
            
            if cnt > 0:
                ocr_text_distance /= cnt
            ocr_distance += ocr_text_distance
        
        if len(query_ocr) > 0:
            ocr_distance /= len(query_ocr)
        
        # Combine CLIP and OCR scores
        res[i].distance = (record.distance + ocr_distance * ocr_weight) / (1 + ocr_weight)
    
    return sorted(res, key=lambda x: x.distance, reverse=True)
```

#### Temporal Consistency Algorithm:
```python
def _combine_temporal_results(self, results, temporal_k, max_interval):
    # Parse frame IDs to extract video and frame numbers
    for res_list in results:
        for item in res_list:
            frame_id = item.frame_id
            video_id, frame_num = frame_id.split("#")
            video_id = int(video_id.replace("L", "").replace("_V", ""))
            item._id = (video_id, int(frame_num))
    
    # Iteratively combine results with temporal windows
    best = None
    for res in results[::-1]:
        if best is None:
            best = res[:temporal_k]
            continue
        
        # Sort by temporal ID for efficient window processing
        res = sorted(res, key=lambda x: x._id)
        best = sorted(best, key=lambda x: x._id)
        
        # Sliding window combination
        combined = []
        for cur in res:
            cur_vid, cur_frame = cur._id
            
            # Find temporal neighbors in previous results
            for best_item in best:
                best_vid, best_frame = best_item._id
                
                if (best_vid == cur_vid and 
                    abs(best_frame - cur_frame) <= max_interval):
                    # Combine scores for temporally consistent results
                    new_item = deepcopy(cur)
                    new_item.distance = cur.distance + best_item.distance
                    combined.append(new_item)
        
        # Keep only highest scoring items per unique frame
        best = self._deduplicate_by_frame(combined)[:temporal_k]
    
    return best
```

---

## Configuration System

### Hierarchical Configuration Loading

```python
# src/config/loader.py
class GlobalConfig:
    def _load_config(cls):
        config_path = cls._work_dir / "layout/config.yaml"
        
        if not config_path.exists():
            cls._create_default_config(config_path)
        
        with open(config_path, "r", encoding="utf-8") as f:
            cls._config = yaml.safe_load(f)
    
    def get(cls, *args, default=None):
        # Dot notation access: GlobalConfig.get("webui", "database")
        result = cls._config
        for arg in args:
            if isinstance(result, dict) and arg in result:
                result = result[arg]
            else:
                return default
        return result
```

### Performance Tuning Configuration

```yaml
# Performance-based FAISS configuration
faiss:
  # Dataset size < 10k vectors
  index_type: "Flat"
  
  # Dataset size 10k-1M vectors  
  index_type: "IVF"
  nlist: 128
  nprobe: 8
  
  # Dataset size > 1M vectors
  index_type: "HNSW"
  M: 32
  efConstruction: 400
  efSearch: 200
  
  # Memory-constrained environments
  index_type: "PQ"
  m: 8
  nbits: 8

# Hardware optimization
hardware:
  gpu:
    enabled: true
    mixed_precision: true  # FP16 for 2x speed
  cpu:
    num_threads: -1       # Use all cores
  memory:
    prefetch_factor: 4    # Overlap I/O with compute
```

---

## Performance Considerations

### Parallel Processing Strategy

1. **Video Processing**: Multi-threaded keyframe extraction
2. **Feature Extraction**: GPU batching with optimized batch sizes
3. **Database Operations**: Batch insertions and parallel queries
4. **Search**: Asynchronous request handling

### Memory Optimization

1. **Streaming Processing**: Process videos in chunks to avoid memory overflow
2. **Feature Caching**: LRU cache for frequently accessed features
3. **Database Connection Pooling**: Reuse connections for better performance
4. **Lazy Loading**: Load features only when needed

### Scalability Features

1. **Configurable Workers**: Adjust parallelism based on hardware
2. **Database Selection**: Choose optimal database for dataset size
3. **Index Optimization**: Automatic index type selection based on data size
4. **Batch Size Tuning**: Dynamic batch size adjustment for GPU memory

---

This comprehensive workflow and architecture guide provides deep insight into how each component works under the hood, enabling effective development, debugging, and optimization of the AIC25 multimedia retrieval system.