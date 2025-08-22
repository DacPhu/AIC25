# Component Extension Guide - Enhancing Search Effectiveness

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Feature Extraction Components](#feature-extraction-components)
3. [Search Algorithm Components](#search-algorithm-components)
4. [Database Backend Components](#database-backend-components)
5. [UI/Frontend Components](#uifrontend-components)
6. [Integration Best Practices](#integration-best-practices)
7. [Performance Optimization](#performance-optimization)
8. [Practical Examples](#practical-examples)

---

## Architecture Overview

The AIC25 system is designed with extensibility in mind. Here are the main extension points:

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTENSIBILITY POINTS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│ │   EXTRACTORS    │  │   ALGORITHMS    │  │   BACKENDS    │ │
│ │                 │  │                 │  │               │ │
│ │ • CLIP          │  │ • Vector Search │  │ • FAISS       │ │
│ │ • OCR           │  │ • Fuzzy Match   │  │ • Milvus      │ │
│ │ • Audio         │  │ • Temporal      │  │ • Pinecone    │ │
│ │ • Objects       │  │ • Reranking     │  │ • Weaviate    │ │
│ │ • Faces         │  │ • Fusion        │  │ • Custom      │ │
│ └─────────────────┘  └─────────────────┘  └───────────────┘ │
│                                                             │
│ ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│ │   UI COMPONENTS │  │   MIDDLEWARE    │  │   CONFIGS     │ │
│ │                 │  │                 │  │               │ │
│ │ • Search Forms  │  │ • Caching       │  │ • Models      │ │
│ │ • Result Cards  │  │ • Rate Limiting │  │ • Parameters  │ │
│ │ • Filters       │  │ • Auth          │  │ • Thresholds  │ │
│ │ • Analytics     │  │ • Monitoring    │  │ • Hardware    │ │
│ └─────────────────┘  └─────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature Extraction Components

### 1. Adding New Feature Extractors

The system uses a plugin-based architecture for feature extractors. Here's how to add new ones:

#### Base Extractor Interface

```python
# src/services/analyse/features/base_extractor.py
from abc import ABC, abstractmethod
from typing import List, Any, Union, Callable
from pathlib import Path

class BaseFeatureExtractor(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def get_image_features(self, image_paths: List[Path], batch_size: int, 
                          progress_callback: Callable = None) -> List[Any]:
        """Extract features from images"""
        pass
    
    @abstractmethod
    def get_text_features(self, texts: List[str]) -> Any:
        """Extract features from text (for search queries)"""
        pass
    
    def to(self, device: str):
        """Move model to device"""
        self.device = device
        return self
```

#### Example 1: Audio Feature Extractor

```python
# src/services/analyse/features/audio.py
import librosa
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class AudioExtractor(BaseFeatureExtractor):
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        model_name = self.config.get("model_name", "facebook/wav2vec2-base-960h")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
    
    def get_image_features(self, image_paths: List[Path], batch_size: int, 
                          progress_callback: Callable = None) -> List[Any]:
        """Extract audio from video frames' corresponding time segments"""
        features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = []
            
            for img_path in batch_paths:
                # Extract corresponding video segment
                video_path = self._get_video_path(img_path)
                frame_time = self._get_frame_time(img_path)
                
                # Extract audio segment (±2 seconds around frame)
                audio, sr = librosa.load(str(video_path), 
                                       offset=max(0, frame_time - 2),
                                       duration=4, sr=16000)
                
                # Process with Wav2Vec2
                inputs = self.processor(audio, sampling_rate=16000, 
                                      return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of hidden states
                    audio_features = outputs.last_hidden_state.mean(dim=1)
                    audio_features = F.normalize(audio_features, dim=-1)
                
                batch_features.append(audio_features.cpu().numpy().flatten())
            
            features.extend(batch_features)
            
            if progress_callback:
                progress_callback(i // batch_size + 1, 
                                len(image_paths) // batch_size + 1)
        
        return features
    
    def get_text_features(self, texts: List[str]) -> Any:
        """Convert text queries to comparable audio features"""
        # This could use a text-to-audio embedding model
        # or semantic similarity with audio transcriptions
        pass
    
    def _get_video_path(self, img_path: Path) -> Path:
        """Get video path from image path"""
        video_id = img_path.parent.name
        return img_path.parent.parent.parent / "videos" / f"{video_id}.mp4"
    
    def _get_frame_time(self, img_path: Path) -> float:
        """Calculate frame timestamp from filename"""
        frame_num = int(img_path.stem)
        fps = 30  # Could be extracted from video metadata
        return frame_num / fps
```

#### Example 2: Object Detection Extractor

```python
# src/services/analyse/features/objects.py
import torch
from transformers import DETRImageProcessor, DETRForObjectDetection
from PIL import Image

class ObjectExtractor(BaseFeatureExtractor):
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        model_name = self.config.get("model_name", "facebook/detr-resnet-50")
        self.processor = DETRImageProcessor.from_pretrained(model_name)
        self.model = DETRForObjectDetection.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
    
    def get_image_features(self, image_paths: List[Path], batch_size: int, 
                          progress_callback: Callable = None) -> List[Any]:
        features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load images
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            
            # Process batch
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process detections
            batch_features = []
            for j, image in enumerate(images):
                # Extract bounding boxes, labels, and scores
                target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
                results = self.processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
                )[j]
                
                # Format detections
                detections = []
                for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
                    detection = {
                        "bbox": box.cpu().numpy().tolist(),
                        "label": self.model.config.id2label[label.item()],
                        "confidence": score.item()
                    }
                    detections.append(detection)
                
                batch_features.append(detections)
            
            features.extend(batch_features)
            
            if progress_callback:
                progress_callback(i // batch_size + 1, 
                                len(image_paths) // batch_size + 1)
        
        return features
    
    def get_text_features(self, texts: List[str]) -> Any:
        """Match text queries with detected object labels"""
        # Use semantic similarity or exact matching
        return texts  # Simplified - would use embeddings in practice
```

#### Registering New Extractors

```python
# src/services/analyse/features/__init__.py
from .clip import CLIP
from .trorc import TrOCR
from .audio import AudioExtractor
from .objects import ObjectExtractor

# Extractor registry
EXTRACTORS = {
    "clip": CLIP,
    "ocr": TrOCR,
    "audio": AudioExtractor,
    "objects": ObjectExtractor,
}

def get_extractor(name: str, config: dict = None):
    if name.lower() not in EXTRACTORS:
        raise ValueError(f"Unknown extractor: {name}")
    return EXTRACTORS[name.lower()](config)
```

#### Configuration Update

```yaml
# aic25_workspace/layout/config.yaml
analyse:
  features:
    - name: "clip"
      pretrained_model: "openai/clip-vit-base-patch16"
      batch_size: 16
      
    - name: "ocr"
      batch_size: 8
      confidence_threshold: 0.7
      languages: ["vi", "en"]
      
    - name: "audio"  # NEW
      model_name: "facebook/wav2vec2-base-960h"
      batch_size: 4
      
    - name: "objects"  # NEW
      model_name: "facebook/detr-resnet-50"
      batch_size: 8
      confidence_threshold: 0.7
```

---

## Search Algorithm Components

### 1. Adding Advanced Search Algorithms

#### Query Expansion Component

```python
# src/services/search/components/query_expansion.py
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QueryExpansionComponent:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = SentenceTransformer(
            self.config.get("model", "all-MiniLM-L6-v2")
        )
        self.expansion_ratio = self.config.get("expansion_ratio", 0.3)
        self.synonyms_db = self._load_synonyms()
    
    def expand_query(self, query: str, corpus_texts: List[str]) -> List[str]:
        """Expand query using semantic similarity and synonyms"""
        expanded_terms = [query]
        
        # 1. Synonym expansion
        synonyms = self._get_synonyms(query)
        expanded_terms.extend(synonyms)
        
        # 2. Semantic expansion using corpus
        if corpus_texts:
            semantic_terms = self._semantic_expansion(query, corpus_texts)
            expanded_terms.extend(semantic_terms)
        
        return list(set(expanded_terms))
    
    def _semantic_expansion(self, query: str, corpus_texts: List[str]) -> List[str]:
        # Encode query and corpus
        query_embedding = self.model.encode([query])
        corpus_embeddings = self.model.encode(corpus_texts)
        
        # Find most similar texts
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5
        
        # Extract expansion terms
        expansion_terms = []
        for idx in top_indices:
            if similarities[idx] > 0.6:  # Threshold
                expansion_terms.append(corpus_texts[idx])
        
        return expansion_terms[:int(len(expansion_terms) * self.expansion_ratio)]
    
    def _get_synonyms(self, query: str) -> List[str]:
        # Load from synonyms database or use WordNet
        return self.synonyms_db.get(query.lower(), [])
    
    def _load_synonyms(self) -> dict:
        # Load synonym dictionary
        return {
            "car": ["automobile", "vehicle", "sedan"],
            "person": ["human", "individual", "people"],
            # ... more synonyms
        }
```

#### Pseudo Relevance Feedback

```python
# src/services/search/components/prf.py
class PseudoRelevanceFeedbackComponent:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.top_k = self.config.get("top_k", 5)
        self.expansion_terms = self.config.get("expansion_terms", 10)
        self.weight = self.config.get("weight", 0.3)
    
    def refine_query(self, original_query: str, initial_results: List[dict], 
                    feature_extractor) -> str:
        """Refine query based on top results"""
        
        # Extract features from top results
        top_results = initial_results[:self.top_k]
        relevant_features = []
        
        for result in top_results:
            # Extract text content from result
            text_content = self._extract_text_content(result)
            if text_content:
                relevant_features.append(text_content)
        
        # Generate expansion terms
        expansion_terms = self._generate_expansion_terms(relevant_features)
        
        # Combine with original query
        refined_query = f"{original_query} {' '.join(expansion_terms[:self.expansion_terms])}"
        
        return refined_query
    
    def _extract_text_content(self, result: dict) -> str:
        """Extract searchable text from result"""
        content_parts = []
        
        # Extract OCR text
        if "ocr" in result.get("entity", {}):
            ocr_results = result["entity"]["ocr"]
            for ocr_item in ocr_results:
                if isinstance(ocr_item, list) and len(ocr_item) > 1:
                    content_parts.append(ocr_item[1])  # Text part
        
        # Extract object labels
        if "objects" in result.get("entity", {}):
            objects = result["entity"]["objects"]
            for obj in objects:
                if "label" in obj:
                    content_parts.append(obj["label"])
        
        return " ".join(content_parts)
    
    def _generate_expansion_terms(self, relevant_features: List[str]) -> List[str]:
        """Generate expansion terms from relevant documents"""
        from collections import Counter
        import re
        
        # Tokenize and count terms
        all_terms = []
        for text in relevant_features:
            terms = re.findall(r'\b\w+\b', text.lower())
            all_terms.extend(terms)
        
        # Get most frequent terms (excluding stopwords)
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        filtered_terms = [term for term in all_terms if term not in stopwords and len(term) > 2]
        
        term_counts = Counter(filtered_terms)
        expansion_terms = [term for term, count in term_counts.most_common(self.expansion_terms)]
        
        return expansion_terms
```

#### Multi-Stage Reranking

```python
# src/services/search/components/reranker.py
from sentence_transformers import CrossEncoder

class RerankingComponent:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model = CrossEncoder(
            self.config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self.enabled = self.config.get("enabled", False)
        self.top_k = self.config.get("top_k", 100)
    
    def rerank(self, query: str, results: List[dict]) -> List[dict]:
        """Rerank results using cross-encoder"""
        if not self.enabled or len(results) == 0:
            return results
        
        # Prepare query-document pairs
        query_doc_pairs = []
        for result in results[:self.top_k]:
            doc_text = self._extract_document_text(result)
            query_doc_pairs.append([query, doc_text])
        
        # Get reranking scores
        scores = self.model.predict(query_doc_pairs)
        
        # Update results with new scores
        reranked_results = []
        for i, result in enumerate(results[:self.top_k]):
            result_copy = result.copy()
            result_copy["rerank_score"] = float(scores[i])
            # Combine original distance with rerank score
            result_copy["final_score"] = (
                result["distance"] * 0.7 + scores[i] * 0.3
            )
            reranked_results.append(result_copy)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Add remaining results
        reranked_results.extend(results[self.top_k:])
        
        return reranked_results
    
    def _extract_document_text(self, result: dict) -> str:
        """Extract text representation of document for reranking"""
        text_parts = []
        
        entity = result.get("entity", {})
        
        # Add OCR text
        if "ocr" in entity:
            for ocr_item in entity["ocr"]:
                if isinstance(ocr_item, list) and len(ocr_item) > 1:
                    text_parts.append(ocr_item[1])
        
        # Add object labels
        if "objects" in entity:
            for obj in entity["objects"]:
                text_parts.append(obj.get("label", ""))
        
        # Add frame context
        text_parts.append(f"Frame {result.get('frame_id', '')}")
        
        return " ".join(filter(None, text_parts))
```

### 2. Integrating Search Components

```python
# src/services/search/enhanced_searcher.py
class EnhancedSearcher:
    def __init__(self, collection_name: str, database_type: str = None):
        self.base_searcher = SearchFactory.create_searcher(collection_name, database_type)
        
        # Initialize components based on config
        self.query_expansion = QueryExpansionComponent(
            GlobalConfig.get("advanced", "query_expansion")
        ) if GlobalConfig.get("advanced", "query_expansion", "enabled") else None
        
        self.prf = PseudoRelevanceFeedbackComponent(
            GlobalConfig.get("advanced", "prf")
        ) if GlobalConfig.get("advanced", "prf", "enabled") else None
        
        self.reranker = RerankingComponent(
            GlobalConfig.get("webui", "search", "rerank")
        ) if GlobalConfig.get("webui", "search", "rerank", "enabled") else None
    
    def search(self, query: str, **kwargs) -> dict:
        """Enhanced search with multiple components"""
        
        # 1. Query expansion
        if self.query_expansion:
            # Get corpus for semantic expansion (could be cached)
            corpus_texts = self._get_corpus_sample()
            expanded_queries = self.query_expansion.expand_query(query, corpus_texts)
            
            # Search with expanded queries
            all_results = []
            for exp_query in expanded_queries:
                results = self.base_searcher.search(exp_query, **kwargs)
                all_results.extend(results.get("results", []))
            
            # Deduplicate and merge
            unique_results = self._deduplicate_results(all_results)
        else:
            # Standard search
            search_result = self.base_searcher.search(query, **kwargs)
            unique_results = search_result.get("results", [])
        
        # 2. Pseudo Relevance Feedback (if enabled and we have results)
        if self.prf and len(unique_results) >= self.prf.top_k:
            refined_query = self.prf.refine_query(
                query, unique_results, self.base_searcher._models.get("clip")
            )
            
            # Re-search with refined query
            refined_results = self.base_searcher.search(refined_query, **kwargs)
            
            # Combine original and refined results
            combined_results = self._combine_results(
                unique_results, refined_results.get("results", [])
            )
        else:
            combined_results = unique_results
        
        # 3. Reranking
        if self.reranker:
            combined_results = self.reranker.rerank(query, combined_results)
        
        return {
            "results": combined_results,
            "total": len(combined_results),
            "offset": kwargs.get("offset", 0),
            "enhanced": True
        }
    
    def _get_corpus_sample(self) -> List[str]:
        """Get sample texts from corpus for query expansion"""
        # This could be cached or pre-computed
        return []
    
    def _deduplicate_results(self, results: List[dict]) -> List[dict]:
        """Remove duplicate results based on frame_id"""
        seen_frames = set()
        unique_results = []
        
        for result in results:
            frame_id = result.get("frame_id") or result.get("entity", {}).get("frame_id")
            if frame_id not in seen_frames:
                seen_frames.add(frame_id)
                unique_results.append(result)
        
        return unique_results
    
    def _combine_results(self, original: List[dict], refined: List[dict]) -> List[dict]:
        """Combine original and refined results with weighted scoring"""
        # Implementation for combining results
        combined = {}
        
        # Add original results with weight 0.7
        for result in original:
            frame_id = result.get("frame_id")
            combined[frame_id] = result.copy()
            combined[frame_id]["combined_score"] = result["distance"] * 0.7
        
        # Add refined results with weight 0.3
        for result in refined:
            frame_id = result.get("frame_id")
            if frame_id in combined:
                combined[frame_id]["combined_score"] += result["distance"] * 0.3
            else:
                result_copy = result.copy()
                result_copy["combined_score"] = result["distance"] * 0.3
                combined[frame_id] = result_copy
        
        # Sort by combined score
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return final_results
```

---

## Database Backend Components

### 1. Adding New Database Backends

#### Example: Pinecone Backend

```python
# src/services/index/pinecone_db.py
import pinecone
from typing import List, Dict, Any, Optional
from entities.search import SearchResult

class PineconeDatabase:
    def __init__(self, collection_name: str, do_overwrite: bool = False):
        self.collection_name = collection_name
        
        # Initialize Pinecone
        api_key = GlobalConfig.get("pinecone", "api_key")
        environment = GlobalConfig.get("pinecone", "environment")
        
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create or connect to index
        dimension = GlobalConfig.get("pinecone", "dimension", 512)
        metric = GlobalConfig.get("pinecone", "metric", "cosine")
        
        if collection_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=collection_name,
                dimension=dimension,
                metric=metric
            )
        
        self.index = pinecone.Index(collection_name)
        
        if do_overwrite:
            self.index.delete(delete_all=True)
    
    def insert(self, data: List[Dict], do_update: bool = False):
        """Insert data into Pinecone"""
        vectors = []
        
        for item in data:
            vector_data = {
                "id": item["frame_id"],
                "values": item["clip"],  # Vector values
                "metadata": {
                    k: v for k, v in item.items() 
                    if k != "clip"  # Exclude vector from metadata
                }
            }
            vectors.append(vector_data)
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query_vectors: List[List[float]], 
               filter_func=None, offset: int = 0, 
               limit: int = 50, **kwargs) -> List[List[SearchResult]]:
        """Search similar vectors in Pinecone"""
        results = []
        
        for query_vector in query_vectors:
            # Build filter from filter_func if provided
            pinecone_filter = self._build_pinecone_filter(filter_func)
            
            # Search
            search_results = self.index.query(
                vector=query_vector,
                top_k=offset + limit,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Convert to SearchResult objects
            query_results = []
            for match in search_results["matches"][offset:offset + limit]:
                result = SearchResult(
                    frame_id=match["id"],
                    distance=float(match["score"]),
                    entity=match["metadata"]
                )
                query_results.append(result)
            
            results.append(query_results)
        
        return results
    
    def get(self, frame_id: str) -> Optional[Dict]:
        """Get entity by frame_id"""
        try:
            result = self.index.fetch(ids=[frame_id])
            if frame_id in result["vectors"]:
                return result["vectors"][frame_id]["metadata"]
        except:
            pass
        return None
    
    def get_total(self) -> int:
        """Get total number of vectors"""
        return self.index.describe_index_stats()["total_vector_count"]
    
    def _build_pinecone_filter(self, filter_func) -> Optional[Dict]:
        """Convert filter function to Pinecone filter format"""
        if not filter_func:
            return None
        
        # This is a simplified example - you'd need to implement
        # proper filter translation based on your filter functions
        return {}
```

#### Example: Weaviate Backend

```python
# src/services/index/weaviate_db.py
import weaviate
from typing import List, Dict, Any, Optional

class WeaviateDatabase:
    def __init__(self, collection_name: str, do_overwrite: bool = False):
        self.collection_name = collection_name.capitalize()  # Weaviate class names are capitalized
        
        # Connect to Weaviate
        weaviate_url = GlobalConfig.get("weaviate", "url", "http://localhost:8080")
        self.client = weaviate.Client(weaviate_url)
        
        # Create schema if it doesn't exist
        if not self.client.schema.exists(self.collection_name):
            self._create_schema()
        elif do_overwrite:
            self.client.schema.delete_class(self.collection_name)
            self._create_schema()
    
    def _create_schema(self):
        """Create Weaviate schema"""
        schema = {
            "class": self.collection_name,
            "description": "Video frame features",
            "properties": [
                {
                    "name": "frame_id",
                    "dataType": ["string"],
                    "description": "Unique frame identifier"
                },
                {
                    "name": "ocr_text",
                    "dataType": ["text"],
                    "description": "OCR extracted text"
                },
                {
                    "name": "objects",
                    "dataType": ["string[]"],
                    "description": "Detected objects"
                }
            ],
            "vectorizer": "none"  # We'll provide our own vectors
        }
        
        self.client.schema.create_class(schema)
    
    def insert(self, data: List[Dict], do_update: bool = False):
        """Insert data into Weaviate"""
        with self.client.batch as batch:
            batch.batch_size = 100
            
            for item in data:
                # Prepare properties
                properties = {
                    "frame_id": item["frame_id"],
                    "ocr_text": self._extract_ocr_text(item.get("ocr", [])),
                    "objects": self._extract_object_labels(item.get("objects", []))
                }
                
                # Add to batch with vector
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.collection_name,
                    vector=item["clip"]  # CLIP embedding
                )
    
    def search(self, query_vectors: List[List[float]], 
               filter_func=None, offset: int = 0, 
               limit: int = 50, **kwargs) -> List[List[SearchResult]]:
        """Search in Weaviate"""
        results = []
        
        for query_vector in query_vectors:
            # Build GraphQL query
            near_vector = {"vector": query_vector}
            
            query_result = (
                self.client.query
                .get(self.collection_name, ["frame_id", "ocr_text", "objects"])
                .with_near_vector(near_vector)
                .with_limit(offset + limit)
                .with_additional(["distance"])
                .do()
            )
            
            # Convert results
            query_results = []
            if "data" in query_result and "Get" in query_result["data"]:
                items = query_result["data"]["Get"][self.collection_name][offset:offset + limit]
                
                for item in items:
                    result = SearchResult(
                        frame_id=item["frame_id"],
                        distance=float(item["_additional"]["distance"]),
                        entity={
                            "frame_id": item["frame_id"],
                            "ocr_text": item["ocr_text"],
                            "objects": item["objects"]
                        }
                    )
                    query_results.append(result)
            
            results.append(query_results)
        
        return results
    
    def _extract_ocr_text(self, ocr_data: List) -> str:
        """Extract text from OCR data"""
        texts = []
        for item in ocr_data:
            if isinstance(item, list) and len(item) > 1:
                texts.append(item[1])  # Text part
        return " ".join(texts)
    
    def _extract_object_labels(self, objects_data: List) -> List[str]:
        """Extract object labels"""
        labels = []
        for obj in objects_data:
            if isinstance(obj, dict) and "label" in obj:
                labels.append(obj["label"])
        return labels
```

#### Updating Database Factory

```python
# src/services/index/database_factory.py
from .pinecone_db import PineconeDatabase
from .weaviate_db import WeaviateDatabase

class DatabaseFactory:
    @staticmethod
    def create_database(database_type=None, collection_name="default", 
                       do_overwrite=False, work_dir=None):
        if database_type is None:
            database_type = GlobalConfig.get("webui", "database") or "faiss"
        
        database_type = database_type.lower()
        
        if database_type == "faiss":
            return FAISSDatabase(collection_name, do_overwrite, work_dir)
        elif database_type == "milvus":
            return MilvusDatabase(collection_name, do_overwrite)
        elif database_type == "pinecone":
            return PineconeDatabase(collection_name, do_overwrite)
        elif database_type == "weaviate":
            return WeaviateDatabase(collection_name, do_overwrite)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
    
    @staticmethod
    def get_supported_databases():
        return ["faiss", "milvus", "pinecone", "weaviate"]
```

---

## UI/Frontend Components

### 1. Advanced Search Interface Components

#### Multi-Modal Search Form

```jsx
// src/entry/web/view/src/components/AdvancedSearchForm.jsx
import React, { useState } from 'react';
import { Upload, Mic, Image, Type } from 'lucide-react';

const AdvancedSearchForm = ({ onSearch }) => {
  const [searchMode, setSearchMode] = useState('text');
  const [textQuery, setTextQuery] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [filters, setFilters] = useState({
    videoIds: [],
    objects: [],
    confidence: 0.5,
    timeRange: [0, 100]
  });

  const handleSearch = () => {
    const searchParams = {
      mode: searchMode,
      query: textQuery,
      filters: filters
    };

    if (searchMode === 'image' && imageFile) {
      searchParams.imageFile = imageFile;
    }
    if (searchMode === 'audio' && audioFile) {
      searchParams.audioFile = audioFile;
    }

    onSearch(searchParams);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <div className="flex flex-wrap gap-4 mb-6">
        {/* Search Mode Tabs */}
        <div className="flex bg-gray-100 rounded-lg p-1">
          {[
            { id: 'text', label: 'Text', icon: Type },
            { id: 'image', label: 'Image', icon: Image },
            { id: 'audio', label: 'Audio', icon: Mic }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setSearchMode(id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md transition-colors ${
                searchMode === id
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:bg-gray-200'
              }`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Search Input Area */}
      <div className="mb-6">
        {searchMode === 'text' && (
          <div>
            <input
              type="text"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              placeholder="Enter your search query... (use 'video:id' for specific videos, 'OCR:text' for text search)"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <div className="mt-2 text-sm text-gray-600">
              <p><strong>Examples:</strong></p>
              <p>• "person walking" - Find scenes with people walking</p>
              <p>• "video:L01 OCR:sign" - Find signs in video L01</p>
              <p>• "car; person" - Multiple queries with temporal consistency</p>
            </div>
          </div>
        )}

        {searchMode === 'image' && (
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <Upload className="mx-auto mb-4 text-gray-400" size={48} />
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setImageFile(e.target.files[0])}
              className="hidden"
              id="image-upload"
            />
            <label
              htmlFor="image-upload"
              className="cursor-pointer text-blue-600 hover:text-blue-800"
            >
              Upload an image to find similar frames
            </label>
            {imageFile && (
              <div className="mt-4">
                <p className="text-green-600">Selected: {imageFile.name}</p>
              </div>
            )}
          </div>
        )}

        {searchMode === 'audio' && (
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <Mic className="mx-auto mb-4 text-gray-400" size={48} />
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => setAudioFile(e.target.files[0])}
              className="hidden"
              id="audio-upload"
            />
            <label
              htmlFor="audio-upload"
              className="cursor-pointer text-blue-600 hover:text-blue-800"
            >
              Upload audio to find matching segments
            </label>
            {audioFile && (
              <div className="mt-4">
                <p className="text-green-600">Selected: {audioFile.name}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Advanced Filters */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Confidence Threshold */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Confidence Threshold
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={filters.confidence}
            onChange={(e) => setFilters(prev => ({
              ...prev,
              confidence: parseFloat(e.target.value)
            }))}
            className="w-full"
          />
          <span className="text-sm text-gray-600">{filters.confidence}</span>
        </div>

        {/* Object Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Objects
          </label>
          <select
            multiple
            value={filters.objects}
            onChange={(e) => setFilters(prev => ({
              ...prev,
              objects: Array.from(e.target.selectedOptions, option => option.value)
            }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          >
            <option value="person">Person</option>
            <option value="car">Car</option>
            <option value="sign">Sign</option>
            <option value="building">Building</option>
          </select>
        </div>

        {/* Time Range */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Time Range (%)
          </label>
          <div className="flex gap-2">
            <input
              type="number"
              min="0"
              max="100"
              value={filters.timeRange[0]}
              onChange={(e) => setFilters(prev => ({
                ...prev,
                timeRange: [parseInt(e.target.value), prev.timeRange[1]]
              }))}
              className="w-full px-2 py-1 border border-gray-300 rounded"
            />
            <span>-</span>
            <input
              type="number"
              min="0"
              max="100"
              value={filters.timeRange[1]}
              onChange={(e) => setFilters(prev => ({
                ...prev,
                timeRange: [prev.timeRange[0], parseInt(e.target.value)]
              }))}
              className="w-full px-2 py-1 border border-gray-300 rounded"
            />
          </div>
        </div>
      </div>

      {/* Search Button */}
      <button
        onClick={handleSearch}
        className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-medium"
      >
        Search
      </button>
    </div>
  );
};

export default AdvancedSearchForm;
```

#### Enhanced Result Display

```jsx
// src/entry/web/view/src/components/EnhancedResultCard.jsx
import React, { useState } from 'react';
import { Eye, Download, Share, Info, Play } from 'lucide-react';

const EnhancedResultCard = ({ result, onView, onDownload }) => {
  const [showDetails, setShowDetails] = useState(false);
  const { frame_id, distance, entity } = result;

  const formatConfidence = (score) => {
    return (score * 100).toFixed(1) + '%';
  };

  const renderFeatureInfo = () => {
    const features = [];
    
    if (entity.ocr && entity.ocr.length > 0) {
      features.push(
        <div key="ocr" className="mb-3">
          <h5 className="font-medium text-gray-700 mb-1">Detected Text:</h5>
          <div className="flex flex-wrap gap-1">
            {entity.ocr.map((ocrItem, idx) => (
              <span key={idx} className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs">
                {Array.isArray(ocrItem) ? ocrItem[1] : ocrItem}
              </span>
            ))}
          </div>
        </div>
      );
    }
    
    if (entity.objects && entity.objects.length > 0) {
      features.push(
        <div key="objects" className="mb-3">
          <h5 className="font-medium text-gray-700 mb-1">Detected Objects:</h5>
          <div className="flex flex-wrap gap-1">
            {entity.objects.map((obj, idx) => (
              <span key={idx} className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs">
                {obj.label} ({formatConfidence(obj.confidence)})
              </span>
            ))}
          </div>
        </div>
      );
    }
    
    if (entity.audio_transcript) {
      features.push(
        <div key="audio" className="mb-3">
          <h5 className="font-medium text-gray-700 mb-1">Audio Transcript:</h5>
          <p className="text-sm text-gray-600 italic">"{entity.audio_transcript}"</p>
        </div>
      );
    }
    
    return features;
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow">
      {/* Image */}
      <div className="relative">
        <img
          src={`/api/frame/${frame_id}`}
          alt={`Frame ${frame_id}`}
          className="w-full h-48 object-cover"
        />
        
        {/* Overlay with confidence score */}
        <div className="absolute top-2 right-2">
          <span className="bg-black bg-opacity-75 text-white px-2 py-1 rounded-full text-xs">
            {formatConfidence(distance)}
          </span>
        </div>
        
        {/* Play button overlay */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black bg-opacity-50">
          <button
            onClick={() => onView(result)}
            className="bg-white rounded-full p-3 hover:bg-gray-100 transition-colors"
          >
            <Play className="text-blue-600" size={24} />
          </button>
        </div>
      </div>
      
      {/* Content */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-medium text-gray-900">{frame_id}</h4>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-blue-600 hover:text-blue-800"
          >
            <Info size={16} />
          </button>
        </div>
        
        {/* Quick feature preview */}
        <div className="text-sm text-gray-600 mb-3">
          {entity.ocr && entity.ocr.length > 0 && (
            <div className="truncate">
              <strong>Text:</strong> {entity.ocr.map(item => 
                Array.isArray(item) ? item[1] : item
              ).join(', ')}
            </div>
          )}
          {entity.objects && entity.objects.length > 0 && (
            <div className="truncate">
              <strong>Objects:</strong> {entity.objects.map(obj => obj.label).join(', ')}
            </div>
          )}
        </div>
        
        {/* Detailed features (collapsible) */}
        {showDetails && (
          <div className="border-t pt-3 mt-3">
            {renderFeatureInfo()}
            
            <div className="text-xs text-gray-500">
              <p><strong>Similarity Score:</strong> {distance.toFixed(4)}</p>
              {result.rerank_score && (
                <p><strong>Rerank Score:</strong> {result.rerank_score.toFixed(4)}</p>
              )}
              {result.final_score && (
                <p><strong>Final Score:</strong> {result.final_score.toFixed(4)}</p>
              )}
            </div>
          </div>
        )}
        
        {/* Action buttons */}
        <div className="flex gap-2 mt-4">
          <button
            onClick={() => onView(result)}
            className="flex-1 bg-blue-600 text-white py-2 px-3 rounded-md hover:bg-blue-700 transition-colors text-sm flex items-center justify-center gap-1"
          >
            <Eye size={14} />
            View
          </button>
          <button
            onClick={() => onDownload(result)}
            className="bg-gray-100 text-gray-700 py-2 px-3 rounded-md hover:bg-gray-200 transition-colors"
          >
            <Download size={14} />
          </button>
          <button className="bg-gray-100 text-gray-700 py-2 px-3 rounded-md hover:bg-gray-200 transition-colors">
            <Share size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default EnhancedResultCard;
```

---

## Integration Best Practices

### 1. Component Registration System

```python
# src/services/registry.py
from typing import Dict, Type, Any
import importlib

class ComponentRegistry:
    def __init__(self):
        self._extractors: Dict[str, Type] = {}
        self._databases: Dict[str, Type] = {}
        self._search_components: Dict[str, Type] = {}
    
    def register_extractor(self, name: str, extractor_class: Type):
        """Register a feature extractor"""
        self._extractors[name.lower()] = extractor_class
    
    def register_database(self, name: str, database_class: Type):
        """Register a database backend"""
        self._databases[name.lower()] = database_class
    
    def register_search_component(self, name: str, component_class: Type):
        """Register a search component"""
        self._search_components[name.lower()] = component_class
    
    def get_extractor(self, name: str, config: dict = None):
        """Get extractor instance"""
        if name.lower() not in self._extractors:
            # Try dynamic loading
            self._try_load_component("extractors", name)
        
        extractor_class = self._extractors.get(name.lower())
        if not extractor_class:
            raise ValueError(f"Unknown extractor: {name}")
        
        return extractor_class(config)
    
    def get_database(self, name: str, *args, **kwargs):
        """Get database instance"""
        if name.lower() not in self._databases:
            self._try_load_component("databases", name)
        
        database_class = self._databases.get(name.lower())
        if not database_class:
            raise ValueError(f"Unknown database: {name}")
        
        return database_class(*args, **kwargs)
    
    def _try_load_component(self, component_type: str, name: str):
        """Try to dynamically load component"""
        try:
            if component_type == "extractors":
                module = importlib.import_module(f"services.analyse.features.{name}")
                # Look for class with similar name
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "get_image_features"):
                        self.register_extractor(name, attr)
                        break
            
            elif component_type == "databases":
                module = importlib.import_module(f"services.index.{name}_db")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "insert"):
                        self.register_database(name, attr)
                        break
        
        except ImportError:
            pass

# Global registry instance
registry = ComponentRegistry()

# Auto-register built-in components
from services.analyse.features import CLIP, TrOCR
from services.index import FAISSDatabase, MilvusDatabase

registry.register_extractor("clip", CLIP)
registry.register_extractor("ocr", TrOCR)
registry.register_database("faiss", FAISSDatabase)
registry.register_database("milvus", MilvusDatabase)
```

### 2. Plugin System

```python
# src/plugins/plugin_manager.py
import os
import importlib.util
from pathlib import Path
from typing import List, Dict

class PluginManager:
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.loaded_plugins: Dict[str, Any] = {}
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins"""
        plugins = []
        
        if not self.plugin_dir.exists():
            return plugins
        
        for plugin_path in self.plugin_dir.glob("*.py"):
            if plugin_path.name.startswith("_"):
                continue
            plugins.append(plugin_path.stem)
        
        return plugins
    
    def load_plugin(self, plugin_name: str):
        """Load a specific plugin"""
        plugin_path = self.plugin_dir / f"{plugin_name}.py"
        
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin {plugin_name} not found")
        
        # Load the plugin module
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Call plugin's register function if it exists
        if hasattr(module, "register_plugin"):
            module.register_plugin(registry)
        
        self.loaded_plugins[plugin_name] = module
        
        print(f"Loaded plugin: {plugin_name}")
    
    def load_all_plugins(self):
        """Load all discovered plugins"""
        plugins = self.discover_plugins()
        
        for plugin_name in plugins:
            try:
                self.load_plugin(plugin_name)
            except Exception as e:
                print(f"Failed to load plugin {plugin_name}: {e}")

# Global plugin manager
plugin_manager = PluginManager()
```

#### Example Plugin

```python
# plugins/yolo_detector.py
import torch
from ultralytics import YOLO
from services.analyse.features.base_extractor import BaseFeatureExtractor
from services.registry import registry

class YOLODetector(BaseFeatureExtractor):
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        model_name = self.config.get("model_name", "yolov8n.pt")
        self.model = YOLO(model_name)
        
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
    
    def get_image_features(self, image_paths, batch_size, progress_callback=None):
        features = []
        
        for i, img_path in enumerate(image_paths):
            # Run YOLO detection
            results = self.model(str(img_path))
            
            # Extract detections
            detections = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        if box.conf.item() > self.confidence_threshold:
                            detection = {
                                "bbox": box.xyxy.cpu().numpy().tolist()[0],
                                "confidence": box.conf.item(),
                                "class_id": int(box.cls.item()),
                                "label": self.model.names[int(box.cls.item())]
                            }
                            detections.append(detection)
            
            features.append(detections)
            
            if progress_callback:
                progress_callback(i + 1, len(image_paths))
        
        return features
    
    def get_text_features(self, texts):
        # Simple label matching for search
        return texts

def register_plugin(registry):
    """Register this plugin with the component registry"""
    registry.register_extractor("yolo", YOLODetector)
    print("Registered YOLO detector plugin")
```

### 3. Configuration Management

```yaml
# aic25_workspace/layout/config.yaml - Extended Configuration
analyse:
  features:
    # Built-in extractors
    - name: "clip"
      pretrained_model: "openai/clip-vit-base-patch16"
      batch_size: 16
    
    - name: "ocr"
      batch_size: 8
      confidence_threshold: 0.7
      languages: ["vi", "en"]
    
    # Plugin extractors
    - name: "yolo"  # From plugin
      model_name: "yolov8s.pt"
      confidence_threshold: 0.5
      batch_size: 4
    
    - name: "audio"  # Custom extractor
      model_name: "facebook/wav2vec2-base-960h"
      batch_size: 2

# Enhanced search configuration
webui:
  database: "faiss"
  
  search:
    # Multi-stage pipeline
    pipeline:
      - type: "query_expansion"
        enabled: true
        model: "all-MiniLM-L6-v2"
        expansion_ratio: 0.3
      
      - type: "vector_search"
        enabled: true
        top_k: 100
      
      - type: "reranking"
        enabled: true
        model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        top_k: 50
      
      - type: "temporal_fusion"
        enabled: true
        window_size: 25
        consistency_weight: 0.15

# Plugin configuration
plugins:
  enabled: true
  directory: "plugins"
  auto_load: true
  
  # Plugin-specific settings
  yolo:
    models:
      - "yolov8n.pt"  # Nano - fastest
      - "yolov8s.pt"  # Small - balanced
      - "yolov8m.pt"  # Medium - better accuracy
```

---

This comprehensive guide shows you how to extend the AIC25 system with new components to enhance search effectiveness. The modular architecture allows you to add feature extractors, search algorithms, database backends, and UI components while maintaining system coherence and performance.