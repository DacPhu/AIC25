import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from config import GlobalConfig
from entities.search import SearchResult


class FAISSDatabase:
    def __init__(
        self, collection_name: str, do_overwrite: bool = False, db_dir: str = None
    ):
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)

        if db_dir:
            self.db_dir = Path(db_dir) / collection_name
        else:
            self.db_dir = Path("faiss_db") / collection_name

        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.db_dir / "index.faiss"
        self.metadata_path = self.db_dir / "metadata.pkl"
        self.config_path = self.db_dir / "config.json"

        self.dimension = None
        self.index = None
        self.metadata = {}  # frame_id -> {entity data}
        self.id_to_frame = {}  # faiss_id -> frame_id
        self.frame_to_id = {}  # frame_id -> faiss_id
        self._lock = threading.Lock()  # Thread safety for concurrent access

        if do_overwrite or not self._database_exists():
            self._create_new_database()
        else:
            self._load_existing_database()

    def _database_exists(self) -> bool:
        return (
            self.index_path.exists()
            and self.metadata_path.exists()
            and self.config_path.exists()
        )

    def _create_new_database(self):
        """Create a new FAISS database"""
        self.logger.info(f"Creating new FAISS database: {self.collection_name}")

        fields = GlobalConfig.get("faiss", "fields") or []
        vector_fields = [f for f in fields if f.get("datatype") == "FLOAT_VECTOR"]

        if not vector_fields:
            raise ValueError("No FLOAT_VECTOR field found in config")

        self.dimension = vector_fields[0]["dim"]

        index_type = GlobalConfig.get("faiss", "index_type") or "IVF"
        self._create_index(index_type)

        config = {
            "dimension": self.dimension,
            "index_type": index_type,
            "collection_name": self.collection_name,
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

    def _load_existing_database(self):
        """Load existing FAISS database"""
        self.logger.info(f"Loading existing FAISS database: {self.collection_name}")

        with open(self.config_path, "r") as f:
            config = json.load(f)
        self.dimension = config["dimension"]

        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.id_to_frame = data["id_to_frame"]
            self.frame_to_id = data["frame_to_id"]

    def _create_index(self, index_type: str):
        """Create FAISS index based on type"""
        if index_type == "Flat":
            # Exact search (slower but accurate)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product
        elif index_type == "IVF":
            # Inverted File Index (faster approximate search)
            nlist = GlobalConfig.get("faiss", "nlist") or 100
            # Ensure nlist is reasonable for the data size - FAISS needs at least 39*nlist training vectors
            # For safety, we'll use a smaller nlist for smaller datasets
            nlist = min(nlist, 64)  # Cap at 64 clusters to avoid training issues
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif index_type == "HNSW":
            # Hierarchical Navigable Small World (very fast)
            M = GlobalConfig.get("faiss", "M") or 16
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
        elif index_type == "PQ":
            # Product Quantization (memory efficient)
            m = GlobalConfig.get("faiss", "m") or 8
            self.index = faiss.IndexPQ(self.dimension, m, 8)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def insert(self, data: List[Dict], do_update: bool = False):
        """Insert data into FAISS database"""
        if not data:
            return

        with self._lock:  # Thread-safe insertion
            vectors = []
            new_metadata = {}

            for item in data:
                frame_id = item["frame_id"]

                if frame_id in self.frame_to_id and not do_update:
                    continue

                vector_field = None
                for key, value in item.items():
                    if key != "frame_id" and isinstance(value, (list, np.ndarray)):
                        if len(value) == self.dimension:
                            vector_field = key
                            break

                if vector_field is None:
                    self.logger.warning(f"No vector found for {frame_id}")
                    continue

                vector = np.array(item[vector_field], dtype=np.float32)
                if vector.ndim == 1:
                    vector = vector.reshape(1, -1)

                vectors.append(vector[0])
                new_metadata[frame_id] = item

            if not vectors:
                return

            vectors = np.array(vectors, dtype=np.float32)

            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)

            if not self.index.is_trained:
                self.logger.info("Training FAISS index...")
                try:
                    # For IVF index, ensure we have enough training data
                    if hasattr(self.index, "nlist"):
                        min_required = self.index.nlist * 39  # FAISS requirement
                        if len(vectors) < min_required:
                            self.logger.warning(
                                f"Not enough training data ({len(vectors)} < {min_required}). Using Flat index instead."
                            )
                            # Fall back to Flat index if not enough training data
                            self.index = faiss.IndexFlatIP(self.dimension)
                        else:
                            self.index.train(vectors)
                    else:
                        self.index.train(vectors)
                except Exception as e:
                    self.logger.error(
                        f"Index training failed: {e}. Falling back to Flat index."
                    )
                    # Fall back to Flat index on training failure
                    self.index = faiss.IndexFlatIP(self.dimension)

            start_id = self.index.ntotal
            self.index.add(vectors)

            for i, frame_id in enumerate(new_metadata.keys()):
                faiss_id = start_id + i
                self.metadata[frame_id] = new_metadata[frame_id]
                self.id_to_frame[faiss_id] = frame_id
                self.frame_to_id[frame_id] = faiss_id

            self._save_database()

    def search(
        self,
        query_vectors: List[List[float]],
        filter_func=None,
        offset: int = 0,
        limit: int = 50,
        nprobe: int = 8,
    ) -> List[List[SearchResult]]:
        """Search similar vectors"""
        if not query_vectors:
            return []

        query_vectors = np.array(query_vectors, dtype=np.float32)
        faiss.normalize_L2(query_vectors)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        k = min(offset + limit, self.index.ntotal)

        # Handle empty index case
        if k <= 0 or self.index.ntotal == 0:
            # Return empty results for each query
            return [[] for _ in range(len(query_vectors))]

        distances, indices = self.index.search(query_vectors, k)

        results = []
        for i in range(len(query_vectors)):
            query_results = []

            for j in range(len(indices[i])):
                if indices[i][j] == -1:
                    break

                faiss_id = indices[i][j]
                frame_id = self.id_to_frame.get(faiss_id)

                if frame_id is None:
                    continue

                entity = self.metadata[frame_id]
                if filter_func and not filter_func(entity):
                    continue

                result = SearchResult(
                    frame_id=frame_id, distance=float(distances[i][j]), entity=entity
                )
                query_results.append(result)

            query_results = query_results[offset : offset + limit]
            results.append(query_results)

        return results

    def get(self, frame_id: str) -> Optional[Dict]:
        """Get entity by frame_id"""
        return self.metadata.get(frame_id)

    def query(self, filter_func=None, offset: int = 0, limit: int = 50) -> List[Dict]:
        """Query entities with filter"""
        results = []
        count = 0

        for frame_id, entity in self.metadata.items():
            if filter_func and not filter_func(entity):
                continue

            if count >= offset:
                results.append(entity)
                if len(results) >= limit:
                    break
            count += 1

        return results

    def get_total(self) -> int:
        """Get the total number of entities"""
        return len(self.metadata)

    def _save_database(self):
        """Save FAISS index and metadata to disk"""
        try:
            if hasattr(self, "index") and self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
                self.logger.debug(f"Saved FAISS index to {self.index_path}")

            if hasattr(self, "metadata") and hasattr(self, "metadata_path"):
                # Create a copy of the metadata to avoid modification during iteration
                metadata_data = {
                    "metadata": dict(self.metadata),
                    "id_to_frame": dict(self.id_to_frame),
                    "frame_to_id": dict(self.frame_to_id),
                }
                # Use regular open for better reliability
                with open(self.metadata_path, "wb") as f:
                    pickle.dump(metadata_data, f)
                self.logger.debug(f"Saved metadata to {self.metadata_path}")
        except Exception as e:
            # Log errors for debugging
            self.logger.error(f"Error saving database: {e}")
            # Don't raise during garbage collection

    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, "index") and self.index is not None:
                self._save_database()
        except:
            # Ignore all errors during garbage collection
            pass
