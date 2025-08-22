import time
from typing import Any, Optional


class gCacheManager:
    """Simple in-memory cache with TTL support"""

    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None

        if time.time() - self.timestamps[key] > self.ttl:
            self.remove(key)
            return None

        return self.cache[key]

    def set(self, key: str, value: Any):
        while len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self.remove(oldest_key)

        self.cache[key] = value
        self.timestamps[key] = time.time()

    def remove(self, key: str):
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]

    def clear(self):
        self.cache.clear()
        self.timestamps.clear()
