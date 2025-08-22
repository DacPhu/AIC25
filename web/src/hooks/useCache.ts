import { useState, useEffect, useRef } from 'react';

// Simple in-memory cache with TTL support
class Cache {
  constructor(defaultTTL = 5 * 60 * 1000) { // 5 minutes default
    this.cache = new Map();
    this.defaultTTL = defaultTTL;
  }

  set(key, value, ttl = this.defaultTTL) {
    const expiresAt = Date.now() + ttl;
    this.cache.set(key, {
      value,
      expiresAt,
    });
  }

  get(key) {
    const item = this.cache.get(key);
    if (!item) return null;

    if (Date.now() > item.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return item.value;
  }

  has(key) {
    return this.get(key) !== null;
  }

  clear() {
    this.cache.clear();
  }

  size() {
    return this.cache.size;
  }
}

const globalCache = new Cache();

// Hook for caching API requests
export function useCache(key, fetcher, options = {}) {
  const {
    ttl = 5 * 60 * 1000, // 5 minutes
    staleWhileRevalidate = false,
    dependencies = [],
  } = options;

  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);

  useEffect(() => {
    const cacheKey = typeof key === 'function' ? key() : key;
    if (!cacheKey) return;

    // Check cache first
    const cachedData = globalCache.get(cacheKey);
    if (cachedData && !staleWhileRevalidate) {
      setData(cachedData);
      setIsLoading(false);
      setError(null);
      return;
    }

    // If we have stale data, show it while revalidating
    if (cachedData && staleWhileRevalidate) {
      setData(cachedData);
      setError(null);
    } else {
      setIsLoading(true);
    }

    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    // Fetch new data
    const fetchData = async () => {
      try {
        const result = await fetcher(abortController.signal);
        
        if (!abortController.signal.aborted) {
          globalCache.set(cacheKey, result, ttl);
          setData(result);
          setError(null);
        }
      } catch (err) {
        if (!abortController.signal.aborted) {
          setError(err);
          // If we had stale data and fetch failed, keep showing stale data
          if (!cachedData) {
            setData(null);
          }
        }
      } finally {
        if (!abortController.signal.aborted) {
          setIsLoading(false);
        }
      }
    };

    fetchData();

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [key, ttl, staleWhileRevalidate, ...dependencies]);

  const invalidate = () => {
    const cacheKey = typeof key === 'function' ? key() : key;
    if (cacheKey) {
      globalCache.cache.delete(cacheKey);
    }
  };

  const mutate = (newData) => {
    const cacheKey = typeof key === 'function' ? key() : key;
    if (cacheKey && newData !== undefined) {
      globalCache.set(cacheKey, newData, ttl);
      setData(newData);
    }
  };

  return {
    data,
    isLoading,
    error,
    invalidate,
    mutate,
  };
}

// Hook for caching search results specifically
export function useSearchCache(searchParams) {
  const cacheKey = () => {
    if (!searchParams || !searchParams.q) return null;
    return `search:${JSON.stringify(searchParams)}`;
  };

  return useCache(
    cacheKey,
    async (signal) => {
      const { search } = await import('../services/search.js');
      return search(
        searchParams.q,
        searchParams.offset,
        searchParams.limit,
        searchParams.nprobe,
        searchParams.model,
        searchParams.temporal_k,
        searchParams.ocr_weight,
        searchParams.ocr_threshold,
        searchParams.max_interval,
        searchParams.selected
      );
    },
    {
      ttl: 10 * 60 * 1000, // 10 minutes for search results
      staleWhileRevalidate: true,
      dependencies: [searchParams]
    }
  );
}

export { globalCache };