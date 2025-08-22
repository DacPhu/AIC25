import axios, { AxiosError, AxiosInstance } from "axios";
import type { SearchResponse, Frame } from "@/types";
import { backendDiscovery, BackendService } from "./discovery";

const PORT = import.meta.env.VITE_PORT || 5000;

// Create axios instance with optimizations
const api = axios.create({
  baseURL: `http://127.0.0.1:${PORT}/api`,
  timeout: 30000, // 30 second timeout
  // Enable response compression
  headers: {
    'Accept-Encoding': 'gzip, deflate, br'
  }
});

// Enhanced API client for v1 endpoints
const apiV1 = axios.create({
  baseURL: `http://127.0.0.1:${PORT}/api/v1`,
  timeout: 30000,
  headers: {
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/json'
  }
});

// Initialize backend discovery
let discoveryInitialized = false;

async function ensureDiscoveryInitialized() {
  if (!discoveryInitialized) {
    await backendDiscovery.start();
    discoveryInitialized = true;
  }
}

/**
 * Create an axios instance for a specific backend
 */
function createBackendClient(backend: BackendService, version: 'v0' | 'v1' = 'v1'): AxiosInstance {
  const baseURL = version === 'v1' ? `${backend.base_url}/api/v1` : `${backend.base_url}/api`;
  
  return axios.create({
    baseURL,
    timeout: 30000,
    headers: {
      'Accept-Encoding': 'gzip, deflate, br',
      'Content-Type': 'application/json'
    }
  });
}

/**
 * Execute a request with automatic backend selection and failover
 */
async function executeWithBackendFailover<T>(
  requestFn: (client: AxiosInstance, backend: BackendService) => Promise<T>,
  capabilities?: string[],
  maxRetries: number = 3,
  version: 'v0' | 'v1' = 'v1'
): Promise<T> {
  await ensureDiscoveryInitialized();
  
  const excludeServices: string[] = [];
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const backend = backendDiscovery.selectBestBackend(capabilities);
    
    if (!backend) {
      throw new Error('No healthy backends available');
    }
    
    // Skip if this backend was already tried and failed
    if (excludeServices.includes(backend.service_id)) {
      continue;
    }
    
    try {
      const client = createBackendClient(backend, version);
      const result = await requestFn(client, backend);
      
      // Request succeeded, return result
      return result;
      
    } catch (error) {
      console.error(`Request failed for backend ${backend.base_url}:`, error);
      
      // Add this backend to exclude list for subsequent attempts
      excludeServices.push(backend.service_id);
      
      // If this is the last attempt, throw the error
      if (attempt === maxRetries - 1) {
        throw error;
      }
      
      // Wait a bit before retrying
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  throw new Error('All backend requests failed');
}

/**
 * Fallback to static configuration if discovery fails
 */
async function executeWithFallback<T>(
  requestFn: (client: AxiosInstance) => Promise<T>,
  staticClient: AxiosInstance
): Promise<T> {
  try {
    return await executeWithBackendFailover(async (client) => {
      return await requestFn(client);
    });
  } catch (error) {
    console.warn('Backend discovery failed, falling back to static configuration:', error);
    return await requestFn(staticClient);
  }
}

// Request deduplication
const pendingRequests = new Map<string, Promise<any>>();

function createRequestKey(url: string, params: Record<string, any>): string {
  return `${url}:${JSON.stringify(params)}`;
}

function deduplicateRequest<T>(
  url: string, 
  params: Record<string, any>, 
  requestFn: () => Promise<T>
): Promise<T> {
  const key = createRequestKey(url, params);
  
  if (pendingRequests.has(key)) {
    return pendingRequests.get(key) as Promise<T>;
  }
  
  const promise = requestFn().finally(() => {
    pendingRequests.delete(key);
  });
  
  pendingRequests.set(key, promise);
  return promise;
}

export async function search(
  q: string,
  offset: number,
  limit: number,
  nprobe: number,
  model: string,
  temporal_k: number,
  ocr_weight: number,
  ocr_threshold: number,
  max_interval: number,
  selected: string | null,
): Promise<SearchResponse> {
  const params = {
    q,
    offset,
    limit,
    nprobe,
    model,
    temporal_k,
    ocr_weight,
    ocr_threshold,
    max_interval,
    selected,
  };
  
  return deduplicateRequest('/search', params, async () => {
    try {
      const res = await api.get('/search', { params });
      return res.data;
    } catch (error) {
      console.error('Search request failed:', error);
      const axiosError = error as AxiosError<{ message?: string }>;
      throw new Error(
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Search request failed'
      );
    }
  });
}
export async function searchSimilar(
  id: string,
  offset: number,
  limit: number,
  nprobe: number,
  model: string,
  temporal_k: number,
  ocr_weight: number,
  ocr_threshold: number,
  max_interval: number,
): Promise<SearchResponse> {
  const params = {
    id,
    offset,
    limit,
    nprobe,
    model,
    temporal_k,
    ocr_weight,
    ocr_threshold,
    max_interval,
  };
  
  return deduplicateRequest('/similar', params, async () => {
    try {
      const res = await api.get('/similar', { params });
      return res.data;
    } catch (error) {
      console.error('Similar search request failed:', error);
      const axiosError = error as AxiosError<{ message?: string }>;
      throw new Error(
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Similar search request failed'
      );
    }
  });
}

export async function getFrameInfo(videoId: string, frameId: string): Promise<Frame> {
  const params = {
    video_id: videoId,
    frame_id: frameId,
  };
  
  return deduplicateRequest('/frame_info', params, async () => {
    try {
      const res = await api.get('/frame_info', { params });
      return res.data;
    } catch (error) {
      console.error('Frame info request failed:', error);
      const axiosError = error as AxiosError<{ message?: string }>;
      throw new Error(
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Failed to get frame info'
      );
    }
  });
}

export async function getAvailableModels(): Promise<{ models: string[] }> {
  return deduplicateRequest('/models', {}, async () => {
    try {
      const res = await api.get('/models');
      return res.data;
    } catch (error) {
      console.error('Models request failed:', error);
      const axiosError = error as AxiosError<{ message?: string }>;
      throw new Error(
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Failed to get available models'
      );
    }
  });
}

// Text-based audio search function
export async function searchByAudioText(
  query: string,
  offset: number = 0,
  limit: number = 50,
  nprobe: number = 8,
  model: string = "audio"
): Promise<SearchResponse> {
  const params = {
    q: query,
    offset,
    limit,
    nprobe,
    model
  };
  
  return deduplicateRequest('/search/audio', params, async () => {
    try {
      const res = await api.get('/search/audio', { params });
      return res.data;
    } catch (error) {
      console.error('Audio text search request failed:', error);
      const axiosError = error as AxiosError<{ message?: string }>;
      throw new Error(
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Audio text search request failed'
      );
    }
  });
}

// Enhanced API functions using v1 endpoints

export async function searchV1(
  q: string,
  offset: number = 0,
  limit: number = 50,
  nprobe: number = 8,
  model: string = "clip",
  temporal_k: number = 10000,
  ocr_weight: number = 1.0,
  ocr_threshold: number = 40,
  max_interval: number = 250,
  selected: string | null = null
): Promise<SearchResponse> {
  const params = {
    q,
    offset,
    limit,
    nprobe,
    model,
    temporal_k,
    ocr_weight,
    ocr_threshold,
    max_interval,
    selected
  };
  
  return deduplicateRequest('/v1/search', params, async () => {
    return executeWithFallback(
      async (client) => {
        const res = await client.get('/search', { params });
        return res.data;
      },
      apiV1
    );
  });
}

export async function searchSimilarV1(
  id: string,
  offset: number = 0,
  limit: number = 50,
  nprobe: number = 8,
  model: string = "clip"
): Promise<SearchResponse> {
  const params = {
    id,
    offset,
    limit,
    nprobe,
    model
  };
  
  return deduplicateRequest('/v1/search/similar', params, async () => {
    try {
      const res = await apiV1.get('/search/similar', { params });
      return res.data;
    } catch (error) {
      console.error('Enhanced similar search request failed:', error);
      const axiosError = error as AxiosError<{ error?: string, message?: string }>;
      throw new Error(
        axiosError.response?.data?.error || 
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Enhanced similar search request failed'
      );
    }
  });
}

export async function searchAudioV1(
  q: string,
  offset: number = 0,
  limit: number = 50,
  nprobe: number = 8,
  model: string = "audio"
): Promise<SearchResponse> {
  const params = {
    q,
    offset,
    limit,
    nprobe,
    model
  };
  
  return deduplicateRequest('/v1/search/audio', params, async () => {
    try {
      const res = await apiV1.get('/search/audio', { params });
      return res.data;
    } catch (error) {
      console.error('Enhanced audio search request failed:', error);
      const axiosError = error as AxiosError<{ error?: string, message?: string }>;
      throw new Error(
        axiosError.response?.data?.error || 
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Enhanced audio search request failed'
      );
    }
  });
}

export async function getSearchSuggestions(
  q: string,
  limit: number = 10
): Promise<{ suggestions: Array<{ query: string; score: number; category: string }> }> {
  const params = { q, limit };
  
  return deduplicateRequest('/v1/search/suggestions', params, async () => {
    try {
      const res = await apiV1.get('/search/suggestions', { params });
      return res.data;
    } catch (error) {
      console.error('Search suggestions request failed:', error);
      const axiosError = error as AxiosError<{ error?: string, message?: string }>;
      throw new Error(
        axiosError.response?.data?.error || 
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Failed to get search suggestions'
      );
    }
  });
}

export async function batchSearch(
  queries: string[],
  params: {
    model?: string;
    limit?: number;
    nprobe?: number;
    temporal_k?: number;
    ocr_weight?: number;
    ocr_threshold?: number;
    max_interval?: number;
  } = {}
): Promise<{ results: SearchResponse[] }> {
  const requestData = {
    queries,
    params: {
      model: "clip",
      limit: 50,
      nprobe: 8,
      temporal_k: 10000,
      ocr_weight: 1.0,
      ocr_threshold: 40,
      max_interval: 250,
      ...params
    }
  };
  
  return deduplicateRequest('/v1/search/batch', requestData, async () => {
    try {
      const res = await apiV1.post('/search/batch', requestData);
      return res.data;
    } catch (error) {
      console.error('Batch search request failed:', error);
      const axiosError = error as AxiosError<{ error?: string, message?: string }>;
      throw new Error(
        axiosError.response?.data?.error || 
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Batch search request failed'
      );
    }
  });
}

export async function getSystemHealth(): Promise<{
  status: string;
  database_type: string;
  total_frames?: number;
  version: string;
}> {
  return deduplicateRequest('/v1/system/health', {}, async () => {
    try {
      const res = await apiV1.get('/system/health');
      return res.data;
    } catch (error) {
      console.error('System health request failed:', error);
      const axiosError = error as AxiosError<{ error?: string, message?: string }>;
      throw new Error(
        axiosError.response?.data?.error || 
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Failed to get system health'
      );
    }
  });
}

export async function getSystemStats(): Promise<{
  system: { cpu_percent: number; memory: any; disk: any };
  database: { type: string; total_frames: number };
  content: { video_count: number; indexed_videos: number };
}> {
  return deduplicateRequest('/v1/system/stats', {}, async () => {
    try {
      const res = await apiV1.get('/system/stats');
      return res.data;
    } catch (error) {
      console.error('System stats request failed:', error);
      const axiosError = error as AxiosError<{ error?: string, message?: string }>;
      throw new Error(
        axiosError.response?.data?.error || 
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Failed to get system stats'
      );
    }
  });
}

export async function getVideoList(
  offset: number = 0,
  limit: number = 50,
  search?: string
): Promise<{
  videos: Array<{
    video_id: string;
    title: string;
    fps: number;
    duration?: number;
    file_size?: number;
  }>;
  total: number;
}> {
  const params = { offset, limit, search };
  
  return deduplicateRequest('/v1/videos', params, async () => {
    try {
      const res = await apiV1.get('/videos', { params });
      return res.data;
    } catch (error) {
      console.error('Video list request failed:', error);
      const axiosError = error as AxiosError<{ error?: string, message?: string }>;
      throw new Error(
        axiosError.response?.data?.error || 
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Failed to get video list'
      );
    }
  });
}

export async function getFrameInfoV1(
  videoId: string,
  frameId: string
): Promise<{ frame?: Frame; message: string }> {
  return deduplicateRequest(`/v1/frames/${videoId}/${frameId}`, {}, async () => {
    try {
      const res = await apiV1.get(`/frames/${videoId}/${frameId}`);
      return res.data;
    } catch (error) {
      console.error('Enhanced frame info request failed:', error);
      const axiosError = error as AxiosError<{ error?: string, message?: string }>;
      throw new Error(
        axiosError.response?.data?.error || 
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Failed to get frame info'
      );
    }
  });
}
