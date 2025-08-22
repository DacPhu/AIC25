/**
 * Distributed Search Service - Simple Manual Backend Discovery
 * Frontend can connect to any FastAPI backend running on any device
 */

import axios, { AxiosError } from "axios";
import type { SearchResponse, Frame } from "@/types";
import { simpleBackendManager, SimpleBackend } from "./simple_discovery";

// Initialize and start health checks
simpleBackendManager.startHealthChecks(30000); // Check every 30 seconds

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

/**
 * Execute request with automatic backend selection and failover
 */
async function executeDistributedRequest<T>(
  endpoint: string,
  params: Record<string, any>,
  method: 'GET' | 'POST' = 'GET',
  data?: any
): Promise<T> {
  return simpleBackendManager.executeRequest(async (client, backend) => {
    console.log(`Making ${method} request to ${backend.name} (${backend.base_url})`);
    
    if (method === 'GET') {
      const response = await client.get(endpoint, { params });
      return response.data;
    } else {
      const response = await client.post(endpoint, data, { params });
      return response.data;
    }
  });
}

// =============== ENHANCED V1 API FUNCTIONS (DISTRIBUTED) ===============

export async function searchDistributed(
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
  
  return deduplicateRequest('/search', params, () =>
    executeDistributedRequest<SearchResponse>('/search', params)
  );
}

export async function searchSimilarDistributed(
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
  
  return deduplicateRequest('/search/similar', params, () =>
    executeDistributedRequest<SearchResponse>('/search/similar', params)
  );
}

export async function searchAudioDistributed(
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
  
  return deduplicateRequest('/search/audio', params, () =>
    executeDistributedRequest<SearchResponse>('/search/audio', params)
  );
}

export async function getSearchSuggestionsDistributed(
  q: string,
  limit: number = 10
): Promise<{ suggestions: Array<{ query: string; score: number; category: string }> }> {
  const params = { q, limit };
  
  return deduplicateRequest('/search/suggestions', params, () =>
    executeDistributedRequest('/search/suggestions', params)
  );
}

export async function batchSearchDistributed(
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
  
  return deduplicateRequest('/search/batch', requestData, () =>
    executeDistributedRequest('/search/batch', {}, 'POST', requestData)
  );
}

export async function getFrameInfoDistributed(
  videoId: string,
  frameId: string
): Promise<{ frame?: Frame; message: string }> {
  return deduplicateRequest(`/frames/${videoId}/${frameId}`, {}, () =>
    executeDistributedRequest(`/frames/${videoId}/${frameId}`, {})
  );
}

export async function getVideoListDistributed(
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
  
  return deduplicateRequest('/videos', params, () =>
    executeDistributedRequest('/videos', params)
  );
}

export async function getSystemHealthDistributed(): Promise<{
  status: string;
  database_type: string;
  total_frames?: number;
  version: string;
}> {
  return deduplicateRequest('/system/health', {}, () =>
    executeDistributedRequest('/system/health', {})
  );
}

export async function getSystemStatsDistributed(): Promise<{
  system: { cpu_percent: number; memory: any; disk: any };
  database: { type: string; total_frames: number };
  content: { video_count: number; indexed_videos: number };
}> {
  return deduplicateRequest('/system/stats', {}, () =>
    executeDistributedRequest('/system/stats', {})
  );
}

export async function getAvailableModelsDistributed(): Promise<{ models: string[] }> {
  return deduplicateRequest('/system/models', {}, () =>
    executeDistributedRequest('/system/models', {})
  );
}

// =============== BACKEND MANAGEMENT FUNCTIONS ===============

/**
 * Add a new backend manually
 */
export function addBackend(name: string, host: string, port: number): void {
  simpleBackendManager.addBackend(name, host, port);
}

/**
 * Remove a backend
 */
export function removeBackend(name: string): void {
  simpleBackendManager.removeBackend(name);
}

/**
 * Get all available backends
 */
export function getBackends(): any[] {
  return simpleBackendManager.getBackends();
}

/**
 * Get healthy backends only
 */
export function getHealthyBackends(): any[] {
  return simpleBackendManager.getHealthyBackends();
}

/**
 * Get backend statistics
 */
export function getBackendStats() {
  return simpleBackendManager.getStats();
}

/**
 * Manually trigger backend discovery
 */
export async function discoverBackends(): Promise<void> {
  await simpleBackendManager.discoverLocalBackends();
}

/**
 * Check health of all backends
 */
export async function checkBackendHealth(): Promise<void> {
  await simpleBackendManager.checkAllBackends();
}

// =============== LEGACY API FUNCTIONS (FALLBACK) ===============

// Fallback to the first healthy backend for legacy endpoints
async function getLegacyClient() {
  const healthyBackends = simpleBackendManager.getHealthyBackends();
  if (healthyBackends.length === 0) {
    throw new Error('No healthy backends available');
  }
  
  const backend = healthyBackends[0];
  return axios.create({
    baseURL: `${backend.base_url}/api`,
    timeout: 30000,
    headers: {
      'Accept-Encoding': 'gzip, deflate, br'
    }
  });
}

export async function searchLegacy(
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
      const client = await getLegacyClient();
      const res = await client.get('/search', { params });
      return res.data;
    } catch (error) {
      console.error('Legacy search request failed:', error);
      const axiosError = error as AxiosError<{ message?: string }>;
      throw new Error(
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Search request failed'
      );
    }
  });
}

export async function searchSimilarLegacy(
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
      const client = await getLegacyClient();
      const res = await client.get('/similar', { params });
      return res.data;
    } catch (error) {
      console.error('Legacy similar search request failed:', error);
      const axiosError = error as AxiosError<{ message?: string }>;
      throw new Error(
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Similar search request failed'
      );
    }
  });
}

export async function getFrameInfoLegacy(videoId: string, frameId: string): Promise<Frame> {
  const params = {
    video_id: videoId,
    frame_id: frameId,
  };
  
  return deduplicateRequest('/frame_info', params, async () => {
    try {
      const client = await getLegacyClient();
      const res = await client.get('/frame_info', { params });
      return res.data;
    } catch (error) {
      console.error('Legacy frame info request failed:', error);
      const axiosError = error as AxiosError<{ message?: string }>;
      throw new Error(
        axiosError.response?.data?.message || 
        axiosError.message || 
        'Failed to get frame info'
      );
    }
  });
}

// Auto-discovery on module load
setTimeout(() => {
  discoverBackends().catch(console.error);
}, 1000);

export default {
  // Distributed API (recommended)
  search: searchDistributed,
  searchSimilar: searchSimilarDistributed,
  searchAudio: searchAudioDistributed,
  getSearchSuggestions: getSearchSuggestionsDistributed,
  batchSearch: batchSearchDistributed,
  getFrameInfo: getFrameInfoDistributed,
  getVideoList: getVideoListDistributed,
  getSystemHealth: getSystemHealthDistributed,
  getSystemStats: getSystemStatsDistributed,
  getAvailableModels: getAvailableModelsDistributed,
  
  // Legacy API (fallback)
  searchLegacy,
  searchSimilarLegacy,
  getFrameInfoLegacy,
  
  // Backend management
  addBackend,
  removeBackend,
  getBackends,
  getHealthyBackends,
  getBackendStats,
  discoverBackends,
  checkBackendHealth
};