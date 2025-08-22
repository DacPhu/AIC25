// Core data types
export interface Frame {
  id: string;
  video_id: string;
  frame_id: string;
  frame_uri: string;
  video_uri: string;
  fps: number;
}

export interface SearchResult {
  distance: number;
  entity: Frame;
}

export interface SearchResponse {
  total: number;
  frames: Frame[];
  params: SearchParams;
  offset: number;
  search_type?: string;
}

export interface SearchParams {
  model: string;
  limit: number;
  nprobe: number;
  temporal_k?: number;
  ocr_weight?: number;
  ocr_threshold?: number;
  max_interval?: number;
  query?: string;
}

export interface Query {
  q?: string;
  id?: string | null;
}

// Cache-related types
export interface CacheItem<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

export interface CacheEntry<T> {
  value: T;
  timestamp: number;
}

// Component prop types
export interface FrameItemProps {
  video_id: string;
  frame_id: string;
  thumbnail: string;
  onPlay: () => void;
  onSearchSimilar: () => void;
  onSelect: () => void;
  selected?: boolean;
}

export interface FrameContainerProps {
  id: string;
  isLoading: boolean;
  className?: string;
  children: React.ReactNode;
}

export interface VirtualGridProps {
  items: Frame[];
  itemHeight: number;
  itemWidth: number;
  containerHeight: number;
  gap: number;
  renderItem: (item: Frame, index: number) => React.ReactNode;
  onLoadMore?: () => void;
  hasNextPage?: boolean;
  isLoadingMore?: boolean;
  className?: string;
}

export interface AudioSearchProps {
  onResults: (results: SearchResponse) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

// Video player types
export interface VideoPlayerProps {
  frame: Frame;
  onClose: () => void;
}

export interface UsePlayVideoReturn {
  (frame: Frame): void;
}

// Filter types
export interface DropdownProps {
  name: string;
  options: (string | number)[];
}

export interface EditableProps {
  name: string;
  defaultValue: string | number;
}

// Options types
export interface ModelOption {
  value: string;
  label: string;
}

// API response types
export interface ApiError {
  error: string;
  message?: string;
}

// Hook types
export interface UseCacheOptions {
  ttl?: number;
  staleWhileRevalidate?: boolean;
}

export interface UseCacheReturn<T> {
  data: T | undefined;
  isLoading: boolean;
  error: Error | null;
  mutate: (data: T) => void;
  invalidate: () => void;
}