"""
Pydantic models for API request/response validation and documentation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ModelType(str, Enum):
    """Available model types for search."""

    CLIP = "clip"
    AUDIO = "audio"
    ENHANCED_CLIP = "enhanced_clip"
    SENTENCE_TRANSFORMER = "sentence_transformer"


class SearchType(str, Enum):
    """Types of search operations."""

    TEXT = "text"
    AUDIO_TEXT = "audio_text"
    SIMILAR = "similar"
    VIDEO_FILTER = "video_filter"
    SEMANTIC = "semantic"


class BaseAPIResponse(BaseModel):
    """Base response model with common fields."""

    success: bool = True
    message: Optional[str] = None
    timestamp: Optional[str] = None


class ErrorResponse(BaseAPIResponse):
    """Error response model."""

    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SearchParams(BaseModel):
    """Search parameters model."""

    model: ModelType = Field(
        default=ModelType.CLIP, description="Model type to use for search"
    )
    limit: int = Field(
        default=50, ge=1, le=200, description="Maximum number of results"
    )
    nprobe: int = Field(
        default=8, ge=1, le=128, description="Number of probes for FAISS search"
    )
    temporal_k: int = Field(
        default=10000, ge=1, description="Temporal search parameter"
    )
    ocr_weight: float = Field(
        default=1.0, ge=0.0, le=10.0, description="OCR weight for search"
    )
    ocr_threshold: int = Field(
        default=40, ge=0, le=100, description="OCR threshold percentage"
    )
    max_interval: int = Field(
        default=250, ge=1, description="Maximum interval for temporal search"
    )


class SearchRequest(BaseModel):
    """Search request model."""

    q: str = Field(default="", description="Search query")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")
    selected: Optional[str] = Field(default=None, description="Selected frame ID")
    params: SearchParams = Field(default_factory=SearchParams)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "q": "person walking",
                "offset": 0,
                "params": {"model": "clip", "limit": 50, "nprobe": 8},
            }
        }
    )


class AudioSearchRequest(BaseModel):
    """Audio search request model."""

    q: str = Field(..., min_length=1, description="Audio search query description")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")
    limit: int = Field(
        default=50, ge=1, le=200, description="Maximum number of results"
    )
    nprobe: int = Field(
        default=8, ge=1, le=128, description="Number of probes for FAISS search"
    )
    model: ModelType = Field(default=ModelType.AUDIO, description="Audio model type")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "q": "music playing",
                "offset": 0,
                "limit": 50,
                "nprobe": 8,
                "model": "audio",
            }
        }
    )


class FrameData(BaseModel):
    """Frame data model."""

    id: str = Field(..., description="Unique frame identifier")
    video_id: str = Field(..., description="Video identifier")
    frame_id: str = Field(..., description="Frame identifier")
    frame_uri: str = Field(..., description="Frame image URI")
    video_uri: str = Field(..., description="Video streaming URI")
    fps: int = Field(default=25, ge=1, description="Video frame rate")
    distance: Optional[float] = Field(
        default=None, description="Search similarity distance"
    )


class SearchResponse(BaseAPIResponse):
    """Search response model."""

    total: int = Field(..., ge=0, description="Total number of results")
    frames: List[FrameData] = Field(
        default_factory=list, description="List of frame results"
    )
    params: SearchParams = Field(..., description="Search parameters used")
    offset: int = Field(..., ge=0, description="Current result offset")
    search_type: SearchType = Field(
        default=SearchType.TEXT, description="Type of search performed"
    )
    query: Optional[str] = Field(default=None, description="Original search query")
    has_more: bool = Field(
        default=False, description="Whether more results are available"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "total": 150,
                "frames": [
                    {
                        "id": "video1#frame001",
                        "video_id": "video1",
                        "frame_id": "frame001",
                        "frame_uri": "/api/files/keyframes/video1/frame001.jpg",
                        "video_uri": "/api/stream/videos/video1.mp4",
                        "fps": 25,
                        "distance": 0.85,
                    }
                ],
                "offset": 0,
                "has_more": True,
                "search_type": "text",
                "query": "person walking",
            }
        }
    )


class SimilarSearchRequest(BaseModel):
    """Similar search request model."""

    id: str = Field(..., description="Frame ID to find similar frames")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")
    limit: int = Field(
        default=50, ge=1, le=200, description="Maximum number of results"
    )
    nprobe: int = Field(
        default=8, ge=1, le=128, description="Number of probes for FAISS search"
    )
    model: ModelType = Field(default=ModelType.CLIP, description="Model type to use")


class FrameInfoRequest(BaseModel):
    """Frame info request model."""

    video_id: str = Field(..., description="Video identifier")
    frame_id: str = Field(..., description="Frame identifier")


class FrameInfoResponse(BaseAPIResponse):
    """Frame info response model."""

    frame: Optional[FrameData] = Field(default=None, description="Frame information")


class ModelsResponse(BaseAPIResponse):
    """Available models response."""

    models: List[str] = Field(
        default_factory=list, description="List of available models"
    )


class HealthResponse(BaseAPIResponse):
    """Health check response."""

    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="API version")
    database_type: str = Field(..., description="Database type in use")
    total_frames: Optional[int] = Field(
        default=None, description="Total frames indexed"
    )


class PaginationInfo(BaseModel):
    """Pagination information."""

    current_page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    total_items: int = Field(..., ge=0, description="Total number of items")
    has_previous: bool = Field(..., description="Whether previous page exists")
    has_next: bool = Field(..., description="Whether next page exists")


class SearchResponseWithPagination(SearchResponse):
    """Search response with pagination info."""

    pagination: PaginationInfo = Field(..., description="Pagination information")


class BatchSearchRequest(BaseModel):
    """Batch search request for multiple queries."""

    queries: List[str] = Field(
        ..., min_length=1, max_length=10, description="List of search queries"
    )
    params: SearchParams = Field(default_factory=SearchParams)


class BatchSearchResponse(BaseAPIResponse):
    """Batch search response."""

    results: List[SearchResponse] = Field(
        default_factory=list, description="List of search results"
    )
    total_queries: int = Field(..., description="Total number of queries processed")


class SearchSuggestion(BaseModel):
    """Search suggestion model."""

    query: str = Field(..., description="Suggested query")
    score: float = Field(..., ge=0.0, le=1.0, description="Suggestion relevance score")
    category: Optional[str] = Field(default=None, description="Suggestion category")


class SearchSuggestionsResponse(BaseAPIResponse):
    """Search suggestions response."""

    suggestions: List[SearchSuggestion] = Field(
        default_factory=list, description="List of suggestions"
    )
    query: str = Field(..., description="Original query")


class VideoInfo(BaseModel):
    """Video information model."""

    video_id: str = Field(..., description="Video identifier")
    title: Optional[str] = Field(default=None, description="Video title")
    duration: Optional[float] = Field(
        default=None, description="Video duration in seconds"
    )
    fps: int = Field(default=25, description="Video frame rate")
    total_frames: Optional[int] = Field(
        default=None, description="Total number of frames"
    )
    file_size: Optional[int] = Field(default=None, description="File size in bytes")


class VideoListResponse(BaseAPIResponse):
    """Video list response."""

    videos: List[VideoInfo] = Field(default_factory=list, description="List of videos")
    total: int = Field(..., ge=0, description="Total number of videos")


class UploadStatus(str, Enum):
    """Upload status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseAPIResponse):
    """File upload response."""

    upload_id: str = Field(..., description="Unique upload identifier")
    status: UploadStatus = Field(..., description="Upload status")
    filename: str = Field(..., description="Uploaded filename")
    file_size: int = Field(..., description="File size in bytes")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Upload progress percentage"
    )
