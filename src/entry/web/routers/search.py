"""
Search API routes with enhanced functionality.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_400_BAD_REQUEST

from config import GlobalConfig
from entry.web.models import (
    BatchSearchRequest,
    BatchSearchResponse,
    FrameData,
    PaginationInfo,
    SearchParams,
    SearchResponseWithPagination,
    SearchSuggestion,
    SearchSuggestionsResponse,
    SearchType,
)
from services.search import SearchFactory

# Setup
router = APIRouter(prefix="/api/v1/search", tags=["search"])
LOGGER = logging.getLogger(__name__)
WORK_DIR = Path(os.getenv("AIC25_WORK_DIR") or ".")

_searcher = None


def _get_database_type():
    """Get database type from config, with fallback."""
    try:
        return GlobalConfig.get("webui", "database") or "faiss"
    except Exception as e:
        LOGGER.warning(f"Could not get database type from config: {e}")
        LOGGER.warning("Using default database type: faiss")
        return "faiss"


def _initialize_searcher():
    """Initialize searcher instance lazily."""
    global _searcher
    if _searcher is None:
        database_type = _get_database_type()
        _searcher = SearchFactory.create_searcher("default", database_type)
    return _searcher


def get_searcher():
    """Dependency to get searcher instance."""
    return _initialize_searcher()


def calculate_pagination(offset: int, limit: int, total: int) -> PaginationInfo:
    """Calculate pagination information."""
    current_page = (offset // limit) + 1
    total_pages = (total + limit - 1) // limit if total > 0 else 1

    return PaginationInfo(
        current_page=current_page,
        per_page=limit,
        total_pages=total_pages,
        total_items=total,
        has_previous=offset > 0,
        has_next=offset + limit < total,
    )


def process_search_results(
    request: Request,
    results: dict,
    search_type: SearchType = SearchType.TEXT,
    query: Optional[str] = None,
) -> List[FrameData]:
    """Process search results into FrameData objects."""
    frames = []
    for record in results["results"]:
        data = record["entity"]
        video_frame_str = data["frame_id"]
        video_id, frame_id = video_frame_str.split("#")

        frame_uri = f"{request.base_url}api/files/keyframes/{video_id}/{frame_id}.jpg"
        video_uri = f"{request.base_url}api/stream/videos/{video_id}.mp4"

        fps = 25  # default
        try:
            video_info_path = WORK_DIR / "videos_info" / f"{video_id}.json"
            if video_info_path.exists():
                with open(video_info_path, "r") as f:
                    fps = json.load(f).get("frame_rate", 25)
        except Exception as e:
            LOGGER.warning(f"Could not load FPS for {video_id}: {e}")

        distance = getattr(record, "distance", record.get("distance", None))

        frames.append(
            FrameData(
                id=video_frame_str,
                video_id=video_id,
                frame_id=frame_id,
                frame_uri=frame_uri,
                video_uri=video_uri,
                fps=fps,
                distance=distance,
            )
        )

    return frames


@router.get("/", response_model=SearchResponseWithPagination)
async def search_frames(
    request: Request,
    q: str = Query(default="", description="Search query"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results"),
    nprobe: int = Query(default=8, ge=1, le=128, description="FAISS nprobe"),
    model: str = Query(default="clip", description="Model type"),
    temporal_k: int = Query(default=10000, ge=1, description="Temporal parameter"),
    ocr_weight: float = Query(default=1.0, ge=0.0, le=10.0, description="OCR weight"),
    ocr_threshold: int = Query(default=40, ge=0, le=100, description="OCR threshold"),
    max_interval: int = Query(default=250, ge=1, description="Max interval"),
    selected: Optional[str] = Query(default=None, description="Selected frame"),
    searcher_instance=Depends(get_searcher),
):
    """
    Enhanced search endpoint with comprehensive validation and pagination.

    Supports:
    - Text-based search
    - OCR search with OCR:"text" syntax
    - Video filtering with video:id syntax
    - Advanced temporal search parameters
    """
    try:
        # Perform search
        results = searcher_instance.search(
            q,
            "",
            offset,
            limit,
            nprobe,
            model,
            temporal_k,
            ocr_weight,
            ocr_threshold,
            max_interval,
            selected,
        )

        frames = process_search_results(request, results, SearchType.TEXT, q)

        search_params = SearchParams(
            model=model,
            limit=limit,
            nprobe=nprobe,
            temporal_k=temporal_k,
            ocr_weight=ocr_weight,
            ocr_threshold=ocr_threshold,
            max_interval=max_interval,
        )

        total = results["total"]
        pagination = calculate_pagination(offset, limit, total)

        return SearchResponseWithPagination(
            success=True,
            total=total,
            frames=frames,
            params=search_params,
            offset=results["offset"],
            search_type=SearchType.TEXT,
            query=q if q else None,
            has_more=offset + limit < total,
            pagination=pagination,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        LOGGER.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Search operation failed", "message": str(e), "query": q},
        )


@router.get("/similar", response_model=SearchResponseWithPagination)
async def search_similar_frames(
    request: Request,
    id: str = Query(..., description="Frame ID to find similar frames"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results"),
    nprobe: int = Query(default=8, ge=1, le=128, description="FAISS nprobe"),
    model: str = Query(default="clip", description="Model type"),
    temporal_k: int = Query(default=10000, description="Temporal parameter"),
    ocr_weight: float = Query(default=1.0, description="OCR weight"),
    ocr_threshold: int = Query(default=40, description="OCR threshold"),
    max_interval: int = Query(default=250, description="Max interval"),
    searcher_instance=Depends(get_searcher),
):
    """
    Find frames similar to a given frame.

    Uses the selected frame's features to find semantically similar frames
    in the database using vector similarity search.
    """
    try:
        results = searcher_instance.search_similar(id, offset, limit, nprobe, model)

        frames = process_search_results(request, results, SearchType.SIMILAR)

        search_params = SearchParams(
            model=model,
            limit=limit,
            nprobe=nprobe,
            temporal_k=temporal_k,
            ocr_weight=ocr_weight,
            ocr_threshold=ocr_threshold,
            max_interval=max_interval,
        )

        total = results["total"]
        pagination = calculate_pagination(offset, limit, total)

        return SearchResponseWithPagination(
            success=True,
            total=total,
            frames=frames,
            params=search_params,
            offset=results["offset"],
            search_type=SearchType.SIMILAR,
            query=f"similar:{id}",
            has_more=offset + limit < total,
            pagination=pagination,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        LOGGER.error(f"Similar search failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Similar search operation failed",
                "message": str(e),
                "frame_id": id,
            },
        )


@router.get("/audio", response_model=SearchResponseWithPagination)
async def search_audio_by_text(
    request: Request,
    q: str = Query(..., min_length=1, description="Audio search query"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results"),
    nprobe: int = Query(default=8, ge=1, le=128, description="FAISS nprobe"),
    model: str = Query(default="audio", description="Audio model type"),
    searcher_instance=Depends(get_searcher),
):
    """
    Search for audio content using text descriptions.

    This endpoint allows searching for video frames containing specific audio
    content by describing what you're looking for (e.g., "music playing",
    "people talking", "car engine").
    """
    try:
        if not q.strip():
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Query text is required for audio search",
            )

        results = searcher_instance.search(
            q, "", offset, limit, nprobe, model, 10000, 1.0, 40, 250, None
        )

        frames = process_search_results(request, results, SearchType.AUDIO_TEXT, q)

        search_params = SearchParams(
            model=model,
            limit=limit,
            nprobe=nprobe,
            temporal_k=10000,
            ocr_weight=1.0,
            ocr_threshold=40,
            max_interval=250,
        )

        total = results["total"]
        pagination = calculate_pagination(offset, limit, total)

        return SearchResponseWithPagination(
            success=True,
            total=total,
            frames=frames,
            params=search_params,
            offset=results["offset"],
            search_type=SearchType.AUDIO_TEXT,
            query=q,
            has_more=offset + limit < total,
            pagination=pagination,
            timestamp=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Audio search failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Audio search operation failed",
                "message": str(e),
                "query": q,
            },
        )


@router.post("/batch", response_model=BatchSearchResponse)
async def batch_search(
    request: Request,
    batch_request: BatchSearchRequest,
    searcher_instance=Depends(get_searcher),
):
    """
    Perform a batch search for multiple queries simultaneously.

    This endpoint allows searching for multiple queries in a single request,
    which can be more efficient for certain use cases.
    """
    try:
        results = []

        for query in batch_request.queries:
            try:
                search_results = searcher_instance.search(
                    query,
                    "",
                    0,
                    batch_request.params.limit,
                    batch_request.params.nprobe,
                    batch_request.params.model,
                    batch_request.params.temporal_k,
                    batch_request.params.ocr_weight,
                    batch_request.params.ocr_threshold,
                    batch_request.params.max_interval,
                    None,
                )

                frames = process_search_results(
                    request, search_results, SearchType.TEXT, query
                )

                total = search_results["total"]
                pagination = calculate_pagination(0, batch_request.params.limit, total)

                result = SearchResponseWithPagination(
                    success=True,
                    total=total,
                    frames=frames,
                    params=batch_request.params,
                    offset=0,
                    search_type=SearchType.TEXT,
                    query=query,
                    has_more=batch_request.params.limit < total,
                    pagination=pagination,
                    timestamp=datetime.utcnow().isoformat(),
                )
                results.append(result)

            except Exception as e:
                LOGGER.error(f"Batch search failed for query '{query}': {e}")
                error_result = SearchResponseWithPagination(
                    success=False,
                    message=f"Search failed: {str(e)}",
                    total=0,
                    frames=[],
                    params=batch_request.params,
                    offset=0,
                    search_type=SearchType.TEXT,
                    query=query,
                    has_more=False,
                    pagination=calculate_pagination(0, batch_request.params.limit, 0),
                    timestamp=datetime.utcnow().isoformat(),
                )
                results.append(error_result)

        return BatchSearchResponse(
            success=True,
            results=results,
            total_queries=len(batch_request.queries),
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        LOGGER.error(f"Batch search failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Batch search operation failed", "message": str(e)},
        )


@router.get("/semantic", response_model=SearchResponseWithPagination)
async def semantic_search(
    request: Request,
    q: str = Query(..., min_length=1, description="Semantic search query"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results"),
    nprobe: int = Query(default=8, ge=1, le=128, description="FAISS nprobe"),
    model: str = Query(
        default="sentence_transformer", description="Sentence transformer model"
    ),
    temporal_k: int = Query(default=10000, description="Temporal parameter"),
    ocr_weight: float = Query(default=1.0, description="OCR weight"),
    ocr_threshold: int = Query(default=40, description="OCR threshold"),
    max_interval: int = Query(default=250, description="Max interval"),
    searcher_instance=Depends(get_searcher),
):
    """
    Semantic search using sentence transformers for natural language understanding.

    This endpoint provides advanced semantic search capabilities using pre-trained
    sentence transformer models. It's particularly effective for:
    - Natural language queries
    - Complex sentence understanding
    - Semantic similarity rather than keyword matching
    - Cross-lingual search (with multilingual models)

    Example queries:
    - "A person walking through a busy street"
    - "Someone cooking in a modern kitchen"
    - "Cars moving on a highway during sunset"
    """
    try:
        if not q.strip():
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Query text is required for semantic search",
            )

        available_models = searcher_instance.get_models()
        if model not in available_models:
            sentence_models = [
                m
                for m in available_models
                if "sentence" in m.lower() or "transformer" in m.lower()
            ]
            if sentence_models:
                model = sentence_models[0]
                LOGGER.info(f"Using available sentence transformer model: {model}")
            else:
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail={
                        "error": f"Sentence transformer model '{model}' not available",
                        "available_models": available_models,
                        "suggestion": "Configure sentence_transformer in your config.yaml or use available models",
                    },
                )

        results = searcher_instance.search(
            q,
            "",
            offset,
            limit,
            nprobe,
            model,
            temporal_k,
            ocr_weight,
            ocr_threshold,
            max_interval,
            None,
        )

        frames = process_search_results(request, results, SearchType.SEMANTIC, q)

        search_params = SearchParams(
            model=model,
            limit=limit,
            nprobe=nprobe,
            temporal_k=temporal_k,
            ocr_weight=ocr_weight,
            ocr_threshold=ocr_threshold,
            max_interval=max_interval,
        )

        total = results["total"]
        pagination = calculate_pagination(offset, limit, total)

        return SearchResponseWithPagination(
            success=True,
            total=total,
            frames=frames,
            params=search_params,
            offset=results["offset"],
            search_type=SearchType.SEMANTIC,
            query=q,
            has_more=offset + limit < total,
            pagination=pagination,
            timestamp=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Semantic search failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Semantic search operation failed",
                "message": str(e),
                "query": q,
                "suggestion": "Check if sentence transformer models are properly configured",
            },
        )


@router.get("/suggestions", response_model=SearchSuggestionsResponse)
async def get_search_suggestions(
    q: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum suggestions"),
):
    """
    Get search suggestions based on partial query input.

    This endpoint provides intelligent search suggestions to help users
    discover content and improve their search experience.
    """
    try:
        # Simple suggestion logic - in production, this could be more sophisticated
        # using ML-based suggestion systems, popular queries, etc.

        suggestions = []

        suggestion_categories = {
            "people": [
                "person walking",
                "people talking",
                "crowd gathering",
                "person sitting",
            ],
            "objects": ["car driving", "phone ringing", "door opening", "book reading"],
            "activities": [
                "music playing",
                "cooking food",
                "writing text",
                "playing sports",
            ],
            "scenes": ["outdoor scene", "indoor scene", "night scene", "city view"],
            "animals": ["dog barking", "cat meowing", "bird singing", "horse running"],
            "vehicles": [
                "car engine",
                "airplane flying",
                "train moving",
                "bicycle riding",
            ],
            "sounds": ["applause", "laughter", "crying", "shouting"],
            "weather": ["rain falling", "wind blowing", "snow falling", "sunny day"],
        }

        query_lower = q.lower().strip()

        for category, examples in suggestion_categories.items():
            for example in examples:
                if query_lower in example.lower() or example.lower().startswith(
                    query_lower
                ):
                    score = 1.0 if example.lower().startswith(query_lower) else 0.7
                    if (
                        query_lower in example.lower()
                        and not example.lower().startswith(query_lower)
                    ):
                        score = 0.5

                    suggestions.append(
                        SearchSuggestion(query=example, score=score, category=category)
                    )

        suggestions.sort(key=lambda x: x.score, reverse=True)
        suggestions = suggestions[:limit]

        return SearchSuggestionsResponse(
            success=True,
            suggestions=suggestions,
            query=q,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        LOGGER.error(f"Suggestions failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get search suggestions", "message": str(e)},
        )
