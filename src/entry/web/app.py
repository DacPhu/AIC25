import json
import logging
import os
import traceback
from pathlib import Path
from typing import List

from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.status import HTTP_206_PARTIAL_CONTENT, HTTP_404_NOT_FOUND, HTTP_200_OK

from config import GlobalConfig
from services.search import SearchFactory

from .routers import frames, registry, search, system, videos

WORK_DIR = Path(os.getenv("AIC25_WORK_DIR") or ".")
logger = logging.getLogger(__name__)

GlobalConfig.initialize(WORK_DIR)

_searcher = None


def _get_database_type():
    """Get database type from config, with fallback."""
    try:
        return GlobalConfig.get("webui", "database") or "faiss"
    except Exception as e:
        logger.warning(f"Could not get database type from config: {e}")
        logger.warning("Using default database type: faiss")
        return "faiss"


def _get_searcher():
    """Get a searcher instance, initializing if needed."""
    global _searcher
    if _searcher is None:
        database_type = _get_database_type()
        _searcher = SearchFactory.create_searcher("default_collection", database_type)
    return _searcher


app = FastAPI(
    title="AIC25 Multimedia Retrieval API",
    description="Enhanced multimedia search and retrieval system with video, frame, and audio search capabilities",
    version="2.0.0",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(frames.router)
app.include_router(videos.router)
app.include_router(system.router)
app.include_router(registry.router)


@app.get("/api/search")
async def search(
    request: Request,
    model: str = "clip",
    q: str = "",
    offset: int = 0,
    limit: int = 50,
    nprobe: int = 8,
    temporal_k: int = 10000,
    ocr_weight: float = 1.0,
    ocr_threshold: int = 40,
    max_interval: int = 250,
    selected: str | None = None,
    use_sentence_transformer: bool = False,
):
    try:
        searcher = _get_searcher()
        available_models = searcher.get_models()

        if use_sentence_transformer and "sentence_transformer" in available_models:
            model = "sentence_transformer"
            logger.info(f"Using sentence transformer model for semantic search")
        elif use_sentence_transformer:
            logger.warning(
                "Sentence transformer requested but not available, using default model"
            )

        res = searcher.search(
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
        list_frames = []
        for record in res["results"]:
            data = record["entity"]
            video_frame_str = data["frame_id"]
            video_id, frame_id = video_frame_str.split("#")
            frame_uri = (
                f"{request.base_url}api/files/keyframes/{video_id}/{frame_id}.jpg"
            )
            video_uri = f"{request.base_url}api/stream/videos/{video_id}.mp4"
            try:
                with open(WORK_DIR / "videos_info" / f"{video_id}.json", "r") as f:
                    fps = json.load(f)["frame_rate"]
            except Exception as e:
                logger.warning(f"Could not load FPS for {video_id}: {e}")
                fps = 25

            list_frames.append(
                dict(
                    id=video_frame_str,
                    video_id=video_id,
                    frame_id=frame_id,
                    frame_uri=frame_uri,
                    video_uri=video_uri,
                    fps=fps,
                )
            )

        params = {
            "model": model,
            "limit": limit,
            "nprobe": nprobe,
            "temporal_k": temporal_k,
            "ocr_weight": ocr_weight,
            "ocr_threshold": ocr_threshold,
            "max_interval": max_interval,
        }
        return {
            "total": res["total"],
            "frames": list_frames,
            "params": params,
            "offset": res["offset"],
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        if "AssertionError" in str(e) or "ntotal" in str(e):
            return {
                "error": "No data indexed yet. Please run 'aic25-cli analyse' and 'aic25-cli index' first.",
                "total": 0,
                "frames": [],
                "params": {
                    "model": model,
                    "limit": limit,
                    "nprobe": nprobe,
                    "temporal_k": temporal_k,
                    "ocr_weight": ocr_weight,
                    "ocr_threshold": ocr_threshold,
                    "max_interval": max_interval,
                },
                "offset": offset,
            }
        else:
            return {
                "error": f"Search failed: {str(e)}",
                "total": 0,
                "frames": [],
                "params": {
                    "model": model,
                    "limit": limit,
                    "nprobe": nprobe,
                    "temporal_k": temporal_k,
                    "ocr_weight": ocr_weight,
                    "ocr_threshold": ocr_threshold,
                    "max_interval": max_interval,
                },
                "offset": offset,
            }


@app.get("/api/similar")
async def similar(
    request: Request,
    id: str,
    model: str = "clip",
    offset: int = 0,
    limit: int = 50,
    nprobe: int = 8,
    temporal_k: int = 10000,
    ocr_weight: float = 1.0,
    ocr_threshold: int = 40,
    max_interval: int = 250,
):
    res = _get_searcher().search_similar(id, offset, limit, nprobe, model)
    list_frames = []
    for record in res["results"]:
        data = record["entity"]
        video_frame_str = data["frame_id"]
        video_id, frame_id = video_frame_str.split("#")
        frame_uri = f"{request.base_url}api/files/keyframes/{video_id}/{frame_id}.jpg"
        video_uri = f"{request.base_url}api/stream/videos/{video_id}.mp4"
        try:
            with open(WORK_DIR / "videos_info" / f"{video_id}.json", "r") as f:
                fps = json.load(f)["frame_rate"]
        except Exception as e:
            logger.warning(f"Could not load FPS for {video_id}: {e}")
            logger.warning(f"Using default FPS: 25")
            fps = 25

        list_frames.append(
            dict(
                id=video_frame_str,
                video_id=video_id,
                frame_id=frame_id,
                frame_uri=frame_uri,
                video_uri=video_uri,
                fps=fps,
            )
        )

    params = {
        "model": model,
        "limit": limit,
        "nprobe": nprobe,
        "temporal_k": temporal_k,
        "ocr_weight": ocr_weight,
        "ocr_threshold": ocr_threshold,
        "max_interval": max_interval,
    }
    return {
        "total": res["total"],
        "frames": list_frames,
        "params": params,
        "offset": res["offset"],
    }


@app.get("/api/frame_info")
async def frame_info(request: Request, video_id: str, frame_id: str):
    id = f"{video_id}#{frame_id}"
    record = _get_searcher().get(id)
    frame_uri = f"{request.base_url}api/files/keyframes/{video_id}/{frame_id}.jpg"
    video_uri = f"{request.base_url}api/stream/videos/{video_id}.mp4"
    try:
        with open(WORK_DIR / "videos_info" / f"{video_id}.json", "r") as f:
            fps = json.load(f)["frame_rate"]
    except Exception as e:
        logger.warning(f"Could not load FPS for {video_id}: {e}")
        logger.warning(f"Using default FPS: 25")
        fps = 25
    return dict(
        id=id if len(record) > 0 else None,
        video_id=video_id,
        frame_id=frame_id,
        frame_uri=frame_uri if len(record) > 0 else None,
        video_uri=video_uri,
        fps=fps,
    )


@app.get("/api/files/{file_path:path}")
async def get_file(file_path):
    return FileResponse(str(WORK_DIR / file_path))


CHUNK_SIZE = 1024 * 1024


@app.get("/api/stream/{file_path:path}")
async def video_endpoint(
    file_path: str, param_range: str = Header(None, alias="range")
):
    video_path = WORK_DIR / file_path

    if not video_path.exists():
        logger.warning(f"Video not found: {video_path}")
        return Response("Video not found", status_code=HTTP_404_NOT_FOUND)

    filesize = video_path.stat().st_size

    if param_range:
        # Parse range header: "bytes=start-end"
        range_match = param_range.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = (
            int(range_match[1])
            if range_match[1]
            else min(start + CHUNK_SIZE, filesize - 1)
        )

        # Ensure end doesn't exceed file size
        end = min(end, filesize - 1)

        with open(video_path, "rb") as video:
            video.seek(start)
            data = video.read(end - start + 1)

        headers = {
            "Content-Range": f"bytes {start}-{end}/{filesize}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        }

        return Response(
            data,
            status_code=HTTP_206_PARTIAL_CONTENT,
            headers=headers,
            media_type="video/mp4",
        )
    else:
        # If no Range header, serve the entire file
        with open(video_path, "rb") as video:
            data = video.read()

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(filesize),
        }

        return Response(
            data,
            status_code=HTTP_200_OK,
            headers=headers,
            media_type="video/mp4",
        )


@app.get("/api/search/audio")
async def search_audio_by_text(
    request: Request,
    q: str = "",
    offset: int = 0,
    limit: int = 50,
    nprobe: int = 8,
    model: str = "clip"
):
    """Text-based audio search endpoint - search for audio content using text query"""
    try:
        if not q.strip():
            return {"error": "Query text is required for audio search"}

        searcher = _get_searcher()
        available_models = searcher.get_models()

        if "audio" in available_models:
            model = "audio"
        elif model not in available_models:
            logger.warning(
                f"Model '{model}' not available. Available models: {available_models}. Using 'clip' instead."
            )
            model = "clip"

        res = searcher.search(
            q,
            "",  # image_path (not used for audio)
            offset,
            limit,
            nprobe,
            model,  # Use available model
            10000,  # temporal_k
            1.0,  # ocr_weight
            40,  # ocr_threshold
            250,  # max_interval
            None,  # selected
        )

        list_frames: List[dict] = []
        for record in res["results"]:
            data = record["entity"]
            video_frame_str = data["frame_id"]
            video_id, frame_id = video_frame_str.split("#")
            frame_uri = (
                f"{request.base_url}api/files/keyframes/{video_id}/{frame_id}.jpg"
            )
            video_uri = f"{request.base_url}api/stream/videos/{video_id}.mp4"
            try:
                with open(WORK_DIR / "videos_info" / f"{video_id}.json", "r") as f:
                    fps = json.load(f)["frame_rate"]
            except Exception as e:
                logger.warning(f"Could not load FPS for {video_id}: {e}")
                logger.warning(f"Using default FPS: 25")
                fps = 25

            list_frames.append(
                dict(
                    id=video_frame_str,
                    video_id=video_id,
                    frame_id=frame_id,
                    frame_uri=frame_uri,
                    video_uri=video_uri,
                    fps=fps,
                )
            )

        params = {
            "model": model,
            "limit": limit,
            "nprobe": nprobe,
            "search_type": "audio_text",
            "query": q,
        }

        return {
            "total": res["total"],
            "frames": list_frames,
            "params": params,
            "offset": res["offset"],
            "search_type": "audio_text",
        }

    except Exception as e:
        error_trace = traceback.format_exc()

        logger.error(f"Audio text search failed: {e}")
        logger.debug(f"Full traceback: {error_trace}")
        if (
            "AssertionError" in str(e)
            or "ntotal" in str(e)
            or "audio" in str(e).lower()
        ):
            return {
                "error": "Audio features not indexed yet. Please run 'aic25-cli analyse' with audio model and 'aic25-cli index' first.",
                "total": 0,
                "frames": [],
                "params": {
                    "model": model,
                    "limit": limit,
                    "nprobe": nprobe,
                    "search_type": "audio_text",
                    "query": q,
                },
                "offset": offset,
                "search_type": "audio_text",
            }
        else:
            return {"error": f"Audio text search failed: {str(e)}"}


@app.get("/api/models")
async def models():
    return {"models": _get_searcher().get_models()}


WEB_DIR = WORK_DIR / ".web"
DIST_DIR = WEB_DIR / "dist"

if DIST_DIR.exists():
    assets_dir = DIST_DIR / "assets"
    icon_dir = DIST_DIR / "icon"

    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            "assets",
        )

    if icon_dir.exists():
        app.mount(
            "/icon",
            StaticFiles(directory=icon_dir),
            "icon",
        )

    index_file = DIST_DIR / "index.html"
    if index_file.exists():

        @app.get("/{rest_of_path:path}")
        async def client_app():
            return FileResponse(index_file)

    else:

        @app.get("/")
        async def root():
            return {
                "message": "AIC25 Multimedia Retrieval API",
                "status": "running",
                "frontend": "not built",
            }

else:

    @app.get("/")
    async def root():
        return {
            "message": "AIC25 Multimedia Retrieval API",
            "status": "running",
            "frontend": "not built",
        }
