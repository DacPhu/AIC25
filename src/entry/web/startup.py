"""
Startup script for distributed AIC25 backend with service discovery.
"""

import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config import GlobalConfig
from services.discovery import MulticastDiscovery, ServiceDiscovery, service_registry
from services.load_balancer import load_balancer

logger = logging.getLogger(__name__)

service_discovery = None
multicast_discovery = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for service discovery"""
    global service_discovery, multicast_discovery

    logger.info("Starting AIC25 Multimedia Retrieval Service...")

    host = os.getenv("AIC25_HOST", "0.0.0.0")
    port = int(os.getenv("AIC25_PORT", "5000"))
    registry_endpoints = os.getenv("AIC25_REGISTRY_ENDPOINTS", "").split(",")
    registry_endpoints = [
        endpoint.strip() for endpoint in registry_endpoints if endpoint.strip()
    ]

    service_registry.start()

    load_balancer.start()

    service_discovery = ServiceDiscovery("backend", host, port)
    service_discovery.set_capabilities(
        ["search", "frames", "videos", "system", "audio", "similar"]
    )
    service_discovery.set_metadata(
        {
            "version": "2.0.0",
            "database_type": GlobalConfig.get("webui", "database") or "faiss",
            "features": ["enhanced_api", "load_balancing", "service_discovery"],
        }
    )

    for endpoint in registry_endpoints:
        service_discovery.add_registry_endpoint(endpoint)

    service_discovery.register_service()
    service_discovery.start_heartbeat()

    multicast_discovery = MulticastDiscovery(service_discovery)
    multicast_discovery.start()

    logger.info(f"Service registered and discovery started on {host}:{port}")

    yield

    logger.info("Shutting down service discovery...")

    if service_discovery:
        service_discovery.stop_heartbeat()
        service_discovery.unregister_service()

    if multicast_discovery:
        multicast_discovery.stop()

    load_balancer.stop()
    service_registry.stop()

    logger.info("Service discovery shutdown complete")


def create_app():
    """Create FastAPI app with service discovery"""
    from .app import app
    from .routers import frames, registry, search, system, videos

    new_app = FastAPI(
        title="AIC25 Multimedia Retrieval API",
        description="Enhanced multimedia search and retrieval system with video, frame, and audio search capabilities",
        version="2.0.0",
        lifespan=lifespan,
    )

    new_app.middleware_stack = app.middleware_stack

    new_app.include_router(search.router)
    new_app.include_router(frames.router)
    new_app.include_router(videos.router)
    new_app.include_router(system.router)
    new_app.include_router(registry.router)

    for route in app.router.routes:
        new_app.router.routes.append(route)

    return new_app


def run_server():
    """Run the server with appropriate configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    host = os.getenv("AIC25_HOST", "0.0.0.0")
    port = int(os.getenv("AIC25_PORT", "5000"))
    workers = int(os.getenv("AIC25_WORKERS", "1"))
    reload = os.getenv("AIC25_RELOAD", "false").lower() == "true"

    app = create_app()

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Reload only works with 1 worker
        reload=reload,
        loop="asyncio",
        lifespan="on",
    )

    def signal_handler(signum, _frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server = uvicorn.Server(config)

    try:
        logger.info(f"Starting AIC25 server on {host}:{port}")
        server.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Server shutdown")


if __name__ == "__main__":
    run_server()
