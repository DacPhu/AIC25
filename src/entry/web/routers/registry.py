"""
Service Registry API routes for distributed service discovery.
"""

import logging
import aiohttp
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from starlette.status import (
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
)

from services.discovery import ServiceInstance, service_registry

# Setup
router = APIRouter(prefix="/api/v1/registry", tags=["service-registry"])
logger = logging.getLogger(__name__)


@router.get("/services", response_model=dict)
async def list_services(
    service_type: Optional[str] = Query(None, description="Filter by service type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
):
    """
    List all registered services with optional filtering.

    Returns a list of all services currently registered in the service registry.
    Supports filtering by service type, status, and device ID.
    """
    try:
        services = service_registry.get_all_services()

        if service_type:
            services = [s for s in services if s.service_type == service_type]

        if status:
            services = [s for s in services if s.status == status]

        if device_id:
            services = [s for s in services if s.device_id == device_id]

        services_data = [service.to_dict() for service in services]

        return {
            "success": True,
            "services": services_data,
            "total": len(services_data),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to list services", "message": str(e)},
        )


@router.post("/services")
async def register_service(service_data: dict = Body(...)):
    """
    Register a new service instance.

    Accepts service registration data and adds the service to the registry.
    The service will be monitored via heartbeats.
    """
    try:
        service = ServiceInstance.from_dict(service_data)

        success = service_registry.register_service(service)

        if success:
            return {
                "success": True,
                "message": "Service registered successfully",
                "service_id": service.service_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Failed to register service",
                    "message": "Service registration was rejected",
                },
            )

    except ValueError as e:
        logger.error(f"Invalid service data: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid service data", "message": str(e)},
        )
    except Exception as e:
        logger.error(f"Failed to register service: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to register service", "message": str(e)},
        )


@router.delete("/services/{service_id}")
async def unregister_service(service_id: str):
    """
    Unregister a service instance.

    Removes the specified service from the registry. This should be called
    when a service is shutting down gracefully.
    """
    try:
        success = service_registry.unregister_service(service_id)

        if success:
            return {
                "success": True,
                "message": "Service unregistered successfully",
                "service_id": service_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail={"error": "Service not found", "service_id": service_id},
            )

    except Exception as e:
        logger.error(f"Failed to unregister service {service_id}: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to unregister service",
                "message": str(e),
                "service_id": service_id,
            },
        )


@router.get("/services/{service_id}")
async def get_service(service_id: str):
    """
    Get details of a specific service instance.

    Returns detailed information about a single service identified by its ID.
    """
    try:
        service = service_registry.get_service_by_id(service_id)

        if service:
            return {
                "success": True,
                "service": service.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail={"error": "Service not found", "service_id": service_id},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service {service_id}: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to get service",
                "message": str(e),
                "service_id": service_id,
            },
        )


@router.put("/services/{service_id}/heartbeat")
async def update_heartbeat(service_id: str):
    """
    Update the heartbeat timestamp for a service.

    Services should call this endpoint periodically to indicate they are still alive.
    If a service stops sending heartbeats, it will be marked as expired and removed.
    """
    try:
        success = service_registry.update_heartbeat(service_id)

        if success:
            return {
                "success": True,
                "message": "Heartbeat updated successfully",
                "service_id": service_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail={"error": "Service not found", "service_id": service_id},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update heartbeat for service {service_id}: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to update heartbeat",
                "message": str(e),
                "service_id": service_id,
            },
        )


@router.get("/discovery/backends")
async def discover_backends(
    capabilities: Optional[List[str]] = Query(
        None, description="Required capabilities"
    ),
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of backends to return"
    ),
):
    """
    Discover available backend services.

    Returns a list of healthy backend services, optionally filtered by capabilities.
    This is the main endpoint for frontend clients to find available backends.
    """
    try:
        backends = service_registry.get_healthy_backends()

        if capabilities:
            backends = [
                backend
                for backend in backends
                if all(cap in backend.capabilities for cap in capabilities)
            ]

        backends = backends[:limit]

        backends.sort(key=lambda x: x.last_heartbeat, reverse=True)

        return {
            "success": True,
            "backends": [backend.to_dict() for backend in backends],
            "total": len(backends),
            "capabilities_filter": capabilities,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to discover backends: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to discover backends", "message": str(e)},
        )


@router.get("/discovery/best-backend")
async def get_best_backend(
    capabilities: Optional[List[str]] = Query(None, description="Required capabilities")
):
    """
    Get the best available backend service.

    Returns the single best backend service based on load balancing criteria.
    This endpoint implements intelligent backend selection.
    """
    try:
        backend = service_registry.get_best_backend(capabilities)

        if backend:
            return {
                "success": True,
                "backend": backend.to_dict(),
                "selection_criteria": "least_recent_heartbeat",
                "capabilities_filter": capabilities,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "success": False,
                "backend": None,
                "message": "No suitable backend found",
                "capabilities_filter": capabilities,
                "timestamp": datetime.utcnow().isoformat(),
            }

    except Exception as e:
        logger.error(f"Failed to get best backend: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get best backend", "message": str(e)},
        )


@router.get("/health")
async def registry_health():
    """
    Get health status of the service registry.

    Returns information about the registry's health and statistics about
    registered services.
    """
    try:
        services = service_registry.get_all_services()
        healthy_services = [
            s for s in services if s.status == "healthy" and not s.is_expired
        ]

        service_counts = {}
        for service in healthy_services:
            service_type = service.service_type
            service_counts[service_type] = service_counts.get(service_type, 0) + 1

        devices = set(s.device_id for s in healthy_services)

        return {
            "success": True,
            "status": "healthy",
            "registry_uptime": "unknown",
            "statistics": {
                "total_services": len(services),
                "healthy_services": len(healthy_services),
                "service_types": service_counts,
                "unique_devices": len(devices),
                "expired_services": len([s for s in services if s.is_expired]),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get registry health: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get registry health", "message": str(e)},
        )


@router.post("/sync")
async def sync_with_remote_registry(remote_registry_url: str = Body(..., embed=True)):
    """
    Synchronize with a remote service registry.

    Fetches services from a remote registry and merges them with the local registry.
    This enables distributed registry synchronization.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{remote_registry_url}/api/v1/registry/services",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    remote_services = data.get("services", [])

                    synced_count = 0
                    for service_data in remote_services:
                        try:
                            service = ServiceInstance.from_dict(service_data)
                            # Only sync if not already in local registry or if remote is newer
                            existing = service_registry.get_service_by_id(
                                service.service_id
                            )
                            if (
                                not existing
                                or existing.last_heartbeat < service.last_heartbeat
                            ):
                                service_registry.register_service(service)
                                synced_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to sync service: {e}")

                    return {
                        "success": True,
                        "message": f"Synchronized with remote registry",
                        "remote_url": remote_registry_url,
                        "synced_services": synced_count,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                else:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Failed to connect to remote registry",
                            "remote_url": remote_registry_url,
                            "status_code": response.status,
                        },
                    )

    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail={
                "error": "Failed to connect to remote registry",
                "remote_url": remote_registry_url,
                "message": str(e),
            },
        )
    except Exception as e:
        logger.error(f"Failed to sync with remote registry: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to sync with remote registry", "message": str(e)},
        )
