"""
Service Discovery System for Distributed AIC25 Deployment
Supports multiple backend instances across different devices with automatic discovery and load balancing.
"""

import json
import logging
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class ServiceInstance:
    """Represents a service instance in the network"""

    service_id: str
    service_type: str  # "backend", "frontend"
    host: str
    port: int
    version: str
    status: str  # "healthy", "unhealthy", "unknown"
    capabilities: List[str]  # e.g., ["search", "video", "audio"]
    metadata: Dict[str, Any]
    last_heartbeat: float
    device_id: str
    device_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceInstance":
        return cls(**data)

    @property
    def is_expired(self) -> bool:
        """Check if service instance has expired (no heartbeat for 30 seconds)"""
        return time.time() - self.last_heartbeat > 30

    @property
    def base_url(self) -> str:
        """Get the base URL for this service"""
        return f"http://{self.host}:{self.port}"


class ServiceRegistry:
    """Centralized service registry for tracking all instances"""

    def __init__(self):
        self.services: Dict[str, ServiceInstance] = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 10  # seconds
        self.cleanup_thread = None
        self.running = False

    def start(self):
        """Start the registry cleanup thread"""
        self.running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_expired_services, daemon=True
        )
        self.cleanup_thread.start()
        logger.info("Service registry started")

    def stop(self):
        """Stop the registry"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        logger.info("Service registry stopped")

    def register_service(self, service: ServiceInstance) -> bool:
        """Register a new service instance"""
        with self.lock:
            self.services[service.service_id] = service
            logger.info(
                f"Registered service: {service.service_type}@{service.host}:{service.port}"
            )
            return True

    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service instance"""
        with self.lock:
            if service_id in self.services:
                service = self.services.pop(service_id)
                logger.info(
                    f"Unregistered service: {service.service_type}@{service.host}:{service.port}"
                )
                return True
            return False

    def update_heartbeat(self, service_id: str) -> bool:
        """Update the heartbeat timestamp for a service"""
        with self.lock:
            if service_id in self.services:
                self.services[service_id].last_heartbeat = time.time()
                self.services[service_id].status = "healthy"
                return True
            return False

    def get_services_by_type(self, service_type: str) -> List[ServiceInstance]:
        """Get all healthy services of a specific type"""
        with self.lock:
            return [
                service
                for service in self.services.values()
                if service.service_type == service_type
                and service.status == "healthy"
                and not service.is_expired
            ]

    def get_service_by_id(self, service_id: str) -> Optional[ServiceInstance]:
        """Get a specific service by ID"""
        with self.lock:
            return self.services.get(service_id)

    def get_all_services(self) -> List[ServiceInstance]:
        """Get all services"""
        with self.lock:
            return list(self.services.values())

    def get_healthy_backends(self) -> List[ServiceInstance]:
        """Get all healthy backend services"""
        return self.get_services_by_type("backend")

    def get_best_backend(
        self, capabilities: Optional[List[str]] = None
    ) -> Optional[ServiceInstance]:
        """Get the best backend service based on capabilities and load"""
        backends = self.get_healthy_backends()

        if not backends:
            return None

        # Filter by capabilities if specified
        if capabilities:
            backends = [
                b
                for b in backends
                if all(cap in b.capabilities for cap in capabilities)
            ]

        if not backends:
            return None

        # Simple load balancing: return service with least recent heartbeat
        # In production, you might want to use actual load metrics
        return min(backends, key=lambda x: x.last_heartbeat)

    def _cleanup_expired_services(self):
        """Background thread to clean up expired services"""
        while self.running:
            try:
                with self.lock:
                    expired_ids = [
                        service_id
                        for service_id, service in self.services.items()
                        if service.is_expired
                    ]

                    for service_id in expired_ids:
                        service = self.services.pop(service_id)
                        logger.warning(
                            f"Removed expired service: {service.service_type}@{service.host}:{service.port}"
                        )

                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
                time.sleep(self.cleanup_interval)


# Global registry instance
service_registry = ServiceRegistry()


class ServiceDiscovery:
    """Service discovery client for individual services"""

    def __init__(self, service_type: str, host: str = "0.0.0.0", port: int = 5000):
        self.service_type = service_type
        self.host = host if host != "0.0.0.0" else self._get_local_ip()
        self.port = port
        self.service_id = str(uuid.uuid4())
        self.device_id = self._get_device_id()
        self.device_name = socket.gethostname()
        self.heartbeat_interval = 10  # seconds
        self.heartbeat_thread = None
        self.running = False
        self.registry_endpoints: List[str] = []
        self.capabilities: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def _get_local_ip(self) -> str:
        """Get the local IP address"""
        try:
            # Connect to a remote address to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def _get_device_id(self) -> str:
        """Get a unique device identifier"""
        try:
            # Try to get MAC address as device ID
            import uuid

            mac = hex(uuid.getnode())
            return mac
        except Exception:
            return str(uuid.uuid4())

    def set_capabilities(self, capabilities: List[str]):
        """Set service capabilities"""
        self.capabilities = capabilities

    def set_metadata(self, metadata: Dict[str, Any]):
        """Set service metadata"""
        self.metadata = metadata

    def add_registry_endpoint(self, endpoint: str):
        """Add a registry endpoint for distributed discovery"""
        if endpoint not in self.registry_endpoints:
            self.registry_endpoints.append(endpoint)

    def create_service_instance(self, status: str = "healthy") -> ServiceInstance:
        """Create a ServiceInstance object for this service"""
        return ServiceInstance(
            service_id=self.service_id,
            service_type=self.service_type,
            host=self.host,
            port=self.port,
            version="2.0.0",
            status=status,
            capabilities=self.capabilities,
            metadata=self.metadata,
            last_heartbeat=time.time(),
            device_id=self.device_id,
            device_name=self.device_name,
        )

    def register_service(self) -> bool:
        """Register this service with available registries"""
        service = self.create_service_instance()

        # Register with local registry
        success_local = service_registry.register_service(service)

        # Register with remote registries
        success_remote = True
        for endpoint in self.registry_endpoints:
            try:
                response = requests.post(
                    f"{endpoint}/api/v1/registry/services",
                    json=service.to_dict(),
                    timeout=5,
                )
                if response.status_code != 200:
                    logger.warning(
                        f"Failed to register with remote registry {endpoint}"
                    )
                    success_remote = False
            except Exception as e:
                logger.error(f"Error registering with remote registry {endpoint}: {e}")
                success_remote = False

        return success_local and success_remote

    def unregister_service(self) -> bool:
        """Unregister this service from all registries"""
        # Unregister from local registry
        success_local = service_registry.unregister_service(self.service_id)

        # Unregister from remote registries
        success_remote = True
        for endpoint in self.registry_endpoints:
            try:
                response = requests.delete(
                    f"{endpoint}/api/v1/registry/services/{self.service_id}", timeout=5
                )
                if response.status_code != 200:
                    logger.warning(
                        f"Failed to unregister from remote registry {endpoint}"
                    )
                    success_remote = False
            except Exception as e:
                logger.error(
                    f"Error unregistering from remote registry {endpoint}: {e}"
                )
                success_remote = False

        return success_local and success_remote

    def start_heartbeat(self):
        """Start sending periodic heartbeats"""
        self.running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()
        logger.info(f"Started heartbeat for {self.service_type} service")

    def stop_heartbeat(self):
        """Stop sending heartbeats"""
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        logger.info(f"Stopped heartbeat for {self.service_type} service")

    def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while self.running:
            try:
                # Update local registry
                service_registry.update_heartbeat(self.service_id)

                # Update remote registries
                for endpoint in self.registry_endpoints:
                    try:
                        requests.put(
                            f"{endpoint}/api/v1/registry/services/{self.service_id}/heartbeat",
                            timeout=5,
                        )
                    except Exception as e:
                        logger.debug(f"Heartbeat failed for {endpoint}: {e}")

                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(self.heartbeat_interval)

    def discover_services(self, service_type: str) -> List[ServiceInstance]:
        """Discover services of a specific type from all registries"""
        services = []

        # Get from local registry
        local_services = service_registry.get_services_by_type(service_type)
        services.extend(local_services)

        # Get from remote registries
        for endpoint in self.registry_endpoints:
            try:
                response = requests.get(
                    f"{endpoint}/api/v1/registry/services",
                    params={"service_type": service_type},
                    timeout=5,
                )
                if response.status_code == 200:
                    data = response.json()
                    remote_services = [
                        ServiceInstance.from_dict(s) for s in data.get("services", [])
                    ]
                    services.extend(remote_services)
            except Exception as e:
                logger.debug(f"Error discovering from {endpoint}: {e}")

        # Remove duplicates based on service_id
        seen_ids = set()
        unique_services = []
        for service in services:
            if service.service_id not in seen_ids:
                seen_ids.add(service.service_id)
                unique_services.append(service)

        return unique_services

    def find_best_backend(
        self, capabilities: Optional[List[str]] = None
    ) -> Optional[ServiceInstance]:
        """Find the best available backend service"""
        backends = self.discover_services("backend")

        if not backends:
            return None

        # Filter by capabilities
        if capabilities:
            backends = [
                b
                for b in backends
                if all(cap in b.capabilities for cap in capabilities)
            ]

        if not backends:
            return None

        # Simple load balancing strategy
        return min(backends, key=lambda x: x.last_heartbeat)


class MulticastDiscovery:
    """UDP multicast-based service discovery for local network"""

    MULTICAST_GROUP = "224.1.1.1"
    MULTICAST_PORT = 5353

    def __init__(self, service_discovery: ServiceDiscovery):
        self.service_discovery = service_discovery
        self.sock = None
        self.running = False
        self.listen_thread = None
        self.announce_thread = None

    def start(self):
        """Start multicast discovery"""
        try:
            # Create multicast socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind to multicast group
            self.sock.bind(("", self.MULTICAST_PORT))
            mreq = socket.inet_aton(self.MULTICAST_GROUP) + socket.inet_aton("0.0.0.0")
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            self.running = True

            # Start listener thread
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()

            # Start announcement thread
            self.announce_thread = threading.Thread(
                target=self._announce_loop, daemon=True
            )
            self.announce_thread.start()

            logger.info("Started multicast discovery")

        except Exception as e:
            logger.error(f"Failed to start multicast discovery: {e}")

    def stop(self):
        """Stop multicast discovery"""
        self.running = False

        if self.sock:
            try:
                self.sock.close()
            except:
                pass

        if self.listen_thread:
            self.listen_thread.join(timeout=5)

        if self.announce_thread:
            self.announce_thread.join(timeout=5)

        logger.info("Stopped multicast discovery")

    def _listen_loop(self):
        """Listen for service announcements"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                message = json.loads(data.decode())

                if message.get("type") == "service_announcement":
                    service_data = message.get("service")
                    if service_data:
                        service = ServiceInstance.from_dict(service_data)
                        service_registry.register_service(service)

            except socket.timeout:
                continue
            except Exception as e:
                logger.debug(f"Error in multicast listen loop: {e}")

    def _announce_loop(self):
        """Periodically announce this service"""
        while self.running:
            try:
                service = self.service_discovery.create_service_instance()
                message = {
                    "type": "service_announcement",
                    "service": service.to_dict(),
                    "timestamp": time.time(),
                }

                data = json.dumps(message).encode()

                # Send to multicast group
                send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
                send_sock.sendto(data, (self.MULTICAST_GROUP, self.MULTICAST_PORT))
                send_sock.close()

                time.sleep(30)  # Announce every 30 seconds

            except Exception as e:
                logger.debug(f"Error in multicast announce loop: {e}")
                time.sleep(30)
