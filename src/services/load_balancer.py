"""
Load Balancer for distributed AIC25 backend services.
Implements multiple load balancing strategies and health checking.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests

from .discovery import ServiceInstance, service_registry

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Available load balancing strategies"""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    HEALTH_WEIGHTED = "health_weighted"
    RESPONSE_TIME = "response_time"


@dataclass
class BackendHealth:
    """Health metrics for a backend service"""

    service_id: str
    response_time: float  # Average response time in milliseconds
    success_rate: float  # Success rate (0-1)
    active_connections: int
    last_health_check: float
    consecutive_failures: int
    is_healthy: bool


class HealthChecker:
    """Health checker for backend services"""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_data: Dict[str, BackendHealth] = {}
        self.running = False
        self.check_task = None

    def start(self):
        """Start health checking"""
        self.running = True
        self.check_task = threading.Thread(target=self._health_check_loop, daemon=True)
        self.check_task.start()
        logger.info("Health checker started")

    def stop(self):
        """Stop health checking"""
        self.running = False
        if self.check_task:
            self.check_task.join(timeout=5)
        logger.info("Health checker stopped")

    def check_backend_health(self, service: ServiceInstance) -> BackendHealth:
        """Check health of a single backend service"""
        start_time = time.time()

        try:
            response = requests.get(
                f"{service.base_url}/api/v1/system/health", timeout=10
            )
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            is_healthy = response.status_code == 200

            existing = self.health_data.get(service.service_id)
            if existing:
                response_time = (existing.response_time * 0.7) + (response_time * 0.3)
                if is_healthy:
                    success_rate = min(1.0, existing.success_rate + 0.1)
                    consecutive_failures = 0
                else:
                    success_rate = max(0.0, existing.success_rate - 0.2)
                    consecutive_failures = existing.consecutive_failures + 1
            else:
                success_rate = 1.0 if is_healthy else 0.0
                consecutive_failures = 0 if is_healthy else 1

            health = BackendHealth(
                service_id=service.service_id,
                response_time=response_time,
                success_rate=success_rate,
                active_connections=0,  # Would need to implement connection tracking
                last_health_check=time.time(),
                consecutive_failures=consecutive_failures,
                is_healthy=is_healthy and consecutive_failures < 3,
            )

            self.health_data[service.service_id] = health
            return health

        except Exception as e:
            logger.debug(f"Health check failed for {service.service_id}: {e}")

            # Mark as unhealthy
            existing = self.health_data.get(service.service_id)
            consecutive_failures = (
                (existing.consecutive_failures + 1) if existing else 1
            )

            health = BackendHealth(
                service_id=service.service_id,
                response_time=existing.response_time if existing else 5000.0,
                success_rate=max(0.0, existing.success_rate - 0.3) if existing else 0.0,
                active_connections=0,
                last_health_check=time.time(),
                consecutive_failures=consecutive_failures,
                is_healthy=False,
            )

            self.health_data[service.service_id] = health
            return health

    def _health_check_loop(self):
        """Background health checking loop"""
        while self.running:
            try:
                backends = service_registry.get_healthy_backends()

                # Check health of all backends
                for backend in backends:
                    try:
                        self.check_backend_health(backend)
                    except Exception as e:
                        logger.debug(
                            f"Health check failed for {backend.service_id}: {e}"
                        )

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(self.check_interval)

    def get_health(self, service_id: str) -> Optional[BackendHealth]:
        """Get health data for a service"""
        return self.health_data.get(service_id)

    def get_all_health(self) -> Dict[str, BackendHealth]:
        """Get health data for all services"""
        return self.health_data.copy()


class LoadBalancer:
    """Main load balancer class"""

    def __init__(
        self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_WEIGHTED
    ):
        self.strategy = strategy
        self.health_checker = HealthChecker()
        self.round_robin_counter = 0
        self.weights: Dict[str, float] = {}  # Service weights for weighted strategies

        # Strategy implementations
        self.strategies: Dict[LoadBalancingStrategy, Callable] = {
            LoadBalancingStrategy.ROUND_ROBIN: self._round_robin,
            LoadBalancingStrategy.LEAST_CONNECTIONS: self._least_connections,
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: self._weighted_round_robin,
            LoadBalancingStrategy.RANDOM: self._random,
            LoadBalancingStrategy.HEALTH_WEIGHTED: self._health_weighted,
            LoadBalancingStrategy.RESPONSE_TIME: self._response_time,
        }

    def start(self):
        """Start the load balancer"""
        self.health_checker.start()
        logger.info(f"Load balancer started with strategy: {self.strategy.value}")

    def stop(self):
        """Stop the load balancer"""
        self.health_checker.stop()
        logger.info("Load balancer stopped")

    def set_strategy(self, strategy: LoadBalancingStrategy):
        """Change the load balancing strategy"""
        self.strategy = strategy
        logger.info(f"Load balancing strategy changed to: {strategy.value}")

    def set_weight(self, service_id: str, weight: float):
        """Set weight for a service (used in weighted strategies)"""
        self.weights[service_id] = weight

    def select_backend(
        self,
        capabilities: Optional[List[str]] = None,
        exclude_services: Optional[List[str]] = None,
    ) -> Optional[ServiceInstance]:
        """
        Select the best backend service based on the configured strategy.

        Args:
            capabilities: Required capabilities for the backend
            exclude_services: Service IDs to exclude from selection

        Returns:
            Selected ServiceInstance or None if no suitable backend found
        """
        # Get all healthy backends
        backends = service_registry.get_healthy_backends()

        if not backends:
            logger.warning("No healthy backends available")
            return None

        # Filter by capabilities
        if capabilities:
            backends = [
                b
                for b in backends
                if all(cap in b.capabilities for cap in capabilities)
            ]

        # Exclude specific services
        if exclude_services:
            backends = [b for b in backends if b.service_id not in exclude_services]

        if not backends:
            logger.warning("No backends match the specified criteria")
            return None

        # Filter by health status
        healthy_backends = []
        for backend in backends:
            health = self.health_checker.get_health(backend.service_id)
            if not health or health.is_healthy:
                healthy_backends.append(backend)

        if not healthy_backends:
            logger.warning("No healthy backends available after health filtering")
            # If no healthy backends, fall back to registry status
            healthy_backends = backends

        # Apply load balancing strategy
        strategy_func = self.strategies.get(self.strategy)
        if strategy_func:
            return strategy_func(healthy_backends)
        else:
            logger.error(f"Unknown load balancing strategy: {self.strategy}")
            return random.choice(healthy_backends) if healthy_backends else None

    def _round_robin(self, backends: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin load balancing"""
        if not backends:
            return None

        selected = backends[self.round_robin_counter % len(backends)]
        self.round_robin_counter += 1
        return selected

    def _least_connections(self, backends: List[ServiceInstance]) -> ServiceInstance:
        """Least connections load balancing"""
        if not backends:
            return None

        # Select backend with least active connections
        min_connections = float("inf")
        selected = None

        for backend in backends:
            health = self.health_checker.get_health(backend.service_id)
            connections = health.active_connections if health else 0

            if connections < min_connections:
                min_connections = connections
                selected = backend

        return selected or backends[0]

    def _weighted_round_robin(self, backends: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round-robin load balancing"""
        if not backends:
            return None

        # Calculate total weight
        total_weight = 0
        for backend in backends:
            weight = self.weights.get(backend.service_id, 1.0)
            total_weight += weight

        if total_weight == 0:
            return self._round_robin(backends)

        # Select based on weights
        target = random.uniform(0, total_weight)
        current_weight = 0

        for backend in backends:
            weight = self.weights.get(backend.service_id, 1.0)
            current_weight += weight
            if current_weight >= target:
                return backend

        return backends[-1]  # Fallback

    def _random(self, backends: List[ServiceInstance]) -> ServiceInstance:
        """Random load balancing"""
        return random.choice(backends) if backends else None

    def _health_weighted(self, backends: List[ServiceInstance]) -> ServiceInstance:
        """Health-weighted load balancing (considers success rate and response time)"""
        if not backends:
            return None

        # Calculate scores for each backend
        scored_backends = []
        for backend in backends:
            health = self.health_checker.get_health(backend.service_id)

            if health:
                # Score based on success rate and inverse response time
                # Higher success rate and lower response time = higher score
                response_score = 1000.0 / max(
                    health.response_time, 1.0
                )  # Inverse response time
                success_score = health.success_rate * 100  # Success rate weight

                total_score = (response_score * 0.4) + (success_score * 0.6)
            else:
                # Default score for services without health data
                total_score = 50.0

            scored_backends.append((backend, total_score))

        # Select based on weighted random selection
        total_score = sum(score for _, score in scored_backends)
        if total_score == 0:
            return random.choice(backends)

        target = random.uniform(0, total_score)
        current_score = 0

        for backend, score in scored_backends:
            current_score += score
            if current_score >= target:
                return backend

        return backends[-1]  # Fallback

    def _response_time(self, backends: List[ServiceInstance]) -> ServiceInstance:
        """Response time based load balancing (fastest response wins)"""
        if not backends:
            return None

        fastest_backend = None
        fastest_time = float("inf")

        for backend in backends:
            health = self.health_checker.get_health(backend.service_id)
            response_time = health.response_time if health else 1000.0

            if response_time < fastest_time:
                fastest_time = response_time
                fastest_backend = backend

        return fastest_backend or backends[0]

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        backends = service_registry.get_healthy_backends()
        health_data = self.health_checker.get_all_health()

        stats = {
            "strategy": self.strategy.value,
            "total_backends": len(backends),
            "healthy_backends": len(
                [
                    b
                    for b in backends
                    if health_data.get(b.service_id, {}).get("is_healthy", True)
                ]
            ),
            "average_response_time": 0.0,
            "overall_success_rate": 0.0,
            "backend_details": [],
        }

        if health_data:
            total_response_time = sum(h.response_time for h in health_data.values())
            total_success_rate = sum(h.success_rate for h in health_data.values())

            stats["average_response_time"] = total_response_time / len(health_data)
            stats["overall_success_rate"] = total_success_rate / len(health_data)

        # Add individual backend details
        for backend in backends:
            health = health_data.get(backend.service_id)
            backend_stats = {
                "service_id": backend.service_id,
                "host": backend.host,
                "port": backend.port,
                "capabilities": backend.capabilities,
                "device_name": backend.device_name,
                "weight": self.weights.get(backend.service_id, 1.0),
            }

            if health:
                backend_stats.update(
                    {
                        "response_time": health.response_time,
                        "success_rate": health.success_rate,
                        "active_connections": health.active_connections,
                        "is_healthy": health.is_healthy,
                        "consecutive_failures": health.consecutive_failures,
                    }
                )

            stats["backend_details"].append(backend_stats)

        return stats


# Global load balancer instance
load_balancer = LoadBalancer()


def get_best_backend(
    capabilities: Optional[List[str]] = None,
) -> Optional[ServiceInstance]:
    """Convenience function to get the best backend using the global load balancer"""
    return load_balancer.select_backend(capabilities=capabilities)
