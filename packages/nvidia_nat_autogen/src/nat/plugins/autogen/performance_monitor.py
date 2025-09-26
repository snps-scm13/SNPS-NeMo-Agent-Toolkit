# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import functools
import threading
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for AutoGen operations."""

    operation_name: str
    execution_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def average_duration(self) -> float:
        """Calculate average execution duration."""
        return self.total_duration / max(self.execution_count, 1)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.execution_count + self.error_count
        return self.execution_count / max(total, 1)

    @property
    def recent_average(self) -> float:
        """Calculate recent average from last N executions."""
        if not self.recent_durations:
            return 0.0
        return sum(self.recent_durations) / len(self.recent_durations)


class AutoGenPerformanceMonitor:
    """Performance monitoring system for AutoGen integration.

    This class provides comprehensive performance monitoring, optimization
    suggestions, and resource usage tracking for AutoGen operations.
    """

    def __init__(self, enable_monitoring: bool = True):
        """Initialize performance monitor.

        Args:
            enable_monitoring: Whether to enable performance monitoring
        """
        self.enable_monitoring = enable_monitoring
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.resource_cache = weakref.WeakValueDictionary()
        self.connection_pools: Dict[str, Any] = {}
        self._lock = threading.RLock()

        # Performance thresholds
        self.warning_thresholds = {
            'execution_time': 5.0,  # seconds
            'error_rate': 0.1,      # 10%
            'memory_usage': 500 * 1024 * 1024,  # 500MB
        }

        logger.info(f"AutoGen performance monitor initialized (enabled: {enable_monitoring})")

    def performance_monitor(self, operation_name: str):
        """Decorator for monitoring function performance.

        Args:
            operation_name: Name of the operation being monitored

        Returns:
            Decorated function with performance monitoring
        """
        def decorator(func: Callable) -> Callable:
            if not self.enable_monitoring:
                return func

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()

                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.perf_counter() - start_time
                    self._record_success(operation_name, execution_time)
                    return result

                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    self._record_error(operation_name, execution_time)
                    raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.perf_counter() - start_time
                    self._record_success(operation_name, execution_time)
                    return result

                except Exception as e:
                    execution_time = time.perf_counter() - start_time
                    self._record_error(operation_name, execution_time)
                    raise

            # Return appropriate wrapper based on function type
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def _record_success(self, operation_name: str, duration: float) -> None:
        """Record successful operation metrics."""
        with self._lock:
            if operation_name not in self.metrics:
                self.metrics[operation_name] = PerformanceMetrics(operation_name)

            metrics = self.metrics[operation_name]
            metrics.execution_count += 1
            metrics.total_duration += duration
            metrics.min_duration = min(metrics.min_duration, duration)
            metrics.max_duration = max(metrics.max_duration, duration)
            metrics.recent_durations.append(duration)

            # Check for performance warnings
            if duration > self.warning_thresholds['execution_time']:
                logger.warning(f"Slow operation detected: {operation_name} took {duration:.2f}s")

    def _record_error(self, operation_name: str, duration: float) -> None:
        """Record error operation metrics."""
        with self._lock:
            if operation_name not in self.metrics:
                self.metrics[operation_name] = PerformanceMetrics(operation_name)

            metrics = self.metrics[operation_name]
            metrics.error_count += 1

            # Check error rate
            error_rate = metrics.error_count / max(metrics.execution_count + metrics.error_count, 1)
            if error_rate > self.warning_thresholds['error_rate']:
                logger.warning(f"High error rate detected: {operation_name} has {error_rate:.1%} error rate")

    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics.

        Args:
            operation_name: Specific operation to get metrics for, or None for all

        Returns:
            Dictionary containing performance metrics
        """
        with self._lock:
            if operation_name:
                if operation_name not in self.metrics:
                    return {}
                metrics = self.metrics[operation_name]
                return {
                    'operation_name': metrics.operation_name,
                    'execution_count': metrics.execution_count,
                    'error_count': metrics.error_count,
                    'success_rate': metrics.success_rate,
                    'average_duration': metrics.average_duration,
                    'min_duration': metrics.min_duration,
                    'max_duration': metrics.max_duration,
                    'recent_average': metrics.recent_average,
                }
            else:
                return {
                    name: {
                        'operation_name': metrics.operation_name,
                        'execution_count': metrics.execution_count,
                        'error_count': metrics.error_count,
                        'success_rate': metrics.success_rate,
                        'average_duration': metrics.average_duration,
                        'min_duration': metrics.min_duration,
                        'max_duration': metrics.max_duration,
                        'recent_average': metrics.recent_average,
                    }
                    for name, metrics in self.metrics.items()
                }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Performance summary with recommendations
        """
        with self._lock:
            total_operations = sum(m.execution_count for m in self.metrics.values())
            total_errors = sum(m.error_count for m in self.metrics.values())

            # Find slowest operations
            slowest_ops = sorted(
                self.metrics.items(),
                key=lambda x: x[1].average_duration,
                reverse=True
            )[:5]

            # Find operations with highest error rates
            error_prone_ops = sorted(
                [(name, m) for name, m in self.metrics.items() if m.error_count > 0],
                key=lambda x: x[1].error_count / max(x[1].execution_count + x[1].error_count, 1),
                reverse=True
            )[:5]

            recommendations = self._generate_recommendations()

            return {
                'total_operations': total_operations,
                'total_errors': total_errors,
                'overall_success_rate': total_operations / max(total_operations + total_errors, 1),
                'monitored_operations': len(self.metrics),
                'slowest_operations': [
                    {
                        'name': name,
                        'average_duration': metrics.average_duration,
                        'execution_count': metrics.execution_count
                    }
                    for name, metrics in slowest_ops
                ],
                'error_prone_operations': [
                    {
                        'name': name,
                        'error_rate': metrics.error_count / max(metrics.execution_count + metrics.error_count, 1),
                        'error_count': metrics.error_count
                    }
                    for name, metrics in error_prone_ops
                ],
                'recommendations': recommendations
            }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations.

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        with self._lock:
            for name, metrics in self.metrics.items():
                # Check for slow operations
                if metrics.average_duration > self.warning_thresholds['execution_time']:
                    recommendations.append(
                        f"Consider optimizing {name}: average execution time is {metrics.average_duration:.2f}s"
                    )

                # Check for high error rates
                error_rate = metrics.error_count / max(metrics.execution_count + metrics.error_count, 1)
                if error_rate > self.warning_thresholds['error_rate']:
                    recommendations.append(
                        f"Investigate errors in {name}: {error_rate:.1%} error rate"
                    )

                # Check for performance degradation
                if len(metrics.recent_durations) >= 10:
                    recent_avg = metrics.recent_average
                    overall_avg = metrics.average_duration
                    if recent_avg > overall_avg * 1.5:
                        recommendations.append(
                            f"Performance degradation detected in {name}: recent average "
                            f"({recent_avg:.2f}s) is much higher than overall average ({overall_avg:.2f}s)"
                        )

        return recommendations

    def optimize_resource_usage(self) -> Dict[str, Any]:
        """Optimize resource usage for AutoGen operations.

        Returns:
            Optimization results and statistics
        """
        optimizations = []

        # Clean up weak references
        before_cache_size = len(self.resource_cache)
        # Weak references are automatically cleaned up
        after_cache_size = len(self.resource_cache)

        if before_cache_size != after_cache_size:
            cleaned = before_cache_size - after_cache_size
            optimizations.append(f"Cleaned up {cleaned} unused resource references")

        # Optimize connection pools
        for pool_name, pool in self.connection_pools.items():
            if hasattr(pool, 'close_idle_connections'):
                try:
                    pool.close_idle_connections()
                    optimizations.append(f"Closed idle connections in {pool_name} pool")
                except Exception as e:
                    logger.warning(f"Failed to close idle connections in {pool_name}: {e}")

        return {
            'optimizations_applied': optimizations,
            'cache_size_before': before_cache_size,
            'cache_size_after': after_cache_size,
            'active_pools': len(self.connection_pools)
        }

    def reset_metrics(self, operation_name: Optional[str] = None) -> None:
        """Reset performance metrics.

        Args:
            operation_name: Specific operation to reset, or None for all
        """
        with self._lock:
            if operation_name:
                if operation_name in self.metrics:
                    del self.metrics[operation_name]
                    logger.info(f"Reset metrics for {operation_name}")
            else:
                self.metrics.clear()
                logger.info("Reset all performance metrics")

    def register_connection_pool(self, name: str, pool: Any) -> None:
        """Register a connection pool for optimization.

        Args:
            name: Pool name
            pool: Connection pool object
        """
        self.connection_pools[name] = pool
        logger.debug(f"Registered connection pool: {name}")

    def cache_resource(self, key: str, resource: Any) -> None:
        """Cache a resource with automatic cleanup.

        Args:
            key: Cache key
            resource: Resource to cache
        """
        self.resource_cache[key] = resource
        logger.debug(f"Cached resource: {key}")

    def get_cached_resource(self, key: str) -> Optional[Any]:
        """Get cached resource.

        Args:
            key: Cache key

        Returns:
            Cached resource or None if not found
        """
        return self.resource_cache.get(key)


# Global performance monitor instance
_performance_monitor = AutoGenPerformanceMonitor()


def get_performance_monitor() -> AutoGenPerformanceMonitor:
    """Get the global performance monitor instance.

    Returns:
        AutoGen performance monitor instance
    """
    return _performance_monitor


def performance_monitor(operation_name: str):
    """Decorator for monitoring function performance.

    Args:
        operation_name: Name of the operation being monitored

    Returns:
        Performance monitoring decorator
    """
    return _performance_monitor.performance_monitor(operation_name)


def optimize_autogen_performance() -> Dict[str, Any]:
    """Optimize AutoGen integration performance.

    Returns:
        Optimization results
    """
    return _performance_monitor.optimize_resource_usage()


def get_autogen_performance_metrics() -> Dict[str, Any]:
    """Get AutoGen integration performance metrics.

    Returns:
        Performance metrics summary
    """
    return _performance_monitor.get_performance_summary()