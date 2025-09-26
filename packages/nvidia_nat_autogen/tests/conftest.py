# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for AutoGen plugin tests."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Generator, AsyncGenerator

import autogen_core
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import UserMessage, AssistantMessage, SystemMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from nat.builder.builder import Builder
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.data_models.api_server import Message as NATMessage


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def openai_config() -> OpenAIModelConfig:
    """OpenAI model configuration fixture."""
    return OpenAIModelConfig(
        model_name="gpt-4",
        api_key="test-openai-key",
        temperature=0.7,
        max_tokens=1000,
        timeout=30.0
    )


@pytest.fixture
def nim_config() -> NIMModelConfig:
    """NVIDIA NIM model configuration fixture."""
    return NIMModelConfig(
        model_name="llama2-7b-chat",
        base_url="https://nim.nvidia.com/v1",
        api_key="test-nim-key",
        temperature=0.5,
        max_tokens=2000
    )


@pytest.fixture
def builder() -> Builder:
    """NAT builder instance fixture."""
    return Builder()


@pytest.fixture
def sample_nat_messages() -> list[NATMessage]:
    """Sample NAT messages for testing."""
    return [
        NATMessage(
            role="user",
            content="Hello, how can you help me today?",
            metadata={"timestamp": "2025-01-01T00:00:00Z", "user_id": "test_user"}
        ),
        NATMessage(
            role="assistant",
            content="I'm here to help with your AutoGen integration questions!",
            metadata={"model": "gpt-4", "temperature": 0.7}
        ),
        NATMessage(
            role="system",
            content="You are a helpful assistant specialized in AutoGen framework.",
            metadata={"system_prompt": True}
        )
    ]


@pytest.fixture
def sample_autogen_messages() -> list:
    """Sample AutoGen messages for testing."""
    return [
        UserMessage(content="What can AutoGen do?", source="user"),
        AssistantMessage(content="AutoGen enables multi-agent conversations.", source="assistant"),
        SystemMessage(content="You are an AutoGen expert.")
    ]


@pytest.fixture
def mock_autogen_agent() -> Mock:
    """Mock AutoGen agent fixture."""
    agent = Mock(spec=AssistantAgent)
    agent.name = "test_agent"
    agent.description = "Test AutoGen agent"
    agent.on_messages = AsyncMock()
    agent.reset = Mock()
    return agent


@pytest.fixture
def mock_openai_client() -> Mock:
    """Mock OpenAI client fixture."""
    client = Mock(spec=OpenAIChatCompletionClient)
    client.model = "gpt-4"
    client.create = AsyncMock()
    client.acreate = AsyncMock()
    return client


@pytest.fixture
def mock_autogen_runtime():
    """Mock AutoGen runtime fixture."""
    with patch('autogen_core.SingleThreadedAgentRuntime') as mock_runtime:
        runtime_instance = Mock()
        runtime_instance.register = AsyncMock()
        runtime_instance.unregister = AsyncMock()
        runtime_instance.send_message = AsyncMock()
        runtime_instance.stop = AsyncMock()
        runtime_instance.start = AsyncMock()
        mock_runtime.return_value = runtime_instance
        yield runtime_instance


@pytest.fixture
def mock_docker_client():
    """Mock Docker client fixture."""
    with patch('docker.from_env') as mock_docker:
        client = Mock()
        container = Mock()

        # Setup container behavior
        container.logs.return_value = b"Test output"
        container.wait.return_value = {"StatusCode": 0}
        container.remove = Mock()

        # Setup client behavior
        client.containers.run.return_value = container
        client.containers.get.return_value = container
        client.ping.return_value = True

        mock_docker.return_value = client
        yield client


@pytest.fixture
def performance_test_data():
    """Performance test data fixture."""
    return {
        "response_times": [100.0, 150.0, 120.0, 180.0, 95.0],
        "memory_usage": [1024, 1056, 1089, 1123, 1045],
        "cpu_usage": [25.5, 30.2, 28.7, 35.1, 22.8],
        "throughput": [10.5, 9.8, 11.2, 8.9, 12.1]
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    from nat.plugins.autogen.performance_monitor import _performance_monitor_instance

    # Store original state
    original_instance = _performance_monitor_instance._instance if hasattr(_performance_monitor_instance, '_instance') else None

    yield

    # Reset singleton state
    if hasattr(_performance_monitor_instance, '_instance'):
        if original_instance:
            _performance_monitor_instance._instance = original_instance
        else:
            _performance_monitor_instance._instance = None


@pytest.fixture
def async_mock_response():
    """Async mock response for AutoGen agents."""
    async def mock_response(*args, **kwargs):
        return Mock(
            messages=[
                AssistantMessage(
                    content="Mocked response from AutoGen agent",
                    source="assistant"
                )
            ],
            usage=Mock(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
    return mock_response


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "timeout": 30,
        "max_retries": 3,
        "performance_threshold": {
            "max_response_time": 5.0,
            "max_memory_mb": 100,
            "min_throughput": 5.0
        },
        "test_scenarios": [
            {"name": "single_agent", "agents": 1, "messages": 5},
            {"name": "multi_agent", "agents": 3, "messages": 10},
            {"name": "conversation", "agents": 2, "messages": 20}
        ]
    }


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "async_test: Async tests")


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.async_test)

        # Mark tests based on file names
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        elif "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "test_unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)


# Async test utilities
@pytest.fixture
def async_test_timeout():
    """Timeout for async tests."""
    return 30.0


@pytest.fixture
def create_async_context():
    """Create async context for testing."""
    async def _create_context(timeout: float = 30.0):
        """Create an async context with timeout."""
        try:
            async with asyncio.timeout(timeout):
                yield
        except asyncio.TimeoutError:
            pytest.fail(f"Async test timed out after {timeout} seconds")

    return _create_context


# Mock data generators
@pytest.fixture
def generate_test_messages():
    """Generate test messages for various scenarios."""
    def _generate(count: int, message_type: str = "mixed") -> list[NATMessage]:
        messages = []
        for i in range(count):
            if message_type == "user" or (message_type == "mixed" and i % 3 == 0):
                msg = NATMessage(
                    role="user",
                    content=f"User message {i + 1}",
                    metadata={"msg_id": i + 1, "type": "user"}
                )
            elif message_type == "assistant" or (message_type == "mixed" and i % 3 == 1):
                msg = NATMessage(
                    role="assistant",
                    content=f"Assistant response {i + 1}",
                    metadata={"msg_id": i + 1, "type": "assistant"}
                )
            else:
                msg = NATMessage(
                    role="system",
                    content=f"System message {i + 1}",
                    metadata={"msg_id": i + 1, "type": "system"}
                )
            messages.append(msg)
        return messages

    return _generate


@pytest.fixture
def create_mock_agents():
    """Create multiple mock agents for testing."""
    def _create(count: int) -> list[Mock]:
        agents = []
        for i in range(count):
            agent = Mock(spec=AssistantAgent)
            agent.name = f"agent_{i + 1}"
            agent.description = f"Test agent {i + 1}"
            agent.on_messages = AsyncMock()
            agents.append(agent)
        return agents

    return _create


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set test environment variables
    import os
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Setup logging for tests
    import logging
    logging.basicConfig(level=logging.DEBUG)

    yield

    # Cleanup after tests
    os.environ.pop("TESTING", None)
    os.environ.pop("LOG_LEVEL", None)


# Memory leak detection
@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    import gc
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Force garbage collection and get baseline
    gc.collect()
    initial_memory = process.memory_info().rss
    initial_objects = len(gc.get_objects())

    yield {
        "initial_memory": initial_memory,
        "initial_objects": initial_objects
    }

    # Check for memory leaks after test
    gc.collect()
    final_memory = process.memory_info().rss
    final_objects = len(gc.get_objects())

    memory_growth = final_memory - initial_memory
    object_growth = final_objects - initial_objects

    # Assert reasonable memory growth (less than 10MB per test)
    assert memory_growth < 10 * 1024 * 1024, f"Memory leak detected: {memory_growth / 1024 / 1024:.2f}MB growth"

    # Assert reasonable object growth (less than 1000 objects per test)
    assert object_growth < 1000, f"Object leak detected: {object_growth} objects created"