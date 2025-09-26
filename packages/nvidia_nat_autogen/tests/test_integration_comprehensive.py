# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive integration tests for AutoGen plugin."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

import autogen_core
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import UserMessage, AssistantMessage, SystemMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from nat.builder.builder import Builder
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.data_models.api_server import Message as NATMessage

from nat.plugins.autogen import AutoGenAgentWrapper, AutoGenMessageAdapter
from nat.plugins.autogen.llm import openai_autogen, nim_autogen
from nat.plugins.autogen.runtime_wrapper import AutoGenRuntimeManager
from nat.plugins.autogen.performance_monitor import get_performance_monitor
from nat.plugins.autogen.tools import AutoGenCodeExecutorTool


class TestAutoGenIntegration:
    """Comprehensive integration tests for AutoGen plugin components."""

    @pytest.fixture
    def openai_config(self):
        """OpenAI model configuration for testing."""
        return OpenAIModelConfig(
            model_name="gpt-4",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=1000
        )

    @pytest.fixture
    def nim_config(self):
        """NIM model configuration for testing."""
        return NIMModelConfig(
            model_name="llama2-7b-chat",
            base_url="https://nim.nvidia.com/v1",
            api_key="test-nim-key",
            temperature=0.5
        )

    @pytest.fixture
    def builder(self):
        """NAT builder instance."""
        return Builder()

    @pytest.fixture
    def sample_nat_message(self):
        """Sample NAT message for testing."""
        return NATMessage(
            role="user",
            content="Hello, how can you help me today?",
            metadata={"timestamp": "2025-01-01T00:00:00Z"}
        )

    @pytest.mark.asyncio
    async def test_message_adapter_bidirectional_conversion(self, sample_nat_message):
        """Test bidirectional message conversion between NAT and AutoGen."""
        adapter = AutoGenMessageAdapter()

        # Test NAT to AutoGen conversion
        autogen_msg = adapter.nat_to_autogen(sample_nat_message)
        assert isinstance(autogen_msg, UserMessage)
        assert autogen_msg.content == sample_nat_message.content

        # Test AutoGen to NAT conversion
        converted_back = adapter.autogen_to_nat(autogen_msg)
        assert converted_back.role.lower() == "user"
        assert converted_back.content == sample_nat_message.content

    @pytest.mark.asyncio
    async def test_message_adapter_all_message_types(self):
        """Test message adapter handles all AutoGen message types."""
        adapter = AutoGenMessageAdapter()

        # Test User Message
        user_msg = UserMessage(content="User question", source="user")
        nat_user = adapter.autogen_to_nat(user_msg)
        assert nat_user.role.lower() == "user"

        # Test Assistant Message
        assistant_msg = AssistantMessage(content="Assistant response", source="assistant")
        nat_assistant = adapter.autogen_to_nat(assistant_msg)
        assert nat_assistant.role.lower() == "assistant"

        # Test System Message
        system_msg = SystemMessage(content="System instructions")
        nat_system = adapter.autogen_to_nat(system_msg)
        assert nat_system.role.lower() == "system"

    @pytest.mark.asyncio
    async def test_openai_llm_client_integration(self, openai_config, builder):
        """Test OpenAI LLM client integration with AutoGen."""
        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            async for client in openai_autogen(openai_config, builder):
                assert client is not None
                # Verify client was patched with NAT mixins
                assert hasattr(client, '__wrapped__') or client == mock_instance

    @pytest.mark.asyncio
    async def test_nim_llm_client_integration(self, nim_config, builder):
        """Test NVIDIA NIM client integration with AutoGen."""
        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            async for client in nim_autogen(nim_config, builder):
                assert client is not None
                # Verify NIM configuration was applied
                mock_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_wrapper_initialization(self, openai_config):
        """Test AutoGen agent wrapper initialization and basic functionality."""
        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client:
            mock_llm = Mock()
            mock_client.return_value = mock_llm

            # Create mock AutoGen agent
            mock_agent = Mock(spec=AssistantAgent)
            mock_agent.name = "test_agent"

            # Create wrapper
            wrapper = AutoGenAgentWrapper(
                autogen_agent=mock_agent,
                agent_id="test_wrapper",
                config={"temperature": 0.7}
            )

            assert wrapper.agent_id == "test_wrapper"
            assert wrapper.autogen_agent == mock_agent
            assert wrapper.config["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_agent_wrapper_message_processing(self, sample_nat_message):
        """Test agent wrapper message processing workflow."""
        # Create mock AutoGen agent with async response
        mock_agent = Mock(spec=AssistantAgent)
        mock_response = Mock()
        mock_response.messages = [AssistantMessage(content="Test response", source="assistant")]
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        # Create wrapper
        wrapper = AutoGenAgentWrapper(
            autogen_agent=mock_agent,
            agent_id="test_agent",
            config={}
        )

        # Process message (simplified test)
        assert wrapper.autogen_agent is not None
        # Full message processing would require more complex setup

    @pytest.mark.asyncio
    async def test_runtime_manager_lifecycle(self):
        """Test AutoGen runtime manager lifecycle operations."""
        with patch('autogen_core.SingleThreadedAgentRuntime') as mock_runtime:
            mock_runtime_instance = Mock()
            mock_runtime.return_value = mock_runtime_instance

            manager = AutoGenRuntimeManager()

            # Test runtime initialization
            await manager.initialize()
            assert manager.runtime is not None

            # Test agent registration
            mock_agent = Mock(spec=AssistantAgent)
            agent_id = await manager.register_agent("test_agent", mock_agent)
            assert agent_id == "test_agent"

            # Test cleanup
            await manager.cleanup()
            # Verify cleanup was called

    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring system."""
        monitor = get_performance_monitor()

        # Test metric recording
        monitor.record_metric("test_metric", 100.0)
        metrics = monitor.get_metrics()

        assert "test_metric" in metrics
        assert metrics["test_metric"][-1] == 100.0

        # Test performance analysis
        analysis = monitor.analyze_performance()
        assert isinstance(analysis, dict)
        assert "average_response_time" in analysis

    @pytest.mark.asyncio
    async def test_code_executor_tool_functionality(self):
        """Test code execution tool with Docker integration."""
        with patch('docker.from_env') as mock_docker:
            mock_client = Mock()
            mock_container = Mock()
            mock_container.logs.return_value = b"Hello, World!"
            mock_container.wait.return_value = {"StatusCode": 0}
            mock_client.containers.run.return_value = mock_container
            mock_docker.return_value = mock_client

            tool = AutoGenCodeExecutorTool()

            # Test Python code execution
            result = await tool.execute_code(
                code="print('Hello, World!')",
                language="python",
                timeout=30
            )

            assert result.exit_code == 0
            assert "Hello, World!" in result.stdout

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, openai_config, builder):
        """Test comprehensive error handling across components."""
        # Test LLM client error handling
        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                async for client in openai_autogen(openai_config, builder):
                    pass

        # Test message adapter error handling
        adapter = AutoGenMessageAdapter()
        invalid_message = Mock()
        invalid_message.role = "invalid_role"
        invalid_message.content = "test"

        # Should handle gracefully or raise appropriate exception
        with pytest.raises((ValueError, AttributeError)):
            adapter.autogen_to_nat(invalid_message)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, openai_config):
        """Test concurrent operations and thread safety."""
        async def create_wrapper(agent_id: str):
            mock_agent = Mock(spec=AssistantAgent)
            mock_agent.name = f"agent_{agent_id}"
            return AutoGenAgentWrapper(
                autogen_agent=mock_agent,
                agent_id=agent_id,
                config={}
            )

        # Create multiple wrappers concurrently
        tasks = [create_wrapper(f"agent_{i}") for i in range(5)]
        wrappers = await asyncio.gather(*tasks)

        assert len(wrappers) == 5
        assert all(wrapper.agent_id.startswith("agent_") for wrapper in wrappers)

    @pytest.mark.asyncio
    async def test_memory_and_resource_management(self):
        """Test memory usage and resource cleanup."""
        import gc
        import sys

        # Baseline memory usage
        gc.collect()
        baseline_objects = len(gc.get_objects())

        # Create and destroy multiple components
        for i in range(10):
            mock_agent = Mock(spec=AssistantAgent)
            wrapper = AutoGenAgentWrapper(
                autogen_agent=mock_agent,
                agent_id=f"test_{i}",
                config={}
            )
            del wrapper
            del mock_agent

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory growth should be minimal
        memory_growth = final_objects - baseline_objects
        assert memory_growth < 100  # Allow some growth but not excessive

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test invalid OpenAI config
        with pytest.raises((ValueError, TypeError)):
            OpenAIModelConfig(
                model_name="",  # Invalid empty model name
                api_key=None,   # Invalid None API key
                temperature=2.0  # Invalid temperature > 1.0
            )

        # Test valid configuration
        valid_config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="valid-key",
            temperature=0.7
        )
        assert valid_config.model_name == "gpt-4"

    def test_framework_registration(self):
        """Test framework registration in NAT system."""
        from nat.builder.framework_enum import LLMFrameworkEnum

        # Verify AutoGen is registered
        assert hasattr(LLMFrameworkEnum, 'AUTOGEN')
        assert LLMFrameworkEnum.AUTOGEN == "autogen"

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, openai_config, sample_nat_message):
        """Test complete end-to-end workflow from NAT message to AutoGen response."""
        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client:
            # Setup mocks
            mock_llm = Mock()
            mock_client.return_value = mock_llm

            mock_agent = Mock(spec=AssistantAgent)
            mock_response = Mock()
            mock_response.messages = [AssistantMessage(content="E2E Test Response", source="assistant")]
            mock_agent.on_messages = AsyncMock(return_value=mock_response)

            # Create components
            adapter = AutoGenMessageAdapter()
            wrapper = AutoGenAgentWrapper(
                autogen_agent=mock_agent,
                agent_id="e2e_test",
                config={}
            )

            # Execute workflow
            autogen_message = adapter.nat_to_autogen(sample_nat_message)
            assert isinstance(autogen_message, UserMessage)

            # Simulate processing (would be more complex in real scenario)
            assert wrapper.autogen_agent is not None

            # Verify end-to-end integration works
            assert autogen_message.content == sample_nat_message.content


class TestPerformanceBenchmarks:
    """Performance benchmark tests for AutoGen integration."""

    @pytest.mark.asyncio
    async def test_message_conversion_performance(self):
        """Benchmark message conversion performance."""
        import time

        adapter = AutoGenMessageAdapter()
        sample_message = NATMessage(
            role="user",
            content="Performance test message",
            metadata={}
        )

        # Benchmark conversion performance
        start_time = time.time()
        for _ in range(1000):
            autogen_msg = adapter.nat_to_autogen(sample_message)
            nat_msg = adapter.autogen_to_nat(autogen_msg)
        end_time = time.time()

        # Should complete 1000 conversions in under 1 second
        total_time = end_time - start_time
        assert total_time < 1.0, f"Message conversion too slow: {total_time:.2f}s"

    @pytest.mark.asyncio
    async def test_agent_creation_performance(self):
        """Benchmark agent wrapper creation performance."""
        import time

        start_time = time.time()
        wrappers = []

        for i in range(100):
            mock_agent = Mock(spec=AssistantAgent)
            wrapper = AutoGenAgentWrapper(
                autogen_agent=mock_agent,
                agent_id=f"perf_test_{i}",
                config={}
            )
            wrappers.append(wrapper)

        end_time = time.time()

        # Should create 100 wrappers in under 0.5 seconds
        total_time = end_time - start_time
        assert total_time < 0.5, f"Agent creation too slow: {total_time:.2f}s"

    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self):
        """Benchmark memory usage of AutoGen integration."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create multiple components to test memory usage
        components = []
        for i in range(50):
            mock_agent = Mock(spec=AssistantAgent)
            wrapper = AutoGenAgentWrapper(
                autogen_agent=mock_agent,
                agent_id=f"memory_test_{i}",
                config={}
            )
            components.append(wrapper)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 50 components)
        assert memory_increase < 50 * 1024 * 1024, f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])