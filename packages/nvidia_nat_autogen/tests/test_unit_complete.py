# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive unit tests for AutoGen plugin components."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, AsyncGenerator
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import UserMessage, AssistantMessage, SystemMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.data_models.api_server import Message as NATMessage

from nat.plugins.autogen.message_adapter import AutoGenMessageAdapter
from nat.plugins.autogen.agent_wrapper import AutoGenAgentWrapper
from nat.plugins.autogen.runtime_wrapper import AutoGenRuntimeManager
from nat.plugins.autogen.performance_monitor import AutoGenPerformanceMonitor, get_performance_monitor
from nat.plugins.autogen.llm import openai_autogen, nim_autogen, _patch_autogen_client_based_on_config


class TestAutoGenMessageAdapter:
    """Unit tests for AutoGen message adapter."""

    def setup_method(self):
        """Setup test fixtures."""
        self.adapter = AutoGenMessageAdapter()

    def test_nat_to_autogen_user_message(self):
        """Test conversion of NAT user message to AutoGen format."""
        nat_msg = NATMessage(
            role="user",
            content="Hello AutoGen!",
            metadata={"timestamp": "2025-01-01"}
        )

        autogen_msg = self.adapter.nat_to_autogen(nat_msg)

        assert isinstance(autogen_msg, UserMessage)
        assert autogen_msg.content == "Hello AutoGen!"
        assert autogen_msg.source == "user"

    def test_nat_to_autogen_assistant_message(self):
        """Test conversion of NAT assistant message to AutoGen format."""
        nat_msg = NATMessage(
            role="assistant",
            content="Hello from AutoGen!",
            metadata={}
        )

        autogen_msg = self.adapter.nat_to_autogen(nat_msg)

        assert isinstance(autogen_msg, AssistantMessage)
        assert autogen_msg.content == "Hello from AutoGen!"
        assert autogen_msg.source == "assistant"

    def test_nat_to_autogen_system_message(self):
        """Test conversion of NAT system message to AutoGen format."""
        nat_msg = NATMessage(
            role="system",
            content="System instructions",
            metadata={}
        )

        autogen_msg = self.adapter.nat_to_autogen(nat_msg)

        assert isinstance(autogen_msg, SystemMessage)
        assert autogen_msg.content == "System instructions"

    def test_nat_to_autogen_invalid_role(self):
        """Test handling of invalid role in NAT message."""
        nat_msg = NATMessage(
            role="invalid_role",
            content="Test content",
            metadata={}
        )

        with pytest.raises(ValueError, match="Unsupported message role"):
            self.adapter.nat_to_autogen(nat_msg)

    def test_autogen_to_nat_user_message(self):
        """Test conversion of AutoGen user message to NAT format."""
        autogen_msg = UserMessage(content="User question", source="user")

        nat_msg = self.adapter.autogen_to_nat(autogen_msg)

        assert nat_msg.role == "user"
        assert nat_msg.content == "User question"
        assert isinstance(nat_msg.metadata, dict)

    def test_autogen_to_nat_assistant_message(self):
        """Test conversion of AutoGen assistant message to NAT format."""
        autogen_msg = AssistantMessage(content="Assistant response", source="assistant")

        nat_msg = self.adapter.autogen_to_nat(autogen_msg)

        assert nat_msg.role == "assistant"
        assert nat_msg.content == "Assistant response"

    def test_autogen_to_nat_system_message(self):
        """Test conversion of AutoGen system message to NAT format."""
        autogen_msg = SystemMessage(content="System message")

        nat_msg = self.adapter.autogen_to_nat(autogen_msg)

        assert nat_msg.role == "system"
        assert nat_msg.content == "System message"

    def test_autogen_to_nat_unsupported_message(self):
        """Test handling of unsupported AutoGen message type."""
        unsupported_msg = Mock()
        unsupported_msg.__class__.__name__ = "UnsupportedMessage"

        with pytest.raises(ValueError, match="Unsupported AutoGen message type"):
            self.adapter.autogen_to_nat(unsupported_msg)

    def test_bidirectional_conversion_consistency(self):
        """Test that bidirectional conversion maintains consistency."""
        original_nat = NATMessage(
            role="user",
            content="Bidirectional test",
            metadata={"test": True}
        )

        # Convert NAT -> AutoGen -> NAT
        autogen_msg = self.adapter.nat_to_autogen(original_nat)
        converted_nat = self.adapter.autogen_to_nat(autogen_msg)

        assert converted_nat.role == original_nat.role
        assert converted_nat.content == original_nat.content

    def test_batch_conversion(self):
        """Test batch message conversion functionality."""
        nat_messages = [
            NATMessage(role="user", content="Message 1", metadata={}),
            NATMessage(role="assistant", content="Message 2", metadata={}),
            NATMessage(role="system", content="Message 3", metadata={})
        ]

        autogen_messages = [self.adapter.nat_to_autogen(msg) for msg in nat_messages]

        assert len(autogen_messages) == 3
        assert isinstance(autogen_messages[0], UserMessage)
        assert isinstance(autogen_messages[1], AssistantMessage)
        assert isinstance(autogen_messages[2], SystemMessage)


class TestAutoGenAgentWrapper:
    """Unit tests for AutoGen agent wrapper."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_agent = Mock(spec=AssistantAgent)
        self.mock_agent.name = "test_agent"
        self.wrapper = AutoGenAgentWrapper(
            autogen_agent=self.mock_agent,
            agent_id="test_wrapper",
            config={"temperature": 0.7, "max_tokens": 1000}
        )

    def test_initialization(self):
        """Test agent wrapper initialization."""
        assert self.wrapper.agent_id == "test_wrapper"
        assert self.wrapper.autogen_agent == self.mock_agent
        assert self.wrapper.config["temperature"] == 0.7
        assert self.wrapper.config["max_tokens"] == 1000
        assert isinstance(self.wrapper._conversation_history, list)
        assert isinstance(self.wrapper._context, dict)

    def test_initialization_with_runtime(self):
        """Test agent wrapper initialization with runtime."""
        mock_runtime = Mock()
        wrapper = AutoGenAgentWrapper(
            autogen_agent=self.mock_agent,
            agent_id="test_with_runtime",
            runtime=mock_runtime,
            config={}
        )

        assert wrapper.runtime == mock_runtime

    def test_initialization_defaults(self):
        """Test agent wrapper initialization with defaults."""
        wrapper = AutoGenAgentWrapper(
            autogen_agent=self.mock_agent,
            agent_id="test_defaults"
        )

        assert wrapper.config == {}
        assert wrapper.runtime is None
        assert len(wrapper._conversation_history) == 0

    def test_context_management(self):
        """Test context management functionality."""
        # Set context
        self.wrapper.set_context("key1", "value1")
        self.wrapper.set_context("key2", {"nested": "value"})

        assert self.wrapper.get_context("key1") == "value1"
        assert self.wrapper.get_context("key2")["nested"] == "value"
        assert self.wrapper.get_context("nonexistent") is None

        # Clear context
        self.wrapper.clear_context()
        assert len(self.wrapper._context) == 0

    def test_conversation_history_management(self):
        """Test conversation history management."""
        # Initially empty
        assert len(self.wrapper._conversation_history) == 0

        # Add messages
        msg1 = Mock()
        msg2 = Mock()
        self.wrapper._conversation_history.extend([msg1, msg2])

        assert len(self.wrapper._conversation_history) == 2

    def test_agent_properties(self):
        """Test agent property access."""
        assert self.wrapper.name == "test_agent"
        assert self.wrapper.autogen_agent == self.mock_agent

    @pytest.mark.asyncio
    async def test_async_message_processing_setup(self):
        """Test async message processing setup (without full implementation)."""
        # This tests the basic setup for async processing
        # Full implementation would require more complex AutoGen runtime setup
        assert hasattr(self.wrapper, 'autogen_agent')
        assert self.wrapper.autogen_agent is not None


class TestAutoGenRuntimeManager:
    """Unit tests for AutoGen runtime manager."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = AutoGenRuntimeManager()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test runtime manager initialization."""
        with patch('autogen_core.SingleThreadedAgentRuntime') as mock_runtime:
            mock_runtime_instance = Mock()
            mock_runtime.return_value = mock_runtime_instance

            await self.manager.initialize()

            assert self.manager.runtime is not None
            mock_runtime.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_registration(self):
        """Test agent registration functionality."""
        with patch('autogen_core.SingleThreadedAgentRuntime') as mock_runtime:
            mock_runtime_instance = Mock()
            mock_runtime_instance.register = AsyncMock()
            mock_runtime.return_value = mock_runtime_instance

            await self.manager.initialize()

            mock_agent = Mock(spec=AssistantAgent)
            agent_id = await self.manager.register_agent("test_agent", mock_agent)

            assert agent_id == "test_agent"
            assert "test_agent" in self.manager._registered_agents
            assert self.manager._registered_agents["test_agent"] == mock_agent

    @pytest.mark.asyncio
    async def test_agent_unregistration(self):
        """Test agent unregistration functionality."""
        with patch('autogen_core.SingleThreadedAgentRuntime') as mock_runtime:
            mock_runtime_instance = Mock()
            mock_runtime_instance.register = AsyncMock()
            mock_runtime.return_value = mock_runtime_instance

            await self.manager.initialize()

            # Register agent first
            mock_agent = Mock(spec=AssistantAgent)
            await self.manager.register_agent("test_agent", mock_agent)
            assert "test_agent" in self.manager._registered_agents

            # Unregister agent
            success = await self.manager.unregister_agent("test_agent")
            assert success
            assert "test_agent" not in self.manager._registered_agents

    @pytest.mark.asyncio
    async def test_agent_unregistration_nonexistent(self):
        """Test unregistration of non-existent agent."""
        await self.manager.initialize()
        success = await self.manager.unregister_agent("nonexistent")
        assert not success

    @pytest.mark.asyncio
    async def test_get_registered_agents(self):
        """Test getting list of registered agents."""
        with patch('autogen_core.SingleThreadedAgentRuntime') as mock_runtime:
            mock_runtime_instance = Mock()
            mock_runtime_instance.register = AsyncMock()
            mock_runtime.return_value = mock_runtime_instance

            await self.manager.initialize()

            # Register multiple agents
            agents = ["agent1", "agent2", "agent3"]
            for agent_id in agents:
                mock_agent = Mock(spec=AssistantAgent)
                await self.manager.register_agent(agent_id, mock_agent)

            registered = self.manager.get_registered_agents()
            assert set(registered) == set(agents)

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test runtime cleanup functionality."""
        with patch('autogen_core.SingleThreadedAgentRuntime') as mock_runtime:
            mock_runtime_instance = Mock()
            mock_runtime_instance.stop = AsyncMock()
            mock_runtime.return_value = mock_runtime_instance

            await self.manager.initialize()
            await self.manager.cleanup()

            mock_runtime_instance.stop.assert_called_once()
            assert len(self.manager._registered_agents) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test runtime manager as context manager."""
        with patch('autogen_core.SingleThreadedAgentRuntime') as mock_runtime:
            mock_runtime_instance = Mock()
            mock_runtime_instance.stop = AsyncMock()
            mock_runtime.return_value = mock_runtime_instance

            async with self.manager:
                assert self.manager.runtime is not None

            # Cleanup should be called automatically
            mock_runtime_instance.stop.assert_called_once()


class TestAutoGenPerformanceMonitor:
    """Unit tests for AutoGen performance monitor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = AutoGenPerformanceMonitor()

    def test_initialization(self):
        """Test performance monitor initialization."""
        assert isinstance(self.monitor._metrics, dict)
        assert len(self.monitor._metrics) == 0
        assert self.monitor._start_time is not None

    def test_record_metric(self):
        """Test metric recording functionality."""
        self.monitor.record_metric("test_metric", 100.0)
        self.monitor.record_metric("test_metric", 200.0)
        self.monitor.record_metric("another_metric", 50.0)

        metrics = self.monitor.get_metrics()
        assert "test_metric" in metrics
        assert "another_metric" in metrics
        assert len(metrics["test_metric"]) == 2
        assert metrics["test_metric"] == [100.0, 200.0]
        assert metrics["another_metric"] == [50.0]

    def test_record_metric_with_metadata(self):
        """Test metric recording with metadata."""
        metadata = {"operation": "message_conversion", "agent_id": "test_agent"}
        self.monitor.record_metric("response_time", 150.0, metadata)

        # Basic metric should be recorded
        metrics = self.monitor.get_metrics()
        assert "response_time" in metrics
        assert metrics["response_time"] == [150.0]

    def test_get_metric_statistics(self):
        """Test metric statistics calculation."""
        # Record some test data
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            self.monitor.record_metric("test_stat", value)

        stats = self.monitor.get_metric_statistics("test_stat")

        assert stats["count"] == 5
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["average"] == 30.0
        assert stats["sum"] == 150.0

    def test_get_metric_statistics_nonexistent(self):
        """Test statistics for non-existent metric."""
        stats = self.monitor.get_metric_statistics("nonexistent")

        assert stats["count"] == 0
        assert stats["min"] is None
        assert stats["max"] is None
        assert stats["average"] is None
        assert stats["sum"] == 0

    def test_analyze_performance(self):
        """Test performance analysis functionality."""
        # Record various metrics
        self.monitor.record_metric("response_time", 100.0)
        self.monitor.record_metric("response_time", 150.0)
        self.monitor.record_metric("response_time", 200.0)
        self.monitor.record_metric("memory_usage", 1024.0)
        self.monitor.record_metric("cpu_usage", 75.0)

        analysis = self.monitor.analyze_performance()

        assert isinstance(analysis, dict)
        assert "total_metrics" in analysis
        assert "average_response_time" in analysis
        assert "uptime_seconds" in analysis
        assert analysis["total_metrics"] == 5
        assert analysis["average_response_time"] == 150.0

    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        # Record some metrics
        self.monitor.record_metric("test1", 100.0)
        self.monitor.record_metric("test2", 200.0)
        assert len(self.monitor.get_metrics()) == 2

        # Reset metrics
        self.monitor.reset_metrics()
        assert len(self.monitor.get_metrics()) == 0

    def test_singleton_behavior(self):
        """Test that get_performance_monitor returns singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        assert monitor1 is monitor2

        # Record metric in one, should appear in both
        monitor1.record_metric("singleton_test", 42.0)
        metrics = monitor2.get_metrics()
        assert "singleton_test" in metrics
        assert metrics["singleton_test"] == [42.0]


class TestAutoGenLLMIntegration:
    """Unit tests for AutoGen LLM client integration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.builder = Mock(spec=Builder)

    def test_openai_config_creation(self):
        """Test OpenAI configuration creation."""
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000
        )

        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_nim_config_creation(self):
        """Test NIM configuration creation."""
        config = NIMModelConfig(
            model_name="llama2-7b",
            base_url="https://nim.nvidia.com/v1",
            api_key="test-nim-key",
            temperature=0.5
        )

        assert config.model_name == "llama2-7b"
        assert config.base_url == "https://nim.nvidia.com/v1"
        assert config.api_key == "test-nim-key"
        assert config.temperature == 0.5

    @pytest.mark.asyncio
    async def test_openai_autogen_client_creation(self):
        """Test OpenAI AutoGen client creation."""
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.7
        )

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            async for client in openai_autogen(config, self.builder):
                assert client is not None
                # Verify client was created with correct parameters
                mock_client.assert_called_once()
                call_args = mock_client.call_args
                assert call_args[1]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_nim_autogen_client_creation(self):
        """Test NIM AutoGen client creation."""
        config = NIMModelConfig(
            model_name="llama2-7b",
            base_url="https://nim.nvidia.com/v1",
            api_key="test-nim-key"
        )

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            async for client in nim_autogen(config, self.builder):
                assert client is not None
                mock_client.assert_called_once()

    def test_client_patching_with_mixins(self):
        """Test client patching with NAT mixins."""
        mock_client = Mock()
        mock_config = Mock()

        # Test without mixins
        result = _patch_autogen_client_based_on_config(mock_client, mock_config)
        assert result == mock_client

        # Test with retry mixin
        from nat.data_models.retry_mixin import RetryMixin
        mock_config_with_retry = Mock(spec=RetryMixin)
        mock_config_with_retry.num_retries = 3
        mock_config_with_retry.retry_on_status_codes = [500, 502, 503]
        mock_config_with_retry.retry_on_errors = []

        with patch('nat.utils.exception_handlers.automatic_retries.patch_with_retry') as mock_patch:
            mock_patch.return_value = mock_client
            result = _patch_autogen_client_based_on_config(mock_client, mock_config_with_retry)
            mock_patch.assert_called_once()

    def test_framework_enum_registration(self):
        """Test that AutoGen is properly registered in framework enum."""
        assert hasattr(LLMFrameworkEnum, 'AUTOGEN')
        assert LLMFrameworkEnum.AUTOGEN == "autogen"

    @pytest.mark.asyncio
    async def test_async_generator_pattern(self):
        """Test async generator pattern for LLM clients."""
        config = OpenAIModelConfig(model_name="gpt-4", api_key="test")

        with patch('autogen_ext.models.openai.OpenAIChatCompletionClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Test that generator yields exactly one client
            clients = []
            async for client in openai_autogen(config, self.builder):
                clients.append(client)

            assert len(clients) == 1
            assert clients[0] is not None

    def test_config_model_dump_exclusions(self):
        """Test configuration model dump with proper exclusions."""
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000
        )

        # Test that model dump excludes certain fields
        dumped = config.model_dump(
            exclude={"type", "model_name", "thinking"},
            by_alias=True,
            exclude_none=True
        )

        assert "model_name" not in dumped
        assert "type" not in dumped
        assert "thinking" not in dumped
        assert "api_key" in dumped or "api_key" in str(dumped)  # May be aliased


class TestErrorHandling:
    """Unit tests for error handling across components."""

    def test_message_adapter_error_handling(self):
        """Test message adapter error handling."""
        adapter = AutoGenMessageAdapter()

        # Test with None message
        with pytest.raises((AttributeError, TypeError)):
            adapter.nat_to_autogen(None)

        # Test with malformed message
        malformed_msg = Mock()
        malformed_msg.role = None
        malformed_msg.content = "test"

        with pytest.raises((ValueError, AttributeError)):
            adapter.nat_to_autogen(malformed_msg)

    def test_agent_wrapper_error_handling(self):
        """Test agent wrapper error handling."""
        # Test with None agent
        with pytest.raises((TypeError, ValueError)):
            AutoGenAgentWrapper(
                autogen_agent=None,
                agent_id="test"
            )

        # Test with invalid agent_id
        mock_agent = Mock(spec=AssistantAgent)
        with pytest.raises((TypeError, ValueError)):
            AutoGenAgentWrapper(
                autogen_agent=mock_agent,
                agent_id=""  # Empty agent ID
            )

    def test_performance_monitor_error_handling(self):
        """Test performance monitor error handling."""
        monitor = AutoGenPerformanceMonitor()

        # Test with invalid metric name
        with pytest.raises((TypeError, ValueError)):
            monitor.record_metric(None, 100.0)

        # Test with invalid metric value
        with pytest.raises((TypeError, ValueError)):
            monitor.record_metric("test", "invalid_value")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])