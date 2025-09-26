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

"""Tests for AutoGen agent wrapper."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from autogen_core import AgentRuntime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ChatMessage
from autogen_agentchat.base import Response

from nat.plugins.autogen.agent_wrapper import AutoGenAgentWrapper, AutoGenMultiAgentOrchestrator


class TestAutoGenAgentWrapper:
    """Test cases for AutoGen agent wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock AutoGen agent
        self.mock_autogen_agent = Mock(spec=AssistantAgent)
        self.mock_runtime = Mock(spec=AgentRuntime)

        # Create wrapper instance
        self.wrapper = AutoGenAgentWrapper(
            autogen_agent=self.mock_autogen_agent,
            agent_id="test_agent",
            runtime=self.mock_runtime,
            config={"test": "config"}
        )

    def test_agent_wrapper_initialization(self):
        """Test agent wrapper initialization."""
        assert self.wrapper.agent_id == "test_agent"
        assert self.wrapper.autogen_agent == self.mock_autogen_agent
        assert self.wrapper.runtime == self.mock_runtime
        assert self.wrapper.config == {"test": "config"}
        assert len(self.wrapper._conversation_history) == 0

    @pytest.mark.asyncio
    async def test_process_message_with_modern_api(self):
        """Test processing message with modern AutoGen API."""
        # Create mock NAT message
        nat_message = Mock()
        nat_message.content = "Hello, agent!"
        nat_message.role = "user"

        # Mock AutoGen agent response
        mock_response = Mock()
        mock_response.chat_message = ChatMessage(content="Hello back!", source="agent")
        self.mock_autogen_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch('nat.plugins.autogen.agent_wrapper.AutoGenMessageAdapter') as MockAdapter:
            # Set up adapter mocks
            autogen_msg = ChatMessage(content="Hello, agent!", source="user")
            nat_response = Mock()
            nat_response.content = "Hello back!"

            MockAdapter.nat_to_autogen.return_value = autogen_msg
            MockAdapter.autogen_to_nat.return_value = nat_response

            # Process message
            result = await self.wrapper.process_message(nat_message)

            # Verify adapter was called correctly
            MockAdapter.nat_to_autogen.assert_called_once_with(nat_message)
            MockAdapter.autogen_to_nat.assert_called_once()

            # Verify agent was called
            self.mock_autogen_agent.on_messages.assert_called_once()

            # Verify result
            assert result == nat_response

    @pytest.mark.asyncio
    async def test_process_message_with_legacy_api(self):
        """Test processing message with legacy AutoGen API."""
        # Create mock NAT message
        nat_message = Mock()
        nat_message.content = "Hello, agent!"
        nat_message.role = "user"

        # Mock legacy AutoGen agent (no on_messages method)
        legacy_agent = Mock(spec=AssistantAgent)
        legacy_agent.generate_reply = AsyncMock(return_value="Legacy response")
        del legacy_agent.on_messages  # Remove on_messages to simulate legacy API

        wrapper = AutoGenAgentWrapper(
            autogen_agent=legacy_agent,
            agent_id="legacy_agent",
            runtime=self.mock_runtime
        )

        with patch('nat.plugins.autogen.agent_wrapper.AutoGenMessageAdapter') as MockAdapter:
            # Set up adapter mocks
            autogen_msg = ChatMessage(content="Hello, agent!", source="user")
            nat_response = Mock()
            nat_response.content = "Legacy response"

            MockAdapter.nat_to_autogen.return_value = autogen_msg
            MockAdapter.autogen_to_nat.return_value = nat_response

            # Process message
            result = await wrapper.process_message(nat_message)

            # Verify legacy method was called
            legacy_agent.generate_reply.assert_called_once()

            # Verify result
            assert result == nat_response

    @pytest.mark.asyncio
    async def test_process_conversation(self):
        """Test processing multi-message conversation."""
        # Create mock NAT messages
        nat_messages = []
        for i in range(3):
            msg = Mock()
            msg.content = f"Message {i}"
            msg.role = "user"
            nat_messages.append(msg)

        # Mock AutoGen responses
        autogen_responses = []
        nat_responses = []

        for i in range(3):
            autogen_response = ChatMessage(content=f"Response {i}", source="agent")
            autogen_responses.append(autogen_response)

            nat_response = Mock()
            nat_response.content = f"Response {i}"
            nat_responses.append(nat_response)

        # Set up agent to return different responses for each call
        self.mock_autogen_agent.on_messages = AsyncMock(side_effect=[
            Mock(chat_message=autogen_responses[0]),
            Mock(chat_message=autogen_responses[1]),
            Mock(chat_message=autogen_responses[2])
        ])

        with patch('nat.plugins.autogen.agent_wrapper.AutoGenMessageAdapter') as MockAdapter:
            # Set up adapter mocks
            MockAdapter.batch_nat_to_autogen.return_value = [Mock() for _ in range(3)]
            MockAdapter.batch_autogen_to_nat.return_value = nat_responses

            # Process conversation
            results = await self.wrapper.process_conversation(nat_messages)

            # Verify batch conversion was used
            MockAdapter.batch_nat_to_autogen.assert_called_once_with(nat_messages)
            MockAdapter.batch_autogen_to_nat.assert_called_once()

            # Verify results
            assert len(results) == 3
            assert results == nat_responses

    @pytest.mark.asyncio
    async def test_process_message_with_runtime(self):
        """Test processing message using AutoGen runtime."""
        # Set up wrapper with runtime
        nat_message = Mock()
        nat_message.content = "Runtime test"
        nat_message.role = "user"

        # Mock runtime response
        runtime_response = ChatMessage(content="Runtime response", source="agent")
        self.mock_runtime.send_message = AsyncMock(return_value=runtime_response)

        # Mock agent with on_messages that uses runtime
        async def mock_on_messages(messages):
            # Simulate runtime-based processing
            if self.wrapper.runtime:
                from autogen_core import AgentId
                agent_id = AgentId(self.wrapper.agent_id, key="default")
                return await self.wrapper.runtime.send_message(messages[0], agent_id)
            return Mock(chat_message=ChatMessage(content="Direct response", source="agent"))

        self.mock_autogen_agent.on_messages = mock_on_messages

        with patch('nat.plugins.autogen.agent_wrapper.AutoGenMessageAdapter') as MockAdapter:
            # Set up adapter mocks
            autogen_msg = ChatMessage(content="Runtime test", source="user")
            nat_response = Mock()
            nat_response.content = "Runtime response"

            MockAdapter.nat_to_autogen.return_value = autogen_msg
            MockAdapter.autogen_to_nat.return_value = nat_response

            # Process message
            result = await self.wrapper.process_message(nat_message)

            # Verify runtime was used
            self.mock_runtime.send_message.assert_called_once()

            # Verify result
            assert result == nat_response

    def test_conversation_history_management(self):
        """Test conversation history management."""
        # Initially empty
        history = self.wrapper.get_conversation_history()
        assert len(history) == 0

        # Add some messages to internal history
        test_messages = [
            ChatMessage(content="Hello", source="user"),
            ChatMessage(content="Hi there", source="agent")
        ]
        self.wrapper._conversation_history = test_messages

        with patch('nat.plugins.autogen.agent_wrapper.AutoGenMessageAdapter') as MockAdapter:
            # Mock batch conversion
            nat_messages = [Mock(), Mock()]
            MockAdapter.batch_autogen_to_nat.return_value = nat_messages

            # Get history
            history = self.wrapper.get_conversation_history()

            # Verify conversion was called
            MockAdapter.batch_autogen_to_nat.assert_called_once_with(test_messages)
            assert history == nat_messages

        # Test clearing history
        self.wrapper.clear_conversation_history()
        assert len(self.wrapper._conversation_history) == 0

    def test_context_management(self):
        """Test agent context management."""
        # Test getting empty context
        context = self.wrapper.get_context()
        assert context == {}

        # Test updating context
        new_context = {"key1": "value1", "key2": "value2"}
        self.wrapper.update_context(new_context)

        context = self.wrapper.get_context()
        assert context == new_context

        # Test updating with additional context
        additional_context = {"key3": "value3", "key1": "updated_value1"}
        self.wrapper.update_context(additional_context)

        context = self.wrapper.get_context()
        assert context == {"key1": "updated_value1", "key2": "value2", "key3": "value3"}

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test agent shutdown and cleanup."""
        # Add some test data
        self.wrapper._conversation_history = [Mock(), Mock()]
        self.wrapper._context = {"test": "data"}

        # Mock runtime stop
        self.mock_runtime.stop = AsyncMock()

        # Perform shutdown
        await self.wrapper.shutdown()

        # Verify runtime was stopped
        self.mock_runtime.stop.assert_called_once()

        # Verify cleanup
        assert len(self.wrapper._conversation_history) == 0
        assert len(self.wrapper._context) == 0

    @pytest.mark.asyncio
    async def test_error_handling_in_process_message(self):
        """Test error handling during message processing."""
        nat_message = Mock()
        nat_message.content = "Test message"
        nat_message.role = "user"

        # Mock agent to raise exception
        self.mock_autogen_agent.on_messages = AsyncMock(side_effect=Exception("Agent error"))

        with patch('nat.plugins.autogen.agent_wrapper.AutoGenMessageAdapter'):
            # Should raise the exception
            with pytest.raises(Exception, match="Agent error"):
                await self.wrapper.process_message(nat_message)


class TestAutoGenMultiAgentOrchestrator:
    """Test cases for AutoGen multi-agent orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_runtime = Mock(spec=AgentRuntime)
        self.orchestrator = AutoGenMultiAgentOrchestrator(self.mock_runtime)

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.runtime == self.mock_runtime
        assert len(self.orchestrator.agents) == 0

    def test_register_agent(self):
        """Test agent registration."""
        # Create mock agent wrapper
        mock_agent = Mock(spec=AutoGenAgentWrapper)
        mock_agent.agent_id = "test_agent"

        # Register agent
        self.orchestrator.register_agent(mock_agent)

        # Verify registration
        assert "test_agent" in self.orchestrator.agents
        assert self.orchestrator.agents["test_agent"] == mock_agent
        assert mock_agent.runtime == self.mock_runtime

    @pytest.mark.asyncio
    async def test_orchestrate_conversation(self):
        """Test multi-agent conversation orchestration."""
        # Create mock agents
        agents = {}
        for i in range(3):
            agent = Mock(spec=AutoGenAgentWrapper)
            agent.agent_id = f"agent_{i}"
            agent.process_conversation = AsyncMock(return_value=[Mock()])
            agents[f"agent_{i}"] = agent

        self.orchestrator.agents = agents

        # Create mock messages
        messages = [Mock(), Mock()]
        agent_sequence = ["agent_0", "agent_1", "agent_2"]

        # Orchestrate conversation
        results = await self.orchestrator.orchestrate_conversation(
            messages=messages,
            agent_sequence=agent_sequence
        )

        # Verify all agents were called
        for agent_id in agent_sequence:
            agents[agent_id].process_conversation.assert_called_once()

        # Verify results (should have 3 results, one from each agent)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_orchestrate_conversation_missing_agent(self):
        """Test orchestration with missing agent (should skip gracefully)."""
        # Register only one agent
        agent = Mock(spec=AutoGenAgentWrapper)
        agent.agent_id = "existing_agent"
        agent.process_conversation = AsyncMock(return_value=[Mock()])
        self.orchestrator.agents = {"existing_agent": agent}

        messages = [Mock()]
        agent_sequence = ["existing_agent", "missing_agent", "existing_agent"]

        # Should complete successfully, skipping missing agent
        results = await self.orchestrator.orchestrate_conversation(
            messages=messages,
            agent_sequence=agent_sequence
        )

        # Should have 2 results (missing agent skipped)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_shutdown_orchestrator(self):
        """Test orchestrator shutdown."""
        # Create mock agents
        agents = {}
        for i in range(2):
            agent = Mock(spec=AutoGenAgentWrapper)
            agent.agent_id = f"agent_{i}"
            agent.shutdown = AsyncMock()
            agents[f"agent_{i}"] = agent

        self.orchestrator.agents = agents

        # Mock runtime stop
        self.mock_runtime.stop = AsyncMock()

        # Perform shutdown
        await self.orchestrator.shutdown()

        # Verify all agents were shutdown
        for agent in agents.values():
            agent.shutdown.assert_called_once()

        # Verify runtime was stopped
        self.mock_runtime.stop.assert_called_once()

        # Verify cleanup
        assert len(self.orchestrator.agents) == 0