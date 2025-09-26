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

from typing import Any, Dict, List, Optional, Union
import asyncio
import logging

from autogen_core import AgentId, AgentRuntime
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import ChatMessage

from abc import ABC
from nat.data_models.api_server import Message as NATMessage
from .message_adapter import AutoGenMessageAdapter

logger = logging.getLogger(__name__)


class AutoGenAgentWrapper:
    """Wrapper for AutoGen agents to work with NAT framework.

    This wrapper enables AutoGen agents to be used seamlessly within
    NAT's agent orchestration system while preserving AutoGen's
    multi-agent coordination capabilities.
    """

    def __init__(
        self,
        autogen_agent: BaseChatAgent,
        agent_id: str,
        runtime: Optional[AgentRuntime] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize AutoGen agent wrapper.

        Args:
            autogen_agent: The underlying AutoGen agent
            agent_id: Unique identifier for this agent
            runtime: AutoGen runtime for multi-agent scenarios
            config: Additional configuration parameters
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.autogen_agent = autogen_agent
        self.runtime = runtime
        self._conversation_history = []
        self._context: Dict[str, Any] = {}

    async def process_message(
        self,
        message: NATMessage,
        context: Optional[Dict[str, Any]] = None
    ) -> NATMessage:
        """Process a single message through the AutoGen agent.

        Args:
            message: Input message in NAT format
            context: Additional context for processing

        Returns:
            Response message in NAT format
        """
        try:
            # Update context
            if context:
                self._context.update(context)

            # Convert NAT message to AutoGen format
            autogen_message = AutoGenMessageAdapter.nat_to_autogen(message)

            # Add to conversation history
            self._conversation_history.append(autogen_message)

            # Process message through AutoGen agent
            response = await self._process_autogen_message(autogen_message)

            # Convert response back to NAT format
            nat_response = AutoGenMessageAdapter.autogen_to_nat(response)

            # Update conversation history with response
            self._conversation_history.append(response)

            return nat_response

        except Exception as e:
            logger.error(f"Error processing message in AutoGen agent {self.agent_id}: {e}")
            raise

    async def process_conversation(
        self,
        messages: List[NATMessage],
        context: Optional[Dict[str, Any]] = None
    ) -> List[NATMessage]:
        """Process a multi-message conversation.

        Args:
            messages: List of messages in NAT format
            context: Additional context for processing

        Returns:
            List of response messages in NAT format
        """
        try:
            if context:
                self._context.update(context)

            # Convert all messages to AutoGen format
            autogen_messages = AutoGenMessageAdapter.batch_nat_to_autogen(messages)

            # Process conversation through AutoGen agent
            responses = await self._process_autogen_conversation(autogen_messages)

            # Convert responses back to NAT format
            nat_responses = AutoGenMessageAdapter.batch_autogen_to_nat(responses)

            # Update conversation history
            self._conversation_history.extend(autogen_messages + responses)

            return nat_responses

        except Exception as e:
            logger.error(f"Error processing conversation in AutoGen agent {self.agent_id}: {e}")
            raise

    async def _process_autogen_message(self, message: ChatMessage) -> ChatMessage:
        """Process single message through AutoGen agent.

        Args:
            message: AutoGen message

        Returns:
            AutoGen response message
        """
        if hasattr(self.autogen_agent, 'on_messages'):
            # Use runtime-based processing for multi-agent scenarios
            if self.runtime:
                agent_id = AgentId(self.agent_id, key="default")
                response = await self.runtime.send_message(
                    message,
                    agent_id
                )
                return response
            else:
                # Direct agent processing
                response = await self.autogen_agent.on_messages([message])
                return response.chat_message if hasattr(response, 'chat_message') else response
        else:
            # Fallback for older AutoGen API
            response = await self.autogen_agent.generate_reply(messages=[message])
            return ChatMessage(content=response, source=self.agent_id)

    async def _process_autogen_conversation(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Process multi-message conversation through AutoGen agent.

        Args:
            messages: List of AutoGen messages

        Returns:
            List of AutoGen response messages
        """
        responses = []

        for message in messages:
            response = await self._process_autogen_message(message)
            responses.append(response)

        return responses

    def get_conversation_history(self) -> List[NATMessage]:
        """Get conversation history in NAT format.

        Returns:
            List of messages in conversation history
        """
        return AutoGenMessageAdapter.batch_autogen_to_nat(self._conversation_history)

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()

    def get_context(self) -> Dict[str, Any]:
        """Get current context.

        Returns:
            Current context dictionary
        """
        return self._context.copy()

    def update_context(self, context: Dict[str, Any]) -> None:
        """Update agent context.

        Args:
            context: Context updates to apply
        """
        self._context.update(context)

    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        try:
            if self.runtime:
                await self.runtime.stop()

            # Clear conversation history
            self.clear_conversation_history()

            # Clear context
            self._context.clear()

            logger.info(f"AutoGen agent {self.agent_id} shutdown complete")

        except Exception as e:
            logger.error(f"Error during AutoGen agent {self.agent_id} shutdown: {e}")
            raise


class AutoGenMultiAgentOrchestrator:
    """Orchestrator for managing multiple AutoGen agents in NAT workflows."""

    def __init__(self, runtime: AgentRuntime):
        """Initialize multi-agent orchestrator.

        Args:
            runtime: AutoGen runtime for agent coordination
        """
        self.runtime = runtime
        self.agents: Dict[str, AutoGenAgentWrapper] = {}

    def register_agent(self, agent: AutoGenAgentWrapper) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: AutoGen agent wrapper to register
        """
        self.agents[agent.agent_id] = agent
        agent.runtime = self.runtime

    async def orchestrate_conversation(
        self,
        messages: List[NATMessage],
        agent_sequence: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[NATMessage]:
        """Orchestrate multi-agent conversation.

        Args:
            messages: Initial messages to process
            agent_sequence: Sequence of agent IDs to process messages
            context: Shared context for all agents

        Returns:
            Final conversation results in NAT format
        """
        try:
            results = []
            current_messages = messages.copy()

            # If no sequence provided, use all registered agents
            if not agent_sequence:
                agent_sequence = list(self.agents.keys())

            # Process messages through each agent in sequence
            for agent_id in agent_sequence:
                if agent_id not in self.agents:
                    logger.warning(f"Agent {agent_id} not found in orchestrator")
                    continue

                agent = self.agents[agent_id]
                agent_results = await agent.process_conversation(current_messages, context)
                results.extend(agent_results)

                # Use results as input for next agent
                current_messages = agent_results

            return results

        except Exception as e:
            logger.error(f"Error in multi-agent orchestration: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown all agents and cleanup resources."""
        try:
            # Shutdown all registered agents
            for agent in self.agents.values():
                await agent.shutdown()

            # Stop runtime
            if self.runtime:
                await self.runtime.stop()

            # Clear agent registry
            self.agents.clear()

            logger.info("Multi-agent orchestrator shutdown complete")

        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}")
            raise