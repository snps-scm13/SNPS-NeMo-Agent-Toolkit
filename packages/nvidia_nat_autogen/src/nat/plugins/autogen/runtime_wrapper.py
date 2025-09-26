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

from typing import Any, Dict, List, Optional, Set
import logging
from contextlib import asynccontextmanager

from autogen_core import (
    AgentId,
    AgentRuntime,
    SingleThreadedAgentRuntime,
    MessageContext,
    TopicId
)
from autogen_agentchat.agents import AssistantAgent

from nat.data_models.api_server import Message as NATMessage
from .message_adapter import AutoGenMessageAdapter

logger = logging.getLogger(__name__)


class AutoGenRuntimeManager:
    """Manager for AutoGen runtime lifecycle and agent coordination.

    This class provides a bridge between NAT's workflow system and
    AutoGen's runtime-based multi-agent orchestration capabilities.
    """

    def __init__(self, runtime_config: Optional[Dict[str, Any]] = None):
        """Initialize AutoGen runtime manager.

        Args:
            runtime_config: Configuration for runtime behavior
        """
        self.config = runtime_config or {}
        self.runtime: Optional[AgentRuntime] = None
        self.registered_agents: Dict[str, AssistantAgent] = {}
        self.agent_ids: Dict[str, AgentId] = {}
        self.active_topics: Set[TopicId] = set()
        self._is_running = False

    async def start_runtime(self) -> None:
        """Start the AutoGen runtime system."""
        try:
            if self._is_running:
                logger.warning("Runtime already running")
                return

            # Create single-threaded runtime (most common use case)
            self.runtime = SingleThreadedAgentRuntime()
            await self.runtime.start()
            self._is_running = True

            logger.info("AutoGen runtime started successfully")

        except Exception as e:
            logger.error(f"Failed to start AutoGen runtime: {e}")
            raise

    async def stop_runtime(self) -> None:
        """Stop the AutoGen runtime system."""
        try:
            if not self._is_running or not self.runtime:
                logger.warning("Runtime not running")
                return

            # Stop runtime
            await self.runtime.stop()
            self.runtime = None
            self._is_running = False

            # Clear state
            self.registered_agents.clear()
            self.agent_ids.clear()
            self.active_topics.clear()

            logger.info("AutoGen runtime stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping AutoGen runtime: {e}")
            raise

    def register_agent(
        self,
        agent: AssistantAgent,
        agent_name: str,
        topics: Optional[List[str]] = None
    ) -> AgentId:
        """Register an agent with the runtime.

        Args:
            agent: AutoGen conversable agent
            agent_name: Unique name for the agent
            topics: List of topics the agent should subscribe to

        Returns:
            Agent ID for runtime operations

        Raises:
            RuntimeError: If runtime is not started
        """
        if not self._is_running or not self.runtime:
            raise RuntimeError("Runtime must be started before registering agents")

        try:
            # Create agent ID
            agent_id = AgentId(agent_name, key="default")

            # Register with runtime
            self.runtime.add_agent(type(agent), agent_id)

            # Store references
            self.registered_agents[agent_name] = agent
            self.agent_ids[agent_name] = agent_id

            # Subscribe to topics if provided
            if topics:
                for topic_name in topics:
                    topic_id = TopicId(topic_name, source=agent_name)
                    self.runtime.add_subscription(
                        topic_id,
                        agent_id
                    )
                    self.active_topics.add(topic_id)

            logger.info(f"Agent {agent_name} registered successfully with runtime")
            return agent_id

        except Exception as e:
            logger.error(f"Failed to register agent {agent_name}: {e}")
            raise

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent from the runtime.

        Args:
            agent_name: Name of agent to unregister
        """
        try:
            if agent_name not in self.registered_agents:
                logger.warning(f"Agent {agent_name} not found in registry")
                return

            # Remove from runtime if running
            if self._is_running and self.runtime:
                agent_id = self.agent_ids[agent_name]
                # Note: AutoGen runtime doesn't have direct remove_agent method
                # Agents are cleaned up when runtime stops

            # Remove from local registry
            del self.registered_agents[agent_name]
            del self.agent_ids[agent_name]

            # Clean up topics associated with this agent
            topics_to_remove = [
                topic for topic in self.active_topics
                if topic.source == agent_name
            ]
            for topic in topics_to_remove:
                self.active_topics.remove(topic)

            logger.info(f"Agent {agent_name} unregistered successfully")

        except Exception as e:
            logger.error(f"Error unregistering agent {agent_name}: {e}")
            raise

    async def send_message_to_agent(
        self,
        message: NATMessage,
        target_agent: str,
        context: Optional[Dict[str, Any]] = None
    ) -> NATMessage:
        """Send message to specific agent via runtime.

        Args:
            message: Message in NAT format
            target_agent: Name of target agent
            context: Additional context for message

        Returns:
            Response message in NAT format

        Raises:
            RuntimeError: If runtime not running or agent not found
        """
        if not self._is_running or not self.runtime:
            raise RuntimeError("Runtime must be running to send messages")

        if target_agent not in self.registered_agents:
            raise RuntimeError(f"Agent {target_agent} not registered")

        try:
            # Convert to AutoGen message format
            autogen_message = AutoGenMessageAdapter.nat_to_autogen(message)

            # Get agent ID
            agent_id = self.agent_ids[target_agent]

            # Send message via runtime
            response = await self.runtime.send_message(
                autogen_message,
                agent_id
            )

            # Convert response back to NAT format
            nat_response = AutoGenMessageAdapter.autogen_to_nat(response)

            logger.debug(f"Message sent to agent {target_agent} successfully")
            return nat_response

        except Exception as e:
            logger.error(f"Error sending message to agent {target_agent}: {e}")
            raise

    async def broadcast_message(
        self,
        message: NATMessage,
        topic: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[NATMessage]:
        """Broadcast message to all agents subscribed to topic.

        Args:
            message: Message in NAT format
            topic: Topic to publish to
            context: Additional context for message

        Returns:
            List of response messages from subscribed agents

        Raises:
            RuntimeError: If runtime not running
        """
        if not self._is_running or not self.runtime:
            raise RuntimeError("Runtime must be running to broadcast messages")

        try:
            # Convert to AutoGen message format
            autogen_message = AutoGenMessageAdapter.nat_to_autogen(message)

            # Create topic ID
            topic_id = TopicId(topic, source="broadcast")

            # Publish message to topic
            await self.runtime.publish_message(
                autogen_message,
                topic_id
            )

            # Note: Broadcasting doesn't return direct responses
            # Responses would come through event handlers
            logger.info(f"Message broadcasted to topic {topic}")
            return []

        except Exception as e:
            logger.error(f"Error broadcasting message to topic {topic}: {e}")
            raise

    async def orchestrate_multi_agent_workflow(
        self,
        initial_message: NATMessage,
        agent_sequence: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[NATMessage]:
        """Orchestrate multi-agent workflow with sequential processing.

        Args:
            initial_message: Starting message for workflow
            agent_sequence: Ordered list of agents to process message
            context: Shared context for workflow

        Returns:
            List of messages from workflow execution

        Raises:
            RuntimeError: If runtime not running or agents not found
        """
        if not self._is_running or not self.runtime:
            raise RuntimeError("Runtime must be running for workflow orchestration")

        # Verify all agents are registered
        missing_agents = [agent for agent in agent_sequence
                         if agent not in self.registered_agents]
        if missing_agents:
            raise RuntimeError(f"Agents not registered: {missing_agents}")

        try:
            results = []
            current_message = initial_message

            # Process message through each agent in sequence
            for agent_name in agent_sequence:
                logger.debug(f"Processing message with agent: {agent_name}")

                response = await self.send_message_to_agent(
                    current_message,
                    agent_name,
                    context
                )

                results.append(response)
                current_message = response  # Use response as input for next agent

            logger.info(f"Multi-agent workflow completed with {len(results)} steps")
            return results

        except Exception as e:
            logger.error(f"Error in multi-agent workflow orchestration: {e}")
            raise

    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names.

        Returns:
            List of agent names
        """
        return list(self.registered_agents.keys())

    def get_active_topics(self) -> List[str]:
        """Get list of active topic names.

        Returns:
            List of topic names
        """
        return [topic.type for topic in self.active_topics]

    def is_runtime_running(self) -> bool:
        """Check if runtime is currently running.

        Returns:
            True if runtime is running, False otherwise
        """
        return self._is_running

    @asynccontextmanager
    async def runtime_context(self):
        """Context manager for runtime lifecycle.

        Ensures runtime is properly started and stopped.
        """
        await self.start_runtime()
        try:
            yield self
        finally:
            await self.stop_runtime()

    async def get_runtime_stats(self) -> Dict[str, Any]:
        """Get runtime statistics and health information.

        Returns:
            Dictionary containing runtime statistics
        """
        return {
            "is_running": self._is_running,
            "registered_agents": len(self.registered_agents),
            "active_topics": len(self.active_topics),
            "agent_names": list(self.registered_agents.keys()),
            "topic_names": [topic.type for topic in self.active_topics],
            "config": self.config
        }