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

from typing import Any, Dict, List, Union

from autogen_core.models import UserMessage, AssistantMessage, SystemMessage
from nat.data_models.api_server import Message as NATMessage


class AutoGenMessageAdapter:
    """Bidirectional message translation between NAT and AutoGen formats."""

    @staticmethod
    def nat_to_autogen(nat_message: NATMessage) -> Union[UserMessage, AssistantMessage, SystemMessage]:
        """Convert NAT message to AutoGen message format.

        Args:
            nat_message: NAT message object

        Returns:
            Corresponding AutoGen message object

        Raises:
            ValueError: If message role is unsupported
        """
        content = nat_message.content
        role = nat_message.role.lower()

        # Handle metadata preservation
        metadata = getattr(nat_message, 'metadata', {})

        if role == "user":
            return UserMessage(content=content, source="user")
        elif role == "assistant":
            return AssistantMessage(content=content, source="assistant")
        elif role == "system":
            return SystemMessage(content=content)
        else:
            raise ValueError(f"Unsupported message role: {role}")

    @staticmethod
    def autogen_to_nat(autogen_message: Union[UserMessage, AssistantMessage, SystemMessage]) -> NATMessage:
        """Convert AutoGen message to NAT message format.

        Args:
            autogen_message: AutoGen message object

        Returns:
            Corresponding NAT message object
        """
        content = autogen_message.content

        # Determine role based on message type
        if isinstance(autogen_message, UserMessage):
            role = "user"
        elif isinstance(autogen_message, AssistantMessage):
            role = "assistant"
        elif isinstance(autogen_message, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported AutoGen message type: {type(autogen_message)}")

        # Create NAT message with preserved metadata
        nat_message = NATMessage(
            role=role,
            content=content
        )

        # Preserve AutoGen-specific metadata if present
        if hasattr(autogen_message, 'source'):
            nat_message.metadata = {"autogen_source": autogen_message.source}

        return nat_message

    @staticmethod
    def batch_nat_to_autogen(nat_messages: List[NATMessage]) -> List[Union[UserMessage, AssistantMessage, SystemMessage]]:
        """Convert batch of NAT messages to AutoGen format.

        Args:
            nat_messages: List of NAT message objects

        Returns:
            List of corresponding AutoGen message objects
        """
        return [AutoGenMessageAdapter.nat_to_autogen(msg) for msg in nat_messages]

    @staticmethod
    def batch_autogen_to_nat(autogen_messages: List[Union[UserMessage, AssistantMessage, SystemMessage]]) -> List[NATMessage]:
        """Convert batch of AutoGen messages to NAT format.

        Args:
            autogen_messages: List of AutoGen message objects

        Returns:
            List of corresponding NAT message objects
        """
        return [AutoGenMessageAdapter.autogen_to_nat(msg) for msg in autogen_messages]

    @staticmethod
    def validate_message_format(message: Any) -> bool:
        """Validate message format for integrity.

        Args:
            message: Message object to validate

        Returns:
            True if valid, False otherwise
        """
        if isinstance(message, NATMessage):
            return hasattr(message, 'role') and hasattr(message, 'content')
        elif isinstance(message, (UserMessage, AssistantMessage, SystemMessage)):
            return hasattr(message, 'content')
        return False