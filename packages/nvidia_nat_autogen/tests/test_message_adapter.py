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

"""Tests for AutoGen message adapter."""

import pytest
from unittest.mock import Mock

from autogen_core.models import UserMessage, AssistantMessage, SystemMessage

from nat.plugins.autogen.message_adapter import AutoGenMessageAdapter


class TestAutoGenMessageAdapter:
    """Test cases for AutoGen message adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = AutoGenMessageAdapter()

    def test_nat_to_autogen_user_message(self):
        """Test conversion of NAT user message to AutoGen format."""
        # Create mock NAT message
        nat_message = Mock()
        nat_message.content = "Hello, world!"
        nat_message.role = "user"
        nat_message.metadata = {"test": "value"}

        # Convert to AutoGen
        autogen_message = self.adapter.nat_to_autogen(nat_message)

        # Verify result
        assert isinstance(autogen_message, UserMessage)
        assert autogen_message.content == "Hello, world!"
        assert autogen_message.source == "user"

    def test_nat_to_autogen_assistant_message(self):
        """Test conversion of NAT assistant message to AutoGen format."""
        # Create mock NAT message
        nat_message = Mock()
        nat_message.content = "I can help you with that."
        nat_message.role = "assistant"

        # Convert to AutoGen
        autogen_message = self.adapter.nat_to_autogen(nat_message)

        # Verify result
        assert isinstance(autogen_message, AssistantMessage)
        assert autogen_message.content == "I can help you with that."
        assert autogen_message.source == "assistant"

    def test_nat_to_autogen_system_message(self):
        """Test conversion of NAT system message to AutoGen format."""
        # Create mock NAT message
        nat_message = Mock()
        nat_message.content = "You are a helpful assistant."
        nat_message.role = "system"

        # Convert to AutoGen
        autogen_message = self.adapter.nat_to_autogen(nat_message)

        # Verify result
        assert isinstance(autogen_message, SystemMessage)
        assert autogen_message.content == "You are a helpful assistant."

    def test_nat_to_autogen_unsupported_role(self):
        """Test conversion with unsupported role raises ValueError."""
        # Create mock NAT message with unsupported role
        nat_message = Mock()
        nat_message.content = "Test message"
        nat_message.role = "unknown"

        # Should raise ValueError
        with pytest.raises(ValueError, match="Unsupported message role: unknown"):
            self.adapter.nat_to_autogen(nat_message)

    def test_autogen_to_nat_user_message(self):
        """Test conversion of AutoGen user message to NAT format."""
        # Create AutoGen message
        autogen_message = UserMessage(content="Test user message", source="user")

        # Convert to NAT
        nat_message = self.adapter.autogen_to_nat(autogen_message)

        # Verify result
        assert nat_message.role == "user"
        assert nat_message.content == "Test user message"
        assert nat_message.metadata == {"autogen_source": "user"}

    def test_autogen_to_nat_assistant_message(self):
        """Test conversion of AutoGen assistant message to NAT format."""
        # Create AutoGen message
        autogen_message = AssistantMessage(content="Test assistant response", source="assistant")

        # Convert to NAT
        nat_message = self.adapter.autogen_to_nat(autogen_message)

        # Verify result
        assert nat_message.role == "assistant"
        assert nat_message.content == "Test assistant response"
        assert nat_message.metadata == {"autogen_source": "assistant"}

    def test_autogen_to_nat_system_message(self):
        """Test conversion of AutoGen system message to NAT format."""
        # Create AutoGen message
        autogen_message = SystemMessage(content="System instruction")

        # Convert to NAT
        nat_message = self.adapter.autogen_to_nat(autogen_message)

        # Verify result
        assert nat_message.role == "system"
        assert nat_message.content == "System instruction"

    def test_batch_nat_to_autogen(self):
        """Test batch conversion of NAT messages to AutoGen format."""
        # Create mock NAT messages
        nat_messages = []
        for i in range(3):
            msg = Mock()
            msg.content = f"Message {i}"
            msg.role = "user" if i % 2 == 0 else "assistant"
            nat_messages.append(msg)

        # Convert batch
        autogen_messages = self.adapter.batch_nat_to_autogen(nat_messages)

        # Verify results
        assert len(autogen_messages) == 3
        assert isinstance(autogen_messages[0], UserMessage)
        assert isinstance(autogen_messages[1], AssistantMessage)
        assert isinstance(autogen_messages[2], UserMessage)

    def test_batch_autogen_to_nat(self):
        """Test batch conversion of AutoGen messages to NAT format."""
        # Create AutoGen messages
        autogen_messages = [
            UserMessage(content="User message", source="user"),
            AssistantMessage(content="Assistant message", source="assistant"),
            SystemMessage(content="System message")
        ]

        # Convert batch
        nat_messages = self.adapter.batch_autogen_to_nat(autogen_messages)

        # Verify results
        assert len(nat_messages) == 3
        assert nat_messages[0].role == "user"
        assert nat_messages[1].role == "assistant"
        assert nat_messages[2].role == "system"

    def test_validate_message_format_nat_message(self):
        """Test message format validation for NAT messages."""
        # Create valid NAT message
        nat_message = Mock()
        nat_message.role = "user"
        nat_message.content = "Test message"

        # Should be valid
        assert self.adapter.validate_message_format(nat_message) is True

    def test_validate_message_format_autogen_message(self):
        """Test message format validation for AutoGen messages."""
        # Create valid AutoGen message
        autogen_message = UserMessage(content="Test message", source="user")

        # Should be valid
        assert self.adapter.validate_message_format(autogen_message) is True

    def test_validate_message_format_invalid(self):
        """Test message format validation for invalid messages."""
        # Test with string (invalid)
        assert self.adapter.validate_message_format("invalid") is False

        # Test with None
        assert self.adapter.validate_message_format(None) is False

        # Test with incomplete mock
        incomplete_msg = Mock()
        assert self.adapter.validate_message_format(incomplete_msg) is False

    def test_case_insensitive_role_handling(self):
        """Test that role handling is case insensitive."""
        # Create mock NAT message with uppercase role
        nat_message = Mock()
        nat_message.content = "Test message"
        nat_message.role = "USER"

        # Convert to AutoGen
        autogen_message = self.adapter.nat_to_autogen(nat_message)

        # Should work correctly
        assert isinstance(autogen_message, UserMessage)
        assert autogen_message.content == "Test message"