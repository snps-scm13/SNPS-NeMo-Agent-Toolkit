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

"""Tests for AutoGen LLM client integration."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin

from nat.plugins.autogen.llm import openai_autogen, nim_autogen, _patch_autogen_client_based_on_config


class TestAutoGenLLMIntegration:
    """Test cases for AutoGen LLM client integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = Mock(spec=Builder)

    @pytest.mark.asyncio
    async def test_openai_autogen_client_creation(self):
        """Test OpenAI AutoGen client creation."""
        # Create OpenAI config
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            temperature=0.7
        )

        with patch('nat.plugins.autogen.llm.OpenAIChatCompletionClient') as MockClient:
            mock_client_instance = Mock()
            MockClient.return_value = mock_client_instance

            # Test client creation
            async for client in openai_autogen(config, self.builder):
                # Verify client was created with correct parameters
                MockClient.assert_called_once()
                call_args = MockClient.call_args

                # Check that model name was passed
                assert call_args.kwargs['model'] == "gpt-4"

                # Verify client is returned
                assert client is not None
                break

    @pytest.mark.asyncio
    async def test_nim_autogen_client_creation(self):
        """Test NVIDIA NIM AutoGen client creation."""
        # Create NIM config
        config = NIMModelConfig(
            model_name="llama-3-8b-instruct",
            api_key="test-nim-key",
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.5
        )

        with patch('nat.plugins.autogen.llm.OpenAIChatCompletionClient') as MockClient:
            mock_client_instance = Mock()
            MockClient.return_value = mock_client_instance

            # Test client creation
            async for client in nim_autogen(config, self.builder):
                # Verify client was created (NIM uses OpenAI-compatible client)
                MockClient.assert_called_once()
                call_args = MockClient.call_args

                # Check that model name was passed
                assert call_args.kwargs['model'] == "llama-3-8b-instruct"

                # Verify client is returned
                assert client is not None
                break

    def test_patch_autogen_client_with_retry_mixin(self):
        """Test patching AutoGen client with retry mixin."""
        # Create config with retry settings
        class TestConfigWithRetry(RetryMixin):
            num_retries = 3
            retry_on_status_codes = [500, 502, 503]
            retry_on_errors = ["timeout", "connection_error"]

        config = TestConfigWithRetry()
        mock_client = Mock()

        with patch('nat.plugins.autogen.llm.patch_with_retry') as mock_patch_retry:
            mock_patch_retry.return_value = mock_client

            # Apply patching
            result = _patch_autogen_client_based_on_config(mock_client, config)

            # Verify retry patching was applied
            mock_patch_retry.assert_called_once_with(
                mock_client,
                retries=3,
                retry_codes=[500, 502, 503],
                retry_on_messages=["timeout", "connection_error"]
            )
            assert result == mock_client

    def test_patch_autogen_client_with_thinking_mixin(self):
        """Test patching AutoGen client with thinking mixin."""
        # Create config with thinking settings
        class TestConfigWithThinking(ThinkingMixin):
            thinking_system_prompt = "Think step by step before responding."

        config = TestConfigWithThinking()
        mock_client = Mock()

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch_thinking:
            mock_patch_thinking.return_value = mock_client

            # Apply patching
            result = _patch_autogen_client_based_on_config(mock_client, config)

            # Verify thinking patching was applied
            mock_patch_thinking.assert_called_once()
            call_args = mock_patch_thinking.call_args

            # Check that thinking injector was passed
            thinking_injector = call_args[0][1]
            assert thinking_injector.system_prompt == "Think step by step before responding."

            # Check function names are correct for AutoGen
            assert "create" in thinking_injector.function_names
            assert "acreate" in thinking_injector.function_names

            assert result == mock_client

    def test_patch_autogen_client_with_both_mixins(self):
        """Test patching AutoGen client with both retry and thinking mixins."""
        # Create config with both mixins
        class TestConfigWithBoth(RetryMixin, ThinkingMixin):
            num_retries = 2
            retry_on_status_codes = [500]
            retry_on_errors = ["timeout"]
            thinking_system_prompt = "Think carefully."

        config = TestConfigWithBoth()
        mock_client = Mock()

        with patch('nat.plugins.autogen.llm.patch_with_retry') as mock_patch_retry, \
             patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch_thinking:

            # Set up chained patching
            retry_patched_client = Mock()
            thinking_patched_client = Mock()
            mock_patch_retry.return_value = retry_patched_client
            mock_patch_thinking.return_value = thinking_patched_client

            # Apply patching
            result = _patch_autogen_client_based_on_config(mock_client, config)

            # Verify both patches were applied in order
            mock_patch_retry.assert_called_once_with(
                mock_client,
                retries=2,
                retry_codes=[500],
                retry_on_messages=["timeout"]
            )

            mock_patch_thinking.assert_called_once()
            # Thinking should be applied to the retry-patched client
            assert mock_patch_thinking.call_args[0][0] == retry_patched_client

            assert result == thinking_patched_client

    def test_patch_autogen_client_no_mixins(self):
        """Test patching AutoGen client with no mixins returns original client."""
        # Create basic config without mixins
        class TestConfigBasic:
            model_name = "test-model"

        config = TestConfigBasic()
        mock_client = Mock()

        with patch('nat.plugins.autogen.llm.patch_with_retry') as mock_patch_retry, \
             patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch_thinking:

            # Apply patching
            result = _patch_autogen_client_based_on_config(mock_client, config)

            # Verify no patches were applied
            mock_patch_retry.assert_not_called()
            mock_patch_thinking.assert_not_called()

            # Should return original client
            assert result == mock_client

    def test_autogen_thinking_injector_message_handling(self):
        """Test AutoGen thinking injector message handling."""
        from nat.plugins.autogen.llm import _patch_autogen_client_based_on_config
        from autogen_core.models import SystemMessage

        # Create config with thinking
        class TestConfigWithThinking(ThinkingMixin):
            thinking_system_prompt = "Test thinking prompt"

        config = TestConfigWithThinking()
        mock_client = Mock()

        with patch('nat.plugins.autogen.llm.patch_with_thinking') as mock_patch_thinking:
            # Apply patching to capture the injector
            _patch_autogen_client_based_on_config(mock_client, config)

            # Get the thinking injector that was created
            thinking_injector = mock_patch_thinking.call_args[0][1]

            # Test message injection
            test_messages = [Mock(), Mock()]
            result = thinking_injector.inject(test_messages)

            # Verify system message was prepended
            assert len(result.args) == 3  # system message + 2 original messages

            # First message should be system message with thinking prompt
            first_message = result.args[0]
            assert isinstance(first_message, SystemMessage)
            assert first_message.content == "Test thinking prompt"

    @pytest.mark.asyncio
    async def test_openai_autogen_config_filtering(self):
        """Test that OpenAI AutoGen config properly filters excluded fields."""
        # Create config with fields that should be excluded
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.8,
            thinking_system_prompt="Think step by step"  # Should be excluded
        )

        with patch('nat.plugins.autogen.llm.OpenAIChatCompletionClient') as MockClient:
            mock_client_instance = Mock()
            MockClient.return_value = mock_client_instance

            async for client in openai_autogen(config, self.builder):
                # Verify excluded fields are not passed to client
                call_kwargs = MockClient.call_args.kwargs

                assert 'type' not in call_kwargs
                assert 'model_name' not in call_kwargs  # Should be passed as 'model'
                assert 'thinking' not in call_kwargs

                # Verify included fields are passed
                assert call_kwargs['model'] == "gpt-4"
                assert 'api_key' in call_kwargs or 'temperature' in call_kwargs
                break

    @pytest.mark.asyncio
    async def test_autogen_client_generator_pattern(self):
        """Test that AutoGen client functions use proper generator pattern."""
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="test-key"
        )

        with patch('nat.plugins.autogen.llm.OpenAIChatCompletionClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client

            # Test that function returns async generator
            gen = openai_autogen(config, self.builder)
            assert hasattr(gen, '__aiter__')

            # Test that we can iterate through generator
            clients = []
            async for client in gen:
                clients.append(client)

            # Should yield exactly one client
            assert len(clients) == 1
            assert clients[0] == mock_client