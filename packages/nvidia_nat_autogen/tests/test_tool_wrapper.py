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

"""Tests for AutoGen tool wrapper."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from pydantic import BaseModel, Field
from autogen_core.tools import FunctionTool

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.plugins.autogen.tool_wrapper import (
    autogen_tool_wrapper,
    execute_autogen_tool,
    process_autogen_result,
    _convert_pydantic_to_autogen_schema
)


class TestInputSchema(BaseModel):
    """Test input schema for tool testing."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum number of results")


class TestAutoGenToolWrapper:
    """Test cases for AutoGen tool wrapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = Mock(spec=Builder)

        # Create mock NAT function
        self.mock_function = Mock(spec=Function)
        self.mock_function.name = "test_tool"
        self.mock_function.description = "Test tool for AutoGen integration"
        self.mock_function.input_schema = TestInputSchema
        self.mock_function.acall_invoke = AsyncMock(return_value="Test result")

    @pytest.mark.asyncio
    async def test_process_autogen_result_string(self):
        """Test processing string results."""
        result = await process_autogen_result("Simple string result", "test_tool")
        assert result == "Simple string result"

    @pytest.mark.asyncio
    async def test_process_autogen_result_empty_string(self):
        """Test processing empty string results."""
        result = await process_autogen_result("", "test_tool")
        assert "returned an empty result" in result

    @pytest.mark.asyncio
    async def test_process_autogen_result_none(self):
        """Test processing None results."""
        result = await process_autogen_result(None, "test_tool")
        assert result == ""

    @pytest.mark.asyncio
    async def test_process_autogen_result_dict(self):
        """Test processing dictionary results."""
        test_dict = {"key1": "value1", "key2": "value2"}
        result = await process_autogen_result(test_dict, "test_tool")

        # Should be valid JSON
        import json
        parsed = json.loads(result)
        assert parsed == test_dict

    @pytest.mark.asyncio
    async def test_process_autogen_result_list_of_dicts(self):
        """Test processing list of dictionary results."""
        test_list = [
            {"name": "Item 1", "value": 100},
            {"name": "Item 2", "value": 200}
        ]
        result = await process_autogen_result(test_list, "test_tool")

        # Should be formatted nicely
        assert "Result 1:" in result
        assert "Result 2:" in result
        assert "name: Item 1" in result
        assert "value: 100" in result

    @pytest.mark.asyncio
    async def test_process_autogen_result_list_of_strings(self):
        """Test processing list of string results."""
        test_list = ["Item 1", "Item 2", "Item 3"]
        result = await process_autogen_result(test_list, "test_tool")

        assert "Results:" in result
        assert "1. Item 1" in result
        assert "2. Item 2" in result
        assert "3. Item 3" in result

    @pytest.mark.asyncio
    async def test_process_autogen_result_empty_list(self):
        """Test processing empty list results."""
        result = await process_autogen_result([], "test_tool")
        assert "returned an empty list" in result

    def test_convert_pydantic_to_autogen_schema(self):
        """Test Pydantic to AutoGen schema conversion."""
        schema = _convert_pydantic_to_autogen_schema(TestInputSchema)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert schema["required"] == ["query"]
        assert schema["additionalProperties"] is False

    def test_convert_pydantic_to_autogen_schema_error_handling(self):
        """Test schema conversion error handling."""
        # Create invalid schema mock
        invalid_schema = Mock()
        invalid_schema.model_json_schema.side_effect = Exception("Schema error")

        schema = _convert_pydantic_to_autogen_schema(invalid_schema)

        # Should return minimal schema on error
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["required"] == []

    @pytest.mark.asyncio
    async def test_execute_autogen_tool_basic(self):
        """Test basic AutoGen tool execution."""
        # Create async function mock
        async def mock_function(**kwargs):
            return f"Called with: {kwargs}"

        loop = asyncio.get_event_loop()

        # Execute tool
        result = execute_autogen_tool(
            name="test_tool",
            coroutine_fn=mock_function,
            required_fields=["query"],
            loop=loop,
            query="test query",
            max_results=5
        )

        assert "Called with:" in result
        assert "test query" in result

    @pytest.mark.asyncio
    async def test_execute_autogen_tool_missing_required_fields(self):
        """Test tool execution with missing required fields."""
        async def mock_function(**kwargs):
            return "success"

        loop = asyncio.get_event_loop()

        # Should raise ValueError for missing required field
        with pytest.raises(ValueError, match="Missing required parameter 'query'"):
            execute_autogen_tool(
                name="test_tool",
                coroutine_fn=mock_function,
                required_fields=["query"],
                loop=loop,
                max_results=5  # query is missing
            )

    @pytest.mark.asyncio
    async def test_execute_autogen_tool_reserved_keywords_filtering(self):
        """Test that reserved keywords are filtered from tool input."""
        async def mock_function(**kwargs):
            # Should not receive reserved keywords
            assert "type" not in kwargs
            assert "_type" not in kwargs
            assert "model_config" not in kwargs
            return f"Received: {list(kwargs.keys())}"

        loop = asyncio.get_event_loop()

        result = execute_autogen_tool(
            name="test_tool",
            coroutine_fn=mock_function,
            required_fields=["query"],
            loop=loop,
            query="test",
            type="reserved",  # Should be filtered
            _type="reserved",  # Should be filtered
            model_config="reserved"  # Should be filtered
        )

        assert "query" in result
        assert "type" not in result

    @pytest.mark.asyncio
    async def test_execute_autogen_tool_kwargs_unwrapping(self):
        """Test unwrapping of nested kwargs."""
        async def mock_function(**kwargs):
            return f"Unwrapped: {kwargs}"

        loop = asyncio.get_event_loop()

        # Test with wrapped kwargs
        result = execute_autogen_tool(
            name="test_tool",
            coroutine_fn=mock_function,
            required_fields=["query"],
            loop=loop,
            kwargs={  # Wrapped in kwargs
                "query": "test query",
                "max_results": 10
            }
        )

        assert "test query" in result
        assert "max_results" in result or "10" in result

    @pytest.mark.asyncio
    async def test_execute_autogen_tool_empty_query_handling(self):
        """Test handling of empty queries for search tools."""
        async def mock_function(**kwargs):
            return "success"

        loop = asyncio.get_event_loop()

        # First call should be allowed (initialization)
        result1 = execute_autogen_tool(
            name="search_api_tool",  # Tool name suggests it's a search tool
            coroutine_fn=mock_function,
            required_fields=["query"],
            loop=loop,
            query=""  # Empty query
        )

        # Should succeed for first initialization call
        assert "success" in result1

        # Second call with empty query should be blocked
        result2 = execute_autogen_tool(
            name="search_api_tool",
            coroutine_fn=mock_function,
            required_fields=["query"],
            loop=loop,
            query=""  # Empty query again
        )

        assert "ERROR" in result2
        assert "requires a valid query" in result2

    def test_autogen_tool_wrapper_creation(self):
        """Test AutoGen tool wrapper creation."""
        # Create wrapper
        function_tool = autogen_tool_wrapper("test_tool", self.mock_function, self.builder)

        # Verify it's a FunctionTool
        assert isinstance(function_tool, FunctionTool)

        # Verify name and description
        assert function_tool.name == "test_tool"
        assert function_tool.description == "Test tool for AutoGen integration"

    def test_autogen_tool_wrapper_without_input_schema(self):
        """Test wrapper creation fails without input schema."""
        # Remove input schema
        self.mock_function.input_schema = None

        # Should raise assertion error
        with pytest.raises(AssertionError, match="Tool test_tool must have input schema"):
            autogen_tool_wrapper("test_tool", self.mock_function, self.builder)

    @pytest.mark.asyncio
    async def test_autogen_tool_wrapper_execution(self):
        """Test executing wrapped AutoGen tool."""
        # Create wrapper
        function_tool = autogen_tool_wrapper("test_tool", self.mock_function, self.builder)

        # Execute the tool function
        with patch('asyncio.get_running_loop') as mock_loop:
            mock_loop.return_value = asyncio.get_event_loop()

            result = function_tool.func(query="test query", max_results=5)

            # Verify the underlying function was called
            self.mock_function.acall_invoke.assert_called_once()

            # Result should be processed string
            assert isinstance(result, str)

    def test_autogen_tool_wrapper_function_metadata(self):
        """Test that wrapped function has correct metadata."""
        function_tool = autogen_tool_wrapper("test_tool", self.mock_function, self.builder)

        # Check function name and docstring
        assert function_tool.func.__name__ == "test_tool"
        assert function_tool.func.__doc__ == "Test tool for AutoGen integration"

    def test_autogen_tool_wrapper_schema_integration(self):
        """Test schema integration in wrapped tool."""
        function_tool = autogen_tool_wrapper("test_tool", self.mock_function, self.builder)

        # Verify schema is properly converted
        schema = function_tool.parameters_json_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]

    @pytest.mark.asyncio
    async def test_execute_autogen_tool_timeout_handling(self):
        """Test tool execution timeout handling."""
        # Create slow function
        async def slow_function(**kwargs):
            await asyncio.sleep(0.1)  # Simulate slow operation
            return "completed"

        loop = asyncio.get_event_loop()

        # Should complete within timeout
        result = execute_autogen_tool(
            name="slow_tool",
            coroutine_fn=slow_function,
            required_fields=[],
            loop=loop
        )

        assert "completed" in result

    @pytest.mark.asyncio
    async def test_execute_autogen_tool_positional_args_fallback(self):
        """Test fallback to positional arguments on TypeError."""
        # Create function that only accepts positional args
        async def positional_function(input_obj):
            return f"Positional: {input_obj}"

        loop = asyncio.get_event_loop()

        # Should fallback to positional args
        with patch('nat.plugins.autogen.tool_wrapper.asyncio.run_coroutine_threadsafe') as mock_run:
            # First call with kwargs fails, second succeeds
            future1 = Mock()
            future1.result.side_effect = TypeError("missing 1 required positional argument: 'input_obj'")
            future2 = Mock()
            future2.result.return_value = "Positional result"

            mock_run.side_effect = [future1, future2]

            result = execute_autogen_tool(
                name="pos_tool",
                coroutine_fn=positional_function,
                required_fields=[],
                loop=loop,
                test_param="value"
            )

            # Should have tried both calling styles
            assert mock_run.call_count == 2