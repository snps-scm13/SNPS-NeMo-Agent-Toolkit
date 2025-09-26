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

"""Tool wrapper for AutoGen integration with NAT."""

import asyncio
import json
import logging
import textwrap
from typing import Any, Awaitable, Callable, Dict, List, Optional

from autogen_core.tools import FunctionTool
# PythonType not available in AutoGen 0.7.4, using Any instead
from typing import Any as PythonType
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper

logger = logging.getLogger(__name__)

# Tool call tracking for loop prevention (similar to AGNO pattern)
_tool_call_counters = {}
_MAX_EMPTY_CALLS = 1
_tool_initialization_done = {}


async def process_autogen_result(result: Any, name: str) -> str:
    """Process result from AutoGen tool execution.

    Args:
        result: The result to process
        name: Tool name for logging

    Returns:
        Processed result as string
    """
    logger.debug(f"{name} processing result of type {type(result)}")

    # Handle None or empty results
    if result is None:
        logger.warning(f"{name} returned None, converting to empty string")
        return ""

    # If already a string, validate and return
    if isinstance(result, str):
        logger.debug(f"{name} returning string result directly")
        if not result.strip():
            return f"The {name} tool completed successfully but returned an empty result."
        return result

    # Handle structured responses with content
    if hasattr(result, 'content'):
        logger.debug(f"{name} returning result.content")
        content = result.content
        if not isinstance(content, str):
            content = str(content)
        return content

    # Handle list results by formatting
    if isinstance(result, list):
        logger.debug(f"{name} converting list to string")
        if len(result) == 0:
            return f"The {name} tool returned an empty list."

        if all(isinstance(item, dict) for item in result):
            formatted_result = ""
            for i, item in enumerate(result, 1):
                formatted_result += f"Result {i}:\n"
                for k, v in item.items():
                    formatted_result += f"  {k}: {v}\n"
                formatted_result += "\n"
            return formatted_result
        else:
            formatted_result = "Results:\n\n"
            for i, item in enumerate(result, 1):
                formatted_result += f"{i}. {str(item)}\n"
            return formatted_result

    # Handle dictionary results
    if isinstance(result, dict):
        logger.debug(f"{name} converting dictionary to string")
        try:
            return json.dumps(result, indent=2)
        except (TypeError, OverflowError):
            formatted_result = "Result:\n\n"
            for k, v in result.items():
                formatted_result += f"{k}: {v}\n"
            return formatted_result

    # For all other types, convert to string
    logger.debug(f"{name} converting {type(result)} to string")
    return str(result)


def execute_autogen_tool(
    name: str,
    coroutine_fn: Callable[..., Awaitable[Any]],
    required_fields: List[str],
    loop: asyncio.AbstractEventLoop,
    **kwargs: Any
) -> Any:
    """Execute AutoGen tool with NAT function integration.

    Args:
        name: Tool name
        coroutine_fn: Async function to invoke
        required_fields: Required parameter fields
        loop: Event loop for async execution
        **kwargs: Tool arguments

    Returns:
        Tool execution result as string
    """
    try:
        logger.debug(f"Running AutoGen tool {name} with kwargs: {kwargs}")

        # Initialize tracking for this tool
        if name not in _tool_call_counters:
            _tool_call_counters[name] = 0
        if name not in _tool_initialization_done:
            _tool_initialization_done[name] = False

        # Filter reserved keywords and metadata
        reserved_keywords = {'type', '_type', 'model_config', 'model_fields', 'model_dump', 'model_dump_json'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in reserved_keywords}

        # Check for metadata-only calls (potential loop indicator)
        only_metadata = len(filtered_kwargs) == 0 and len(kwargs) > 0

        # Log filtered keywords
        filtered_keys = set(kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_keys:
            logger.debug(f"Filtered reserved keywords from kwargs: {filtered_keys}")

        # Handle empty query for search-like tools
        is_search_tool = any(field in ['query', 'search_term', 'search_query'] for field in required_fields)
        has_empty_query = is_search_tool and any(
            field in filtered_kwargs and (not filtered_kwargs[field] or str(filtered_kwargs[field]).strip() == "")
            for field in ['query', 'search_term', 'search_query']
        )

        # Special handling for search tools with empty queries
        if is_search_tool and (only_metadata or has_empty_query):
            if not _tool_initialization_done[name]:
                logger.info(f"First-time initialization call for AutoGen tool {name}")
                _tool_initialization_done[name] = True
            else:
                logger.error(f"AutoGen tool {name} called with empty query after initialization")
                return f"ERROR: Tool {name} requires a valid search query. Provide specific search terms to continue."

        # Safeguard against infinite loops
        if only_metadata:
            _tool_call_counters[name] += 1
            logger.warning(f"Tool {name} called with only metadata fields (call {_tool_call_counters[name]}/{_MAX_EMPTY_CALLS})")

            if _tool_call_counters[name] >= _MAX_EMPTY_CALLS:
                logger.error(f"Detected potential infinite loop for AutoGen tool {name}")
                _tool_call_counters[name] = 0
                return f"ERROR: Tool {name} appears to be in a loop. Provide valid parameters when calling this tool."
        else:
            _tool_call_counters[name] = 0

        # Handle kwargs wrapper pattern
        if len(filtered_kwargs) == 1 and 'kwargs' in filtered_kwargs and isinstance(filtered_kwargs['kwargs'], dict):
            logger.debug("Detected wrapped kwargs, unwrapping")
            unwrapped_kwargs = filtered_kwargs['kwargs']
            unwrapped_kwargs = {k: v for k, v in unwrapped_kwargs.items() if k not in reserved_keywords}

            # Try to recover missing required fields
            for field in required_fields:
                if field not in unwrapped_kwargs:
                    logger.warning(f"Missing required field '{field}' in unwrapped kwargs")
                    if field in ['query', 'search_term', 'search_query'] and unwrapped_kwargs:
                        query_parts = [f"{k}: {v}" for k, v in unwrapped_kwargs.items()]
                        unwrapped_kwargs[field] = " ".join(query_parts)
                        logger.info(f"Built fallback query: {unwrapped_kwargs[field]}")

            filtered_kwargs = unwrapped_kwargs

        # Validate required fields
        is_initialization = len(filtered_kwargs) == 0
        missing_fields = [field for field in required_fields if field not in filtered_kwargs]

        # Be lenient for search tool initialization
        if not is_initialization and missing_fields and is_search_tool:
            if any(field in missing_fields for field in ['query', 'search_term', 'search_query']):
                logger.info(f"AutoGen tool {name} called without search parameters, treating as initialization")
                is_initialization = True

        # Enforce required fields for non-initialization calls
        if not is_initialization and missing_fields:
            if any(field in missing_fields for field in ['query', 'search_term', 'search_query']):
                raise ValueError(f"Missing required search parameter. Tool {name} requires a search query.")
            else:
                missing_fields_str = ", ".join([f"'{f}'" for f in missing_fields])
                raise ValueError(f"Missing required parameters: {missing_fields_str} for {name}.")

        logger.debug(f"Invoking AutoGen tool function with parameters: {filtered_kwargs}")

        # Execute the async function
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine_fn(**filtered_kwargs), loop)
            result = future.result(timeout=120)  # 2-minute timeout
        except TypeError as e:
            if "missing 1 required positional argument" in str(e):
                logger.debug(f"Retrying with positional argument style for {name}")
                future = asyncio.run_coroutine_threadsafe(coroutine_fn(filtered_kwargs), loop)
                result = future.result(timeout=120)
            else:
                raise

        # Process result for AutoGen format
        process_future = asyncio.run_coroutine_threadsafe(process_autogen_result(result, name), loop)
        return process_future.result(timeout=30)

    except Exception as e:
        logger.error(f"Error executing AutoGen tool {name}: {e}")
        raise


def _convert_pydantic_to_autogen_schema(pydantic_model: BaseModel) -> Dict[str, Any]:
    """Convert Pydantic model schema to AutoGen tool schema format.

    Args:
        pydantic_model: Pydantic model with schema

    Returns:
        AutoGen-compatible schema dictionary
    """
    try:
        schema = pydantic_model.model_json_schema()

        # Convert to AutoGen function tool format
        autogen_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
            "additionalProperties": False
        }

        return autogen_schema

    except Exception as e:
        logger.error(f"Error converting Pydantic schema to AutoGen format: {e}")
        return {"type": "object", "properties": {}, "required": []}


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.AUTOGEN)
def autogen_tool_wrapper(name: str, fn: Function, builder: Builder) -> FunctionTool:
    """Wrap NAT Function as AutoGen FunctionTool.

    This wrapper converts NAT functions to AutoGen's FunctionTool format,
    handling schema conversion, async execution, and result processing.

    Args:
        name: Tool name
        fn: NAT Function to wrap
        builder: Builder instance

    Returns:
        AutoGen FunctionTool instance
    """
    # Ensure input schema is present
    assert fn.input_schema is not None, f"Tool {name} must have input schema"

    # Get event loop for async execution
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Get async function to invoke
    coroutine_fn = fn.acall_invoke

    # Extract tool metadata
    description = fn.description or f"AutoGen tool wrapper for {name}"
    if description:
        description = textwrap.dedent(description).strip()

    # Extract required fields from schema
    required_fields = []
    parameters_schema = {}

    if fn.input_schema is not None:
        try:
            schema_json = fn.input_schema.model_json_schema()
            required_fields = schema_json.get("required", [])
            parameters_schema = _convert_pydantic_to_autogen_schema(fn.input_schema)

            # Add schema description to tool description
            schema_desc = schema_json.get("description")
            if schema_desc and schema_desc not in description:
                description = f"{description}\n\nParameters: {schema_desc}"

        except Exception as e:
            logger.warning(f"Error extracting schema from {name}: {e}")

    # Create wrapper function for AutoGen tool
    def autogen_tool_function(**kwargs: Any) -> str:
        """AutoGen tool function wrapper."""
        return execute_autogen_tool(name, coroutine_fn, required_fields, loop, **kwargs)

    # Set function metadata
    autogen_tool_function.__name__ = name
    autogen_tool_function.__doc__ = description

    # Create AutoGen FunctionTool
    try:
        function_tool = FunctionTool(
            func=autogen_tool_function,
            name=name,
            description=description,
            parameters_json_schema=parameters_schema
        )

        logger.debug(f"Created AutoGen FunctionTool for {name}")
        return function_tool

    except Exception as e:
        logger.error(f"Error creating AutoGen FunctionTool for {name}: {e}")

        # Fallback with minimal schema
        minimal_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        return FunctionTool(
            func=autogen_tool_function,
            name=name,
            description=description,
            parameters_json_schema=minimal_schema
        )