# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import AsyncMock, MagicMock

from nat_autogen_demo.register import AutoGenResearchWorkflowConfig, autogen_research_workflow


class TestAutoGenResearchWorkflow:
    """Test cases for AutoGen research workflow."""

    @pytest.fixture
    def mock_builder(self):
        """Mock builder for testing."""
        builder = MagicMock()
        builder.get_llm = AsyncMock()
        return builder

    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return AutoGenResearchWorkflowConfig(
            llm_name="test_llm",
            max_turns=5,
            verbose=False
        )

    @pytest.mark.asyncio
    async def test_workflow_config_creation(self, config):
        """Test that workflow configuration is created correctly."""
        assert config.llm_name == "test_llm"
        assert config.research_agent_name == "ResearchAgent"
        assert config.analysis_agent_name == "AnalysisAgent"
        assert config.writer_agent_name == "WriterAgent"
        assert config.max_turns == 5
        assert config.verbose is False

    @pytest.mark.asyncio
    async def test_workflow_function_creation(self, config, mock_builder):
        """Test that workflow function is created successfully."""
        # Mock the LLM client
        mock_llm_client = MagicMock()
        mock_builder.get_llm.return_value = mock_llm_client

        # Create the workflow generator
        workflow_gen = autogen_research_workflow(config, mock_builder)

        # Test that we can get a function info object
        function_info = await workflow_gen.__anext__()
        assert function_info is not None
        assert hasattr(function_info, 'fn')

    def test_config_validation(self):
        """Test configuration validation."""
        # Test required fields
        with pytest.raises(ValueError):
            AutoGenResearchWorkflowConfig()

        # Test valid configuration
        config = AutoGenResearchWorkflowConfig(
            llm_name="test_llm"
        )
        assert config.llm_name == "test_llm"