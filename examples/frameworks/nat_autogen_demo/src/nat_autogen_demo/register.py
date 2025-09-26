# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from typing import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AutoGenResearchWorkflowConfig(FunctionBaseConfig, name="autogen_research"):
    llm_name: LLMRef = Field(description="The LLM model to use with AutoGen agents.")
    research_agent_name: str = Field(default="ResearchAgent", description="Name of the research agent.")
    research_agent_instructions: str = Field(
        default="You are a research agent. Your role is to gather comprehensive information about the given topic. "
                "Provide detailed, factual information with sources when possible.",
        description="Instructions for the research agent."
    )
    analysis_agent_name: str = Field(default="AnalysisAgent", description="Name of the analysis agent.")
    analysis_agent_instructions: str = Field(
        default="You are an analysis agent. Your role is to analyze the research provided by the research agent. "
                "Identify key insights, trends, and important findings. Provide structured analysis.",
        description="Instructions for the analysis agent."
    )
    writer_agent_name: str = Field(default="WriterAgent", description="Name of the writer agent.")
    writer_agent_instructions: str = Field(
        default="You are a writer agent. Your role is to synthesize the research and analysis into a "
                "comprehensive, well-structured report. Write clearly and concisely.",
        description="Instructions for the writer agent."
    )
    max_turns: int = Field(default=10, description="Maximum number of conversation turns.")
    verbose: bool = Field(default=False, description="Enable verbose logging for AutoGen.")


@register_function(config_type=AutoGenResearchWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.AUTOGEN])
async def autogen_research_workflow(config: AutoGenResearchWorkflowConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """
    AutoGen multi-agent research workflow that demonstrates collaborative research and analysis.

    This workflow creates three agents:
    1. Research Agent - Gathers information about the topic
    2. Analysis Agent - Analyzes the research findings
    3. Writer Agent - Creates a comprehensive report

    The agents collaborate through AutoGen's conversation system to produce structured research output.
    """

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import TextMessage

    try:
        # Get the LLM client from the builder
        llm_client = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AUTOGEN)

        # Create research agent
        research_agent = AssistantAgent(
            name=config.research_agent_name,
            model_client=llm_client,
            system_message=config.research_agent_instructions
        )

        # Create analysis agent
        analysis_agent = AssistantAgent(
            name=config.analysis_agent_name,
            model_client=llm_client,
            system_message=config.analysis_agent_instructions
        )

        # Create writer agent
        writer_agent = AssistantAgent(
            name=config.writer_agent_name,
            model_client=llm_client,
            system_message=config.writer_agent_instructions
        )

        # Create team with round-robin conversation
        team = RoundRobinGroupChat(
            participants=[research_agent, analysis_agent, writer_agent],
            termination_condition=MaxMessageTermination(max_messages=config.max_turns)
        )

        async def _research_workflow(user_input: str) -> str:
            """Execute the research workflow with the given input."""
            try:
                # Start the research workflow
                initial_message = f"Please research and analyze the following topic: {user_input}"

                if config.verbose:
                    logger.info(f"Starting AutoGen research workflow for topic: {user_input}")

                # Run the team conversation
                result = await team.run(task=TextMessage(content=initial_message, source="user"))

                # Extract the final response
                if hasattr(result, 'messages') and result.messages:
                    # Get the last message from the writer agent
                    final_messages = [msg for msg in result.messages if msg.source == config.writer_agent_name]
                    if final_messages:
                        return final_messages[-1].content
                    else:
                        # Fallback to last message
                        return result.messages[-1].content

                return "Research workflow completed, but no output was generated."

            except Exception as e:
                logger.error(f"Error in AutoGen research workflow: {e}")
                return f"Error occurred during research workflow: {str(e)}"

        def convert_result_to_str(response: str) -> str:
            """Convert the response to string format."""
            return response

        # Yield the function info
        yield FunctionInfo.create(
            single_fn=_research_workflow,
            converters=[convert_result_to_str]
        )

    except GeneratorExit:
        logger.info("AutoGen research workflow exited early")
    except Exception as e:
        logger.error(f"Failed to initialize AutoGen research workflow: {e}")
        raise
    finally:
        logger.debug("AutoGen research workflow cleanup completed")