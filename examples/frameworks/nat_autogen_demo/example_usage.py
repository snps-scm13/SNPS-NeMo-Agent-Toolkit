#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of the AutoGen Research Workflow with NAT.

This script demonstrates how to use the AutoGen multi-agent research workflow
to conduct collaborative research on a given topic.

Usage:
    python example_usage.py --topic "machine learning trends"
"""

import asyncio
import logging
from pathlib import Path

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.component_ref import LLMRef
from nat_autogen_demo.register import AutoGenResearchWorkflowConfig


async def run_research_workflow(topic: str, llm_name: str = "openai_gpt35", verbose: bool = False):
    """
    Run the AutoGen research workflow for a given topic.

    Args:
        topic: The research topic to investigate
        llm_name: Name of the LLM to use (should be configured in NAT)
        verbose: Enable verbose logging
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        async with WorkflowBuilder() as builder:
            # Create configuration
            config = AutoGenResearchWorkflowConfig(
                llm_name=llm_name,
                research_agent_name="ResearchAgent",
                research_agent_instructions=(
                    "You are a research agent specialized in gathering comprehensive information. "
                    "Focus on finding factual, up-to-date information about the given topic. "
                    "Provide sources and context where possible."
                ),
                analysis_agent_name="AnalysisAgent",
                analysis_agent_instructions=(
                    "You are an analysis agent. Examine the research provided and identify "
                    "key patterns, trends, implications, and insights. Structure your analysis clearly."
                ),
                writer_agent_name="WriterAgent",
                writer_agent_instructions=(
                    "You are a writer agent. Create a comprehensive, well-structured report "
                    "based on the research and analysis. Use clear headings and organize information logically."
                ),
                max_turns=12,
                verbose=verbose
            )

            print(f"🔍 Starting AutoGen research workflow for topic: '{topic}'")
            print(f"🤖 Using LLM: {llm_name}")
            print("=" * 60)

            # Add the workflow function to the builder
            workflow_fn = await builder.add_function("autogen_research", config)

            # Execute the research
            result = await workflow_fn(topic)

            print("\n📋 Research Results:")
            print("=" * 60)
            print(result)
            print("=" * 60)

            return result

    except Exception as e:
        print(f"❌ Error running research workflow: {e}")
        logging.error(f"Workflow error: {e}", exc_info=True)
        return None


async def main():
    """Main function to run example workflows."""
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoGen Research Workflow")
    parser.add_argument("--topic", default="artificial intelligence trends in 2024",
                        help="Research topic to investigate")
    parser.add_argument("--llm", default="openai_gpt35",
                        help="LLM model to use (must be configured in NAT)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    # Run the research workflow
    result = await run_research_workflow(
        topic=args.topic,
        llm_name=args.llm,
        verbose=args.verbose
    )

    if result:
        print("✅ Research workflow completed successfully!")
    else:
        print("❌ Research workflow failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())