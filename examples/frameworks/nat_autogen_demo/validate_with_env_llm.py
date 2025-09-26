#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validation script that registers your environment's LLM with NAT's builder system,
then tests the AutoGen integration. This allows testing without modifying the
core integration implementation.
"""

import asyncio
import os
import logging
from openai import OpenAI

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.component_ref import LLMRef
from nat.llm.openai_llm import OpenAIModelConfig
from nat_autogen_demo.register import AutoGenResearchWorkflowConfig

# Force load AutoGen plugin components
import nat.plugins.autogen.register


async def setup_environment_llm(builder: WorkflowBuilder):
    """
    Register your environment's LLM configuration with NAT's builder.
    This mimics how your environment makes LLM calls.
    """

    # Your environment's LLM configuration
    llm_config = OpenAIModelConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model_name="azure-openai/snps-openai-gpt4o"
    )

    # Register this LLM with the builder
    await builder.add_llm("azure-openai/snps-openai-gpt4o", llm_config)

    print(f"✅ Registered LLM: azure-openai/snps-openai-gpt4o")
    print(f"   - API Key: {'***' + os.getenv('OPENAI_API_KEY', '')[-4:] if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"   - Base URL: {os.getenv('OPENAI_BASE_URL', 'Not set')}")


async def test_direct_llm_call():
    """Test direct LLM call to verify your environment setup works."""
    print("\n🔍 Testing direct LLM call (your environment setup)...")

    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI bot."},
                {"role": "user", "content": "Say 'Hello from environment test'"},
            ],
            model="azure-openai/snps-openai-gpt4o",
            stream=False,
            extra_headers={
                "X-TFY-METADATA": '{"your_custom_key":"your_custom_value"}',
                "X-TFY-LOGGING-CONFIG": '{"enabled": true}',
            },
        )

        result = response.choices[0].message.content
        print(f"✅ Direct LLM call successful: {result}")
        return True

    except Exception as e:
        print(f"❌ Direct LLM call failed: {e}")
        return False


async def test_autogen_integration():
    """Test AutoGen integration with your environment's LLM."""
    print("\n🤖 Testing AutoGen integration with environment LLM...")

    try:
        async with WorkflowBuilder() as builder:
            # Register your environment's LLM
            await setup_environment_llm(builder)

            # Create AutoGen workflow configuration
            config = AutoGenResearchWorkflowConfig(
                llm_name="azure-openai/snps-openai-gpt4o",
                research_agent_name="EnvResearcher",
                analysis_agent_name="EnvAnalyst",
                writer_agent_name="EnvWriter",
                max_turns=3,
                verbose=True
            )

            print(f"\n📋 AutoGen Configuration:")
            print(f"   - LLM: {config.llm_name}")
            print(f"   - Agents: {config.research_agent_name}, {config.analysis_agent_name}, {config.writer_agent_name}")
            print(f"   - Max turns: {config.max_turns}")

            # Add the AutoGen workflow
            workflow_fn = await builder.add_function("autogen_research", config)
            print(f"✅ AutoGen workflow registered successfully")

            # Test with a simple topic
            print(f"\n🚀 Testing AutoGen workflow execution...")
            result = await workflow_fn.acall_invoke("artificial intelligence")

            print(f"\n📄 AutoGen Result:")
            print(f"   {result}")

            return True

    except Exception as e:
        print(f"❌ AutoGen integration test failed: {e}")
        logging.error(f"AutoGen test error: {e}", exc_info=True)
        return False


async def main():
    """Main validation function."""
    print("🧪 AutoGen Integration Environment Validation")
    print("=" * 60)

    # Check environment variables
    print("🔧 Environment Check:")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False

    if not base_url:
        print("❌ OPENAI_BASE_URL not set")
        return False

    print(f"✅ OPENAI_API_KEY: {'***' + api_key[-4:]}")
    print(f"✅ OPENAI_BASE_URL: {base_url}")

    # Test direct LLM call first
    direct_success = await test_direct_llm_call()
    if not direct_success:
        print("\n❌ Direct LLM call failed. Fix environment setup first.")
        return False

    # Test AutoGen integration
    autogen_success = await test_autogen_integration()

    print(f"\n{'🎉' if autogen_success else '❌'} Validation {'PASSED' if autogen_success else 'FAILED'}")
    print("=" * 60)

    if autogen_success:
        print("✅ AutoGen integration works with your environment!")
        print("✅ You can now use AutoGen workflows in your NAT setup")
    else:
        print("❌ AutoGen integration needs debugging")

    return autogen_success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = asyncio.run(main())
    exit(0 if success else 1)