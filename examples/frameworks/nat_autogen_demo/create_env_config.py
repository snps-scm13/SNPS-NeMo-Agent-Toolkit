#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper script to create a NAT configuration file that matches your environment's
LLM setup, allowing you to test AutoGen integration without modifying core code.
"""

import os
import yaml
from pathlib import Path


def create_nat_config():
    """Create NAT configuration file for your environment."""

    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key or not base_url:
        print("❌ Required environment variables not set:")
        print("   - OPENAI_API_KEY")
        print("   - OPENAI_BASE_URL")
        return False

    # Create NAT config structure
    config = {
        "llms": {
            "azure-openai/snps-openai-gpt4o": {
                "type": "openai",
                "config": {
                    "api_key": api_key,
                    "base_url": base_url,
                    "model": "azure-openai/snps-openai-gpt4o",
                    "extra_headers": {
                        "X-TFY-METADATA": '{"your_custom_key":"your_custom_value"}',
                        "X-TFY-LOGGING-CONFIG": '{"enabled": true}'
                    }
                }
            }
        }
    }

    # Save to config file
    config_path = Path("nat_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ Created NAT configuration: {config_path.absolute()}")
    print(f"   - LLM name: azure-openai/snps-openai-gpt4o")
    print(f"   - API Key: {'***' + api_key[-4:]}")
    print(f"   - Base URL: {base_url}")

    # Create usage instructions
    instructions = f"""
# Usage Instructions

1. Set environment variable to use this config:
   export NAT_CONFIG_FILE={config_path.absolute()}

2. Test AutoGen integration:
   python examples/frameworks/nat_autogen_demo/example_usage.py \\
     --topic "artificial intelligence" \\
     --llm "azure-openai/snps-openai-gpt4o"

3. Or use the validation script:
   python examples/frameworks/nat_autogen_demo/validate_with_env_llm.py
"""

    with open("USAGE_INSTRUCTIONS.txt", "w") as f:
        f.write(instructions)

    print(f"✅ Created usage instructions: USAGE_INSTRUCTIONS.txt")
    return True


def test_config_creation():
    """Test that the config can be created."""
    print("🔧 Testing NAT Config Creation for Your Environment")
    print("=" * 50)

    success = create_nat_config()

    if success:
        print("\n🎉 Configuration created successfully!")
        print("Follow the instructions in USAGE_INSTRUCTIONS.txt")
    else:
        print("\n❌ Configuration creation failed!")
        print("Make sure your environment variables are set correctly.")

    return success


if __name__ == "__main__":
    test_config_creation()