# NAT AutoGen Demo - Multi-Agent Research Workflow

This example demonstrates how to use Microsoft AutoGen framework with NVIDIA NeMo-Agent-Toolkit (NAT) to create collaborative multi-agent workflows.

## Overview

The AutoGen Research Workflow showcases three specialized agents working together:

1. **Research Agent** - Gathers comprehensive information about a topic
2. **Analysis Agent** - Analyzes research findings and identifies insights
3. **Writer Agent** - Synthesizes everything into a structured report

## Features

- **Multi-Agent Collaboration**: Agents communicate through AutoGen's conversation system
- **Configurable Workflow**: Customize agent instructions, turn limits, and behavior
- **NAT Integration**: Seamless integration with NAT's LLM management and builder system
- **Async Support**: Fully asynchronous execution for better performance

## Installation

1. Install the AutoGen demo package:
```bash
pip install -e .
```

2. Ensure you have the AutoGen integration installed:
```bash
pip install nvidia-nat[autogen]
```

## Usage

### Basic Usage

```python
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.component_ref import LLMRef
from nat_autogen_demo.register import AutoGenResearchWorkflowConfig

async def run_workflow():
    async with WorkflowBuilder() as builder:
        # Create configuration
        config = AutoGenResearchWorkflowConfig(
            llm_name="openai_gpt35",
            max_turns=10
        )

        # Add workflow function to builder
        workflow_fn = await builder.add_function("autogen_research", config)

        # Execute research
        result = await workflow_fn("machine learning trends in 2024")
        print(result)

# Run the workflow
import asyncio
asyncio.run(run_workflow())
```

### Command Line Usage

```bash
# Basic research
python example_usage.py --topic "artificial intelligence trends"

# With custom LLM and verbose logging
python example_usage.py --topic "quantum computing" --llm "nvidia_nim" --verbose
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_name` | LLMRef | Required | LLM model to use for all agents |
| `research_topic` | str | Required | Topic to research and analyze |
| `research_agent_name` | str | "ResearchAgent" | Name of the research agent |
| `research_agent_instructions` | str | Default instructions | Custom instructions for research agent |
| `analysis_agent_name` | str | "AnalysisAgent" | Name of the analysis agent |
| `analysis_agent_instructions` | str | Default instructions | Custom instructions for analysis agent |
| `writer_agent_name` | str | "WriterAgent" | Name of the writer agent |
| `writer_agent_instructions` | str | Default instructions | Custom instructions for writer agent |
| `max_turns` | int | 10 | Maximum conversation turns |
| `verbose` | bool | False | Enable verbose logging |

## Example Output

The workflow produces structured research reports like:

```
# Research Report: Artificial Intelligence Trends

## Research Summary
[Research agent findings...]

## Key Analysis
[Analysis agent insights...]

## Conclusions
[Writer agent synthesis...]
```

## Customization

### Custom Agent Instructions

```python
config = AutoGenResearchWorkflowConfig(
    llm_name=LLMRef(name="openai_gpt4"),
    research_topic="blockchain technology",
    research_agent_instructions="""
    You are a blockchain research specialist. Focus on:
    - Technical developments and innovations
    - Market adoption and use cases
    - Regulatory landscape changes
    Provide detailed technical analysis with sources.
    """,
    analysis_agent_instructions="""
    You are a blockchain analyst. Identify:
    - Investment opportunities and risks
    - Technology maturity indicators
    - Competitive landscape analysis
    Structure your analysis with clear recommendations.
    """,
    max_turns=15
)
```

### Integration with NAT Workflows

This AutoGen workflow can be integrated into larger NAT workflows and chained with other framework components.

## Requirements

- Python 3.11+
- NVIDIA NAT with AutoGen support
- AutoGen 0.7.0+
- Configured LLM clients in NAT

## Troubleshooting

1. **Import Errors**: Ensure both NAT and AutoGen packages are properly installed
2. **LLM Connection Issues**: Verify your LLM configuration in NAT
3. **Agent Communication Problems**: Check the `max_turns` setting and agent instructions

## Related Examples

- [Semantic Kernel Demo](../semantic_kernel_demo/) - Semantic Kernel integration
- [LangChain Examples](../langchain_demo/) - LangChain workflows
- [CrewAI Examples](../crewai_demo/) - CrewAI team coordination