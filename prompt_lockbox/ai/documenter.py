#
# FILE: prompt_lockbox/ai/proompt_documenter.py (Mirascope Version)
#

from mirascope import llm
from pydantic import BaseModel, Field
from typing import List

# Import our logger setup
from .logging import setup_ai_logger
from pathlib import Path

# 1. Define the desired structured output using Pydantic
class PromptDocumentation(BaseModel):
    """A model for the generated documentation of a prompt."""
    description: str = Field(
        ..., description="A concise, one-sentence description of the prompt's purpose."
    )
    tags: List[str] = Field(
        ..., description="A list of 3-5 relevant, lowercase search tags."
    )


def _get_dynamic_documenter(ai_config: dict):
    """
    Dynamically creates and returns a mirascope-decorated function
    based on the provided AI configuration.
    """
    # Get provider and model from config, with sane defaults
    provider = ai_config.get("provider", "openai")
    model = ai_config.get("model", "gpt-4o-mini")

    @llm.call(
        provider=provider,
        model=model,
        response_model=PromptDocumentation
    )
    def generate(prompt_template: str) -> str:
        prompt = f"""
            You are a documentation expert for a prompt engineering framework.
            Your task is to analyze a user-provided prompt template and generate a concise,
            one-sentence description and a list of relevant lowercase search tags.

            # Prompt Template: {prompt_template}
        """
        return prompt
    
    return generate

def get_documentation(prompt_template: str, project_root: Path, ai_config: dict) -> dict:
    """
    The main function that gets documentation and logs the interaction.
    """
    # 3. Call the mirascope-decorated function
    try:
        # Step 1: Get the dynamically configured function from our factory.
        documenter_fn = _get_dynamic_documenter(ai_config)
        
        # Step 2: Call the function just like a normal decorated function.
        response: llm.CallResponse = documenter_fn(prompt_template)
    except Exception as e:
        # This could be an API key error, connection error, etc.
        raise ConnectionError(f"Failed to call LLM: {e}")

    # 4. Log the usage information
    logger = setup_ai_logger(project_root)
    if response._response.usage:
        log_data = {
            "model": response._response.model,
            "input_tokens": response._response.input_tokens,
            "output_tokens": response._response.output_tokens,
            "total_tokens": response._response.usage.total_tokens,
        }
        logger.info("generate_documentation", extra=log_data)
    
    # 5. Return the structured data
    # The response object *is* an instance of our Pydantic model
    return response.model_dump()