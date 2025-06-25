#
# FILE: prompt_lockbox/ai/documenter.py (Mirascope Version)
#

from mirascope import llm
from pydantic import BaseModel, Field
from typing import List

# Import our logger setup
from .logging import setup_ai_logger
from pathlib import Path
from openai import OpenAI, OpenAIError


class PromptDocumentation(BaseModel):
    """A Pydantic model defining the structured output for AI-generated documentation.

    This schema ensures that the language model returns data in a predictable
    and validated format.
    """
    description: str = Field(
        ..., description="A concise, one-sentence description of the prompt's purpose."
    )
    tags: List[str] = Field(
        ..., description="A list of 3-5 relevant, lowercase search tags."
    )


def _get_dynamic_documenter(ai_config: dict):
    """A factory to dynamically create a Mirascope-decorated function.

    This function allows the AI provider and model to be configured at runtime
    based on the project's configuration, rather than being hardcoded in the
    decorator.

    Args:
        ai_config: A dictionary containing 'provider' and 'model' keys to
                   configure the LLM call.

    Returns:
        A callable function decorated with `mirascope.llm.call` and configured
        with the specified provider, model, and response model.
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
        """The internal function that crafts the prompt for the LLM."""
        # This prompt instructs the AI on its role and task.
        # The user's prompt template is injected for analysis.
        prompt = f"""
            You are a documentation expert for a prompt engineering framework.
            Your task is to analyze a user-provided prompt template and generate a concise,
            one-sentence description and a list of relevant lowercase search tags.

            # Prompt Template: {prompt_template}
        """
        return prompt
    
    return generate

def get_documentation(prompt_template: str, project_root: Path, ai_config: dict) -> dict:
    """Generates structured documentation for a given prompt template using an LLM.

    This is the main public function of the module. It orchestrates the AI call,
    handles potential errors, logs the interaction, and returns the structured
    output.

    Args:
        prompt_template: The string content of the prompt template to be documented.
        project_root: The root path of the PromptLockbox project, used for logging.
        ai_config: A dictionary specifying the 'provider' and 'model' for the AI call.

    Returns:
        A dictionary containing the 'description' and 'tags' for the prompt.

    Raises:
        ConnectionError: If there is an issue with the OpenAI API key, a general
                         API error, or other network-related problems.
        ImportError: If the configured AI provider's library is not installed.
    """
    try:
        # Get the dynamically configured function from our factory.
        documenter_fn = _get_dynamic_documenter(ai_config)
        
        # Call the Mirascope-decorated function to interact with the LLM.
        response: llm.CallResponse = documenter_fn(prompt_template)
        
    except OpenAIError as e:
        # Specifically catch errors from the OpenAI library for better feedback.
        if "api_key" in str(e).lower():
            # This is almost certainly an API key issue.
            raise ConnectionError(
                "OpenAI API key is missing or invalid. "
                "Please set the OPENAI_API_KEY environment variable."
            )
        else:
            # It's a different OpenAI error (e.g., rate limit, server error)
            raise ConnectionError(f"An error occurred with the OpenAI API: {e}")
    
    except ImportError as e:
        # Catch errors if the configured provider library (e.g., 'anthropic') is missing.
        raise ImportError(
            f"The '{provider}' provider is configured but its library may be missing. "
            f"Please install it (e.g., `pip install openai anthropic`). Original error: {e}"
        )
            
    except Exception as e:
        # Catch-all for other unexpected errors (e.g., network issues, provider downtime).
        import traceback
        traceback.print_exc()
        raise ConnectionError(f"Failed to call LLM provider '{provider}': {e}")

    # Set up and use the logger to record AI usage statistics.
    logger = setup_ai_logger(project_root)
    if response._response.usage:
        log_data = {
            "model": response._response.model,
            "input_tokens": response._response.input_tokens,
            "output_tokens": response._response.output_tokens,
            "total_tokens": response._response.usage.total_tokens,
        }
        logger.info("generate_documentation", extra=log_data)
    
    # Mirascope automatically parses the LLM's response into the Pydantic model.
    # .model_dump() converts it to a standard dictionary for external use.
    return response.model_dump()