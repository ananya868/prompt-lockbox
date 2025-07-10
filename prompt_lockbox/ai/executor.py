#
# FILE: prompt_lockbox/ai/executor.py (NEW FILE)
#
"""
This module provides the core, centralized function for executing prompts against
a language model. It is designed to handle both general (unstructured text)
and structured (JSON) output, making it a versatile engine for various AI tasks.
"""

from mirascope import llm
from pydantic import BaseModel, create_model, Field
from typing import Dict, Any, Union, Optional, Type, List

from .logging import setup_ai_logger
from pathlib import Path
import traceback

# --- Type Mapping for Dynamic Pydantic Model Creation ---

# A mapping from JSON Schema types to Python types for Pydantic.
# This allows us to dynamically create a Pydantic model from a schema dict.
TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": List,
    "object": Dict,
}

def _create_pydantic_model_from_schema(
    schema: Dict[str, Any], model_name: str = "DynamicResponseModel"
) -> Type[BaseModel]:
    """Dynamically creates a Pydantic model from a JSON schema dictionary.

    This powerful helper function allows users to define an expected output
    structure in their prompt's YAML file, which is then used to generate a
    Pydantic model at runtime for structured LLM calls.

    Args:
        schema: A dictionary representing the JSON schema for the output.
                It must be a JSON Schema object with 'type': 'object'.
        model_name: The internal name for the created Pydantic model class.

    Returns:
        A Pydantic BaseModel class dynamically created from the schema.

    Raises:
        ValueError: If the schema is not a valid object type or is malformed.
    """
    if schema.get("type") != "object" or "properties" not in schema:
        raise ValueError("Schema must be of type 'object' with a 'properties' key.")

    fields = {}
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    for field_name, prop in properties.items():
        field_type_str = prop.get("type")
        if field_type_str not in TYPE_MAP:
            raise ValueError(f"Unsupported schema type for field '{field_name}': {field_type_str}")

        python_type = TYPE_MAP[field_type_str]
        description = prop.get("description", "")

        # Determine if the field is required. If not, it's optional.
        if field_name in required_fields:
            # For required fields, we use Pydantic's ellipsis (...)
            fields[field_name] = (python_type, Field(..., description=description))
        else:
            # For optional fields, we use a default value of None
            fields[field_name] = (Optional[python_type], Field(None, description=description))

    # Use Pydantic's create_model to build the class on the fly.
    return create_model(model_name, **fields)


def _get_dynamic_caller(
    ai_config: Dict[str, str],
    response_model: Optional[Type[BaseModel]] = None
):
    """A single, unified factory for creating a Mirascope-decorated function.

    This factory can generate a caller for either structured or general calls
    based on whether a `response_model` is provided.

    Args:
        ai_config: A dictionary with 'provider' and 'model' keys.
        response_model: An optional Pydantic BaseModel class. If provided,
                        the call will be structured. If None, it will be general.

    Returns:
        A callable function decorated with `mirascope.llm.call`.
    """
    provider = ai_config.get("provider", "openai")
    model = ai_config.get("model", "gpt-4o-mini")

    # The decorator is configured dynamically based on the arguments.
    if provider == "huggingface":
        # LiteLLM expects the model string to be prefixed with "huggingface/"
        # to know which provider to use for the model repo.
        litellm_model_string = f"huggingface/{model}"
        
        @llm.call(
            provider="litellm", # Tell mirascope to use the litellm provider
            model=litellm_model_string,
            response_model=response_model
        )
        def dynamic_executor(prompt_content: str):
            return prompt_content
            
        return dynamic_executor

    @llm.call(
        provider=provider,
        model=model,
        response_model=response_model  # This will be None for general calls
    )
    def dynamic_executor(prompt_content: str):
        """The internal function that wraps the user's prompt."""
        # For structured calls, Mirascope automatically adds instructions
        # for the LLM to respond in the required JSON format.
        return prompt_content

    return dynamic_executor


def execute_prompt(
    rendered_prompt: str,
    ai_config: Dict[str, str],
    project_root: Path,
    output_schema: Optional[Dict[str, Any]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Executes a rendered prompt against an LLM, handling both general and structured output.

    This is the main public function for this module. It determines whether to
    make a general call (returning raw text) or a structured call (returning a
    parsed dictionary) based on the presence of an `output_schema`.

    Args:
        rendered_prompt: The final prompt string to be sent to the LLM.
        ai_config: A dictionary specifying the 'provider' and 'model'.
        project_root: The root path of the project, used for logging.
        output_schema: An optional dictionary representing a JSON schema. If provided,
                       the function will attempt a structured call and return a dict.

    Returns:
        - A string with the LLM's response if `output_schema` is None.
        - A dictionary with the parsed data if `output_schema` is provided.

    Raises:
        ValueError: If the `output_schema` is invalid or the LLM call fails.
    """
    response_model = None
    if output_schema:
        try:
            # If a schema is provided, create a dynamic Pydantic model for it.
            response_model = _create_pydantic_model_from_schema(output_schema)
        except Exception as e:
            raise ValueError(f"Failed to process output schema: {e}")

    try:
        # Get the configured caller from our unified factory.
        caller_fn = _get_dynamic_caller(ai_config, response_model)
        # Execute the call.
        response: llm.CallResponse = caller_fn(rendered_prompt)

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Failed to execute LLM call: {e}")

    # --- Logging ---
    logger = setup_ai_logger(project_root)
    if response._response.usage:
        log_data = {
            "model": response._response.model,
            "input_tokens": response._response.input_tokens,
            "output_tokens": response._response.output_tokens,
            "total_tokens": response._response.usage.total_tokens,
        }
        logger.info("execute_prompt", extra=log_data)

    # --- Return Value ---
    if response_model:
        # For structured calls, return the parsed data as a dictionary.
        return response.model_dump()
    else:
        # For general calls, return the raw text content.
        return response.content