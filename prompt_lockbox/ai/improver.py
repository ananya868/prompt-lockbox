#
# FILE: prompt_lockbox/ai/improver.py (Mirascope Version)
#

from mirascope import llm
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from .logging import setup_ai_logger
from pathlib import Path

# The Pydantic model is unchanged
class PromptCritique(BaseModel):
    """A Pydantic model defining the structured output for an AI-generated prompt critique.
    
    This schema ensures the language model returns a detailed critique, actionable
    suggestions, and an improved template in a validated format.
    """
    critique: str = Field(...)
    suggestions: List[str] = Field(...)
    improved_template: str = Field(...)

# The function factory is updated to accept the new context
def _get_dynamic_improver(ai_config: dict):
    """A factory to dynamically create a Mirascope-decorated function for prompt improvement.

    This function allows the AI provider and model to be configured at runtime
    based on the project's configuration. It generates a callable that accepts
    the prompt and its surrounding context for a more informed analysis.

    Args:
        ai_config: A dictionary containing 'provider' and 'model' keys to
                   configure the LLM call.

    Returns:
        A callable function decorated with `mirascope.llm.call` and configured
        for the prompt improvement task.
    """
    provider = ai_config.get("provider", "openai")
    model = ai_config.get("model", "gpt-4o-mini")

    @llm.call(provider=provider, model=model, response_model=PromptCritique)
    def _dynamic_improve_prompt(
        prompt_template: str,
        user_note: str,
        description: Optional[str] = None,
        inputs: Optional[Dict[str, str]] = None,
    ):
        """The internal function that crafts the detailed prompt for the LLM.

        Args:
            prompt_template: The user's original prompt template.
            user_note: A specific instruction from the user on how to improve it.
            description: The prompt's intended purpose.
            inputs: Example variables and their values for the prompt.
        """
        # Dynamically build a context block to provide the LLM with all available info.
        context_parts = []
        if description:
            context_parts.append(f"# PROMPT'S INTENDED PURPOSE (DESCRIPTION):\n{description}")
        if inputs:
            # Format the inputs nicely for the LLM
            input_examples = "\n".join([f"- {key}: (e.g., '{value}')" for key, value in inputs.items()])
            context_parts.append(f"# PROMPT'S INPUT VARIABLES AND EXAMPLES:\n{input_examples}")
        
        context_str = "\n\n".join(context_parts)
        
        # This is the final prompt sent to the LLM, combining the static instructions
        # with the dynamic context and user-provided template.
        return f"""
        You are a world-class prompt engineering expert. Your task is to analyze a
        user-provided prompt template and improve it based on established principles
        like clarity, specificity, persona setting, and providing examples.
        You must also consider all available context: the prompt's description,
        its input variables, and the user's specific note for improvement.
        Your response must be a valid JSON object.
        # AVAILABLE CONTEXT:

        {context_str if context_str else "No additional context was provided."}

        # USER'S PROMPT TEMPLATE TO IMPROVE:
        {prompt_template}

        # USER'S NOTE FOR IMPROVEMENT:
        {user_note}
        """

    return _dynamic_improve_prompt

# The main public function is updated to pass the new context
def get_critique(
    prompt_template: str,
    note: str,
    project_root: Path,
    ai_config: dict,
    description: Optional[str] = None,
    default_inputs: Optional[Dict[str, str]] = None,
) -> dict:
    """Gets a critique and improvement for a prompt, and logs the interaction.

    This is the main public function for this module. It orchestrates the AI call,
    passing all available context (description, inputs) to the LLM for a
    high-quality analysis. It handles errors and logs usage.

    Args:
        prompt_template: The string content of the prompt template to improve.
        note: A specific instruction from the user on the desired improvements.
        project_root: The root path of the PromptLockbox project, used for logging.
        ai_config: A dictionary specifying the 'provider' and 'model' for the AI call.
        description: An optional description of the prompt's purpose.
        default_inputs: An optional dictionary of default input variables and examples.

    Returns:
        A dictionary containing the 'critique', 'suggestions', and 'improved_template'.

    Raises:
        ConnectionError: If the call to the language model provider fails for any reason.
    """
    try:
        # Get the dynamically configured function from our factory.
        improver_fn = _get_dynamic_improver(ai_config)
        
        # Call the Mirascope-decorated function, passing all context.
        call_response: llm.CallResponse = improver_fn(
            prompt_template,
            user_note=note,
            description=description,
            inputs=default_inputs,
        )
    except Exception as e:
        raise ConnectionError(f"Failed to call LLM for prompt improvement: {e}")

    # Set up and use the logger to record AI usage statistics for this call.
    logger = setup_ai_logger(project_root)
    if call_response._response.usage:
        log_data = {
            "model": call_response._response.model,
            "input_tokens": call_response._response.input_tokens,
            "output_tokens": call_response._response.output_tokens,
            "total_tokens": call_response._response.usage.total_tokens,
        }
        logger.info("get_prompt_critique", extra=log_data)
        
    # Mirascope automatically parses the response into our Pydantic model.
    # .model_dump() converts it to a standard dictionary for return.
    return call_response.model_dump()