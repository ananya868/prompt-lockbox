from mirascope import llm
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from .logging import setup_ai_logger
from pathlib import Path

# The Pydantic model is unchanged
class PromptCritique(BaseModel):
    """A structured critique and improvement for a prompt template."""
    critique: str = Field(...)
    suggestions: List[str] = Field(...)
    improved_template: str = Field(...)

# The function factory is updated to accept the new context
def _get_dynamic_improver(ai_config: dict):
    """Dynamically creates a mirascope-decorated function for prompt improvement."""
    provider = ai_config.get("provider", "openai")
    model = ai_config.get("model", "gpt-4o-mini")

    @llm.call(provider=provider, model=model, response_model=PromptCritique)
    def _dynamic_improve_prompt(
        prompt_template: str,
        user_note: str,
        description: Optional[str] = None,
        inputs: Optional[Dict[str, str]] = None,
    ) -> str:
        # --- NEW: Build the prompt with optional context ---
        context_parts = []
        if description:
            context_parts.append(f"# PROMPT'S INTENDED PURPOSE (DESCRIPTION):\n{description}")
        if inputs:
            # Format the inputs nicely for the LLM
            input_examples = "\n".join([f"- {key}: (e.g., '{value}')" for key, value in inputs.items()])
            context_parts.append(f"# PROMPT'S INPUT VARIABLES AND EXAMPLES:\n{input_examples}")
        
        context_str = "\n\n".join(context_parts)
        
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
        # --- END OF NEW PROMPT ---

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
    """Gets a critique and improvement for a prompt, and logs the interaction."""
    try:
        improver_fn = _get_dynamic_improver(ai_config)
        # --- THE FIX IS HERE ---
        # The result of the call is the full CallResponse object
        call_response: llm.CallResponse = improver_fn(
            prompt_template,
            user_note=note,
            description=description,
            inputs=default_inputs,
        )
    except Exception as e:
        raise ConnectionError(f"Failed to call LLM for prompt improvement: {e}")

    logger = setup_ai_logger(project_root)
    if call_response._response.usage:
        log_data = {
            "model": call_response._response.model,
            "input_tokens": call_response._response.input_tokens,
            "output_tokens": call_response._response.output_tokens,
            "total_tokens": call_response._response.usage.total_tokens,
        }
        logger.info("get_prompt_critique", extra=log_data)
        
    # Return the structured data by dumping the Pydantic model
    return call_response.model_dump()