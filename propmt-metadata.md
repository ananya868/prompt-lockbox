The Final Metadata Schema

This is the blueprint for every prompt file in your framework.

name (Required): The unique identifier for the prompt (e.g., summarize-ticket).

version (Required): The semantic version of the prompt (e.g., 1.0.0).

description (Optional): A short, one-line explanation of the prompt's purpose.

tags (Optional): A list of keywords for categorization and discovery (e.g., Summarization, Support).

status (Optional): The prompt's current lifecycle stage (e.g., Draft, In-Review, Production).

author (Optional): The email or username of the prompt's creator or maintainer.

last_update (Optional): An ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SSZ) of the last significant change.

intended_model (Optional): The specific LLM this prompt was designed and tested for (e.g., openai/gpt-4-turbo).

model_parameters (Optional): A dictionary of recommended settings like temperature or max_tokens.

linked_prompts (Optional): A list of other prompts that this one depends on.

notes (Optional): Any detailed comments, warnings, or usage instructions.

template (Required): The core prompt text, using Jinja2 syntax for variables and logic.