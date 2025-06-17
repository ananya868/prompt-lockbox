#
# FILE: prompt_lockbox/core/templating.py
#

import jinja2
from jinja2 import meta

def get_jinja_env() -> jinja2.Environment:
    """
    Creates and returns a Jinja2 environment configured for ${...} variables.
    """
    # Create a custom environment and change the variable delimiters
    env = jinja2.Environment(
        variable_start_string="${",
        variable_end_string="}",
        # We keep the comment syntax standard for Jinja2
        comment_start_string="{#",
        comment_end_string="#}",
        # Automatically remove trailing newlines to prevent extra blank lines
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env

def get_template_variables(template_string: str) -> set[str]:
    """
    Parses a template string and returns a set of all undeclared variables.
    """
    env = get_jinja_env()
    try:
        ast = env.parse(template_string)
        return meta.find_undeclared_variables(ast)
    except jinja2.exceptions.TemplateSyntaxError:
        return set()

def render_prompt(template_string: str, variables: dict) -> str:
    """
    Renders a template string with the given variables.

    Args:
        template_string: The raw prompt template.
        variables: A dictionary of variables to substitute.

    Returns:
        The final, rendered prompt text as a string.

    Raises:
        jinja2.exceptions.TemplateSyntaxError: If the template has syntax errors.
        jinja2.exceptions.UndefinedError: If a required variable is not provided.
    """
    env = get_jinja_env()
    template = env.from_string(template_string)
    return template.render(variables)