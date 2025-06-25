#
# FILE: prompt_lockbox/core/templating.py
#
"""
This core module provides centralized functions for handling Jinja2 templating.

It defines a custom Jinja2 environment with specific delimiters (`${...}`),
and provides utilities for extracting variables from templates and rendering them.
"""

import jinja2
from jinja2 import meta


def get_jinja_env() -> jinja2.Environment:
    """Creates and returns a shared Jinja2 environment with custom settings.

    This function centralizes the configuration for all Jinja2 operations,
    ensuring consistency across the application. It sets up a non-standard
    variable syntax to be more user-friendly and less likely to conflict with
    other syntaxes (like shell scripts).

    Returns:
        A `jinja2.Environment` object configured for the project.
    """
    # Create a custom environment to control template behavior.
    env = jinja2.Environment(
        # Use a shell-like syntax for variables, e.g., ${variable_name}.
        variable_start_string="${",
        variable_end_string="}",
        # Keep the standard Jinja2 comment syntax, e.g., {# a comment #}.
        comment_start_string="{#",
        comment_end_string="#}",
        # Automatically remove trailing newlines from blocks to prevent extra blank lines.
        trim_blocks=True,
        # Strip leading whitespace from a block's start for cleaner template logic.
        lstrip_blocks=True,
    )
    return env


def get_template_variables(template_string: str) -> set[str]:
    """Parses a template string and returns a set of all undeclared variables.

    This function uses Jinja2's Abstract Syntax Tree (AST) parsing to reliably
    find all variable names that are used but not defined within the template itself
    (e.g., via `{% set ... %}`). It gracefully handles syntax errors by returning
    an empty set.

    Args:
        template_string: The raw prompt template string to be analyzed.

    Returns:
        A set of strings, where each string is a required variable name. Returns
        an empty set if the template has syntax errors or no variables.
    """
    env = get_jinja_env()
    try:
        # Parse the template into an AST.
        ast = env.parse(template_string)
        # Use Jinja2's metadata helper to find all required variables.
        return meta.find_undeclared_variables(ast)
    except jinja2.exceptions.TemplateSyntaxError:
        # If the template is invalid, there are no valid variables to find.
        return set()


def render_prompt(template_string: str, variables: dict) -> str:
    """Renders a template string with the given variables in strict mode.

    This function substitutes the variables (e.g., `${user_input}`) in the template
    with their corresponding values from the `variables` dictionary.

    Args:
        template_string: The raw prompt template to be rendered.
        variables: A dictionary where keys are variable names and values are
                   the content to be substituted.

    Returns:
        The final, rendered prompt as a string.

    Raises:
        jinja2.exceptions.TemplateSyntaxError: If the template has syntax errors.
        jinja2.exceptions.UndefinedError: If a required variable in the template
                                          is not provided in the `variables` dict.
    """
    env = get_jinja_env()
    template = env.from_string(template_string)
    # The `render` method performs the substitution.
    return template.render(variables)