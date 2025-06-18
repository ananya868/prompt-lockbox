#
# FILE: prompt_lockbox/__init__.py (Final Version)
#

"""
A framework to secure, manage, and develop prompts programmatically.
"""

# Import the public-facing classes from your API layer
# This makes them available at the top-level of the package.
from .api import Project, Prompt

# You can define a package version here, which is a common practice.
# Poetry may also manage this for you in pyproject.toml.
__version__ = "0.1.0" 

# Define what gets imported when a user does 'from prompt_lockbox import *'
# This is a best practice for defining the public API of a package.
__all__ = [
    "Project",
    "Prompt",
]