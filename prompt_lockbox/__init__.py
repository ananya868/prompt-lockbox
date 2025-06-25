#
# FILE: prompt_lockbox/__init__.py 
#

"""
A framework to secure, manage, and develop prompts programmatically.
"""

# Import the public-facing classes from your API layer
from .api import Project, Prompt

# You can define a package version here, which is a common practice.
__version__ = "0.1.0" 

# Define what gets imported when a user does 'from prompt_lockbox import *'
__all__ = [
    "Project",
    "Prompt",
]