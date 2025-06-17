#
# In FILE: prompt_lockbox/core/prompt.py
# ADD THE FOLLOWING FUNCTIONS to the bottom of the file:
#

import uuid
import copy
import textwrap
import json
from datetime import datetime, timezone
from packaging.version import Version, InvalidVersion

# (Your existing functions like load_prompt_data, find_prompt_file, etc., remain at the top)

def generate_prompt_file_content(
    name: str,
    version: str,
    author: str,
    description: str = "",
    namespace: list[str] | None = None,
    tags: list[str] | None = None,
    intended_model: str = "",
    notes: str = "",
) -> str:
    """
    Generates the complete, formatted string content for a new prompt YAML file,
    including the helpful comments and structure from the CLI.
    """
    if namespace is None:
        namespace = []
    if tags is None:
        tags = []
        
    prompt_id = f"prm_{uuid.uuid4().hex}"
    last_update = datetime.now(timezone.utc).isoformat()

    # The default template that will be embedded in the new file
    default_template_str = """\
You are a helpful assistant.

-- Now you can start writing your prompt template! --

How to use this template:
- To input a value: ${user_input}
- To add a comment: {# This is a comment #}

Create awesome prompts! :)"""

    # Format lists into YAML-compatible strings, ensuring quotes for safety
    namespace_str = f"[{', '.join(json.dumps(n) for n in namespace)}]"
    tags_str = f"[{', '.join(json.dumps(t) for t in tags)}]"
    indented_template = textwrap.indent(default_template_str, '  ')

    # Use an f-string to build the exact file layout, using json.dumps
    # to safely handle quotes in any of the string inputs.
    file_content = f"""\
# PROMPT IDENTITY
# --------------------
id: {prompt_id}
name: {json.dumps(name)}
version: "{version}"

# DOCUMENTATION
# --------------------
description: {json.dumps(description)}
namespace: {namespace_str}
tags: {tags_str}

# OWNERSHIP & STATUS
# --------------------
status: "Draft"
author: {json.dumps(author)}
last_update: "{last_update}"

# CONFIGURATION
# --------------------
intended_model: {json.dumps(intended_model)}
model_parameters:
  temperature: 0.7

# NOTES & LINKS
# --------------------
linked_prompts: []
notes: {json.dumps(notes)}

# - - - 💖 THE PROMPT 💖 - - -
# ---------------------------------
# NOTE - Comments inside the prompt are automatically removed on prompt call.
default_inputs:
  user_input: "Sample Input"

template: |
{indented_template}
"""
    return file_content


def create_new_version_data(
    source_data: dict, 
    bump_type: str = "minor"
) -> tuple[dict, str]:
    """
    Takes existing prompt data and returns new data and a new filename for a version bump.

    Args:
        source_data: The dictionary data from the source prompt.
        bump_type: 'major', 'minor', or 'patch'.

    Returns:
        A tuple of (new_data_dict, new_filename_string).

    Raises:
        ValueError: If bump_type is invalid or source data is missing keys.
    """
    current_version_str = source_data.get("version")
    prompt_name = source_data.get("name")

    if not all([current_version_str, prompt_name]):
        raise ValueError("Source prompt data must contain 'name' and 'version' keys.")

    try:
        v = Version(current_version_str)
        if bump_type == "major":
            new_version_str = f"{v.major + 1}.0.0"
        elif bump_type == "patch":
            new_version_str = f"{v.major}.{v.minor}.{v.patch + 1}"
        elif bump_type == "minor":
            new_version_str = f"{v.major}.{v.minor + 1}.0"
        else:
            raise ValueError("bump_type must be 'major', 'minor', or 'patch'.")
    except InvalidVersion:
        raise ValueError(f"Invalid version format '{current_version_str}' in source file.")

    new_filename = f"{prompt_name}.v{new_version_str}.yml"
    
    new_data = copy.deepcopy(source_data)
    new_data['version'] = new_version_str
    new_data['status'] = 'Draft'
    new_data['last_update'] = datetime.now(timezone.utc).isoformat()
    # Each prompt version gets its own unique ID
    new_data['id'] = f"prm_{uuid.uuid4().hex}"

    return (new_data, new_filename)