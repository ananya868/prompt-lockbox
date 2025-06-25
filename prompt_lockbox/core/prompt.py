#
# FILE: prompt_lockbox/core/prompt.py
#
"""
This core module provides low-level functions for finding, loading, validating,
and creating prompt files and their data structures. It serves as the engine
behind the higher-level API and CLI operations.
"""

import yaml
import re
import uuid
import copy
import textwrap
import json
from pathlib import Path
from packaging.version import Version, InvalidVersion
from datetime import datetime, timezone

from . import templating as core_templating


def load_prompt_data(file_path: Path) -> dict:
    """Safely loads and parses a YAML prompt file into a dictionary.

    Handles file-not-found and YAML parsing errors by returning an empty dict,
    ensuring downstream functions can operate without crashing.

    Args:
        file_path: The `Path` object to the YAML prompt file.

    Returns:
        A dictionary of the parsed YAML content, or an empty dictionary on failure.
    """
    if not file_path.is_file():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except (yaml.YAMLError, IOError):
        return {}


def find_prompt_file(
    project_root: Path,
    name: str | None = None,
    id: str | None = None,
    path: Path | None = None
) -> Path | None:
    """Scans the `prompts/` directory to find a prompt file by name, ID, or path.

    This function provides a flexible way to locate a specific prompt file.
    Matching by ID is the fastest and most exact method. Matching by name
    will return the file with the highest semantic version number.

    Args:
        project_root: The root path of the PromptLockbox project.
        name: The `name` of the prompt to find (e.g., 'summarize-ticket').
        id: The unique `id` of the prompt to find (e.g., 'prm_...').
        path: A direct file `Path` to the prompt file.

    Returns:
        A `Path` object to the found file, or `None` if no match is found.
    """
    prompts_dir = project_root / "prompts"
    if not prompts_dir.exists():
        return None

    # Handle direct path identifier first for performance.
    if path:
        # Resolve the path relative to the project root for consistency.
        absolute_path = (project_root / path).resolve()
        return absolute_path if absolute_path.exists() else None

    # If not a direct path, scan all .yml files in the prompts directory.
    all_prompt_files = list(prompts_dir.glob("**/*.yml"))

    # If an ID is provided, it's a fast and exact match.
    if id:
        for f in all_prompt_files:
            data = load_prompt_data(f)
            if data.get("id") == id:
                return f
        return None

    # If a name is provided, find all candidates and select the latest version.
    if name:
        candidate_files = []
        for f in all_prompt_files:
            data = load_prompt_data(f)
            if data.get("name") == name:
                try:
                    version = Version(data.get("version", "0.0.0"))
                    candidate_files.append((f, version))
                except InvalidVersion:
                    continue  # Ignore files with malformed versions.

        if not candidate_files:
            return None

        # Sort by version, descending, to find the latest one.
        candidate_files.sort(key=lambda x: x[1], reverse=True)
        return candidate_files[0][0]

    return None


def find_all_prompt_files(project_root: Path) -> list[Path]:
    """Returns a sorted list of all .yml prompt files in the project.

    Args:
        project_root: The root path of the PromptLockbox project.

    Returns:
        A sorted list of `Path` objects for every found `.yml` file.
    """
    prompts_dir = project_root / "prompts"
    if not prompts_dir.is_dir():
        return []
    return sorted(list(prompts_dir.glob("**/*.yml")))


def validate_prompt_file(file_path: Path) -> dict:
    """Validates a single prompt file against a strict schema and best practices.

    This function checks for YAML validity, presence of required keys, correct
    data types, valid value formats, and Jinja2 template syntax.

    Args:
        file_path: The `Path` to the prompt file to validate.

    Returns:
        A dictionary of results categorized by check type, containing lists
        of 'errors' and 'warnings'.
    """
    categories = [
        "YAML Structure & Parsing", "Schema: Required Keys", "Schema: Data Types",
        "Schema: Value Formats", "Template: Jinja2 Syntax", "Best Practices & Logic",
    ]
    results = {cat: {"errors": [], "warnings": []} for cat in categories}

    # Define the schema rules for validation.
    REQUIRED_KEYS = {
        "id", "name", "version", "description", "namespace", "tags",
        "status", "author", "last_update", "intended_model",
        "model_parameters", "linked_prompts", "notes", "default_inputs", "template"
    }
    EXPECTED_TYPES = {
        "namespace": list, "tags": list, "model_parameters": dict,
        "linked_prompts": list, "default_inputs": dict
    }
    VALID_STATUSES = {"Draft", "In-Review", "Staging", "Production", "Deprecated", "Archived"}

    data = load_prompt_data(file_path)
    if not data:
        results["YAML Structure & Parsing"]["errors"].append("File is empty or contains invalid YAML.")
        return results

    # CHECK 2: Required Keys
    missing_keys = REQUIRED_KEYS - set(data.keys())
    for key in sorted(list(missing_keys)):
        results["Schema: Required Keys"]["errors"].append(f"Missing required key: '{key}'")

    # CHECK 3: Data Types
    for key, expected_type in EXPECTED_TYPES.items():
        if key in data and data.get(key) is not None and not isinstance(data[key], expected_type):
            results["Schema: Data Types"]["errors"].append(
                f"Key '{key}' must be a {expected_type.__name__}, but found {type(data[key]).__name__}."
            )

    # CHECK 4: Value Formats (e.g., SemVer, ISO 8601)
    if "version" in data and data.get("version"):
        try:
            Version(str(data["version"]))
        except InvalidVersion:
            results["Schema: Value Formats"]["errors"].append(f"Invalid semantic version for 'version': '{data['version']}'.")

    if "status" in data and data.get("status") and data["status"] not in VALID_STATUSES:
        results["Schema: Value Formats"]["errors"].append(f"Value '{data['status']}' is not a valid status.")

    if "last_update" in data and data.get("last_update"):
        try:
            datetime.fromisoformat(str(data["last_update"]).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            results["Schema: Value Formats"]["errors"].append(f"Invalid ISO 8601 format for 'last_update': '{data['last_update']}'.")

    if "id" in data and data.get("id"):
        if not re.match(r"^prm_[a-f0-9]{32}$", str(data["id"])):
            results["Schema: Value Formats"]["errors"].append(f"Invalid 'id' format: '{data['id']}'.")

    # CHECK 5: Jinja2 Template Syntax
    if "template" in data and data.get("template"):
        try:
            env = core_templating.get_jinja_env()
            env.parse(data["template"])
        except Exception as e:
            results["Template: Jinja2 Syntax"]["errors"].append(f"Invalid Jinja2 syntax: {e}")

    # CHECK 6: Best Practices & Logic
    if "template" in data and data.get("template") and "default_inputs" in data and data.get("default_inputs"):
        template_vars = core_templating.get_template_variables(data["template"])
        for key in data["default_inputs"]:
            if key not in template_vars:
                results["Best Practices & Logic"]["warnings"].append(f"Default input '{key}' is defined but not used in the template.")

    return results


def generate_prompt_file_content(
    name: str,
    version: str,
    author: str,
    description: str = "",
    namespace: list[str] | None = None,
    tags: list[str] | None = None,
    intended_model: str = "",
    notes: str = "",
    model_parameters: dict | None = None,
    linked_prompts: list[str] | None = None
) -> str:
    """Generates the complete, formatted string content for a new prompt YAML file.

    This function constructs a well-formatted and commented YAML string that
    serves as the starting template for a new prompt.

    Args:
        name: The name of the new prompt.
        version: The starting version string.
        author: The author of the prompt.
        description: A short description.
        namespace: A list of namespace parts.
        tags: A list of search tags.
        intended_model: The target model string.
        notes: Any additional notes.
        model_parameters: A dictionary of model parameters.
        linked_prompts: A list of linked prompt IDs.

    Returns:
        A string containing the full content for the new `.yml` file.
    """
    if namespace is None: namespace = []
    if tags is None: tags = []
    if linked_prompts is None: linked_prompts = []
    if model_parameters is None: model_parameters = {"temperature": 0.7}

    prompt_id = f"prm_{uuid.uuid4().hex}"
    last_update = datetime.now(timezone.utc).isoformat()

    default_template_str = "You are a helpful assistant.\n\n-- Now you can start writing your prompt template! --\n\nHow to use this template:\n- To input a value: ${user_input}\n- To add a comment: {# This is a comment #}\n\nCreate awesome prompts! :)"

    # Format list-like and dict-like fields for clean YAML output.
    linked_prompts_str = f"[{', '.join(json.dumps(lp) for lp in linked_prompts)}]"
    namespace_str = f"[{', '.join(json.dumps(n) for n in namespace)}]"
    tags_str = f"[{', '.join(json.dumps(t) for t in tags)}]"
    indented_template = textwrap.indent(default_template_str, '  ')

    # Construct the final file content using an f-string.
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
{textwrap.indent(yaml.dump(model_parameters, indent=2), '  ')}

# NOTES & LINKS
# --------------------
linked_prompts: {linked_prompts_str}
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
    """Takes existing prompt data and returns new data and a new filename for a version bump.

    This function calculates the new version string based on the `bump_type`,
    creates a new unique ID, resets the status to 'Draft', and updates the
    timestamp.

    Args:
        source_data: The dictionary data from the original prompt file.
        bump_type: The type of version increment ('major', 'minor', or 'patch').

    Returns:
        A tuple containing:
        - dict: The data for the new version.
        - str: The filename for the new version.

    Raises:
        ValueError: If the source data is missing keys or the version is invalid.
    """
    current_version_str = source_data.get("version")
    prompt_name = source_data.get("name")

    if not all([current_version_str, prompt_name]):
        raise ValueError("Source prompt data must contain 'name' and 'version' keys.")

    try:
        v = Version(current_version_str)

        # Calculate the new version string based on the bump type.
        if bump_type == "major":
            new_version_str = f"{v.major + 1}.0.0"
        elif bump_type == "patch":
            patch_num = v.release[2] if len(v.release) > 2 else 0
            new_version_str = f"{v.major}.{v.minor}.{patch_num + 1}"
        elif bump_type == "minor":
            new_version_str = f"{v.major}.{v.minor + 1}.0"
        else:
            raise ValueError("bump_type must be 'major', 'minor', or 'patch'.")

    except InvalidVersion:
        raise ValueError(f"Invalid version format '{current_version_str}' in source file.")

    # Generate the new filename.
    new_filename = f"{prompt_name}.v{new_version_str}.yml"

    # Create the new data dictionary.
    new_data = copy.deepcopy(source_data)
    new_data['version'] = new_version_str
    new_data['status'] = 'Draft'
    new_data['last_update'] = datetime.now(timezone.utc).isoformat()
    # Give each version a unique ID for better tracking.
    new_data['id'] = f"prm_{uuid.uuid4().hex}"

    return (new_data, new_filename)


def find_next_template_index(prompts_dir: Path) -> int:
    """Finds the highest index for existing 'prompt_template_*.yml' files.

    This is used by the `plb create --bulk` command to generate new template
    files with unique, incrementing names (e.g., prompt_template_1,
    prompt_template_2, etc.) without overwriting existing ones.

    Args:
        prompts_dir: The path to the `prompts/` directory.

    Returns:
        The next available integer index for a new template file.
    """
    if not prompts_dir.exists():
        return 1

    highest_index = 0
    # Scan for files matching the template naming convention.
    for f in prompts_dir.glob("prompt_template_*.yml"):
        try:
            # Assumes a filename like 'prompt_template_123.v1.0.0.yml'.
            base_name = f.name.split('.v')[0]
            index_str = base_name.split('_')[-1]
            index = int(index_str)
            if index > highest_index:
                highest_index = index
        except (ValueError, IndexError):
            # Ignore files that don't match the expected naming pattern.
            continue
    return highest_index + 1