#
# FILE: prompt_lockbox/core/project.py 
#
"""
This core module provides utility functions for interacting with the overall
PromptLockbox project, such as finding the project root, reading and writing
the main configuration file, and retrieving user information from Git.
"""

import tomli
import tomli_w
import subprocess
from pathlib import Path


def get_project_root() -> Path | None:
    """Finds the project root by searching upwards for a 'plb.toml' file.

    This function starts from the current working directory and traverses up
    the directory tree until it finds a directory containing `plb.toml`. This
    allows commands to be run from any subdirectory within the project.

    Returns:
        The `Path` object of the project's root directory, or `None` if
        `plb.toml` is not found in any parent directory.
    """
    # Start from the current working directory from where the script is run.
    current_path = Path.cwd().resolve()
    # Loop upwards until the config file is found.
    while not (current_path / "plb.toml").exists():
        # If we have reached the filesystem root, the project was not found.
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent
    return current_path


def get_config(project_root: Path) -> dict:
    """Reads and parses the plb.toml configuration file from the project root.

    Args:
        project_root: The root path of the PromptLockbox project.

    Returns:
        A dictionary containing the parsed TOML configuration, or an empty
        dictionary if the file doesn't exist or is malformed.
    """
    config_path = project_root / "plb.toml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except tomli.TOMLDecodeError:
        # Return a safe default if the TOML file is invalid.
        return {}


def write_config(config: dict, project_root: Path):
    """Writes the given dictionary to plb.toml in the project root.

    Args:
        config: The dictionary object to be written to the config file.
        project_root: The root path of the PromptLockbox project.
    """
    config_path = project_root / "plb.toml"
    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)


def get_git_author() -> str | None:
    """Tries to get the current user's name and email from the git config.

    This is used to automatically populate the 'author' field when creating
    or versioning prompts, if the project is a git repository.

    Returns:
        A formatted string "Name <email>" if git is installed and user.name
        and user.email are set, otherwise `None`.
    """
    try:
        # Run 'git config' commands to get user name and email.
        name = subprocess.check_output(
            ["git", "config", "user.name"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        email = subprocess.check_output(
            ["git", "config", "user.email"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return f"{name} <{email}>"
    except (FileNotFoundError, subprocess.CalledProcessError):
        # This handles cases where git is not installed or config is not set.
        return None