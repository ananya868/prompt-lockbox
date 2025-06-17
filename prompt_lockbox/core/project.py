#
# FILE: prompt_lockbox/core/project.py
#

import tomli
import tomli_w
import subprocess
from pathlib import Path

def get_project_root() -> Path | None:
    """
    Finds the project root by searching upwards for the 'plb.toml' file.
    
    Returns the Path to the root directory, or None if not found.
    """
    current_path = Path.cwd().resolve()
    while not (current_path / "plb.toml").exists():
        # Stop if we have reached the filesystem root
        if current_path.parent == current_path:
            return None
        current_path = current_path.parent
    return current_path

def get_config(project_root: Path) -> dict:
    """Reads the plb.toml file from the project root."""
    config_path = project_root / "plb.toml"
    if not config_path.exists():
        return {}
    with open(config_path, "rb") as f:
        return tomli.load(f)

def write_config(config: dict, project_root: Path):
    """Writes the given dictionary to plb.toml in the project root."""
    config_path = project_root / "plb.toml"
    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)
        
def get_git_author() -> str | None:
    """Tries to get the current user's name and email from git config."""
    try:
        name = subprocess.check_output(["git", "config", "user.name"], text=True).strip()
        email = subprocess.check_output(["git", "config", "user.email"], text=True).strip()
        return f"{name} <{email}>"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None