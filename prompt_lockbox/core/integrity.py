#
# FILE: prompt_lockbox/core/integrity.py
#
"""
This core module provides functions for ensuring the integrity of prompt files.

It handles the calculation of file hashes, reading from and writing to the
project's lockfile (`.plb.lock`), and verifying whether a prompt file has been
modified since it was last locked.
"""

import hashlib
import tomli
import tomli_w
from pathlib import Path
from datetime import datetime, timezone


def calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file.

    Reads the file in binary mode and in chunks to efficiently handle
    potentially large files without consuming excessive memory.

    Args:
        file_path: The `Path` object pointing to the file to be hashed.

    Returns:
        The hex digest of the SHA256 hash as a string, or an empty string
        if an `IOError` occurs.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K for memory efficiency.
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError:
        return ""


def read_lockfile(project_root: Path) -> dict:
    """Reads and parses the .plb.lock file from the project root.

    This function safely handles cases where the lockfile does not exist or
    is malformed, returning a default dictionary structure.

    Args:
        project_root: The root path of the PromptLockbox project.

    Returns:
        A dictionary containing the parsed TOML data from the lockfile.
    """
    lockfile_path = project_root / ".plb.lock"
    if not lockfile_path.is_file():
        return {"locked_prompts": {}}
    try:
        with open(lockfile_path, "rb") as f:
            return tomli.load(f)
    except tomli.TOMLDecodeError:
        # If the lockfile is corrupted or invalid, return a safe default.
        return {"locked_prompts": {}}


def write_lockfile(data: dict, project_root: Path):
    """Writes a dictionary of data to the .plb.lock file in TOML format.

    Args:
        data: The dictionary object to be written to the lockfile.
        project_root: The root path of the PromptLockbox project.
    """
    lockfile_path = project_root / ".plb.lock"
    try:
        with open(lockfile_path, "wb") as f:
            tomli_w.dump(data, f)
    except IOError:
        # In a real SDK, you might want to log this error.
        pass


def check_prompt_integrity(prompt_path: Path, project_root: Path) -> tuple[bool, str]:
    """Checks a single prompt's integrity against its entry in the lockfile.

    This function determines if a prompt is unlocked, locked and unmodified,
    or locked and tampered with.

    Args:
        prompt_path: The absolute path to the prompt file being checked.
        project_root: The root path of the PromptLockbox project.

    Returns:
        A tuple containing:
        - bool: True if the prompt is considered secure (i.e., it is unlocked
          or it is locked and its hash matches), False otherwise.
        - str: A status message: 'UNLOCKED', 'OK', 'TAMPERED', or 'MISSING'.
    """
    lock_data = read_lockfile(project_root)
    locked_prompts = lock_data.get("locked_prompts", {})

    # Use relative paths for keys in the lockfile to ensure portability.
    relative_path_str = str(prompt_path.relative_to(project_root))

    # If the prompt is not in the lockfile, it's considered 'UNLOCKED'.
    if relative_path_str not in locked_prompts:
        return (True, "UNLOCKED")

    # If the file is in the lockfile but doesn't exist on disk, it's 'MISSING'.
    if not prompt_path.exists():
        return (False, "MISSING")

    # Compare the stored hash with the file's current hash.
    lock_info = locked_prompts[relative_path_str]
    stored_hash = lock_info.get("hash", "").split(":")[-1]

    current_hash = calculate_sha256(prompt_path)

    # If the hashes match, the file is secure and 'OK'.
    if stored_hash and stored_hash == current_hash:
        return (True, "OK")
    # If the hashes do not match, the file has been 'TAMPERED' with.
    else:
        return (False, "TAMPERED")