#
# FILE: prompt_lockbox/core/integrity.py
#

import hashlib
import tomli
import tomli_w
from pathlib import Path
from datetime import datetime, timezone

def calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError:
        return ""

def read_lockfile(project_root: Path) -> dict:
    """Reads and parses the .plb.lock file."""
    lockfile_path = project_root / ".plb.lock"
    if not lockfile_path.is_file():
        return {"locked_prompts": {}}
    try:
        with open(lockfile_path, "rb") as f:
            return tomli.load(f)
    except tomli.TOMLDecodeError:
        return {"locked_prompts": {}}

def write_lockfile(data: dict, project_root: Path):
    """Writes data to the .plb.lock file."""
    lockfile_path = project_root / ".plb.lock"
    try:
        with open(lockfile_path, "wb") as f:
            tomli_w.dump(data, f)
    except IOError:
        # In a real SDK, you might want to log this error
        pass

def check_prompt_integrity(prompt_path: Path, project_root: Path) -> tuple[bool, str]:
    """
    Checks a single prompt's integrity if it is locked.

    Returns:
        A tuple containing:
        - bool: True if the prompt is secure (unlocked or locked and OK), False otherwise.
        - str: A status message ('UNLOCKED', 'OK', 'TAMPERED', 'MISSING').
    """
    lock_data = read_lockfile(project_root)
    locked_prompts = lock_data.get("locked_prompts", {})
    
    relative_path_str = str(prompt_path.relative_to(project_root))

    if relative_path_str not in locked_prompts:
        return (True, "UNLOCKED")

    if not prompt_path.exists():
        return (False, "MISSING")

    lock_info = locked_prompts[relative_path_str]
    stored_hash = lock_info.get("hash", "").split(":")[-1]
    
    current_hash = calculate_sha256(prompt_path)
    
    if stored_hash and stored_hash == current_hash:
        return (True, "OK")
    else:
        return (False, "TAMPERED")