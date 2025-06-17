#
# FILE: prompt_lockbox/api.py
#

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timezone
import jinja2
import yaml

# Import our "engine" modules
from .core import project as core_project
from .core import prompt as core_prompt
from .core import integrity as core_integrity
from .core import templating as core_templating


class Prompt:
    """
    Represents a single, versioned prompt file within a project.

    This object contains the prompt's metadata and content, and provides
    methods for integrity checks, versioning, and rendering.
    """
    def __init__(self, path: Path, data: Dict[str, Any], project_root: Path):
        self.path = path
        self.data = data
        self._project_root = project_root

    @property
    def name(self) -> Optional[str]:
        """The short, unique name of the prompt."""
        return self.data.get("name")

    @property
    def version(self) -> Optional[str]:
        """The semantic version of the prompt (e.g., '1.0.0')."""
        return self.data.get("version")

    @property
    def description(self) -> Optional[str]:
        """A human-readable description of what the prompt does."""
        return self.data.get("description")

    @property
    def required_variables(self) -> Set[str]:
        """
        Parses the template and returns a set of required variable names.
        """
        template_string = self.data.get("template", "")
        return core_templating.get_template_variables(template_string)

    def verify(self) -> tuple[bool, str]:
        """
        Checks the integrity of this prompt against the project's lockfile.

        Returns:
            A tuple containing (is_secure, status_message).
            - is_secure (bool): True if the prompt is unlocked or locked and valid.
            - status_message (str): 'UNLOCKED', 'OK', 'TAMPERED', or 'MISSING'.
        """
        return core_integrity.check_prompt_integrity(self.path, self._project_root)

    def lock(self):
        """Calculates the prompt's hash and locks it in the project's lockfile."""
        lock_data = core_integrity.read_lockfile(self._project_root)
        if "locked_prompts" not in lock_data:
            lock_data["locked_prompts"] = {}

        relative_path = str(self.path.relative_to(self._project_root))
        file_hash = core_integrity.calculate_sha256(self.path)
        
        lock_entry = {
            "hash": f"sha256:{file_hash}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        lock_data["locked_prompts"][relative_path] = lock_entry
        core_integrity.write_lockfile(lock_data, self._project_root)

    def unlock(self):
        """Removes this prompt from the project's lockfile."""
        lock_data = core_integrity.read_lockfile(self._project_root)
        relative_path = str(self.path.relative_to(self._project_root))

        if "locked_prompts" in lock_data and relative_path in lock_data["locked_prompts"]:
            del lock_data["locked_prompts"][relative_path]
            core_integrity.write_lockfile(lock_data, self._project_root)

    def render(self, **kwargs: Any) -> str:
        """
        Renders the prompt's template with the given variables.
        """
        template_string = self.data.get("template", "")
        if not template_string:
            return ""
        default_vars = self.data.get("default_inputs", {})
        final_vars = {**default_vars, **kwargs}
        return core_templating.render_prompt(template_string, final_vars)
    
    # --- NEW METHOD START HERE ---

    def new_version(self, bump_type: str = "minor", author: Optional[str] = None) -> Prompt:
        """
        Creates and saves a new version of this prompt.

        Args:
            bump_type: The type of version bump: 'major', 'minor', or 'patch'.
                       Defaults to 'minor'.
            author: The author of the new version. If None, tries to get from
                    git config, otherwise uses the source prompt's author.

        Returns:
            A new `Prompt` object representing the newly created file.

        Raises:
            FileExistsError: If a file for the new version already exists.
            ValueError: If the source prompt is missing name/version or bump_type is invalid.
        """
        new_data, new_filename = core_prompt.create_new_version_data(self.data, bump_type)

        # Update the author intelligently
        if author:
            new_data['author'] = author
        elif new_author := core_project.get_git_author():
            new_data['author'] = new_author
        
        prompts_dir = self._project_root / "prompts"
        new_filepath = prompts_dir / new_filename

        if new_filepath.exists():
            raise FileExistsError(f"A file for this version already exists: {new_filepath}")

        with open(new_filepath, "w", encoding="utf-8") as f:
            # Use yaml.dump for controlled, clean output
            yaml.dump(new_data, f, sort_keys=False, indent=2, width=80, allow_unicode=True)

        # Return a new Prompt object for the file we just created
        return Prompt(path=new_filepath, data=new_data, project_root=self._project_root)

    # --- END OF NEW METHOD ---
    
    def __repr__(self) -> str:
        return f"<Prompt name='{self.name}' version='{self.version}'>"


class Project:
    """
    The main entry point for a PromptLockbox project.
    """
    def __init__(self, path: str | Path | None = None):
        self._root = core_project.get_project_root()
        if not self._root:
            raise FileNotFoundError(
                "Could not find a PromptLockbox project. "
                "Ensure you are inside a directory with a 'plb.toml' file."
            )

    @property
    def root(self) -> Path:
        """Returns the resolved absolute path to the project root directory."""
        return self._root

    # --- NEW METHOD START HERE ---

    def create_prompt(
        self,
        name: str,
        version: str = "1.0.0",
        author: Optional[str] = None,
        **kwargs: Any
    ) -> Prompt:
        """
        Creates and saves a new prompt file from scratch.

        Args:
            name: The machine-friendly name of the new prompt.
            version: The starting semantic version. Defaults to '1.0.0'.
            author: The author of the prompt. If None, will try to get from git config.
            **kwargs: Other optional metadata like 'description', 'tags', etc.
                      that match the arguments of `generate_prompt_file_content`.

        Returns:
            A `Prompt` object representing the newly created file.

        Raises:
            FileExistsError: If a file with the same name and version already exists.
        """
        if author is None:
            author = core_project.get_git_author() or "Unknown Author"

        file_content = core_prompt.generate_prompt_file_content(
            name=name, version=version, author=author, **kwargs
        )

        prompts_dir = self.root / "prompts"
        prompts_dir.mkdir(exist_ok=True) # Ensure prompts directory exists
        
        filename = f"{name}.v{version}.yml"
        filepath = prompts_dir / filename

        if filepath.exists():
            raise FileExistsError(f"A prompt file with this name and version already exists: {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(file_content)

        # Load the data we just wrote to create a consistent Prompt object
        newly_loaded_data = core_prompt.load_prompt_data(filepath)
        return Prompt(path=filepath, data=newly_loaded_data, project_root=self.root)

    # --- END OF NEW METHOD ---
    
    def get_prompt(self, identifier: str) -> Optional[Prompt]:
        """
        Finds and loads a single prompt by its name, ID, or file path.
        """
        prompt_path = core_prompt.find_prompt_file(self.root, name=identifier)
        if not prompt_path:
            prompt_path = core_prompt.find_prompt_file(self.root, id=identifier)
        if not prompt_path:
            prompt_path = core_prompt.find_prompt_file(self.root, path=Path(identifier))
        
        if not prompt_path:
            return None

        prompt_data = core_prompt.load_prompt_data(prompt_path)
        if not prompt_data:
            return None

        return Prompt(path=prompt_path, data=prompt_data, project_root=self.root)

    def list_prompts(self) -> List[Prompt]:
        """
        Finds and loads all valid prompts within the project.
        """
        all_prompt_paths = core_prompt.find_all_prompt_files(self.root)
        
        prompts = []
        for path in all_prompt_paths:
            data = core_prompt.load_prompt_data(path)
            if data:
                prompts.append(Prompt(path=path, data=data, project_root=self.root))
        return prompts
        
    def __repr__(self) -> str:
        return f"<Project root='{self.root}'>"