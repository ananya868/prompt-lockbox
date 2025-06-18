#
# FILE: prompt_lockbox/api.py
#

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
import jinja2
import yaml

# Import our "engine" modules
from .core import project as core_project
from .core import prompt as core_prompt
from .core import integrity as core_integrity
from .core import templating as core_templating
from .search import hybrid, splade


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

    def render(self, strict: bool = True, **kwargs: Any) -> str:
        """
        Renders the prompt's template with the given variables.

        This method intelligently combines the prompt's `default_inputs`
        with any variables provided as keyword arguments.

        Args:
            strict: If True (default), raises an error for missing variables.
                    If False, substitutes missing variables with a placeholder.
            **kwargs: Keyword arguments corresponding to the variables
                      in the prompt's template.

        Returns:
            The final, rendered prompt text.

        Raises:
            jinja2.exceptions.TemplateSyntaxError: If the template has syntax errors.
            jinja2.exceptions.UndefinedError: If a required variable is not provided
                                              and `strict` is True.
        """
        template_string = self.data.get("template", "")
        if not template_string:
            return ""

        # Smartly merge default variables with user-provided ones.
        # User-provided kwargs take precedence.
        default_vars = self.data.get("default_inputs", {})
        final_vars = {**default_vars, **kwargs}

        if strict:
            # The original, safe behavior
            return core_templating.render_prompt(template_string, final_vars)
        else:
            # New "partial" rendering behavior
            env = core_templating.get_jinja_env()
            
            # Create a custom Undefined class that returns a placeholder
            class PlaceholderUndefined(jinja2.Undefined):
                def __str__(self):
                    return f"<<{self._undefined_name}>>"

            env.undefined = PlaceholderUndefined
            template = env.from_string(template_string)
            return template.render(final_vars)
    
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
        description: str = "",
        namespace: Optional[Union[str, List[str]]] = None,
        tags: Optional[List[str]] = None,
        intended_model: str = "",
        notes: str = "",
        model_parameters: Optional[Dict[str, Any]] = None,
        linked_prompts: Optional[List[str]] = None,
    ) -> Prompt:
        """
        Creates and saves a new prompt file from scratch with explicit parameters.
        ... (docstring is the same) ...
        """
        if author is None:
            author = core_project.get_git_author() or "Unknown Author"

        final_namespace = []
        if isinstance(namespace, str):
            final_namespace = [namespace]
        elif isinstance(namespace, list):
            final_namespace = namespace

        # --- THE FIX IS HERE ---

        # 1. Call the core engine to generate the complete file content string.
        file_content = core_prompt.generate_prompt_file_content(
            name=name, version=version, author=author, description=description,
            namespace=final_namespace, tags=tags, intended_model=intended_model,
            notes=notes, model_parameters=model_parameters, linked_prompts=linked_prompts,
        )

        # 2. Write the generated content to the file.
        prompts_dir = self.root / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        filename = f"{name}.v{version}.yml"
        filepath = prompts_dir / filename
        if filepath.exists():
            raise FileExistsError(f"A prompt file with this name and version already exists: {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(file_content)

        # 3. Instead of re-reading the file, parse the content string we already have.
        # This avoids any filesystem race conditions.
        data_from_content = yaml.safe_load(file_content)

        # 4. Create the Prompt object with the guaranteed-correct data.
        return Prompt(path=filepath, data=data_from_content, project_root=self.root)

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

    def get_status_report(self) -> dict:
        """
        Checks the integrity status of all prompts in the project.

        Returns:
            A dictionary containing lists of prompts categorized by status:
            'locked', 'unlocked', and 'tampered'.
        """
        report = {
            "locked": [],
            "unlocked": [],
            "tampered": [],
            "missing": [] # For locked files that have been deleted
        }
        
        all_prompts = self.list_prompts()
        for prompt in all_prompts:
            is_secure, status = prompt.verify()
            
            if status == "OK":
                report["locked"].append(prompt)
            elif status == "UNLOCKED":
                report["unlocked"].append(prompt)
            elif status == "TAMPERED":
                report["tampered"].append(prompt)
        
        # Now, check for missing files that are still in the lockfile
        lock_data = core_integrity.read_lockfile(self.root)
        locked_paths = set(lock_data.get("locked_prompts", {}).keys())
        found_paths = {str(p.path.relative_to(self.root)) for p in all_prompts}
        
        missing_paths = locked_paths - found_paths
        for path_str in missing_paths:
            report["missing"].append({"path": path_str}) # Add placeholder for missing prompts

        return report

    def lint(self) -> dict:
        """
        Validates all prompts in the project for schema and best practice compliance.

        Returns:
            A dictionary containing aggregated errors and warnings from all files.
        """
        all_prompts = self.list_prompts()
        
        categories = [
            "YAML Structure & Parsing", "Schema: Required Keys", "Schema: Data Types",
            "Schema: Value Formats", "Template: Jinja2 Syntax", "Best Practices & Logic",
        ]
        project_results = {cat: {"errors": [], "warnings": []} for cat in categories}
        
        for prompt in all_prompts:
            file_results = core_prompt.validate_prompt_file(prompt.path)
            relative_path = str(prompt.path.relative_to(self.root))
            for cat in categories:
                for error in file_results[cat]["errors"]:
                    project_results[cat]["errors"].append((relative_path, error))
                for warning in file_results[cat]["warnings"]:
                    project_results[cat]["warnings"].append((relative_path, warning))
                    
        return project_results

    def index(self, method: str = "hybrid"):
        """Builds a search index for all prompts in the project."""
        prompt_paths = [p.path for p in self.list_prompts()]
        if not prompt_paths:
            raise ValueError("No prompts found to index.")

        if method.lower() == 'hybrid':
            hybrid.build_hybrid_index(prompt_paths, self.root)
        elif method.lower() == 'splade':
            splade.build_splade_index(prompt_paths, self.root)
        else:
            raise ValueError(f"Invalid indexing method: '{method}'. Choose 'hybrid' or 'splade'.")

    def search(self, query: str, method: str = "hybrid", limit: int = 10, **kwargs) -> list[dict]:
        """Searches for prompts using a specified search engine."""
        if method.lower() == 'hybrid':
            alpha = kwargs.get('alpha', 0.5)
            return hybrid.search_hybrid(query, limit, self.root, alpha=alpha)
        elif method.lower() == 'splade':
            return splade.search_with_splade(query, limit, self.root)
        else:
            raise ValueError(f"Invalid search method: '{method}'. Choose 'hybrid' or 'splade'.")