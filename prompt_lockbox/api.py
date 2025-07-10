#
# FILE: prompt_lockbox/api.py 
#
"""
This file defines the primary public-facing API for the PromptLockbox SDK.

It contains the `Project` and `Prompt` classes, which are the main entry
points for developers to interact with a prompt library, manage individual
prompts, and perform various operations like rendering, versioning, and
AI-assisted documentation and improvement.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Set
from datetime import datetime, timezone
import jinja2
import yaml
import re
import textwrap
from rich import print as rprint
from rich.console import Console

# Import our engine modules
from .core import project as core_project
from .core import prompt as core_prompt
from .core import integrity as core_integrity
from .core import templating as core_templating
from .search import hybrid, splade, fuzzy
from .ai import documenter, improver

console = Console()


class Prompt:
    """Represents a single, versioned prompt file within a project.

    This class provides an interface to interact with a prompt's data,
    render its template, manage its version, and perform AI-assisted
    operations like documentation and improvement.

    Attributes:
        path (Path): The absolute path to the prompt's YAML file.
        data (Dict[str, Any]): The parsed content of the prompt file.
    """

    def __init__(self, path: Path, data: Dict[str, Any], project: 'Project'):
        """Initializes a Prompt object.

        Args:
            path: The file path to the prompt's YAML file.
            data: The dictionary of data loaded from the prompt file.
            project: The parent Project instance to which this prompt belongs.
        """
        self.path = path
        self.data = data
        self._project = project
        self._project_root = project.root

    @property
    def name(self) -> Optional[str]:
        """The name of the prompt."""
        return self.data.get("name")

    @property
    def version(self) -> Optional[str]:
        """The version string of the prompt (e.g., '1.0.0')."""
        return self.data.get("version")

    @property
    def description(self) -> Optional[str]:
        """The description of the prompt."""
        return self.data.get("description")

    @property
    def required_variables(self) -> Set[str]:
        """A set of all required Jinja2 template variables."""
        return core_templating.get_template_variables(self.data.get("template", ""))

    def verify(self) -> tuple[bool, str]:
        """Verifies the prompt's integrity against the project's lockfile.

        Checks if the prompt file has been tampered with since it was last locked.

        Returns:
            A tuple containing a boolean (True if secure) and a status string
            ('OK', 'UNLOCKED', 'TAMPERED').
        """
        return core_integrity.check_prompt_integrity(self.path, self._project_root)

    def lock(self):
        """Creates a lock entry for the prompt in the project's lockfile.

        This records the file's current SHA256 hash and a timestamp, marking it
        as secure and verified.
        """
        lock_data=core_integrity.read_lockfile(self._project_root);lock_data.setdefault("locked_prompts",{})
        rel_path=str(self.path.relative_to(self._project_root));hash=core_integrity.calculate_sha256(self.path)
        lock_data["locked_prompts"][rel_path]={"hash":f"sha256:{hash}","timestamp":datetime.now(timezone.utc).isoformat()}
        core_integrity.write_lockfile(lock_data,self._project_root)

    def unlock(self):
        """Removes the lock entry for the prompt from the project's lockfile."""
        lock_data=core_integrity.read_lockfile(self._project_root);rel_path=str(self.path.relative_to(self._project_root))
        if "locked_prompts" in lock_data and rel_path in lock_data["locked_prompts"]:
            del lock_data["locked_prompts"][rel_path];core_integrity.write_lockfile(lock_data,self._project_root)

    def render(self, strict: bool = True, **kwargs: Any) -> str:
        """Renders the prompt's template with the provided variables.

        Args:
            strict: If True, raises an error for missing variables. If False,
              it will render missing variables as '<<variable_name>>'.
            **kwargs: The variables to inject into the template. These will
              override any 'default_inputs' specified in the prompt file.

        Returns:
            The final, rendered prompt as a string.
        """
        template_string=self.data.get("template","");
        if not template_string:return ""
        default_vars=self.data.get("default_inputs",{});final_vars={**default_vars,**kwargs}
        if strict:return core_templating.render_prompt(template_string,final_vars)
        else:
            env=core_templating.get_jinja_env();
            class PU(jinja2.Undefined):__str__=lambda s:f"<<{s._undefined_name}>>"
            env.undefined=PU;return env.from_string(template_string).render(final_vars)

    def document(self):
        """Uses an AI to automatically generate and save a description and tags.

        This method analyzes the prompt's template, generates new documentation,
        and then carefully overwrites the `description` and `tags` fields in
        the original file, preserving all comments and existing layout.

        Raises:
            ValueError: If the prompt's template content is empty.
        """
        print(f"📄 Analyzing '{self.name}' to generate documentation...")
        template_content = self.data.get("template", "")
        if not template_content: raise ValueError("Cannot document an empty prompt.")

        ai_config = self._project.get_ai_config()
        new_docs = documenter.get_documentation(
            template_content,
            project_root=self._project_root,
            ai_config=ai_config
        )

        if not new_docs.get("description") and not new_docs.get("tags"):
            print("🟡 Warning: AI did not return valid documentation."); return

        # This layout-preserving logic reads the file, replaces specific lines,
        # and writes the modified content back, keeping all other lines intact.

        # 1. Read the entire file as a list of text lines.
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            # 2. Use regex to find and replace the 'description' and 'tags' lines.
            if re.match(r"^\s*description:", line):
                new_desc = yaml.dump({"description": new_docs['description']}).strip()
                new_lines.append(new_desc + "\n")
            elif re.match(r"^\s*tags:", line):
                new_tags = yaml.dump({"tags": new_docs['tags']}).strip()
                new_lines.append(new_tags + "\n")
            else:
                # If it's not a line we're changing, keep it exactly as it was.
                new_lines.append(line)

        # 3. Write the modified list of lines back to the file.
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        # 4. Update the in-memory `self.data` object to be consistent with the disk.
        self.data['description'] = new_docs['description']
        self.data['tags'] = new_docs.get('tags', [])

        print("✅ Success! Description and tags updated while preserving layout.")

    def new_version(self, bump_type: str = "minor", author: Optional[str] = None) -> Prompt:
        """Creates a new version of the prompt file.

        This duplicates the current prompt and increments the version number
        according to the `bump_type`.

        Args:
            bump_type: The type of version bump ('major', 'minor', 'patch').
            author: The author of the new version. If None, it attempts to
              use the current Git user's name.

        Returns:
            A new `Prompt` object representing the newly created file.

        Raises:
            FileExistsError: If a file for the new version already exists.
        """
        new_data, new_filename = core_prompt.create_new_version_data(self.data, bump_type)
        if author: new_data['author'] = author
        elif new_author := core_project.get_git_author(): new_data['author'] = new_author
        new_filepath = self._project_root / "prompts" / new_filename
        if new_filepath.exists(): raise FileExistsError(f"File exists: {new_filepath}")
        with open(new_filepath, "w", encoding="utf-8") as f:
            yaml.dump(new_data, f, sort_keys=False, indent=2, width=80, allow_unicode=True)
        # Return a new Prompt object, passing the current project context.
        return Prompt(path=new_filepath, data=new_data, project=self._project)

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the Prompt."""
        return f"<Prompt name='{self.name}' version='{self.version}'>"

    def get_critique(self, note: str = "General improvements") -> dict:
        """Calls an AI to get a critique and an improved version of the prompt.

        This method sends the prompt's template, description, and default inputs
        to an AI for analysis. It does NOT save any changes to the file.

        Args:
            note: A specific instruction for the AI on how to improve the prompt.

        Returns:
            A dictionary containing the 'critique', 'suggestions', and
            'improved_template' from the AI.

        Raises:
            ValueError: If the prompt's template content is empty.
        """
        template_content = self.data.get("template", "")
        if not template_content:
            raise ValueError("Cannot get critique for an empty prompt.")

        ai_config = self._project.get_ai_config()

        # Call the AI with the full prompt context for a better critique.
        critique_data = improver.get_critique(
            prompt_template=template_content,
            note=note,
            project_root=self._project_root,
            ai_config=ai_config,
            # Pass the description and default_inputs to the AI engine.
            description=self.data.get("description"),
            default_inputs=self.data.get("default_inputs")
        )

        return critique_data

    def improve(self, improved_template: str):
        """Overwrites the prompt's template and updates its timestamp.

        This method carefully replaces the `template` block in the prompt's YAML
        file with the new content, preserving all other fields, comments, and
        the overall file layout.

        Args:
            improved_template: The new template string to write to the file.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Error: Could not find file {self.path} to save improvements.")
            return

        new_last_update = datetime.now(timezone.utc).isoformat()
        new_lines = []
        in_template_block = False

        # Reconstruct the file line by line to preserve structure.
        for line in lines:
            # Once we enter the old template block, skip all its lines.
            if in_template_block:
                continue

            # Identify the start of the template block.
            if re.match(r"^\s*template:", line):
                in_template_block = True
                # Write the header for the new template block.
                new_lines.append("template: |\n")
                # Add the new, improved template with correct indentation.
                indented_template = textwrap.indent(improved_template, '  ')
                new_lines.append(indented_template + "\n")
            # Find and replace the last_update timestamp.
            elif re.match(r"^\s*last_update:", line):
                new_lines.append(f'last_update: "{new_last_update}"\n')
            # Keep all other lines as they are.
            else:
                new_lines.append(line)

        # Write the reconstructed content back to the file.
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        # Update the in-memory object to match the file.
        self.data['template'] = improved_template
        self.data['last_update'] = new_last_update


class Project:
    """The main entry point for interacting with a PromptLockbox project.

    This class represents a project on the filesystem and provides methods
    for finding, creating, and managing all prompts within it.
    """

    def __init__(self, path: str | Path | None = None):
        """Initializes the Project, finding its root directory and loading config.

        Args:
            path: An optional path to a directory within the project. If None,
              it searches upwards from the current working directory.

        Raises:
            FileNotFoundError: If no '.promptlockbox' directory or `plb.toml` is
              found, indicating it's not a valid project.
        """
        self._root = core_project.get_project_root()
        if not self._root: raise FileNotFoundError("Not a PromptLockbox project. Have you run `plb init`?")
        self._config = core_project.get_config(self.root)

    @property
    def root(self) -> Path:
        """The root `Path` of the PromptLockbox project."""
        return self._root

    def get_ai_config(self) -> dict:
        """Retrieves the AI configuration from the project's `plb.toml`.

        Returns a dictionary with default values if no configuration is found.
        """
        ai_config = self._config.get("ai", {})
        return {"provider": ai_config.get("provider", "openai"), "model": ai_config.get("model", "gpt-4o-mini")}

    def get_prompt(self, identifier: str) -> Optional[Prompt]:
        """Finds a single prompt by its name, ID, or file path.

        If a name is provided, it returns the latest version of that prompt.

        Args:
            identifier: The name, ID, or file path string of the prompt to find.

        Returns:
            A `Prompt` object if found, otherwise `None`.
        """
        prompt_path = core_prompt.find_prompt_file(self.root, name=identifier) or \
                      core_prompt.find_prompt_file(self.root, id=identifier) or \
                      core_prompt.find_prompt_file(self.root, path=Path(identifier))
        if not prompt_path: return None
        prompt_data = core_prompt.load_prompt_data(prompt_path)
        # Pass `project=self` to the constructor to link the prompt to its project.
        return Prompt(path=prompt_path, data=prompt_data, project=self) if prompt_data else None

    def list_prompts(self) -> List[Prompt]:
        """Lists all prompts found in the project.

        Returns:
            A list of `Prompt` objects for every valid prompt file found.
        """
        all_prompt_paths = core_prompt.find_all_prompt_files(self.root)
        prompts = []
        for path in all_prompt_paths:
            data = core_prompt.load_prompt_data(path)
            if data:
                # Pass `project=self` to the constructor.
                prompts.append(Prompt(path=path, data=data, project=self))
        return prompts

    def create_prompt(
        self,
        name: str, version: str = "1.0.0", author: Optional[str] = None,
        description: str = "", namespace: Optional[Union[str, List[str]]] = None,
        tags: Optional[List[str]] = None, intended_model: str = "",
        notes: str = "", model_parameters: Optional[Dict[str, Any]] = None,
        linked_prompts: Optional[List[str]] = None,
    ) -> Prompt:
        """Creates a new prompt file on disk from the given metadata.

        Args:
            name: The name of the new prompt.
            version: The starting version string.
            author: The author of the prompt.
            description: A short description.
            namespace: A list or single string for the namespace.
            tags: A list of search tags.
            intended_model: The target model string.
            notes: Any additional notes.
            model_parameters: A dictionary of model parameters.
            linked_prompts: A list of linked prompt IDs.

        Returns:
            A `Prompt` object representing the newly created file.

        Raises:
            FileExistsError: If a prompt file with that name and version already exists.
        """
        if author is None:
            author = core_project.get_git_author() or "Unknown Author"

        final_namespace = [namespace] if isinstance(namespace, str) else (namespace or [])
        file_content = core_prompt.generate_prompt_file_content(
            name=name, version=version, author=author, description=description, namespace=final_namespace,
            tags=tags, intended_model=intended_model, notes=notes,
            model_parameters=model_parameters, linked_prompts=linked_prompts,
        )
        prompts_dir = self.root / "prompts"; prompts_dir.mkdir(exist_ok=True)
        filename = f"{name}.v{version}.yml"; filepath = prompts_dir / filename
        if filepath.exists(): raise FileExistsError(f"File exists: {filepath}")
        with open(filepath, "w", encoding="utf-8") as f: f.write(file_content)
        data_from_content = yaml.safe_load(file_content)
        # Pass `project=self` to the constructor.
        return Prompt(path=filepath, data=data_from_content, project=self)

    def get_status_report(self) -> dict:
        """Generates a report of the lock status of all prompts.

        Returns:
            A dictionary categorizing prompts into 'locked', 'unlocked',
            'tampered', and 'missing' lists.
        """
        report={"locked":[],"unlocked":[],"tampered":[],"missing":[]}
        all_prompts=self.list_prompts()
        for p in all_prompts:
            is_secure,status=p.verify()
            if status=="OK":report["locked"].append(p)
            elif status=="UNLOCKED":report["unlocked"].append(p)
            elif status=="TAMPERED":report["tampered"].append(p)
        lock_data=core_integrity.read_lockfile(self.root);locked_paths=set(lock_data.get("locked_prompts",{}).keys())
        found_paths={str(p.path.relative_to(self.root)) for p in all_prompts};missing_paths=locked_paths-found_paths
        for path_str in missing_paths:report["missing"].append({"path":path_str})
        return report

    def lint(self) -> dict:
        """Validates all prompt files in the project for correctness and consistency.

        Returns:
            A dictionary of results categorized by check type, containing lists
            of errors and warnings for all prompts.
        """
        all_prompts=self.list_prompts();categories=["YAML Structure & Parsing","Schema: Required Keys","Schema: Data Types","Schema: Value Formats","Template: Jinja2 Syntax","Best Practices & Logic"]
        project_results={cat:{"errors":[],"warnings":[]} for cat in categories}
        for p in all_prompts:
            file_results=core_prompt.validate_prompt_file(p.path);relative_path=str(p.path.relative_to(self.root))
            for cat in categories:
                for e in file_results[cat]["errors"]:project_results[cat]["errors"].append((relative_path,e))
                for w in file_results[cat]["warnings"]:project_results[cat]["warnings"].append((relative_path,w))
        return project_results

    def index(self, method: str = "hybrid"):
        """Builds a search index for all prompts.

        Args:
            method: The indexing method to use ('hybrid' or 'splade').

        Raises:
            ValueError: If no prompts are found or the method is invalid.
        """
        prompt_paths=[p.path for p in self.list_prompts()];
        if not prompt_paths:raise ValueError("No prompts to index.")
        if method.lower()=='hybrid':hybrid.build_hybrid_index(prompt_paths,self.root)
        elif method.lower()=='splade':splade.build_splade_index(prompt_paths,self.root)
        else:raise ValueError(f"Invalid index method: '{method}'.")

    def search(self, query: str, method: str = "fuzzy", limit: int = 10, **kwargs) -> list[dict]:
        """Searches for prompts using a specified method.

        Args:
            query: The search query string.
            method: The search method to use ('fuzzy', 'hybrid', 'splade').
            limit: The maximum number of results to return.
            **kwargs: Additional arguments for specific search methods (e.g., 'alpha' for hybrid).

        Returns:
            A list of result dictionaries, sorted by relevance.
        """
        if method.lower()=='fuzzy':return fuzzy.search_fuzzy(query,self.list_prompts(),limit=limit)
        elif method.lower()=='hybrid':return hybrid.search_hybrid(query,limit,self.root,alpha=kwargs.get('alpha',0.5))
        elif method.lower()=='splade':return splade.search_with_splade(query,limit,self.root)
        else:raise ValueError(f"Invalid search method: '{method}'.")

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the Project."""
        return f"<Project root='{self.root}'>"

    def write_config(self, config: Dict[str, Any]):
        """
        A pass-through method to write to the project's config file.
        
        Args:
            config: The dictionary object to be written.
        """
        core_project.write_config(config, self.root)

    def document_all(self, prompts_to_document: Optional[List[Prompt]] = None):
        """Uses an AI to automatically generate and save documentation for multiple prompts.

        If documenting more than one prompt, a progress bar will be displayed.

        Args:
            prompts_to_document: A specific list of Prompt objects to document.
                                 If None, all prompts in the project will be documented.
        """
        # Determine if this is a bulk operation to decide whether to show a progress bar.
        is_bulk_operation = False

        if prompts_to_document is None:
            prompts_to_document = self.list_prompts()
            print(f"Found {len(prompts_to_document)} prompts to document...")
            if len(prompts_to_document) > 1:
                is_bulk_operation = True
        else:
            if len(prompts_to_document) > 1:
                is_bulk_operation = True

        if not prompts_to_document:
            print("No matching prompts found to document.")
            return

        # Show a progress bar only for true bulk operations (more than one prompt).
        if is_bulk_operation:
            from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
            progress_bar = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), TextColumn("[progress.percentage]{task.gsu>3.0f}%"),
                TimeRemainingColumn(),
            )

            with progress_bar as progress:
                task = progress.add_task("[cyan]Documenting...", total=len(prompts_to_document))
                for prompt in prompts_to_document:
                    progress.update(task, description=f"[cyan]Documenting [bold]{prompt.name}[/bold]")
                    try:
                        # The single prompt.document() method prints its own status.
                        prompt.document()
                    except Exception as e:
                        console.print(f"❌ [bold red]Could not document '{prompt.name}':[/] {e}", style="red")
                    progress.advance(task)
        else:
            # If it's not a bulk operation (i.e., only one prompt), just process it directly.
            for prompt in prompts_to_document:
                try:
                    prompt.document()
                except Exception as e:
                    console.print(f"❌ [bold red]Could not document '{prompt.name}':[/] {e}", style="red")

        # Only print the "Bulk complete" message if it was actually a bulk operation.
        if is_bulk_operation:
            rprint(f"\n✅ [bold green]Bulk documentation complete.[/bold green]")