#
# FILE: prompt_lockbox/api.py (Corrected and Final)
#

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Set
from datetime import datetime, timezone
import jinja2
import yaml

# Import our engine modules
from .core import project as core_project
from .core import prompt as core_prompt
from .core import integrity as core_integrity
from .core import templating as core_templating
from .search import hybrid, splade, fuzzy
# Let's be consistent with the filename
from .ai import documenter

class Prompt:
    """Represents a single, versioned prompt file within a project."""
    
    # --- FIX 1: Correct the constructor ---
    def __init__(self, path: Path, data: Dict[str, Any], project: 'Project'):
        self.path = path
        self.data = data
        # This correctly stores the whole project object
        self._project = project
        # This correctly gets the root path from the passed project object
        self._project_root = project.root

    # --- All properties and methods down to `new_version` are correct and unchanged ---
    @property
    def name(self) -> Optional[str]: return self.data.get("name")
    @property
    def version(self) -> Optional[str]: return self.data.get("version")
    @property
    def description(self) -> Optional[str]: return self.data.get("description")
    @property
    def required_variables(self) -> Set[str]:
        return core_templating.get_template_variables(self.data.get("template", ""))
    def verify(self) -> tuple[bool, str]:
        return core_integrity.check_prompt_integrity(self.path, self._project_root)
    def lock(self): # ... (implementation is correct)
        lock_data=core_integrity.read_lockfile(self._project_root);lock_data.setdefault("locked_prompts",{})
        rel_path=str(self.path.relative_to(self._project_root));hash=core_integrity.calculate_sha256(self.path)
        lock_data["locked_prompts"][rel_path]={"hash":f"sha256:{hash}","timestamp":datetime.now(timezone.utc).isoformat()}
        core_integrity.write_lockfile(lock_data,self._project_root)
    def unlock(self): # ... (implementation is correct)
        lock_data=core_integrity.read_lockfile(self._project_root);rel_path=str(self.path.relative_to(self._project_root))
        if "locked_prompts" in lock_data and rel_path in lock_data["locked_prompts"]:
            del lock_data["locked_prompts"][rel_path];core_integrity.write_lockfile(lock_data,self._project_root)
    def render(self, strict: bool = True, **kwargs: Any) -> str: # ... (implementation is correct)
        template_string=self.data.get("template","");
        if not template_string:return ""
        default_vars=self.data.get("default_inputs",{});final_vars={**default_vars,**kwargs}
        if strict:return core_templating.render_prompt(template_string,final_vars)
        else:
            env=core_templating.get_jinja_env();
            class PU(jinja2.Undefined):__str__=lambda s:f"<<{s._undefined_name}>>"
            env.undefined=PU;return env.from_string(template_string).render(final_vars)

    def document(self):
        print(f"📄 Analyzing '{self.name}' to generate documentation...");template_content=self.data.get("template","")
        if not template_content:raise ValueError("Cannot document an empty prompt.")
        ai_config=self._project.get_ai_config()
        new_docs=documenter.get_documentation(template_content,project_root=self._project_root,ai_config=ai_config)
        if not new_docs.get("description") and not new_docs.get("tags"):
            print("🟡 Warning: AI did not return valid documentation.");return
        self.data['description']=new_docs['description'];self.data['tags']=new_docs.get('tags',[])
        self.data['last_update']=datetime.now(timezone.utc).isoformat()
        with open(self.path,"w",encoding="utf-8") as f:yaml.dump(self.data,f,sort_keys=False,indent=2,width=80)
        print("✅ Success! Description and tags updated.")

    # --- FIX 2: Correct the return statement in `new_version` ---
    def new_version(self, bump_type: str = "minor", author: Optional[str] = None) -> Prompt:
        new_data, new_filename = core_prompt.create_new_version_data(self.data, bump_type)
        if author: new_data['author'] = author
        elif new_author := core_project.get_git_author(): new_data['author'] = new_author
        new_filepath = self._project_root / "prompts" / new_filename
        if new_filepath.exists(): raise FileExistsError(f"File exists: {new_filepath}")
        with open(new_filepath, "w", encoding="utf-8") as f:
            yaml.dump(new_data, f, sort_keys=False, indent=2, width=80, allow_unicode=True)
        # Correctly pass `project=self._project`
        return Prompt(path=new_filepath, data=new_data, project=self._project)
    
    def __repr__(self) -> str: return f"<Prompt name='{self.name}' version='{self.version}'>"


class Project:
    """The main entry point for a PromptLockbox project."""
    def __init__(self, path: str | Path | None = None):
        self._root = core_project.get_project_root()
        if not self._root: raise FileNotFoundError("Not a PromptLockbox project. Have you run `plb init`?")
        self._config = core_project.get_config(self.root)
    
    @property
    def root(self) -> Path: return self._root
    
    def get_ai_config(self) -> dict:
        ai_config = self._config.get("ai", {})
        return {"provider": ai_config.get("provider", "openai"), "model": ai_config.get("model", "gpt-4o-mini")}

    # --- FIX 3: Correct how Prompt objects are created in all Project methods ---
    def get_prompt(self, identifier: str) -> Optional[Prompt]:
        prompt_path = core_prompt.find_prompt_file(self.root, name=identifier) or \
                      core_prompt.find_prompt_file(self.root, id=identifier) or \
                      core_prompt.find_prompt_file(self.root, path=Path(identifier))
        if not prompt_path: return None
        prompt_data = core_prompt.load_prompt_data(prompt_path)
        # Pass `project=self` to the constructor
        return Prompt(path=prompt_path, data=prompt_data, project=self) if prompt_data else None

    def list_prompts(self) -> List[Prompt]:
        all_prompt_paths = core_prompt.find_all_prompt_files(self.root)
        prompts = []
        for path in all_prompt_paths:
            data = core_prompt.load_prompt_data(path)
            if data:
                # Pass `project=self` to the constructor
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
        # Pass `project=self` to the constructor
        return Prompt(path=filepath, data=data_from_content, project=self)
        
    def get_status_report(self) -> dict: # ... (implementation is correct)
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

    def lint(self) -> dict: # ... (implementation is correct)
        all_prompts=self.list_prompts();categories=["YAML Structure & Parsing","Schema: Required Keys","Schema: Data Types","Schema: Value Formats","Template: Jinja2 Syntax","Best Practices & Logic"]
        project_results={cat:{"errors":[],"warnings":[]} for cat in categories}
        for p in all_prompts:
            file_results=core_prompt.validate_prompt_file(p.path);relative_path=str(p.path.relative_to(self.root))
            for cat in categories:
                for e in file_results[cat]["errors"]:project_results[cat]["errors"].append((relative_path,e))
                for w in file_results[cat]["warnings"]:project_results[cat]["warnings"].append((relative_path,w))
        return project_results

    def index(self, method: str = "hybrid"): # ... (implementation is correct)
        prompt_paths=[p.path for p in self.list_prompts()];
        if not prompt_paths:raise ValueError("No prompts to index.")
        if method.lower()=='hybrid':hybrid.build_hybrid_index(prompt_paths,self.root)
        elif method.lower()=='splade':splade.build_splade_index(prompt_paths,self.root)
        else:raise ValueError(f"Invalid index method: '{method}'.")

    def search(self, query: str, method: str = "fuzzy", limit: int = 10, **kwargs) -> list[dict]: # ... (implementation is correct)
        if method.lower()=='fuzzy':return fuzzy.search_fuzzy(query,self.list_prompts(),limit=limit)
        elif method.lower()=='hybrid':return hybrid.search_hybrid(query,limit,self.root,alpha=kwargs.get('alpha',0.5))
        elif method.lower()=='splade':return splade.search_with_splade(query,limit,self.root)
        else:raise ValueError(f"Invalid search method: '{method}'.")

    def __repr__(self) -> str: return f"<Project root='{self.root}'>"