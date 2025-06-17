import typer
import copy
from lunr import lunr
import re
from rich.syntax import Syntax
import json
from rich.text import Text
from rich.tree import Tree 
from rich.columns import Columns
import jinja2
from rich import print
from jinja2 import meta
# from jinja2 import Template
from rich.markup import escape
from rich.panel import Panel
from pathlib import Path
import hashlib
import tomli
import tomli_w
import yaml  # PyYAML, which we added as a dependency
from string import Template # For simple variable substitution
from datetime import datetime, timezone
import subprocess # To get git author info
import uuid
from rich.table import Table
from packaging.version import Version # A robust version parsing tool
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
import tomli
import tomli_w
import os
import sqlite3
# Whoosh for lightweight search
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser
import zoneinfo
import textwrap


# -- Main Application Object --
app = typer.Typer(
    help="A framework to secure, manage, and develop prompts.",
    add_completion=False
)


# --- Helper Functions ---
def _get_project_root() -> Path | None:
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

def _get_jinja_env() -> jinja2.Environment:
    """
    Creates and returns a Jinja2 environment configured for ${...} variables.
    """
    # Create a custom environment and change the variable delimiters
    env = jinja2.Environment(
        variable_start_string="${",
        variable_end_string="}",
        # We keep the comment syntax standard for Jinja2
        comment_start_string="{#",
        comment_end_string="#}"
    )
    return env

def _generate_prompt_file_content(
    prompt_id: str,
    name: str,
    version: str,
    description: str,
    namespace: list[str],
    tags: list[str],
    author: str,
    last_update: str,
    intended_model: str,
    notes: str,
    default_template_str: str  # The new parameter
) -> str:
    """
    Generates the complete, formatted string content for a new prompt YAML file.
    """
    # Format lists into YAML-compatible strings
    namespace_str = f"[{', '.join(namespace)}]" if namespace else "[]"
    tags_str = f"[{', '.join(tags)}]" if tags else "[]"

    # Indent the template block correctly for clean YAML output
    indented_template = textwrap.indent(default_template_str, '  ')

    # Use a triple-quoted f-string to build the exact file layout.
    file_content = f"""\
# PROMPT IDENTITY
# --------------------
id: {prompt_id}
name: "{name}"             # Your prompt's unique, short name.
version: "{version}"     # Let's start with version 1.0.0!

# DOCUMENTATION
# --------------------
description: "{description}"      # What does this prompt do? (e.g., "Summarizes user feedback into bullet points.")
namespace: {namespace_str}        # Where does this prompt live? (e.g., [Support, Tickets])
tags: {tags_str}             # Add some search tags! (e.g., [Summarization, Triage])

# OWNERSHIP & STATUS
# --------------------
status: "Draft"      # Current stage: Draft, In-Review, Production, etc.
author: "{author}"           # Who created this? (e.g., your-name@email.com)
last_update: "{last_update}"      # When was this last edited?

# CONFIGURATION
# --------------------
intended_model: "{intended_model}"   # Which AI model is this for? (e.g., anthropic/claude-3-opus)
model_parameters:
  temperature: 0.7   # You can set model params like temperature, top_p, etc. here.

# NOTES & LINKS
# --------------------
linked_prompts: []   # Any other prompts related to this one?
notes: "{notes}"

# - - - 💖 THE PROMPT 💖 - - - 
# ---------------------------------
# NOTE - Comments inside the prompt are automatically removed on prompt call. 
default_inputs:
  user_input: "Sample Input"

template: |
{indented_template}
"""
    return file_content

def _calculate_sha256(file_path: Path) -> str:
    # ... (this function is unchanged)
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def _check_lock_integrity(prompt_path: Path, project_root: Path) -> bool:
    """Checks a single prompt's integrity if it is locked. Returns True if OK."""
    lockfile_path = project_root / ".plb.lock"
    with open(lockfile_path, "rb") as f:
        lock_data = tomli.load(f)
    
    locked_prompts = lock_data.get("locked_prompts", {})
    relative_path_str = str(prompt_path.relative_to(project_root))

    if relative_path_str in locked_prompts:
        # MODIFICATION: Read the hash from the dictionary value
        lock_info = locked_prompts[relative_path_str]
        stored_hash = lock_info.get("hash", "").split(":")[-1]
        
        current_hash = _calculate_sha256(prompt_path)
        if stored_hash != current_hash:
            print(f"❌ [bold red]Integrity Check FAILED:[/bold red] '{escape(relative_path_str)}' has been modified since it was locked.")
            return False
    return True

def _get_git_author() -> str | None:
    """Tries to get the current user's name and email from git config."""
    try:
        name = subprocess.check_output(["git", "config", "user.name"], text=True).strip()
        email = subprocess.check_output(["git", "config", "user.email"], text=True).strip()
        return f"{name} <{email}>"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

def _find_next_template_index(prompts_dir: Path) -> int:
    """Finds the highest index for existing prompt_template_*.yml files."""
    if not prompts_dir.exists():
        return 1
    
    highest_index = 0
    for f in prompts_dir.glob("prompt_template_*.yml"):
        try:
            index_str = f.stem.split('_')[-1]
            index = int(index_str)
            if index > highest_index:
                highest_index = index
        except (ValueError, IndexError):
            continue
    return highest_index + 1

def _find_prompt_file(
    project_root: Path,
    name: str | None = None,
    id: str | None = None,
    path: Path | None = None
) -> Path | None:
    """Scans the prompts/ directory to find a prompt file by name, id, or path."""
    prompts_dir = project_root / "prompts"
    if not prompts_dir.exists():
        return None

    if path:
        return path if path.exists() else None

    candidate_files = []
    for f in prompts_dir.glob("**/*.yml"):
        try:
            with open(f, "r", encoding="utf-8") as prompt_file:
                data = yaml.safe_load(prompt_file) or {}
            
            if id and data.get("id") == id:
                return f
            
            if name and data.get("name") == name:
                candidate_files.append((f, data.get("version", "0.0.0")))
        except (yaml.YAMLError, IOError):
            continue
    
    if not candidate_files:
        return None

    candidate_files.sort(key=lambda x: Version(x[1]), reverse=True)
    return candidate_files[0][0]


def _validate_prompt_file(file_path: Path) -> dict:
    """
    Validates a single prompt file against a strict schema and best practices.
    
    Returns a dictionary of results categorized by check type.
    """
    # --- CHANGE 1: Define categories and initialize a structured result object ---
    categories = [
        "YAML Structure & Parsing",
        "Schema: Required Keys",
        "Schema: Data Types",
        "Schema: Value Formats",
        "Template: Jinja2 Syntax",
        "Best Practices & Logic",
    ]
    results = {cat: {"errors": [], "warnings": []} for cat in categories}

    # --- Rule Groups ---
    REQUIRED_KEYS = {
        "id", "name", "version", "description", "namespace", "tags",
        "status", "author", "last_update", "intended_model",
        "model_parameters", "linked_prompts", "notes", "default_inputs", "template"
    }
    EXPECTED_TYPES = {"namespace": list, "tags": list, "model_parameters": dict, "linked_prompts": list, "default_inputs": dict}
    VALID_STATUSES = {"Draft", "In-Review", "Staging", "Production", "Deprecated", "Archived"}

    # --- CHECK 1: YAML Validity ---
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        results["YAML Structure & Parsing"]["errors"].append(f"Invalid YAML format: {e}")
        return results # Fatal error for this file, return immediately.

    # --- CHECK 2: Required Keys ---
    missing_keys = REQUIRED_KEYS - set(data.keys())
    for key in sorted(list(missing_keys)):
        results["Schema: Required Keys"]["errors"].append(f"Missing required key: '{key}'")

    # --- CHECK 3: Data Types ---
    for key, expected_type in EXPECTED_TYPES.items():
        if key in data and data.get(key) is not None and not isinstance(data[key], expected_type):
            results["Schema: Data Types"]["errors"].append(f"Key '{key}' must be a {expected_type.__name__}, but found {type(data[key]).__name__}.")

    # --- CHECK 4: Value Formats ---
    if "version" in data and data.get("version"):
        try:
            from packaging.version import Version, InvalidVersion
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

    # --- CHECK 5: Jinja2 Syntax ---
    if "template" in data and data.get("template"):
        try:
            env = _get_jinja_env()
            env.parse(data["template"])
        except jinja2.exceptions.TemplateSyntaxError as e:
            results["Template: Jinja2 Syntax"]["errors"].append(f"Invalid Jinja2 syntax in 'template': {e}")
            
    # --- CHECK 6: Best Practices & Logic ---
    if "template" in data and data.get("template") and "default_inputs" in data and data.get("default_inputs"):
        try:
            env = _get_jinja_env()
            ast = env.parse(data["template"])
            template_vars = meta.find_undeclared_variables(ast)
            for key in data["default_inputs"]:
                if key not in template_vars:
                    results["Best Practices & Logic"]["warnings"].append(f"Default input '{key}' is defined but not used in the template.")
        except jinja2.exceptions.TemplateSyntaxError:
            pass # Error is already caught by the syntax check

    return results

def _get_config(project_root: Path) -> dict:
    """Reads the plb.toml file from the project root."""
    config_path = project_root / "plb.toml"
    if not config_path.exists():
        return {}
    with open(config_path, "rb") as f:
        return tomli.load(f)

def _write_config(config: dict, project_root: Path):
    """Writes the given dictionary to plb.toml in the project root."""
    config_path = project_root / "plb.toml"
    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)


def _get_progress_bar():
    """Returns a pre-configured rich Progress object."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        transient=False  # Keep the "completed" message
    )

def _resolve_prompt_path(project_root: Path, identifier: str) -> Path | None:
    """
    Finds a prompt file's full path from a user-provided identifier.

    The identifier can be:
    1. A direct file path (e.g., 'prompts/my-prompt.v1.yml').
    2. A short name (e.g., 'my-prompt'), in which case it finds the latest version.

    Returns the resolved Path object or None if not found.
    """
    # Case 1: The identifier is a direct, existing file path.
    direct_path = Path(identifier)
    if direct_path.is_file():
        return direct_path.resolve()

    # Case 2: The identifier is a short name. We need to find matching files.
    prompts_dir = project_root / "prompts"
    if not prompts_dir.exists():
        return None

    candidate_files = []
    for f in prompts_dir.glob(f"{identifier}*.yml"):
        try:
            # We need to extract the version to sort correctly.
            # Example: "my-prompt.v1.2.3.yml" -> "1.2.3"
            version_str = f.stem.split('.v')[-1]
            candidate_files.append((f, Version(version_str)))
        except (InvalidVersion, IndexError):
            # Ignore files that don't match the "name.vA.B.C.yml" pattern
            continue
    
    if not candidate_files:
        print(f"❌ [bold red]Error:[/bold red] No prompt found matching identifier '{identifier}'.")
        return None

    # Sort candidates by version, descending (latest first)
    candidate_files.sort(key=lambda x: x[1], reverse=True)
    
    latest_path, latest_version = candidate_files[0]

    # If there was more than one version, inform the user we picked the latest.
    if len(candidate_files) > 1:
        print(f"🟡 [dim]Warning: Multiple versions found for '{identifier}'. Using latest: [bold]{latest_version}[/bold].[/dim]")
        print(f"   [dim]To target a specific version, use the full filename.[/dim]")

    return latest_path

def _build_hybrid_index(prompt_files: list[Path], project_root: Path):
    """
    Builds a complete hybrid search index using TF-IDF (sparse) and
    SentenceTransformers + FAISS (dense), stored in SQLite and flat files.
    """
    print("[dim]Loading dependencies for hybrid search index...[/dim]")
    try:
        import pickle
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        import scipy.sparse
    except ImportError as e:
        print(f"❌ [bold red]Missing Libraries for Hybrid Search:[/bold red] {e}")
        print("   Please run: [bold]pip install sentence-transformers faiss-cpu scikit-learn[/bold]")
        raise typer.Exit(code=1)

    print("🚀 [bold cyan]Building Hybrid Search Index (TF-IDF + FAISS)...[/bold cyan]")
    
    # Define file paths for our index components
    index_dir = project_root / ".plb"
    index_dir.mkdir(exist_ok=True)
    db_path = index_dir / "prompts.db"
    faiss_path = index_dir / "prompts.faiss"
    vectorizer_path = index_dir / "tfidf.pkl"
    
    # Clean up old index files
    for path in [db_path, faiss_path, vectorizer_path]:
        if path.exists():
            path.unlink()

    print("[dim]Initializing sentence transformer model...[/dim]")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embedding_size = model.get_sentence_embedding_dimension()

    # --- 1. Data Collection Phase ---
    # We'll collect all data first to process it in batches, which is more efficient.
    prompts_data = []
    sparse_corpus = []  # Documents for TF-IDF (full text)
    dense_corpus = []   # Documents for embeddings (metadata only)

    print("[dim]Reading and preparing prompt files...[/dim]")
    for i, p_file in enumerate(prompt_files):
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            # Prepare data for sparse (keyword) indexing
            sparse_text = " ".join([
                data.get('name', ''), data.get('description', ''),
                " ".join(data.get('namespace', [])), " ".join(data.get('tags', [])),
                data.get('template', '')
            ])
            sparse_corpus.append(sparse_text)

            # Prepare data for dense (semantic) indexing
            dense_text = f"Name: {data.get('name', '')}. Description: {data.get('description', '')}. Tags: {', '.join(data.get('tags', []))}"
            dense_corpus.append(dense_text)
            
            # Store metadata for our database
            prompts_data.append({
                'faiss_idx': i,
                'path': str(p_file.relative_to(project_root)),
                'name': data.get('name', '[n/a]'),
                'description': data.get('description', '')
            })
        except (yaml.YAMLError, IOError):
            continue
    
    if not prompts_data:
        print("❌ [bold red]No valid prompts found to index.[/bold red]")
        raise typer.Exit(code=1)

    # --- 2. Sparse Indexing (TF-IDF) ---
    print("🚀 [bold blue]Creating TF-IDF sparse index...[/bold blue]")
    vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(sparse_corpus)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # --- 3. Dense Indexing (FAISS) ---
    print("🚀 [bold green]Creating FAISS dense index...[/bold green]")
    with _get_progress_bar() as progress:
        task = progress.add_task("[green]Generating embeddings...", total=len(dense_corpus))
        embeddings = np.empty((0, embedding_size), dtype='float32')
        for i in range(0, len(dense_corpus), 32): # Process in batches of 32 for efficiency
            batch = dense_corpus[i:i+32]
            batch_embeddings = model.encode(batch, convert_to_numpy=True)
            embeddings = np.vstack([embeddings, batch_embeddings])
            progress.update(task, advance=len(batch))

    faiss.normalize_L2(embeddings) # Normalize for cosine similarity
    faiss_index = faiss.IndexFlatIP(embedding_size) # IP (Inner Product) is equivalent to cosine sim for normalized vectors
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(faiss_path))

    # --- 4. Storing Metadata and Sparse Vectors in SQLite ---
    print("🚀 [bold purple]Storing metadata and indexes in SQLite DB...[/bold purple]")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE prompts (
                id INTEGER PRIMARY KEY,
                faiss_idx INTEGER UNIQUE NOT NULL,
                path TEXT NOT NULL,
                name TEXT,
                description TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE sparse_vectors (
                prompt_id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            );
        """)
        
        # Insert metadata
        for p_data in prompts_data:
            cursor.execute(
                "INSERT INTO prompts (faiss_idx, path, name, description) VALUES (?, ?, ?, ?)",
                (p_data['faiss_idx'], p_data['path'], p_data['name'], p_data['description'])
            )
            prompt_id = cursor.lastrowid
            
            # Serialize and insert the corresponding sparse vector
            sparse_vector = tfidf_matrix[p_data['faiss_idx']]
            vector_blob = pickle.dumps(sparse_vector)
            cursor.execute("INSERT INTO sparse_vectors (prompt_id, vector) VALUES (?, ?)", (prompt_id, vector_blob))


def _search_hybrid(query: str, limit: int, project_root: Path, alpha: float = 0.7):
    """
    Performs a hybrid search and fuses the results.
    `alpha` controls the weight of dense (semantic) vs. sparse (keyword) search.
    """
    print("[dim]Loading dependencies for hybrid search...[/dim]")
    try:
        import pickle
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import scipy.sparse
    except ImportError as e:
        print(f"❌ [bold red]Missing Libraries for Hybrid Search:[/bold red] {e}")
        print("   Please run: [bold]pip install sentence-transformers faiss-cpu scikit-learn[/bold]")
        raise typer.Exit(code=1)
    print("🚀 [bold cyan]Performing Hybrid Search (TF-IDF + FAISS)...[/bold cyan]")
    index_dir = project_root / ".plb"
    db_path = index_dir / "prompts.db"
    faiss_path = index_dir / "prompts.faiss"
    vectorizer_path = index_dir / "tfidf.pkl"
    
    if not all(p.exists() for p in [db_path, faiss_path, vectorizer_path]):
        print("❌ [bold red]Search index is missing or incomplete.[/bold red]")
        print("   Please run `plb index` to build it.")
        raise typer.Exit(code=1)

    # --- 1. Load all index components ---
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    faiss_index = faiss.read_index(str(faiss_path))
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    # --- 2. Generate query vectors ---
    query_dense = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_dense)
    query_sparse = vectorizer.transform([query])

    # --- 3. Perform Searches ---
    # Dense search (FAISS)
    k_dense = min(limit * 3, faiss_index.ntotal) # Search for more results to give the fusion more to work with
    distances, faiss_indices = faiss_index.search(query_dense, k_dense)
    
    # Sparse search (TF-IDF)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sparse_rows = conn.execute("SELECT prompt_id, vector FROM sparse_vectors").fetchall()
    
    doc_ids = [row['prompt_id'] for row in sparse_rows]
    doc_vectors = [pickle.loads(row['vector']) for row in sparse_rows]
    all_doc_sparse = scipy.sparse.vstack(doc_vectors)
    
    similarities = cosine_similarity(query_sparse, all_doc_sparse)[0]
    
    # --- 4. Fuse Results ---
    dense_scores = {faiss_idx: score for faiss_idx, score in zip(faiss_indices[0], distances[0])}
    sparse_scores = {doc_id: score for doc_id, score in zip(doc_ids, similarities)}
    
    # Get mapping from faiss_idx to prompt_id
    with sqlite3.connect(db_path) as conn:
        faiss_to_prompt_id = dict(conn.execute("SELECT faiss_idx, id FROM prompts").fetchall())

    hybrid_scores = {}
    all_ids = set(doc_ids)
    
    for prompt_id in all_ids:
        # Find the faiss_idx for the current prompt_id
        current_faiss_idx = next((fi for fi, pi in faiss_to_prompt_id.items() if pi == prompt_id), None)

        dense_score = dense_scores.get(current_faiss_idx, 0.0)
        sparse_score = sparse_scores.get(prompt_id, 0.0)
        
        hybrid_scores[prompt_id] = (alpha * dense_score) + ((1 - alpha) * sparse_score)
        
    sorted_results = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    
    if not sorted_results:
        print("No results found.")
        return

    # --- 5. Fetch details and display ---
    top_ids = [item[0] for item in sorted_results]
    placeholders = ",".join("?" for _ in top_ids)
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        details_map = {row['id']: row for row in conn.execute(f"SELECT id, path, name, description FROM prompts WHERE id IN ({placeholders})", top_ids).fetchall()}

    table = Table(title=f"Top {len(sorted_results)} Hybrid Search Results", show_header=True, highlight=True)
    table.add_column("Score", style="magenta", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Path", style="cyan")
    table.add_column("Description", style="yellow")
    
    for prompt_id, score in sorted_results:
        details = details_map.get(prompt_id)
        if details:
            table.add_row(f"{score:.3f}", details['name'], details['path'], details['description'])
    
    print(table)


def _splade_encode(texts: list[str], tokenizer, model):
    """Encodes a batch of texts into SPLADE sparse vectors."""
    import torch 
    import torch.nn.functional as F
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
        out = model(**inputs).last_hidden_state
        # Use the CLS token representation and apply ReLU, as per SPLADE's method
        scores = F.relu(out[:, 0, :])
        return scores

def _build_splade_index(prompt_files: list[Path], project_root: Path):
    """Builds and saves a SPLADE index for all prompts."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        print(f"❌ [bold red]Missing Libraries for SPLADE Search:[/bold red] {e}")
        print("   Please run: [bold]pip install torch transformers[/bold]")
        raise typer.Exit(code=1)
    print("🚀 [bold magenta]Building SPLADE Sparse Vector Index...[/bold magenta]")
    
    index_dir = project_root / ".plb"
    index_dir.mkdir(exist_ok=True)
    vectors_path = index_dir / "splade_vectors.pt"
    metadata_path = index_dir / "splade_metadata.json"
    
    for path in [vectors_path, metadata_path]:
        if path.exists(): path.unlink()

    print("[dim]This may take a moment as the SPLADE model is loaded...[/dim]")
    model_name = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    prompts_data = []
    corpus = []
    for i, p_file in enumerate(prompt_files):
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            # Create a single text block for each prompt
            text_representation = " ".join([
                data.get('name', ''), data.get('description', ''),
                " ".join(data.get('tags', [])), data.get('template', '')
            ])
            corpus.append(text_representation)
            
            prompts_data.append({
                'index': i,
                'path': str(p_file.relative_to(project_root)),
                'name': data.get('name', '[n/a]'),
                'description': data.get('description', '')
            })
        except (yaml.YAMLError, IOError):
            continue
            
    if not prompts_data:
        print("❌ [bold red]No valid prompts found to index.[/bold red]")
        raise typer.Exit(code=1)

    all_vectors = []
    with _get_progress_bar() as progress:
        task = progress.add_task("[magenta]Encoding with SPLADE...", total=len(corpus))
        # Process in batches for efficiency
        for i in range(0, len(corpus), 16):
            batch = corpus[i:i+16]
            batch_vectors = _splade_encode(batch, tokenizer, model)
            all_vectors.append(batch_vectors)
            progress.update(task, advance=len(batch))
    
    doc_vectors_tensor = torch.cat(all_vectors, dim=0)

    print(f"\n[dim]Saving {doc_vectors_tensor.shape[0]} vectors to disk...[/dim]")
    torch.save(doc_vectors_tensor, vectors_path)
    with open(metadata_path, "w") as f:
        json.dump(prompts_data, f)

def _search_with_splade(query: str, limit: int, project_root: Path):
    """Searches for prompts using the pre-built SPLADE index."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        print(f"❌ [bold red]Missing Libraries for SPLADE Search:[/bold red] {e}")
        print("   Please run: [bold]pip install torch transformers[/bold]")
        raise typer.Exit(code=1)
    index_dir = project_root / ".plb"
    vectors_path = index_dir / "splade_vectors.pt"
    metadata_path = index_dir / "splade_metadata.json"
    
    if not all(p.exists() for p in [vectors_path, metadata_path]):
        print("❌ [bold red]SPLADE search index is missing.[/bold red]")
        print("   Please run `plb index --method=splade` to build it.")
        raise typer.Exit(code=1)
        
    print("[dim]Loading SPLADE model and index...[/dim]")
    model_name = "naver/splade-cocondenser-ensembledistil"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    doc_vectors = torch.load(vectors_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Encode query and perform search
    query_vector = _splade_encode([query], tokenizer, model)
    scores = torch.matmul(query_vector, doc_vectors.T)
    top_results = torch.topk(scores, k=min(limit, len(metadata)))
    
    table = Table(title=f"Top {len(top_results.indices[0])} SPLADE Search Results", show_header=True, highlight=True)
    table.add_column("Score", style="magenta", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Path", style="cyan")
    table.add_column("Description", style="yellow")

    for i in range(len(top_results.indices[0])):
        score = top_results.values[0][i].item()
        doc_index = top_results.indices[0][i].item()
        details = metadata[doc_index]
        table.add_row(f"{score:.2f}", details['name'], details['path'], details['description'])
    
    print(table)


# --- CLI Commands (init, lock, unlock, verify are unchanged) ---
@app.command(rich_help_panel="[italic grey100]Project & Integrity[/italic grey100]")
def init(path: str = typer.Argument(".", help="The directory to initialize PromptLockbox in.")):
    """
    Initializes a PromptLockbox Project 
    """
    print(f"[bold green]Initializing PromptLockbox in '{path}'...[/bold green] :rocket:")
    base_path = Path(path)
    base_path.mkdir(exist_ok=True)
    
    # Create directories
    (base_path / "prompts").mkdir(exist_ok=True)
    (base_path / ".plb").mkdir(exist_ok=True)
    (base_path / "prompts" / ".gitkeep").touch()
    print("  [cyan]✓[/cyan] Created [bold]prompts/[/bold] and [bold].plb/[/bold] directories.")

    # Create config files
    if not (base_path / "plb.toml").exists():
        # Create a more comprehensive default config
        default_config = """\
        # PromptLockbox Configuration

        [search]
        # The active search index. Can be "whoosh" or "vector".
        active_index = "whoosh"
        """
        (base_path / "plb.toml").write_text(default_config)
        print("  [cyan]✓[/cyan] Created [bold]plb.toml[/bold] config file.")

    if not (base_path / ".plb.lock").exists():
        (base_path / ".plb.lock").write_text("# Auto-generated lock file\n[locked_prompts]\n")
        print("  [cyan]✓[/cyan] Created [bold].plb.lock[/bold] file for integrity checks.")

    # --- MODIFIED PART: .gitignore handling ---
    gitignore = base_path / ".gitignore"
    entry_to_add = "\n# Ignore PromptLockbox internal directory\n.plb/\n"
    
    should_add_entry = not gitignore.exists() or ".plb/" not in gitignore.read_text()

    if should_add_entry:
        update_gitignore = typer.confirm(
            "May I add the '.plb/' directory to your .gitignore file?",
            default=True # Default to 'yes' for convenience
        )
        if update_gitignore:
            with gitignore.open("a", encoding="utf-8") as f:
                f.write(entry_to_add)
            print("  [cyan]✓[/cyan] Updated [bold].gitignore[/bold].")
        else:
            print("  [yellow]! Skipped .gitignore update. Please add '.plb/' manually.[/yellow]")

    print("\n[bold green]Initialization complete![/bold green]")


@app.command(rich_help_panel="[italic grey100]Project & Integrity[/italic grey100]")
def lock(
    prompt_identifier: str = typer.Argument(
        ...,
        help="Name or path of the prompt to lock, or 'all' to lock every prompt."
    )
):
    """Validate and lock prompts to ensure integrity"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)

    # --- NEW: "all" keyword logic ---
    if prompt_identifier.lower() == "all":
        prompts_dir = project_root / "prompts"
        all_files = list(prompts_dir.glob("**/*.yml"))
        
        if not all_files:
            print("🟡 [yellow]Warning:[/yellow] No prompt files found in the 'prompts/' directory. Nothing to lock.")
            raise typer.Exit()

        # --- CORRECTED VALIDATION LOGIC FOR BULK LOCK ---
        print(f"🔍 Validating all {len(all_files)} prompts before locking...")
        errors_by_file = {}
        for file in all_files:
            # 1. Get the structured results from the validator
            validation_results = _validate_prompt_file(file)
            
            # 2. Aggregate only the critical error messages into a single list
            file_errors = []
            for category_results in validation_results.values():
                file_errors.extend(category_results["errors"])
            
            # 3. If there were any errors, store them for this file
            if file_errors:
                errors_by_file[file] = file_errors
        
        if errors_by_file:
            print(f"\n❌ [bold red]Validation failed. Found issues in {len(errors_by_file)} file(s):[/bold red]")
            for file_path, error_list in errors_by_file.items():
                # The panel now correctly displays the list of error strings
                print(Panel("\n".join([f"  - {error}" for error in error_list]),
                            title=f"[cyan]{escape(str(file_path.relative_to(project_root)))}[/cyan]",
                            border_style="red", title_align="left"))
            print("\nPlease fix the errors above or run `plb lint` for a full report. Aborting lock operation.")
            raise typer.Exit(code=1)

        print("✅ [green]All prompts passed validation.[/green]")
        
        # ... (The rest of the bulk lock logic is unchanged and correct) ...
        typer.confirm(f"\nAre you sure you want to lock all {len(all_files)} prompts?", abort=True)
        lockfile_path = project_root / ".plb.lock"
        with open(lockfile_path, "rb") as f: lock_data = tomli.load(f)
        if "locked_prompts" not in lock_data: lock_data["locked_prompts"] = {}
        locked_count = 0
        for file in all_files:
            relative_prompt_path = str(file.relative_to(project_root))
            file_hash = _calculate_sha256(file)
            utc_now = datetime.now(timezone.utc)
            lock_entry = {"hash": f"sha256:{file_hash}", "timestamp": utc_now.isoformat()}
            lock_data["locked_prompts"][relative_prompt_path] = lock_entry
            locked_count += 1
        with open(lockfile_path, "wb") as f: tomli_w.dump(lock_data, f)
        print(f"\n🔒 [bold green]Successfully locked {locked_count} prompts.[/bold green]")
        raise typer.Exit()
        
    # --- Original logic for a single prompt ---
    target_path = _resolve_prompt_path(project_root, prompt_identifier)
    if not target_path:
        raise typer.Exit(code=1)

    # --- CORRECTED VALIDATION LOGIC FOR SINGLE LOCK ---
    print(f"🔍 Validating '{escape(str(target_path.relative_to(project_root)))}' before locking...")
    validation_results = _validate_prompt_file(target_path)
    all_errors = []
    for category_results in validation_results.values():
        all_errors.extend(category_results["errors"])

    if all_errors:
        panel_content = "\n".join([f"  - {error}" for error in all_errors])
        print(Panel(panel_content, 
                    title="[bold red]Validation Failed[/bold red]", 
                    subtitle="[dim]Cannot lock a file with schema violations.[/dim]", 
                    border_style="red"))
        raise typer.Exit(code=1)
    
    # ... (The rest of the single lock logic is unchanged and correct) ...
    print("✅ [green]Validation successful.[/green]")
    lockfile_path = project_root / ".plb.lock"
    relative_prompt_path = str(target_path.relative_to(project_root))
    with open(lockfile_path, "rb") as f: lock_data = tomli.load(f)
    is_already_locked = "locked_prompts" in lock_data and relative_prompt_path in lock_data["locked_prompts"]
    if is_already_locked:
        print(f"🟡 [yellow]Warning:[/yellow] '{escape(relative_prompt_path)}' is already locked.")
        re_lock = typer.confirm("Are you sure you want to update the lock to the current version?")
        if not re_lock: print("Aborted lock update."); raise typer.Exit()
    file_hash = _calculate_sha256(target_path)
    utc_now = datetime.now(timezone.utc)
    lock_entry = {"hash": f"sha256:{file_hash}", "timestamp": utc_now.isoformat()}
    if "locked_prompts" not in lock_data: lock_data["locked_prompts"] = {}
    lock_data["locked_prompts"][relative_prompt_path] = lock_entry
    with open(lockfile_path, "wb") as f: tomli_w.dump(lock_data, f)
    action_word = "Updated lock for" if is_already_locked else "Locked"
    print(f"\n🔒 [bold green]{action_word} '[/bold green]{escape(relative_prompt_path)}[bold green]' successfully.[/bold green]")
    
@app.command(rich_help_panel="[italic grey100]Project & Integrity[/italic grey100]")
def unlock(
    prompt_identifier: str = typer.Argument(
        ...,
        help="Name or path of the prompt to unlock, or 'all' to unlock every prompt."
    )
):
    """Unlock a previously locked prompts"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)

    lockfile_path = project_root / ".plb.lock"
    
    # --- NEW: "all" keyword logic ---
    if prompt_identifier.lower() == "all":
        with open(lockfile_path, "rb") as f:
            lock_data = tomli.load(f)
            
        if not lock_data.get("locked_prompts"):
            print("✅ [green]No prompts are currently locked. Nothing to unlock.[/green]")
            raise typer.Exit()

        typer.confirm(f"Are you sure you want to unlock ALL {len(lock_data['locked_prompts'])} prompts?", abort=True)
        
        lock_data["locked_prompts"] = {} # Clear the dictionary
        
        with open(lockfile_path, "wb") as f:
            tomli_w.dump(lock_data, f)
            
        print(f"\n🔓 [bold green]Successfully unlocked all prompts.[/bold green]")
        raise typer.Exit()

    # --- Original logic for a single prompt ---
    target_path = _resolve_prompt_path(project_root, prompt_identifier)
    if not target_path:
        raise typer.Exit(code=1)

    relative_prompt_path = str(target_path.relative_to(project_root))
    
    with open(lockfile_path, "rb") as f:
        lock_data = tomli.load(f)
    if "locked_prompts" not in lock_data or relative_prompt_path not in lock_data["locked_prompts"]:
        print(f"🔓 [yellow]Warning:[/yellow] '{escape(str(target_path.relative_to(project_root)))}' is not currently locked.")
        raise typer.Exit()
    
    typer.confirm(f"Are you sure you want to unlock '{escape(str(target_path.relative_to(project_root)))}'?", abort=True)
    del lock_data["locked_prompts"][relative_prompt_path]
    with open(lockfile_path, "wb") as f:
        tomli_w.dump(lock_data, f)
    
    print(f"🔓 [bold green]Unlocked '[/bold green]{escape(str(target_path.relative_to(project_root)))}[bold green]' successfully.[/bold green]")

@app.command(rich_help_panel="[italic grey100]Project & Integrity[/italic grey100]")
def status():
    """Displays lock status for prompts"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)

    prompts_dir = project_root / "prompts"
    if not prompts_dir.exists():
        print("🟡 [yellow]Warning:[/yellow] The 'prompts' directory was not found.")
        raise typer.Exit()

    prompt_files = sorted(list(prompts_dir.glob("**/*.yml")))
    if not prompt_files:
        print("✅ [green]The 'prompts' directory is empty. Nothing to show status for.[/green]")
        raise typer.Exit()

    lockfile_path = project_root / ".plb.lock"
    locked_prompts = {}
    if lockfile_path.exists():
        with open(lockfile_path, "rb") as f:
            lock_data = tomli.load(f)
        locked_prompts = lock_data.get("locked_prompts", {})

    table = Table(title="Prompt Lock Status", show_header=True, highlight=True, border_style="dim")
    table.add_column("Status", justify="center")
    # MODIFICATION 1: Changed column header for clarity
    table.add_column("Prompt File", style="white", no_wrap=True)
    table.add_column("Locked At (IST)", style="yellow")
    
    try:
        ist_tz = zoneinfo.ZoneInfo("Asia/Kolkata")
    except zoneinfo.ZoneInfoNotFoundError:
        print("[bold red]Error:[/bold red] Timezone data not found. Please install the `tzdata` package:")
        print("   [bold cyan]pip install tzdata[/bold cyan]")
        raise typer.Exit(code=1)

    for prompt_file in prompt_files:
        # We still need the relative path for the lookup
        relative_path_str = str(prompt_file.relative_to(project_root))
        # MODIFICATION 2: Get just the filename for display
        display_name = prompt_file.name

        if relative_path_str in locked_prompts:
            lock_info = locked_prompts[relative_path_str]
            timestamp_str = lock_info.get("timestamp")
            
            if timestamp_str:
                dt_utc = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                dt_ist = dt_utc.astimezone(ist_tz)
                date_formatted = dt_ist.strftime("%d/%m/%y")
                time_formatted = dt_ist.strftime("%I:%M %p").lower()
                if time_formatted.startswith('0'):
                    time_formatted = time_formatted[1:]
                locked_at_display = f"[light_cyan1]{date_formatted} at {time_formatted}[/light_cyan1]"
            else:
                locked_at_display = "[dim]Timestamp not available[/dim]"
            
            # MODIFICATION 3: Use the display_name in the table row
            table.add_row("[bright_white]Locked[/bright_white]", display_name, locked_at_display)
        else:
            # MODIFICATION 4: Use the display_name in the table row
            table.add_row("[bright_white]Unlocked[/bright_white]", display_name, "[dim]--[/dim]")

    print(table)

@app.command(rich_help_panel="[italic grey100]Project & Integrity[/italic grey100]")
def verify():
    """Verifies the integrity of locked prompts."""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)
        
    print("🛡️  [bold]Verifying integrity of all locked prompts...[/bold]")
    lockfile_path = project_root / ".plb.lock"
    
    if not lockfile_path.exists():
        print("✅ [green]No lock file found. Nothing to verify.[/green]")
        raise typer.Exit()
        
    with open(lockfile_path, "rb") as f:
        lock_data = tomli.load(f)
        
    locked_prompts = lock_data.get("locked_prompts", {})
    if not locked_prompts:
        print("✅ [green]No prompts are currently locked. Nothing to verify.[/green]")
        raise typer.Exit()
        
    all_ok = True
    # --- The key change is in this loop ---
    for relative_path, lock_info in locked_prompts.items():
        full_path = project_root / relative_path
        if not full_path.exists():
            print(f"❌ [bold red]FAILED:[/bold red] '{escape(relative_path)}' is locked but the file is missing.")
            all_ok = False
            continue
            
        current_hash = _calculate_sha256(full_path)
        
        # --- THE FIX ---
        # 1. Get the hash string from the `lock_info` dictionary.
        # 2. Add a check to handle case where 'hash' key might be missing.
        stored_hash_string = lock_info.get("hash", "")
        if not stored_hash_string or ":" not in stored_hash_string:
            print(f"❌ [bold red]FAILED:[/bold red] Invalid lock entry for '{escape(relative_path)}'. Hash is missing or malformed.")
            all_ok = False
            continue

        # 3. Split the hash string to get the value.
        stored_hash_value = stored_hash_string.split(":")[-1]
        
        if current_hash == stored_hash_value:
            print(f"✅ [green]OK:[/green] '{escape(relative_path)}'")
        else:
            print(f"❌ [bold red]FAILED:[/bold red] '{escape(relative_path)}' has been modified since it was locked.")
            all_ok = False
            
    if not all_ok:
        print("\n[bold red]Verification failed. One or more locked files have been tampered with or are missing.[/bold red]")
        raise typer.Exit(code=1)
    else:
        print("\n[bold green]✅ All locked files verified successfully.[/bold green]")

@app.command(name="lint", rich_help_panel="[italic grey100]Project & Integrity[/italic grey100]")
def lint_prompts():
    """Validates all prompts for compliances and consistency"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)
    
    prompts_dir = project_root / "prompts"
    if not prompts_dir.exists():
        print("🟡 [yellow]Warning:[/yellow] The 'prompts' directory was not found.")
        raise typer.Exit()
        
    prompt_files = sorted(list(prompts_dir.glob("**/*.yml")))
    if not prompt_files:
        print("✅ [green]The 'prompts' directory is empty. Nothing to lint.[/green]")
        raise typer.Exit()
        
    print(f"🔍 [bold]Scanning {len(prompt_files)} prompt file(s)...[/bold]\n")

    # --- CHANGE 1: Initialize a master aggregator for all results ---
    categories = [
        "YAML Structure & Parsing", "Schema: Required Keys", "Schema: Data Types",
        "Schema: Value Formats", "Template: Jinja2 Syntax", "Best Practices & Logic",
    ]
    project_results = {cat: {"errors": [], "warnings": []} for cat in categories}
    
    # --- CHANGE 2: Aggregate results from all files ---
    for prompt_file in prompt_files:
        file_results = _validate_prompt_file(prompt_file)
        relative_path = str(prompt_file.relative_to(project_root))
        for cat in categories:
            for error in file_results[cat]["errors"]:
                project_results[cat]["errors"].append((relative_path, error))
            for warning in file_results[cat]["warnings"]:
                project_results[cat]["warnings"].append((relative_path, warning))

    # --- CHANGE 3: Display the new summary table ---
    summary_table = Table(title="Linting Report Summary", show_header=True, highlight=True, border_style="dim")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Checklist", style="bright_white", no_wrap=True)
    summary_table.add_column("Details", style="white")
    
    total_errors = 0
    total_warnings = 0

    for cat in categories:
        errors = project_results[cat]["errors"]
        warnings = project_results[cat]["warnings"]
        total_errors += len(errors)
        total_warnings += len(warnings)

        status_emoji = "✅"
        status_style = "green"
        if errors:
            status_emoji = "❌"
            status_style = "red"
        elif warnings:
            status_emoji = "☢️"
            status_style = "yellow"

        details = f"[white]{len(errors)}[/white] errors, [white]{len(warnings)} warnings[/white]"
        summary_table.add_row(f"[{status_style}]{status_emoji}[/]", cat, details)
        
    print(summary_table)

    # --- CHANGE 4: Display detailed breakdown if there are issues ---
    if total_errors == 0 and total_warnings == 0:
        print("\n[bold green]🎉 Awesome! All checks passed successfully.[/bold green]")
        raise typer.Exit()

    print("\n" + "-"*50)
    print("[bold]Detailed Report:[/bold]")
    for cat in categories:
        errors = project_results[cat]["errors"]
        warnings = project_results[cat]["warnings"]
        
        if not errors and not warnings:
            continue

        panel_content = ""
        if errors:
            panel_content += "[red]Errors:[/red]\n"
            for path, msg in errors:
                panel_content += f"  - [cyan]{escape(path)}[/cyan]: {escape(msg)}\n"
        if warnings:
            panel_content += "\n[yellow]Warnings:[/yellow]\n" if errors else "[yellow]Warnings:[/yellow]\n"
            for path, msg in warnings:
                panel_content += f"  - [cyan]{escape(path)}[/cyan]: {escape(msg)}\n"
                
        border_style = "red" if errors else "yellow"
        print(Panel(panel_content.strip(), title=f"[bold]{cat}[/bold]", border_style=border_style, title_align="left", expand=False))
        
    print("-" * 50)

    if total_errors > 0:
        print(f"\n[bold red]Linting failed with {total_errors} critical error(s).[/bold red] Please fix the issues marked with ❌.")
        raise typer.Exit(code=1)
    else:
        print(f"\n[bold green]Linting passed with {total_warnings} warning(s).[/bold green] Consider addressing the items marked with ☢️.")


# -- - Prompt Management Commands --- -
@app.command(rich_help_panel="[italic grey100]Prompt Management[/italic grey100]")
def create(bulk: int = typer.Option(None, "--bulk", "-b", help="Create a specified number of blank prompt files non-interactively.")):
    """Interactively create a new prompt file, or in bulk"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)
        
    prompts_dir = project_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Define the exact template content here.
#     default_template_string = """\
# \nYou are a helpful assistant.

# -- Now you can start writing your prompt template! --

# How to use this template:
# - To input a value: ${user_input}
# - To add a comment: {# This is a comment #}

# Create awesome prompts! :)
# """
#     default_template_string = """\
# You are a helpful assistant.

# -- Now you can start writing your prompt template! --

# How to use this template:
# - To input a value: ${user_input}
# - To add a comment: {# This is a comment #}

# Create awesome prompts! :)
# """

    default_template_string = """\
You are a helpful assistant.

-- Now you can start writing your prompt template! --

How to use this template:
- To input a value: ${user_input}
- To add a comment: {# This is a comment #}

Create awesome prompts! :)"""

    if bulk is not None:
        # --- BULK CREATION MODE ---
        if bulk <= 0:
            print("[bold red]Error:[/bold red] Number of files to create must be a positive integer.")
            raise typer.Exit(code=1)
            
        print(Panel(f"[bold yellow]Bulk Creating {bulk} Prompt Template(s)[/bold yellow] 🪄", border_style="cyan", expand=False))
        created_files = []
        start_index = _find_next_template_index(prompts_dir)
        end_index = start_index + bulk
        
        for i in range(start_index, end_index):
            content = _generate_prompt_file_content(
                prompt_id=f"prm_{uuid.uuid4().hex}",
                name="", version="1.0.0", description="", namespace=[], tags=[],
                author=_get_git_author(), last_update=datetime.now(timezone.utc).isoformat(),
                intended_model="", notes="Your notes here!",
                default_template_str=default_template_string
            )
            filename = f"prompt_template_{i}.v1.0.0.yml"
            filepath = prompts_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            created_files.append(filepath)
            
        print("\n" + "-" * 40); print(f"✅ [bold green]Done! Created {len(created_files)} blank prompt file(s).[/bold green]")
        for path in created_files: print(f"   - [cyan]{escape(str(path))}[/cyan]")
        print("\n[dim]Next step: Edit these files to define your prompts.[/dim]"); raise typer.Exit()

    # --- FULL INTERACTIVE CREATION MODE ---
    print(Panel("[bold yellow]Let's create a new prompt![/bold yellow] 🪄 ", border_style="cyan", expand=False))

    # --- Identity ---
    print("\n[bold]1. Prompt Name[/bold] [dim](required)[/dim]")
    print(" [italic]A unique, machine-friendly identifier.[/italic] [dim]e.g., summarize-ticket[/dim]")
    name = typer.prompt("✏️ ")

    print("\n[bold]2. Version[/bold]")
    print(" [italic]The starting semantic version.[/italic] [dim]e.g., 1.0.0[/dim]")
    version = typer.prompt("✏️ ", default="1.0.0")

    # --- Documentation & Organization ---
    print("\n[bold]3. Description[/bold] [dim](optional)[/dim]")
    print(" [italic]A short, human-readable summary of the prompt's purpose.[/italic]")
    description = typer.prompt("✏️ ", default="", show_default=False)

    print("\n[bold]4. Namespace[/bold] [dim](optional)[/dim]")
    print(" [italic]Hierarchical path to organize prompts, comma-separated.[/italic] [dim]e.g., 'Billing,Invoices'[/dim]")
    namespace_str = typer.prompt("✏️ ", default="", show_default=False)
    namespace = [n.strip() for n in namespace_str.split(',') if n.strip()]

    print("\n[bold]5. Tags[/bold] [dim](optional)[/dim]")
    print(" [italic]Keywords for discovery, comma-separated.[/italic] [dim]e.g., Summarization, Code-Gen[/dim]")
    tags_str = typer.prompt("✏️ ", default="", show_default=False)
    tags = [t.strip() for t in tags_str.split(',') if t.strip()]

    # --- Execution ---
    print("\n[bold]6. Intended Model[/bold] [dim](optional)[/dim]")
    print(" [italic]The specific LLM this prompt is designed for.[/italic] [dim]e.g., openai/gpt-4-turbo[/dim]")
    intended_model = typer.prompt("✏️ ", default="", show_default=False)

    print("\n[bold]7. Notes[/bold] [dim](optional)[/dim]")
    print(" [italic]Any extra comments, warnings, or usage instructions.[/italic]")
    notes = typer.prompt("✏️ ", default="", show_default=False)

    prompt_id = f"prm_{uuid.uuid4().hex}"
    
    file_content = _generate_prompt_file_content(
        prompt_id=f"prm_{uuid.uuid4().hex}", name=name, version=version, description=description,
        namespace=[n.strip() for n in namespace_str.split(',') if n.strip()],
        tags=[t.strip() for t in tags_str.split(',') if t.strip()], author=_get_git_author(),
        last_update=datetime.now(timezone.utc).isoformat(), intended_model=intended_model, notes=notes,
        default_template_str=default_template_string  # Pass the template string
    )
    
    filename = f"{name}.v{version}.yml"
    filepath = prompts_dir / filename
    
    if filepath.exists():
        print(f"\n[bold red]Error:[/bold red] File '{escape(str(filepath))}' already exists.")
        raise typer.Exit(code=1)
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(file_content)
        
    print("\n" + "-" * 40)
    print(f"✅ [bold green]Success! Created new prompt file at:[/bold green]")
    print(f"   [cyan]{escape(str(filepath))}[/cyan]")
    print("\n[dim]Next step: Open the file and start editing your new prompt![/dim]")


@app.command(rich_help_panel="[italic grey100]Prompt Management[/italic grey100]")
def run(
    prompt_identifier: str = typer.Argument(..., help="Name or path of the prompt to run."),
    vars: list[str] = typer.Option(None, "--var", "-v", help="Pre-set a template variable, e.g., --var name=Alex"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open an editor to fill in variables for multi-line inputs.", is_flag=True)
):
    """
    Renders a prompt with parameters
    """
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)
        
    target_path = _resolve_prompt_path(project_root, prompt_identifier)
    if not target_path:
        raise typer.Exit(code=1)

    if not _check_lock_integrity(target_path, project_root):
        raise typer.Exit(code=1)

    # 1. Load the YAML file
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[bold red]Error:[/bold red] Failed to parse YAML file '{escape(str(target_path))}': {e}")
        raise typer.Exit(code=1)

    if "template" not in prompt_data:
        print(f"[bold red]Error:[/bold red] Prompt file must contain a 'template' key.")
        raise typer.Exit(code=1)
    
    raw_template = prompt_data.get("template", "")
    
    # --- CHANGE 1: Use the Jinja2 environment and its metadata parser ---
    # This is more robust than regex and understands the full Jinja2 syntax.
    env = _get_jinja_env()
    try:
        ast = env.parse(raw_template)
        required_vars = sorted(list(meta.find_undeclared_variables(ast)))
    except jinja2.exceptions.TemplateSyntaxError as e:
        print(f"❌ [bold red]Template Error in '{escape(str(target_path))}':[/bold red] {e}")
        raise typer.Exit(code=1)
    
    # 3. Handle variables: start with defaults, then override with --var flags
    template_vars = prompt_data.get("default_inputs", {}) or {}
    if vars:
        for var in vars:
            if "=" not in var: continue
            key, value = var.split("=", 1)
            template_vars[key] = value

    # 4. Handle --edit or interactive prompting for any missing variables
    if edit:
        if not required_vars:
            print("✅ [green]Prompt has no variables. Nothing to edit.[/green]")
        else:
            vars_for_editor = {key: template_vars.get(key, "") for key in required_vars}
            editor_header = "# Please fill in the values for the prompt variables.\n# Save and close this file to continue.\n\n"
            editor_content_str = yaml.dump(vars_for_editor, sort_keys=False, indent=2, allow_unicode=True)
            edited_content = typer.edit(editor_header + editor_content_str, extension=".yml")
            if edited_content is None:
                print("Aborted.")
                raise typer.Exit()
            try:
                user_provided_vars = yaml.safe_load(edited_content) or {}
                if user_provided_vars:
                    template_vars.update(user_provided_vars)
            except yaml.YAMLError as e:
                print(f"❌ [bold red]Error:[/bold red] Invalid YAML format in your editor input: {e}")
                raise typer.Exit(code=1)
    else:
        vars_to_ask = [v for v in required_vars if v not in template_vars]
        if vars_to_ask:
            print("\n[bold]Enter your prompt inputs 📌:[/bold] [dim](Press Enter to skip)[/dim]")
            print("[bold cyan]Tip:[/bold cyan] [dim]For large, multi-line inputs, try running with the --edit flag.[/dim]\n")
            for var_name in vars_to_ask:
                prompt_text = Text.assemble((f"{var_name}", "bold cyan"))
                # Fill with a non-empty default if it exists, otherwise empty.
                default_val = template_vars.get(var_name, "")
                user_input = typer.prompt(prompt_text, default=default_val, show_default=False)
                # Only update if the user provided new input
                if user_input != default_val:
                    template_vars[var_name] = user_input

    # --- CHANGE 2: Render using the Jinja2 template object ---
    # This correctly processes comments, variables, and preserves formatting.
    template = env.from_string(raw_template)
    rendered_prompt = template.render(template_vars)
        
    print(Panel(
        rendered_prompt,
        title=f"Rendered Prompt: [cyan]{escape(str(target_path.relative_to(project_root)))}[/cyan]",
        border_style="blue",
        title_align="left",
        expand=False
    ))


@app.command(name="list", rich_help_panel="[italic grey100]Prompt Management[/italic grey100]")
def list_prompts(
    wide: bool = typer.Option(False, "--wide", "-w", help="Display more details in the output.", is_flag=True)
):
    """Lists all prompts in a table"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)
        
    prompts_dir = project_root / "prompts"
    if not prompts_dir.exists():
        print("🟡 [yellow]Warning:[/yellow] The 'prompts' directory was not found.")
        raise typer.Exit()
        
    prompt_files = sorted(list(prompts_dir.glob("**/*.yml")))
    if not prompt_files:
        print("✅ [green]The 'prompts' directory is empty. No prompts to list.[/green]")
        raise typer.Exit()

    table = Table(
        title="Prompt Lockbox Library",
        caption="To see all details for a prompt, run `plb show --name *prompt-name*`",
        expand=False, border_style="dim", show_header=True, highlight=True
    )

    # --- MODIFIED PART 1: Column Definitions ---
    
    # Core columns remain the same
    table.add_column("Name", justify="left", style="grey100", no_wrap=True)
    table.add_column("Version", style="light_cyan3")
    table.add_column("Status", justify="center", style="white")
    
    # If --wide is used, add the new columns
    if wide:
        table.add_column("Intended Model", style="white")
        table.add_column("Default Inputs", style="grey50")
        table.add_column("Last Update", style="white")
    
    table.add_column("Description", style="bright_white")

    for prompt_file in prompt_files:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # --- MODIFIED PART 2: Row Data Assembly ---
            
            # Prepare core data
            name = data.get("name", "[n/a]")
            version = data.get("version", "[n/a]")
            status = data.get("status", "N/A")
            description = data.get("description", "")
            
            row_data = [name, version, status]
            
            # If --wide, gather and append the extra data
            if wide:
                intended_model = data.get("intended_model", "")
                
                # Get the default input keys for a concise display
                defaults = data.get("default_inputs", {}) or {}
                defaults_display = ", ".join(defaults.keys()) if defaults else ""
                
                last_update_iso = data.get("last_update")
                last_update_display = ""
                if last_update_iso:
                    try:
                        dt_obj = datetime.fromisoformat(last_update_iso.replace("Z", "+00:00"))
                        last_update_display = dt_obj.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        last_update_display = "[dim]Invalid[/dim]"
                
                row_data.extend([intended_model, defaults_display, last_update_display])

            row_data.append(description)
            table.add_row(*row_data)

        except (yaml.YAMLError, Exception) as e:
            # Handle malformed files gracefully
            error_row = ["[red]Error in file[/red]", str(prompt_file.name), "[red]Invalid YAML[/red]"]
            if wide:
                # MODIFICATION 3: Update placeholders for the correct number of wide columns
                error_row.extend(["-", "-", "-"])
            error_row.append(f"[dim]{type(e).__name__}[/dim]")
            table.add_row(*error_row)

    print(table)

@app.command(name="version", rich_help_panel="[italic grey100]Prompt Management[/italic grey100]")
def version_prompt(
    prompt_identifier: str = typer.Argument(..., help="Name or path of the prompt to version."),
    major: bool = typer.Option(False, "--major", "-M", help="Perform a major version bump."),
    patch: bool = typer.Option(False, "--patch", "-P", help="Perform a patch version bump."),
):
    """Creates a new version of a prompt"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)

    source_path = _resolve_prompt_path(project_root, prompt_identifier)
    if not source_path:
        raise typer.Exit(code=1)

    if major and patch:
        print("❌ [bold red]Error:[/bold red] Cannot specify --major and --patch at the same time.")
        raise typer.Exit(code=1)
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"❌ [bold red]Error:[/bold red] Invalid YAML in source file '{escape(str(source_path))}': {e}")
        raise typer.Exit(code=1)
        
    # The rest of the logic is unchanged
    current_version_str = data.get("version")
    prompt_name = data.get("name")
    if not current_version_str or not prompt_name:
        print("❌ [bold red]Error:[/bold red] Source file must contain 'name' and 'version' keys.")
        raise typer.Exit(code=1)
    try:
        v = Version(current_version_str)
        if major: new_version = f"{v.major + 1}.0.0"
        elif patch: new_version = f"{v.major}.{v.minor}.{v.patch + 1}"
        else: new_version = f"{v.major}.{v.minor + 1}.0"
    except InvalidVersion:
        print(f"❌ [bold red]Error:[/bold red] Invalid version format '{current_version_str}' in source file.")
        raise typer.Exit(code=1)
    new_filename = f"{prompt_name}.v{new_version}.yml"
    new_filepath = source_path.parent / new_filename
    if new_filepath.exists():
        print(f"❌ [bold red]Error:[/bold red] A file for version {new_version} already exists at '{escape(str(new_filepath))}'.")
        raise typer.Exit(code=1)
    new_data = copy.deepcopy(data)
    new_data['version'] = new_version
    new_data['status'] = 'Draft'
    new_data['last_update'] = datetime.now(timezone.utc).isoformat()
    try:
        with open(new_filepath, "w", encoding="utf-8") as f:
            yaml.dump(new_data, f, sort_keys=False, indent=2, width=80)
    except Exception as e:
        print(f"❌ [bold red]Error:[/bold red] Could not write new file: {e}")
        raise typer.Exit(code=1)
    print(f"✅ [bold green]Success! Created new version at:[/bold green] [cyan]{escape(str(new_filepath))}[/cyan]")
    print("   [dim]Status has been reset to 'Draft'. You can now edit the new file.[/dim]")


@app.command(name="tree", rich_help_panel="[italic grey100]Prompt Management[/italic grey100]")
def tree_prompts():
    """Displays a hierarchical tree of prompts"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)
    prompts_dir = project_root / "prompts"
    if not prompts_dir.exists():
        print("🟡 [yellow]Warning:[/yellow] The 'prompts' directory was not found.")
        raise typer.Exit()
    prompt_files = sorted(list(prompts_dir.glob("**/*.yml")))
    if not prompt_files:
        print("✅ [green]The 'prompts' directory is empty. Nothing to display.[/green]")
        raise typer.Exit()
    namespace_tree = {}
    no_namespace_prompts = []
    invalid_files = []
    for prompt_file in prompt_files:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            namespace = data.get("namespace")
            if not namespace or not isinstance(namespace, list) or not namespace[0]:
                no_namespace_prompts.append(prompt_file.name)
                continue
            current_level = namespace_tree
            for part in namespace:
                current_level = current_level.setdefault(part, {})
            current_level.setdefault("_prompts_", []).append(prompt_file.name)
        except yaml.YAMLError:
            invalid_files.append(prompt_file.name)
    tree = Tree("🥡 [bold dark_slate_gray2]Prompt Library[/bold dark_slate_gray2]", guide_style="cyan")
    def _add_nodes_to_tree(tree_branch: Tree, data_dict: dict):
        sorted_keys = sorted(data_dict.keys(), key=lambda k: (k.startswith('_'), k))
        for key in sorted_keys:
            if key == "_prompts_":
                for prompt_name in sorted(data_dict[key]):
                    tree_branch.add(f"📄 {escape(prompt_name)}")
            else:
                new_branch = tree_branch.add(f"🗂 [bold]{escape(key)}[/bold]")
                _add_nodes_to_tree(new_branch, data_dict[key])
    _add_nodes_to_tree(tree, namespace_tree)
    for prompt_name in sorted(no_namespace_prompts):
        tree.add(f"📄 {escape(prompt_name)} [dim](No Namespace)[/dim]")
    if invalid_files:
        error_branch = tree.add("❌ [bold red]Invalid Files[/bold red]")
        for file_name in sorted(invalid_files):
            error_branch.add(f"📄 {escape(file_name)}")
    print("+" + "-"*60 + "+")
    print(tree)
    print("+" + "-"*60 + "+")

@app.command(name="show", rich_help_panel="[italic grey100]Prompt Management[/italic grey100]")
def show_prompt(
    identifier: str = typer.Argument(
        ...,
        help="The name, ID, or path of the prompt to display (e.g., 'my-prompt', 'prm_...', or 'prompts/my-prompt.v1.yml')."
    )
):
    """Display raw metadata and prompt"""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)

    # (ID and name resolution logic is unchanged)
    prompt_file = None
    if identifier.startswith("prm_"):
        prompts_dir = project_root / "prompts"
        for f in prompts_dir.glob("**/*.yml"):
            try:
                with open(f, "r", encoding="utf-8") as pf:
                    data = yaml.safe_load(pf) or {}
                if data.get("id") == identifier:
                    prompt_file = f
                    break
            except (yaml.YAMLError, IOError):
                continue
    if not prompt_file:
        prompt_file = _resolve_prompt_path(project_root, identifier)
    if not prompt_file:
        print(f"❌ [bold red]Error:[/bold red] Prompt not found with identifier '{identifier}'.")
        raise typer.Exit(code=1)

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"❌ [bold red]Error:[/bold red] Invalid YAML in file '{escape(str(prompt_file))}': {e}")
        raise typer.Exit(code=1)
    
    is_tampered = not _check_lock_integrity(prompt_file.resolve(), project_root)
    if is_tampered:
        print(Panel("🚨 [bold yellow]TAMPERING DETECTED[/bold yellow] 🚨\nThis file has been modified since it was locked.", style="bold red", border_style="red"))

    # --- MODIFIED PART: Safer format_line helper and its usage ---
    
    def format_line(label, value, style="default"):
        # The helper is now simpler. It only handles the text assembly.
        text = Text()
        text.append(f"{label:<18}", style="bold grey100")
        text.append(value, style=style)
        return text

    # Build the list of items to display, checking for empty values *before* calling format_line.
    metadata_items = []
    
    if val := data.get('id'): metadata_items.append(format_line("ID:", val, style="white"))
    if val := data.get('name'): metadata_items.append(format_line("Name:", val, style="bright_cyan"))
    if val := data.get('version'): metadata_items.append(format_line("Version:", val, style="white"))
    if val := data.get('status'): metadata_items.append(format_line("Status:", val, style="white"))
    if val := data.get('description'): metadata_items.append(format_line("Description:", val))
    if val := data.get('namespace'): metadata_items.append(format_line("Namespace:", ", ".join(val)))
    if val := data.get('tags'): metadata_items.append(format_line("Tags:", ", ".join(val), style="white"))
    if val := data.get('author'): metadata_items.append(format_line("Author:", val, style="white"))
    if val := data.get('last_update'): metadata_items.append(format_line("Last Update:", val, style="white"))
    if val := data.get('intended_model'): metadata_items.append(format_line("Intended Model:", val, style="white"))
    
    # Special handling for dictionaries
    if val := data.get('model_parameters'):
        pretty_dict = json.dumps(val, indent=2)
        indented_dict = "\n" + textwrap.indent(pretty_dict, ' ' * 18)
        metadata_items.append(format_line("Model Parameters:", indented_dict, style="white"))
    if val := data.get('default_inputs'):
        pretty_dict = json.dumps(val, indent=2)
        indented_dict = "\n" + textwrap.indent(pretty_dict, ' ' * 18)
        metadata_items.append(format_line("Default Inputs:", indented_dict, style="white"))
        
    if val := data.get('linked_prompts'): metadata_items.append(format_line("Linked Prompts:", ", ".join(val)))
    
    # THE CRITICAL FIX: Check if `notes` exists and is not None before indenting.
    if (notes_val := data.get('notes')) is not None:
        # Only indent and format if notes_val is a string (even an empty one).
        indented_notes = "\n" + textwrap.indent(notes_val, ' ' * 18)
        metadata_items.append(format_line("Notes:", indented_notes))

    metadata_renderable = Text("\n").join(metadata_items)
    
    print(Panel(
        metadata_renderable, 
        title=f"Metadata: [bold cyan]{escape(str(prompt_file.relative_to(project_root)))}[/bold cyan]",
        border_style="blue", 
        expand=False
    ))

    print(Panel(
        Text(data.get("template", ""), style="default"),
        title="[bold]Prompt Template[/bold]",
        border_style="green",
        expand=False
    ))



# --- Search & Indexing Commands ---
@app.command(rich_help_panel="[italic grey100]Search & Indexing[/italic grey100]")
def index(
    method: str = typer.Option(
        "hybrid", 
        "--method", 
        "-m",
        help="The search index to build: 'hybrid' (TF-IDF+FAISS) or 'splade'."
    )
):
    """Builds a search index for prompts"""
    project_root = _get_project_root()
    if not project_root:
        # ... error handling ...
        raise typer.Exit(code=1)
    
    prompts_dir = project_root / "prompts"
    if not prompts_dir.exists() or not any(prompts_dir.iterdir()):
        # ... error handling ...
        raise typer.Exit()
        
    prompt_files = list(prompts_dir.glob("**/*.yml"))
    
    if method.lower() == 'hybrid':
        _build_hybrid_index(prompt_files, project_root)
    elif method.lower() == 'splade':
        _build_splade_index(prompt_files, project_root)
    else:
        print(f"❌ [bold red]Error: Invalid method '{method}'. Choose 'hybrid' or 'splade'.[/bold red]")
        raise typer.Exit(code=1)
        
    print(f"\n✅ [bold green]Successfully built the '{method}' search index.[/bold green]")

# --- Create a new Typer app for the search subcommands ---
search_app = typer.Typer(
    name="search", 
    help="Search for prompts using different methods",
    no_args_is_help=True
)
app.add_typer(search_app, rich_help_panel="[italic grey100]Search & Indexing[/italic grey100]")

@search_app.command(name="hybrid")
def search_hybrid(
    query: str = typer.Argument(..., help="The natural language search query."),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results to return."),
    alpha: float = typer.Option(
        0.5, "--alpha", "-a",
        help="Balance between semantic (1.0) and keyword (0.0) search.",
        min=0.0, max=1.0,
    ),
):
    """Search using the Hybrid (TF-IDF + FAISS) engine."""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error: Not a PromptLockbox project.[/bold red]")
        raise typer.Exit(code=1)
    
    print(f"🔎 Performing [bold]Hybrid[/bold] search for: \"{escape(query)}\" (alpha={alpha})")
    _search_hybrid(query, limit, project_root, alpha=alpha)


@search_app.command(name="splade")
def search_splade(
    query: str = typer.Argument(..., help="The search query."),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results to return."),
):
    """Search using the powerful SPLADE sparse vector engine."""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error: Not a PromptLockbox project.[/bold red]")
        raise typer.Exit(code=1)

    print(f"🔎 Performing [bold]SPLADE[/bold] search for: \"{escape(query)}\"")
    _search_with_splade(query, limit, project_root)


config_app = typer.Typer(name="config", help="Manage search configuration and indexing", no_args_is_help=True)
app.add_typer(config_app, rich_help_panel="[italic grey100]Search & Indexing[/italic grey100]")


@config_app.command(name="status")
def config_status():
    """Shows the build status of the available search indexes."""
    project_root = _get_project_root()
    if not project_root:
        print("[bold red]Error:[/bold red] Not a PromptLockbox project. '.plb.toml' not found.")
        raise typer.Exit(code=1)

    index_dir = project_root / ".plb"

    # Define the files that constitute a "built" index for each method
    index_files = {
        "Hybrid (TF-IDF + FAISS)": [
            index_dir / "prompts.db",
            index_dir / "prompts.faiss",
            index_dir / "tfidf.pkl",
        ],
        "SPLADE": [
            index_dir / "splade_vectors.pt",
            index_dir / "splade_metadata.json",
        ],
    }
    
    table = Table(title="Search Index Status")
    table.add_column("Index Method", style="cyan", no_wrap=True)
    table.add_column("Status", style="yellow")
    
    for method_name, required_files in index_files.items():
        # An index is considered "built" only if ALL its required files exist.
        is_built = all(f.exists() for f in required_files)
        
        if is_built:
            status_text = "✅ Built"
            status_style = "green"
        else:
            status_text = "❌ Not Built"
            status_style = "red"
            
        table.add_row(method_name, f"[{status_style}]{status_text}[/]")

    print(table)
    print("\nTo build an index, run: [bold]plb index --method=<name>[/bold] (e.g., 'hybrid' or 'splade')")
    print("To search, run: [bold]plb search <method> \"your query\"[/bold] (e.g., `plb search hybrid \"...\"`)")

if __name__ == "__main__":
    app()

