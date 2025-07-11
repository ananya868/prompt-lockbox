<div align="center">
  <img src="docs/logo/logo.png" alt="Logo" width=700>  
</div>
<div align="center">
  <h5>Brings structure and reproducibility to your prompts</h5>
  <a href="https://badge.fury.io/py/prompt-lockbox"><img src="https://badge.fury.io/py/prompt-lockbox.svg?v=1?" alt="PyPI version"/></a><a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a><a href="https://pypi.org/project/prompt-lockbox/"><img src="https://img.shields.io/pypi/pyversions/prompt-lockbox.svg" alt="Python versions"/></a>

  A lightweight CLI toolkit and Python SDK to secure, manage, and develop your LLM prompts like reusable code.
</div>

<br></br>
📑 **Explore the full Documenentation here - [Docs](https://prompt-lockbox.mintlify.app)**
>
> *(This README provides an overview. The full site contains detailed guides, API references, and advanced tutorials.)*

<br/>

## Why use it ?

Managing prompts across a team or a large project can be chaotic. Plain text files lack versioning, are prone to accidental changes, and have no built-in quality control. **Prompt Lockbox** turns your prompts from fragile text files into versioned assets.

Here’s a summary of the common pain points it solves:

| The Pain Point                                                                       | The Prompt Lockbox Solution                                                                  | Key Command(s)                                   |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Scattered Prompts**<br/>Prompts are lost across text files, Slack, and Google Docs. | Provides a centralized, version-controlled library for all prompts.                          | `plb init`, `plb list`, `plb tree`               |
| **Untracked Changes**<br/>A great prompt is made worse with no easy way to revert.    | Treats prompts like code with built-in semantic versioning.                                  | `plb version`                                    |
| **Unsafe Edits**<br/>Critical production prompts are modified without approval.       | Locks production-ready prompts to prevent unauthorized edits and detects tampering.          | `plb lock`, `plb unlock`, `plb verify`           |
| **Hard to Discover**<br/>Teams waste time reinventing prompts that are hard to find.  | Makes the entire library instantly searchable with fuzzy, hybrid, and sparse-vector search. | `plb index`, `plb search <method>`               |
| **Poor Documentation**<br/>Writing and maintaining documentation is tedious and skipped. | Uses AI to automate documentation, generating descriptions and tags in seconds.                | `plb prompt document`                            |

## Features

*   **Integrity & Security:** Lock prompts to generate a checksum. The `plb verify` command ensures that production prompts haven't been tampered with since they were last approved.

*   **Version Control First:** Automatically create new, semantically versioned prompt files with `plb version`, making it easy to iterate and experiment safely without breaking existing implementations.

*   **AI features:**
    *   **Auto-Documentation:** `plb prompt document` uses an AI to analyze your prompt and generate a concise description and relevant search tags.
    *   **Auto-Improvement:** `plb prompt improve` provides an expert critique and a suggested, improved version of your prompt template, showing you a diff of the changes.
    *   **Direct Execution:** Execute prompts directly against your configured LLMs (OpenAI, Anthropic, Ollama, HuggingFace) right from the CLI.

*   **Advanced Search:** Don't just `grep`. Build a local search index (`hybrid` TF-IDF + FAISS or `splade`) to find the right prompt instantly using natural language queries.

*   **Flexible Configuration:** An interactive wizard (`plb configure-ai`) makes it trivial to set up any provider and model, securely storing API keys in a local `.env` file.

## Installation

### **Using pip:**

The base toolkit can be installed directly from PyPI:

```bash
pip install prompt-lockbox
```

To include support for specific AI providers or features, install the optional "extras". You can combine multiple extras in one command.

```bash
# To use OpenAI models
pip install 'prompt-lockbox[openai]'

# To use HuggingFace models (includes torch)
pip install 'prompt-lockbox[huggingface]'

# To use the advanced search features (includes faiss-cpu)
pip install 'prompt-lockbox[search]'

# To install everything
pip install 'prompt-lockbox[all]'
```

### **Using this repo:**
   
If you want to install the very latest development version directly from the repository, you can use `pip` with a Git URL. This is useful for testing upcoming features before they are released on PyPI.

```bash
# Install the latest version from the main branch
pip install git+https://github.com/ananya868/prompt-lockbox.git

# To include optional dependencies, you must add them as a comma-separated list
# Example: Installing with support for OpenAI and Search
pip install "prompt-lockbox[openai,search] @ git+https://github.com/ananya868/prompt-lockbox.git"
```

## Quickstart: Your First 5 Minutes

This guide takes you from an empty directory to executing your first AI-powered prompt.

**1. Initialize Your Project**
Create a directory and run `init`. This sets up the required file structure.

```bash
mkdir my-prompt-project && cd my-prompt-project
plb init
```

**2. Configure Your AI Provider**
Run the interactive wizard to connect to your preferred LLM provider. It will securely save your API keys to a local `.env` file.

```bash
# This launches the interactive setup wizard
plb configure-ai
```

**3. Create Your First Prompt**
The `create` command will guide you through making a new prompt file.

```bash
# This launches an interactive prompt creation wizard
plb create
```
Follow the steps to name your prompt (e.g., "email-formatter"). Now, open the new file at `prompts/email-formatter.v1.0.0.yml` and add your template logic.

**4. Run and Execute**
Test your prompt by rendering it with variables, then execute it with the `--execute` flag to get a live AI response.

```bash
# Render the prompt template with a variable
plb run email-formatter --var user_request="draft a polite follow-up email"

# Send the rendered prompt to your configured AI for a response
plb run email-formatter --var user_request="draft a polite follow-up email" --execute
```

## Project Structure

The project is organized into a modular structure designed for clarity and maintainability. The core Python source code is located inside the `prompt_lockbox/` directory, which is the installable package itself.

```bash
prompt-lockbox/
├── .gitignore              # Specifies files for Git to ignore (e.g., .env, __pycache__).
├── CONTRIBUTING.md         # Detailed guide for developers who want to contribute.
├── LICENSE                 # The project's open-source license (MIT).
├── README.md               # This file: an overview of the project for users and developers.
├── poetry.lock             # Locks dependency versions for reproducible builds.
├── pyproject.toml          # Project definition and dependency management via Poetry.
├── docs/                   # Source files for the full documentation website.
├── learn/                  # Tutorials and learning materials.
├── notebooks/              # Jupyter notebooks for experiments and examples.
└── prompt_lockbox/         # The core Python package source code.
    ├── __init__.py         # Makes this directory an installable Python package.
    ├── api.py              # The high-level public SDK (Project and Prompt classes).
    ├── ai/                 # Module for all AI-powered logic.
    │   ├── documenter.py   # AI logic to auto-generate documentation for prompts.
    │   ├── executor.py     # Centralized engine for making calls to LLMs.
    │   ├── improver.py     # AI logic to critique and improve prompts.
    │   └── logging.py      # Handles logging of AI usage and token counts.
    ├── cli/                # Module for all Command-Line Interface logic.
    │   ├── _ai.py          # Defines the `plb prompt` command group.
    │   ├── configure.py    # Logic for the interactive `plb configure-ai` command.
    │   ├── main.py         # The main entry point that assembles all `plb` commands.
    │   ├── manage.py       # Commands for managing prompts (create, list, run, etc.).
    │   ├── project.py      # Commands for managing the project (init, lock, verify, etc.).
    │   └── search.py       # Commands for search and indexing.
    ├── core/               # Low-level engine functions (internal logic).
    │   ├── integrity.py    # Handles file hashing and lockfile verification.
    │   ├── project.py      # Core utilities for finding the project root and config.
    │   ├── prompt.py       # Low-level functions for loading and validating prompt files.
    │   └── templating.py   # Centralized Jinja2 rendering logic.
    ├── search/             # Module for advanced search functionality.
    │   ├── fuzzy.py        # Implements lightweight fuzzy string search.
    │   ├── hybrid.py       # Implements hybrid TF-IDF + FAISS vector search.
    │   └── splade.py       # Implements SPLADE sparse vector search.
    └── ui/                 # Module for formatting rich terminal output.
        └── display.py      # Creates beautiful tables, panels, and trees with Rich.
```

### Top-Level Directory Breakdown
- prompt_lockbox/: This is the heart of the project—the installable Python package containing all the source code.
- docs/ & learn/: Contain the source files for the full documentation site and other learning materials.
- notebooks/: Holds Jupyter notebooks for experimentation, examples, or testing new ideas.
- pyproject.toml & poetry.lock: These files manage the project's dependencies and packaging configuration, controlled by Poetry.
- README.md, CONTRIBUTING.md, LICENSE: Standard repository files for community engagement, contribution guidelines, and legal information.
    
**Package Breakdown**: 

Here is a high-level breakdown of the purpose of each main directory within the package:

| Directory | Purpose                                                                                                                                |
| :-------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| **`ai/`**     | Handles all direct interactions with AI models. This is where `executor.py` lives, along with the logic for documentation and improvement. |
| **`cli/`**    | Contains all the Command-Line Interface logic. Each file corresponds to a `plb` command or command group, built with [Typer](https://typer.tiangolo.com/). |
| **`core/`**   | The internal engine of the toolkit. It contains low-level, non-user-facing functions for file integrity, prompt validation, and templating. |
| **`search/`** | Implements the advanced search backends (`fuzzy`, `hybrid`, `splade`). The heavy dependencies for these modules are loaded on-demand.  |
| **`ui/`**     | Dedicated to presentation logic. This module uses the [Rich](https://rich.readthedocs.io/en/latest/) library to create beautiful terminal tables, panels, and trees. |
| **`api.py`**  | The primary public-facing SDK. It defines the high-level `Project` and `Prompt` classes that developers can import and use in their own Python code. |

## Full Documentation

This README is just the beginning. For detailed guides, the complete command reference, and SDK usage examples, please visit our **[Prompt Lockbox Documentation](https://prompt-lockbox.mintlify.app)**.

## Contributing
We're thrilled you're interested in contributing to Prompt Lockbox! Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Every contribution, from a typo fix to a new feature, is greatly appreciated.

> **To get started, please read our full [Contributing Guide](https://prompt-lockbox.mintlify.app/how_to_contribute).**
>
> This guide contains detailed information on our code of conduct, development setup, and the process for submitting pull requests.

### How You Can Help

Whether you're reporting a bug, suggesting a feature, improving the documentation, or submitting code, your help is welcome. The best place to start is by checking our [GitHub Issues](https://github.com/ananya868/prompt-lockbox/issues) and [Discussions](https://github.com/ananya868/prompt-lockbox/discussions).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
