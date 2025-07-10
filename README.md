<div align="center">
  <img src="docs/logo/logo.png" alt="Logo" width=700>  
</div>
<div align="center">
  <h5>Brings structure and reproducibility to prompt engineering</h5>
</div>


[![PyPI version](https://badge.fury.io/py/prompt-lockbox.svg)](https://badge.fury.io/py/prompt-lockbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/pypi/pyversions/prompt-lockbox.svg)](https://pypi.org/project/prompt-lockbox/)

A powerful CLI toolkit and Python SDK to secure, manage, and develop prompts with AI-powered features.

Prompt Lockbox helps teams and individual developers bring structure and reliability to their prompt engineering workflow. It treats your prompts like code, enabling versioning, integrity checking, and AI-assisted development, all from the command line.

---

## Key Features

*   **🔒 Integrity & Security:** Lock prompts to prevent unintended changes. The `plb verify` command ensures that production prompts haven't been tampered with.
*   **📂 Version Control:** Automatically create new, semantically versioned prompt files with `plb version`, making it easy to iterate and experiment safely.
*   **🤖 AI Superpowers:**
    *   `plb prompt document`: Automatically generate descriptions and tags for your prompts.
    *   `plb prompt improve`: Get an expert critique and suggested improvements for your prompt templates.
    *   `plb run --execute`: Execute prompts directly against configured LLMs (OpenAI, Anthropic, Ollama, HuggingFace).
*   **🔎 Advanced Search:** Build a local search index (`hybrid` or `splade`) to find the right prompt instantly using natural language.
*   **⚙️ Flexible Configuration:** An interactive wizard (`plb configure-ai`) makes it easy to set up any provider and model.
