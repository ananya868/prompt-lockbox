<div align="center">
  <img src="docs/logo/logo.png" alt="Logo" width=700>  
</div>
<div align="center">
  <h5>Brings structure and reproducibility to prompt engineering</h5>
  <a href="https://badge.fury.io/py/prompt-lockbox"><img src="https://badge.fury.io/py/prompt-lockbox.svg" alt="PyPI version"/></a><a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a><a href="https://pypi.org/project/prompt-lockbox/"><img src="https://img.shields.io/pypi/pyversions/prompt-lockbox.svg" alt="Python versions"/></a>

  A powerful CLI toolkit and Python SDK to secure, manage, and develop your LLM prompts like reusable code.
</div>

<br></br>
📑 **Explore the Full Documentation at - [![Docs](https://img.shields.io/badge/Docs-Prompt%20Lockbox-green?style=for-the-badge)](https://prompt-lockbox.mintlify.app)**
>
> *(This README provides an overview. The full site contains detailed guides, API references, and advanced tutorials.)*

<br/>

<div align="center">

**[-- YOUR VIDEO DEMO WILL GO HERE --]**

*A brief video showcasing the core workflow from initialization to AI-powered improvement.*

</div>

<br/>

## Why use it ?

Managing prompts across a team or a large project can be chaotic. Plain text files lack versioning, are prone to accidental changes, and have no built-in quality control. **Prompt Lockbox** brings the discipline of software engineering to your prompt development workflow.

**Scattered Prompts**
> **Problem:** Your team’s best prompts are lost in a chaotic mess of text files, Slack messages, and Google Docs. No one knows where the “official” version is, leading to confusion and duplicated effort.
>
> ✅ **Solution:** Prompt Lockbox creates a centralized and structured library, making it effortless for your entire team to find and use the correct prompt.

**Untracked Versions**
> **Problem:** You tweak a great prompt and accidentally make it worse, with no way to go back to the version that worked.
>
> ✅ **Solution:** Prompt Lockbox provides Git-like versioning for prompts. Safely create new versions and manage modifications with `plb version`, just like code.

**Accidental Production Changes**
> **Problem:** A critical prompt is changed without approval, breaking your application and causing unpredictable outputs.
>
> ✅ **Solution:** Prompt Lockbox lets you lock and secure key prompts with `plb lock`. The system instantly detects any unauthorized edits with `plb verify`, ensuring production reliability.

**Reinventing the Wheel**
> **Problem:** Your team wastes time building prompts that already exist because they’re impossible to find.
>
> ✅ **Solution:** Prompt Lockbox makes your entire library instantly searchable. Find what you need in seconds with powerful, context-aware search (`fuzzy`, `hybrid`, `splade`).

**Poor Documentation**
> **Problem:** You write a brilliant prompt but have no time to document it, making it unusable for others (or your future self).
>
> ✅ **Solution:** Prompt Lockbox uses AI to automate documentation. One command (`plb prompt document`) generates a clear description and search tags, turning your prompts into reusable assets.
---

## Features

*   **🔒 Integrity & Security:** Lock prompts to generate a checksum. The `plb verify` command ensures that production prompts haven't been tampered with since they were last approved.

*   **📂 Version Control First:** Automatically create new, semantically versioned prompt files with `plb version`, making it easy to iterate and experiment safely without breaking existing implementations.

*   **🤖 AI Superpowers:**
    *   **Auto-Documentation:** `plb prompt document` uses an AI to analyze your prompt and generate a concise description and relevant search tags.
    *   **Auto-Improvement:** `plb prompt improve` provides an expert critique and a suggested, improved version of your prompt template, showing you a diff of the changes.
    *   **Direct Execution:** Execute prompts directly against your configured LLMs (OpenAI, Anthropic, Ollama, HuggingFace) right from the CLI.

*   **🔎 Advanced Search:** Don't just `grep`. Build a local search index (`hybrid` TF-IDF + FAISS or `splade`) to find the right prompt instantly using natural language queries.

*   **⚙️ Flexible Configuration:** An interactive wizard (`plb configure-ai`) makes it trivial to set up any provider and model, securely storing API keys in a local `.env` file.

## Installation

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

## ▶️ Quickstart: Your First 5 Minutes

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
