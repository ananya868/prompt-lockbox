Here is a well-structured and comprehensive guide to using this Python SDK, complete with descriptions and examples for each step->

---

# PromptLockbox Python SDK Usage Guide

Welcome to the PromptLockbox SDK! This guide provides a comprehensive overview of how to programmatically manage your prompts directly within your Python applications.

### 1. Initializing Your Project

The first step is always to create a `Project` object. This class is your main entry point to the entire framework. It automatically finds your project's root directory by searching for the `plb.toml` file.

```python
from prompt_lockbox import Project

try:
    # Initialize the project from the current directory
    project = Project()
    print(f"✅ Project loaded successfully from: {project.root}")
except FileNotFoundError:
    print("❌ Not a PromptLockbox project. Run `plb init` first.")
```

### 2. Listing and Retrieving Prompts

Once you have a `project` object, you can easily find and load your prompts.

#### List All Prompts
To get a list of all available prompts in your project:
```python
all_prompts = project.list_prompts()

print(f"Found {len(all_prompts)} prompts:")
for p in all_prompts:
    print(f"  - {p.name} (v{p.version})")
```

#### Get a Specific Prompt
To get a single prompt, you can use its name, ID, or file path. The SDK will automatically find the latest version if you use the name.
```python
# Get by name (most common)
summarizer_prompt = project.get_prompt("message-summarizer")

if summarizer_prompt:
    print(f"Retrieved: {summarizer_prompt.name} v{summarizer_prompt.version}")
    
    # You can access all its metadata as properties
    print(f"Description: {summararizer_prompt.description}")
```

### 3. Rendering a Prompt

This is the core operational feature. The `.render()` method safely injects your variables into the prompt's template.

```python
if summarizer_prompt:
    # Variables to inject into the template
    variables = {
        "user_input": "My application is slow and keeps crashing.",
        "ticket_id": "TICKET-789"
    }

    # Render the prompt
    final_text = summarizer_prompt.render(**variables)
    
    print("\n--- Rendered Prompt ---")
    print(final_text)
```
The SDK automatically handles default values and will raise an error if a required variable is missing. For partial rendering with placeholders, use `prompt.render(strict=False, **variables)`.

### 4. Creating and Versioning Prompts

You can programmatically create new prompts and new versions of existing ones.

#### Create a New Prompt
```python
try:
    new_prompt = project.create_prompt(
        name="email-formatter",
        version="1.0.0",
        description="Formats a user's draft email into a professional tone.",
        tags=["formatting", "email", "nlp"]
    )
    print(f"✅ Successfully created new prompt at: {new_prompt.path}")
except FileExistsError:
    print("🟡 Prompt already exists. Skipping creation.")
```

#### Create a New Version
```python
email_formatter = project.get_prompt("email-formatter")

if email_formatter:
    try:
        # Bump the 'minor' version (e.g., 1.0.0 -> 1.1.0)
        v2_prompt = email_formatter.new_version(bump_type="minor")
        print(f"✅ Created new version: {v2_prompt.name} v{v2_prompt.version}")
    except FileExistsError:
        print(f"🟡 Version {v2_prompt.version} already exists.")
```

### 5. Integrity and Security (Locking)

Ensure your production prompts are safe from accidental changes by using the lockfile mechanism.

```python
production_prompt = project.get_prompt("production-critical-prompt")

if production_prompt:
    # Check the status
    is_secure, status = production_prompt.verify()
    print(f"Initial status: {status}") # e.g., UNLOCKED

    # Lock the prompt
    print("Locking prompt...")
    production_prompt.lock()
    
    is_secure, status = production_prompt.verify()
    print(f"Status after locking: {status}") # e.g., OK

    # Unlock the prompt
    print("Unlocking prompt...")
    production_prompt.unlock()
    
    is_secure, status = production_prompt.verify()
    print(f"Status after unlocking: {status}") # e.g., UNLOCKED
```

### 6. Linting and Searching (CI/CD and Automation)

You can use the SDK in automated scripts, such as a CI/CD pipeline, to enforce quality and search for prompts.

#### Lint the Entire Project
```python
# This is useful for pre-commit hooks or CI checks
lint_report = project.lint()

total_errors = sum(len(cat["errors"]) for cat in lint_report.values())
if total_errors > 0:
    print(f"❌ Found {total_errors} critical errors during linting!")
else:
    print("✅ All prompts passed the linting checks.")
```

#### Search for Prompts
```python
# First, ensure the index is built
# project.index(method="hybrid") # or "splade"

# Now search
search_results = project.search(
    query="A prompt that can translate text to German",
    method="hybrid", # or "splade"
    limit=3
)

print("--- Search Results ---")
for result in search_results:
    print(f"  - Score: {result['score']:.2f}, Name: {result['name']}")
```

---