#
# FILE: prompt_lockbox/cli/configure.py (FINAL, CORRECTED VERSION)
#
"""
This file defines the `plb configure-ai` command for setting up
and checking the AI provider configuration.
"""

import typer
import questionary
import os
from rich import print
from rich.panel import Panel
from rich.text import Text
from dotenv import set_key, find_dotenv, load_dotenv

from prompt_lockbox.api import Project

# (Provider config is unchanged)
PROVIDER_CONFIG = {
    "OpenAI": { "env_var": "OPENAI_API_KEY", "prompt_message": "Please enter your OpenAI API Key", "model_prompt": "Enter the default OpenAI model name (e.g., gpt-4o-mini)", "default_model": "gpt-4o-mini", "toml_provider": "openai" },
    "Anthropic": { "env_var": "ANTHROPIC_API_KEY", "prompt_message": "Please enter your Anthropic API Key", "model_prompt": "Enter the default Anthropic model name (e.g., claude-3-haiku-20240307)", "default_model": "claude-3-haiku-20240307", "toml_provider": "anthropic" },
    "HuggingFace": { "env_var": "HUGGING_FACE_HUB_TOKEN", "prompt_message": "Please enter your Hugging Face Token", "model_prompt": "Enter the Hugging Face model repo ID (e.g., mistralai/Mistral-7B-Instruct-v0.2)", "default_model": "mistralai/Mistral-7B-Instruct-v0.2", "toml_provider": "huggingface" },
    "Ollama (Local)": { "env_var": None, "prompt_message": "", "model_prompt": "Please enter the Ollama model name you have pulled (e.g., llama3)", "default_model": "llama3", "toml_provider": "ollama" }
}

def configure_ai(
    status: bool = typer.Option(
        False, 
        "--status", 
        "-s", 
        help="Display the current AI configuration status instead of running the setup wizard."
    )
):
    """
    Configure the AI provider, model, and API keys.
    Defaults to an interactive setup wizard.
    """
    if status:
        show_status()
    else:
        run_setup_wizard()

def run_setup_wizard():
    """The interactive wizard for configuring AI settings."""
    try:
        project = Project()
    except FileNotFoundError:
        print("❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)

    print(Panel("Welcome to the AI Configuration Wizard 🧙‍♂️", style="bold yellow"))

    provider_choice = questionary.select(
        "Select your LLM Provider:",
        choices=list(PROVIDER_CONFIG.keys())
    ).ask()
    if not provider_choice: raise typer.Exit()

    config = PROVIDER_CONFIG[provider_choice]
    toml_provider, env_var = config["toml_provider"], config["env_var"]
    dotenv_path = project.root / ".env"

    if provider_choice == "Ollama (Local)":
        print("\n[bold cyan]Ollama Setup Note:[/bold cyan]")
        print("To use Ollama, you must have the Ollama server running in a separate terminal via `ollama serve`.")
        questionary.confirm("Press Enter to continue once your server is running...").ask()

    if env_var:
        api_key = questionary.password(f"{config['prompt_message']} (will be stored in .env):").ask()
        if not api_key: raise typer.Exit()
        set_key(dotenv_path, env_var, api_key)
        print(f"✅ [green]API key saved to [cyan]{dotenv_path}[/cyan].[/green]")

    # This is where the typo was fixed: 'model_prompt'
    model_name = questionary.text(config['model_prompt'], default=config['default_model']).ask()
    if not model_name: raise typer.Exit()

    project_config = project._config
    project_config.setdefault("ai", {})
    project_config["ai"]["provider"] = toml_provider
    project_config["ai"]["model"] = model_name
    project.write_config(project_config)
    print(f"✅ [green]Configuration saved to [cyan]plb.toml[/cyan].[/green]")
    print("\n[bold green]✨ Success! Configuration complete. You're ready to use AI features![/bold green]")

def show_status():
    """Displays the current AI provider configuration and API key status."""
    try:
        project = Project()
    except FileNotFoundError:
        print("❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)

    load_dotenv(find_dotenv())
    ai_config = project.get_ai_config()
    provider, model = ai_config.get("provider", "[Not Set]"), ai_config.get("model", "[Not Set]")

    lines = Text()
    lines.append("Provider: ", style="bold dim"); lines.append(f"{provider}\n", style="bright_cyan")
    lines.append("Model:    ", style="bold dim"); lines.append(f"{model}\n\n", style="bright_cyan")
    lines.append("API Key Status (from .env):\n", style="bold dim")

    for p_name, p_conf in PROVIDER_CONFIG.items():
        key_var = p_conf.get("env_var")
        if key_var:
            is_set = os.getenv(key_var) is not None
            status_text, style = ("✅ Set", "green") if is_set else ("❌ Not Set", "red")
            lines.append(f"  - {key_var}: ", style="default"); lines.append(f"[{style}]{status_text}[/]\n")

    print(Panel(lines, title="[yellow]Current AI Configuration Status[/yellow]", border_style="yellow"))