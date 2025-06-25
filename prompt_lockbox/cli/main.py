#
# FILE: prompt_lockbox/cli/main.py
#
"""
This file serves as the main entry point for the PromptLockbox (plb) CLI.

It uses the Typer library to construct the command-line interface by importing
and assembling various command functions and command groups from other modules
within the 'cli' package. It organizes commands into Rich help panels for a
clean and user-friendly --help output.
"""

# Load environment variables from a .env file at the start.
# This is crucial for loading secrets like API keys before any other code runs.
from dotenv import load_dotenv
load_dotenv()

import typer

# Import command functions and groups from their respective modules.
from .project import init, status, lock, unlock, verify, lint
from .manage import list_prompts, show, create, run, version, tree
from .search import index, search_app
from ._ai import prompt_app

# Initialize the main Typer application.
app = typer.Typer(
    name="plb",
    help="A framework to secure, manage, and develop prompts.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="markdown"
)

# --- Attach Commands to Help Panels ---

# Group 1: Core project and file integrity commands.
app.command(rich_help_panel="Project & Integrity")(init)
app.command(rich_help_panel="Project & Integrity")(status)
app.command(rich_help_panel="Project & Integrity")(lock)
app.command(rich_help_panel="Project & Integrity")(unlock)
app.command(rich_help_panel="Project & Integrity")(verify)
app.command(rich_help_panel="Project & Integrity")(lint)

# Group 2: Commands for managing individual prompts.
# To get the desired command name `list`, we specify it here explicitly.
app.command("list", rich_help_panel="Prompt Management")(list_prompts)
app.command(rich_help_panel="Prompt Management")(show)
app.command(rich_help_panel="Prompt Management")(create)
app.command(rich_help_panel="Prompt Management")(run)
app.command(rich_help_panel="Prompt Management")(version)
app.command(rich_help_panel="Prompt Management")(tree)

# Group 3: Search and indexing commands.
# Add the standalone `index` command.
app.command(rich_help_panel="Search & Indexing")(index)
# Add the `search` command group (which contains subcommands like 'fuzzy', 'hybrid').
app.add_typer(search_app, name="search", rich_help_panel="Search & Indexing")

# Group 4: AI-powered commands.
# Add the `prompt` command group (which contains subcommands like 'document', 'improve').
app.add_typer(prompt_app, name="prompt", rich_help_panel="AI Superpowers")


def run():
    """The main entry point function for the console script.

    This function is called when the `plb` command is executed from the terminal.
    It simply invokes the Typer application.
    """
    app()