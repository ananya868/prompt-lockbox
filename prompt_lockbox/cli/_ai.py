#
# FILE: prompt_lockbox/cli/_ai.py (NEW FILE)
#
"""
This file defines the 'plb prompt' command-line interface group, providing
AI-powered actions like automatic documentation and prompt improvement.
"""

import typer
from rich import print
from typing import List, Optional

from prompt_lockbox.api import Project

from rich.panel import Panel
from rich.console import Console
from rich.text import Text
# We use Python's built-in library for diffing
import difflib

# This creates the `plb prompt` command group
prompt_app = typer.Typer(
    name="prompt",
    help="Perform AI-powered actions on prompts.",
    no_args_is_help=True
)


@prompt_app.command()
def document(
    identifiers: List[str] = typer.Argument(
        None,  # Default to None, so we can check if the user provided anything
        help="One or more prompt names, IDs, or paths to document."
    ),
    all: bool = typer.Option(
        False, "--all", "-a",
        help="Document all prompts in the project. Overrides any specific identifiers."
    )
):
    """(AI) Automatically generate a description and tags for one or more prompts.

    This command uses an AI to analyze the content of specified prompts and
    updates their respective files with a generated description and search tags.
    It carefully preserves the existing file layout and comments.

    Args:
        identifiers: A list of prompt names, IDs, or file paths to be documented.
        all: If True, all prompts in the project will be documented, ignoring
             any provided identifiers.
    """
    try:
        project = Project()

        # Determine which prompts to document based on user flags.
        if all:
            # If --all is used, it takes precedence and documents every prompt.
            project.document_all()
        elif identifiers:
            # If specific prompts are listed, find and process them.
            prompts_to_process = []
            for identifier in identifiers:
                prompt = project.get_prompt(identifier)
                if prompt:
                    prompts_to_process.append(prompt)
                else:
                    print(f"🟡 [yellow]Warning:[/yellow] Prompt '{identifier}' not found. Skipping.")

            if prompts_to_process:
                # Pass the collected list of prompts to the bulk documentation method.
                project.document_all(prompts_to_document=prompts_to_process)
        else:
            # If neither --all nor any identifiers are given, show an error and exit.
            print("❌ [bold red]Error:[/bold red] Please specify one or more prompt names, or use the --all flag.")
            raise typer.Exit(code=1)

    except Exception as e:
        print(f"❌ [bold red]An unexpected error occurred:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@prompt_app.command()
def improve(
    identifier: str = typer.Argument(..., help="Name, ID, or path of the prompt to improve."),
    note: str = typer.Option(
        "Make it clearer, more specific, and more robust.",
        "--note", "-n",
        help="A specific note to the AI on how to improve the prompt."
    ),
    apply: bool = typer.Option(
        False, "--apply",
        help="Directly apply the AI's suggestions without asking for confirmation."
    )
):
    """(AI) Get a critique and suggested improvements for a prompt.

    This command sends a prompt's content to an AI for analysis. The AI returns
    a critique and a suggested improved version. The changes are displayed as a
    color-coded diff, and the user is prompted to apply them unless the --apply
    flag is used.

    Args:
        identifier: The name, ID, or file path of the prompt to improve.
        note: A specific instruction to guide the AI's improvement process.
        apply: If True, saves the AI's suggested changes without confirmation.
    """
    try:
        project = Project()
        prompt = project.get_prompt(identifier)

        if not prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        print(f"🤖 Analyzing prompt '[bold cyan]{prompt.name}[/bold cyan]' with your note...")

        # Call the API to get the AI-generated critique and new template.
        critique_data = prompt.get_critique(note=note)

        original_template = prompt.data.get("template", "")
        improved_template = critique_data.get("improved_template", "")

        # Display the AI's text critique in a formatted panel.
        console = Console()
        console.print(Panel(
            f"[bold]Critique:[/bold]\n{critique_data.get('critique', 'N/A')}",
            title="[yellow]AI Analysis[/yellow]",
            border_style="yellow"
        ))

        # Generate and display a color-coded diff of the changes.
        print("\n[bold]Suggested Improvements (Diff View):[/bold]")

        # Use Python's standard difflib to compare the original and improved templates.
        diff = difflib.unified_diff(
            original_template.splitlines(keepends=True),
            improved_template.splitlines(keepends=True),
            fromfile='Original',
            tofile='Improved',
        )

        # Use rich.text.Text to apply color styling to the diff output.
        diff_text = Text()
        for line in diff:
            if line.startswith('+'):
                diff_text.append(line, style="green")
            elif line.startswith('-'):
                diff_text.append(line, style="red")
            elif line.startswith('^'):
                diff_text.append(line, style="blue")
            else:
                diff_text.append(line)

        if diff_text:
            console.print(diff_text)
        else:
            # Handle the case where the AI suggests no changes.
            console.print("[dim]No changes suggested by the AI.[/dim]")

        # Ask the user for confirmation before applying changes, unless --apply is set.
        if apply:
            print("\n--apply flag detected. Saving changes...")
            prompt.improve(improved_template)
            print("✅ [bold green]Success![/bold green] Prompt has been updated.")
        else:
            save_changes = typer.confirm("\nDo you want to apply these improvements and save the file?")
            if save_changes:
                prompt.improve(improved_template)
                print("✅ [bold green]Success![/bold green] Prompt has been updated.")
            else:
                print("Changes discarded.")

    except Exception as e:
        print(f"❌ [bold red]An unexpected error occurred:[/bold red] {e}")
        raise typer.Exit(code=1)