#
# FILE: prompt_lockbox/cli/ai_actions.py (NEW FILE)
#

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
        None, # Default to None, so we can check if the user provided anything
        help="One or more prompt names, IDs, or paths to document."
    ),
    all: bool = typer.Option(
        False, "--all", "-a",
        help="Document all prompts in the project. Overrides any specific identifiers."
    )
):
    """(AI) Automatically generate a description and tags for one or more prompts."""
    try:
        project = Project()

        # New, clearer logic:
        if all:
            # If --all is used, it takes precedence.
            project.document_all()
        elif identifiers:
            # If specific prompts are listed, process them.
            prompts_to_process = []
            for identifier in identifiers:
                prompt = project.get_prompt(identifier)
                if prompt:
                    prompts_to_process.append(prompt)
                else:
                    print(f"🟡 [yellow]Warning:[/yellow] Prompt '{identifier}' not found. Skipping.")
            
            if prompts_to_process:
                project.document_all(prompts_to_document=prompts_to_process)
        else:
            # If neither --all nor any identifiers are given, show an error.
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
    """(AI) Get a critique and suggested improvements for a prompt."""
    try:
        project = Project()
        prompt = project.get_prompt(identifier)

        if not prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        print(f"🤖 Analyzing prompt '[bold cyan]{prompt.name}[/bold cyan]' with your note...")
        
        critique_data = prompt.get_critique(note=note)
        
        original_template = prompt.data.get("template", "")
        improved_template = critique_data.get("improved_template", "")

        # Display the critique panel (unchanged)
        console = Console()
        console.print(Panel(
            f"[bold]Critique:[/bold]\n{critique_data.get('critique', 'N/A')}",
            title="[yellow]AI Analysis[/yellow]",
            border_style="yellow"
        ))
        
        # --- NEW, DEPENDENCY-FREE DIFF LOGIC ---
        print("\n[bold]Suggested Improvements (Diff View):[/bold]")
        
        # Use difflib to get the differences
        diff = difflib.unified_diff(
            original_template.splitlines(keepends=True),
            improved_template.splitlines(keepends=True),
            fromfile='Original',
            tofile='Improved',
        )
        
        # Use rich.text.Text to colorize the output
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
            # Handle case where there are no changes
            console.print("[dim]No changes suggested by the AI.[/dim]")
        # --- END OF NEW DIFF LOGIC ---
        
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
    

