#
# FILE: prompt_lockbox/cli/ai_actions.py (NEW FILE)
#

import typer
from rich import print
from typing import List, Optional

from prompt_lockbox.api import Project

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

    

