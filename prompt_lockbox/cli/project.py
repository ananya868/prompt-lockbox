#
# FILE: prompt_lockbox/cli/project.py (Updated)
#

import typer
from pathlib import Path
from rich import print
from rich.markup import escape

from prompt_lockbox.api import Project # Already imported from `api`
from prompt_lockbox.ui import display # Import our new UI module
from prompt_lockbox.core import prompt as core_prompt # <--- ADD THIS LINE
# REMOVED: The project_app = typer.Typer(...) line

# The function is now a standalone function, not attached to a sub-app here.
def init(
    path: str = typer.Argument(".", help="The directory to initialize PromptLockbox in.")
):
    """Initializes a new PromptLockbox project in the specified directory."""
    # ... the init function's code is completely unchanged ...
    print(f"🚀 [bold green]Initializing PromptLockbox in '{path}'...[/bold green]")
    base_path = Path(path).resolve()
    base_path.mkdir(exist_ok=True)
    
    (base_path / "prompts").mkdir(exist_ok=True)
    (base_path / ".plb").mkdir(exist_ok=True)
    (base_path / "prompts" / ".gitkeep").touch()
    print("  [cyan]✓[/cyan] Created [bold]prompts/[/bold] and [bold].plb/[/bold] directories.")

    if not (base_path / "plb.toml").exists():
        default_config = '# PromptLockbox Configuration\n\n[search]\nactive_index = "hybrid"\n'
        (base_path / "plb.toml").write_text(default_config)
        print("  [cyan]✓[/cyan] Created [bold]plb.toml[/bold] config file.")

    if not (base_path / ".plb.lock").exists():
        (base_path / ".plb.lock").write_text("# Auto-generated lock file\n\n[locked_prompts]\n")
        print("  [cyan]✓[/cyan] Created [bold].plb.lock[/bold] file for integrity checks.")

    gitignore = base_path / ".gitignore"
    entry_to_add = "\n# Ignore PromptLockbox internal directory\n.plb/\n"
    
    if not gitignore.exists() or ".plb/" not in gitignore.read_text():
        update_gitignore = typer.confirm(
            "May I add the '.plb/' directory to your .gitignore file?", default=True
        )
        if update_gitignore:
            with gitignore.open("a", encoding="utf-8") as f:
                f.write(entry_to_add)
            print("  [cyan]✓[/cyan] Updated [bold].gitignore[/bold].")
        else:
            print("  [yellow]! Skipped .gitignore update. Please add '.plb/' manually.[/yellow]")

    print("\n[bold green]Initialization complete![/bold green] You're ready to create prompts.")


def status():
    """Displays the lock status and integrity of all prompts."""
    try:
        # 1. Use SDK to get the data
        project = Project()
        report_data = project.get_status_report()

        # 2. Use UI helper to get the presentation object
        status_table = display.create_status_table(report_data, project.root)

        # 3. Print the result
        print(status_table)

    except FileNotFoundError as e:
        print(f"❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


def lock(
    identifier: str = typer.Argument(
        ...,
        help="Name, ID, or path of the prompt to lock."
    )
):
    """Validates and locks a prompt to ensure its integrity."""
    try:
        project = Project()
        prompt = project.get_prompt(identifier)

        if not prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        # Here we'll add the pre-lock validation from your original CLI
        validation_results = core_prompt.validate_prompt_file(prompt.path)
        all_errors = [
            err for category in validation_results.values() for err in category["errors"]
        ]

        if all_errors:
            print(f"❌ [bold red]Validation Failed for '{escape(prompt.path.name)}'. Cannot lock.[/bold red]")
            for error in all_errors:
                print(f"  - {error}")
            raise typer.Exit(code=1)

        # Call the SDK method to do the actual work
        prompt.lock()

        print(f"🔒 [bold green]Prompt '{escape(prompt.name)}' is now locked.[/bold green]")

    except FileNotFoundError:
        print(f"❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


def unlock(
    identifier: str = typer.Argument(
        ...,
        help="Name, ID, or path of the prompt to unlock."
    )
):
    """Unlocks a previously locked prompt, allowing edits."""
    try:
        project = Project()
        prompt = project.get_prompt(identifier)

        if not prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        # Call the SDK method
        prompt.unlock()

        print(f"🔓 [bold green]Prompt '{escape(prompt.name)}' is now unlocked.[/bold green]")

    except FileNotFoundError:
        print(f"❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


def verify():
    """Verifies the integrity of all locked prompts against the lockfile."""
    print("🛡️  Verifying integrity of all locked prompts...")
    try:
        project = Project()
        report = project.get_status_report()
        
        tampered_count = len(report['tampered'])
        missing_count = len(report['missing'])
        ok_count = len(report['locked'])

        has_issues = tampered_count > 0 or missing_count > 0

        if not has_issues and ok_count == 0:
            print("✅ [green]No prompts are currently locked. Nothing to verify.[/green]")
            raise typer.Exit()
            
        for p in report['locked']:
            print(f"✅ [green]OK:[/green] '{escape(str(p.path.relative_to(project.root)))}'")
        
        for p in report['tampered']:
            print(f"❌ [bold red]FAILED:[/bold red] '{escape(str(p.path.relative_to(project.root)))}' has been modified.")

        for p_info in report['missing']:
             print(f"❌ [bold red]FAILED:[/bold red] '{escape(p_info['path'])}' is locked but the file is missing.")
        
        print("-" * 20)
        if has_issues:
            print(f"❌ [bold red]Verification failed. Found {tampered_count + missing_count} issue(s).[/bold red]")
            raise typer.Exit(code=1)
        else:
            print(f"✅ [bold green]All {ok_count} locked file(s) verified successfully.[/bold green]")

    except FileNotFoundError:
        print(f"❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)    


