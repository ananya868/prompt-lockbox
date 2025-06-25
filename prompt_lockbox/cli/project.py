#
# FILE: prompt_lockbox/cli/project.py (Updated)
#
"""
This file defines the core CLI commands for initializing and managing a
PromptLockbox project's integrity and status. These commands interact
with the project as a whole, rather than individual prompts.
"""

import typer
from pathlib import Path
from rich import print
from rich.markup import escape

from prompt_lockbox.api import Project
from prompt_lockbox.ui import display
from prompt_lockbox.core import prompt as core_prompt


def init(
    path: str = typer.Argument(".", help="The directory to initialize PromptLockbox in.")
):
    """Initializes a new PromptLockbox project in the specified directory.

    This command sets up the necessary directory structure (`prompts/`, `.plb/`)
    and creates the core configuration and lock files (`plb.toml`, `.plb.lock`).
    It will also offer to update the `.gitignore` file.

    Args:
        path: The target directory for initialization. Defaults to the current directory.
    """
    print(f"🚀 [bold green]Initializing PromptLockbox in '{path}'...[/bold green]")
    base_path = Path(path).resolve()
    base_path.mkdir(exist_ok=True)

    # Create the core directories for prompts and internal data.
    (base_path / "prompts").mkdir(exist_ok=True)
    (base_path / ".plb").mkdir(exist_ok=True)
    (base_path / "prompts" / ".gitkeep").touch()
    print("  [cyan]✓[/cyan] Created [bold]prompts/[/bold] and [bold].plb/[/bold] directories.")

    # Create the main project configuration file if it doesn't exist.
    if not (base_path / "plb.toml").exists():
        default_config = '# PromptLockbox Configuration\n\n[search]\nactive_index = "hybrid"\n'
        (base_path / "plb.toml").write_text(default_config)
        print("  [cyan]✓[/cyan] Created [bold]plb.toml[/bold] config file.")

    # Create the lockfile for integrity checking if it doesn't exist.
    if not (base_path / ".plb.lock").exists():
        (base_path / ".plb.lock").write_text("# Auto-generated lock file\n\n[locked_prompts]\n")
        print("  [cyan]✓[/cyan] Created [bold].plb.lock[/bold] file for integrity checks.")

    # Offer to add the internal .plb directory to the project's .gitignore.
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
        # 1. Use the SDK to get the structured status report data.
        project = Project()
        report_data = project.get_status_report()

        # 2. Pass the data to a UI helper to generate a rich Table object.
        status_table = display.create_status_table(report_data, project.root)

        # 3. Print the resulting table to the console.
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
    """Validates and locks a prompt to ensure its integrity.

    Before locking, this command runs a validation check. If any errors are found,
    the lock operation is aborted. A successful lock records the prompt's current
    hash in the `.plb.lock` file.

    Args:
        identifier: The unique name, ID, or file path of the prompt to be locked.
    """
    try:
        project = Project()
        prompt = project.get_prompt(identifier)

        if not prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        # Perform a pre-lock validation to catch schema or syntax errors.
        validation_results = core_prompt.validate_prompt_file(prompt.path)
        all_errors = [
            err for category in validation_results.values() for err in category["errors"]
        ]

        # If validation fails, print errors and abort.
        if all_errors:
            print(f"❌ [bold red]Validation Failed for '{escape(prompt.path.name)}'. Cannot lock.[/bold red]")
            for error in all_errors:
                print(f"  - {error}")
            raise typer.Exit(code=1)

        # If validation passes, call the SDK method to perform the lock.
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
    """Unlocks a previously locked prompt, allowing edits.

    This removes the prompt's entry from the `.plb.lock` file, marking it as
    being in a 'draft' or 'unlocked' state.

    Args:
        identifier: The unique name, ID, or file path of the prompt to be unlocked.
    """
    try:
        project = Project()
        prompt = project.get_prompt(identifier)

        if not prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        # Call the SDK method to remove the entry from the lockfile.
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
        # Get a report of all locked, tampered, and missing files.
        report = project.get_status_report()

        tampered_count = len(report['tampered'])
        missing_count = len(report['missing'])
        ok_count = len(report['locked'])

        has_issues = tampered_count > 0 or missing_count > 0

        if not has_issues and ok_count == 0:
            print("✅ [green]No prompts are currently locked. Nothing to verify.[/green]")
            raise typer.Exit()

        # Print the status for each file.
        for p in report['locked']:
            print(f"✅ [green]OK:[/green] '{escape(str(p.path.relative_to(project.root)))}'")

        for p in report['tampered']:
            print(f"❌ [bold red]FAILED:[/bold red] '{escape(str(p.path.relative_to(project.root)))}' has been modified.")

        for p_info in report['missing']:
             print(f"❌ [bold red]FAILED:[/bold red] '{escape(p_info['path'])}' is locked but the file is missing.")

        print("-" * 20)
        # Provide a final summary and exit with an error code if issues were found.
        if has_issues:
            print(f"❌ [bold red]Verification failed. Found {tampered_count + missing_count} issue(s).[/bold red]")
            raise typer.Exit(code=1)
        else:
            print(f"✅ [bold green]All {ok_count} locked file(s) verified successfully.[/bold green]")

    except FileNotFoundError:
        print(f"❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)


def lint():
    """Validates all prompts for compliance and consistency."""
    try:
        project = Project()
        print(f"🔍 [bold]Scanning all prompts for issues...[/bold]\n")

        # Get the structured linting report from the SDK.
        report_data = project.lint()

        # Delegate the complex printing of the report to a UI helper.
        display.print_lint_report(report_data)

        # Exit with an error code if there were any critical errors.
        total_errors = sum(len(cat["errors"]) for cat in report_data.values())
        if total_errors > 0:
            print(f"\n❌ [bold red]Linting failed with {total_errors} critical error(s).[/bold red]")
            raise typer.Exit(code=1)
        else:
            print("\n✅ [bold green]Linting passed successfully.[/bold green]")

    except FileNotFoundError:
        print(f"❌ [bold red]Error:[/bold red] Not a PromptLockbox project.")
        raise typer.Exit(code=1)