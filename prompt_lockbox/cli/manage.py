#
# FILE: prompt_lockbox/cli/manage.py
#

import typer
from rich import print
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
import textwrap
import json
from typing import List
import jinja2
import yaml

from prompt_lockbox.api import Project
from prompt_lockbox.ui import display # We'll add a new function here
from prompt_lockbox.core import prompt as core_prompt


def create(
    bulk: int = typer.Option(None, "--bulk", "-b", help="Create a specified number of blank prompt files non-interactively.")
):
    """Interactively create a new prompt file, or create template files in bulk."""
    try:
        project = Project()
        prompts_dir = project.root / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        # --- BULK CREATION MODE ---
        if bulk is not None:
            if bulk <= 0:
                print("❌ [bold red]Error:[/bold red] Number of files to create must be a positive integer.")
                raise typer.Exit(code=1)
            
            print(Panel(f"[bold yellow]Bulk Creating {bulk} Prompt Template(s)[/bold yellow] 🪄", border_style="cyan", expand=False))
            
            start_index = core_prompt.find_next_template_index(prompts_dir)
            
            for i in range(start_index, start_index + bulk):
                # We use the SDK to create a blank prompt with a template name
                project.create_prompt(
                    name=f"prompt_template_{i}",
                    version="1.0.0",
                    # Pass empty strings for other fields for a blank template
                    description="", 
                    tags=[],
                    notes="Your notes here!"
                )
            
            print(f"\n✅ [bold green]Done! Created {bulk} blank prompt file(s) in '[cyan]{prompts_dir}[/cyan]'.[/bold green]")
            print("\n[dim]Next step: Edit these files to define your prompts.[/dim]")
            raise typer.Exit()

        # --- FULL INTERACTIVE CREATION MODE ---
        print(Panel("[bold yellow]Let's create a new prompt![/bold yellow]", border_style="yellow", expand=False))
        # 🪄 
        print("\n[bold]1. Prompt Name[/bold] [dim](required)[/dim]")
        print(" [italic]A unique, machine-friendly identifier.[/italic] [dim]e.g., summarize-ticket[/dim]")
        name = typer.prompt("✏️ ")

        print("\n[bold]2. Version[/bold]")
        print(" [italic]The starting semantic version.[/italic] [dim]e.g., 1.0.0[/dim]")
        version = typer.prompt("✏️ ", default="1.0.0")

        # --- Documentation & Organization ---
        print("\n[bold]3. Description[/bold] [dim](optional)[/dim]")
        print(" [italic]A short, human-readable summary of the prompt's purpose.[/italic]")
        description = typer.prompt("✏️ ", default="", show_default=False)

        print("\n[bold]4. Namespace[/bold] [dim](optional)[/dim]")
        print(" [italic]Hierarchical path to organize prompts, comma-separated.[/italic] [dim]e.g., 'Billing,Invoices'[/dim]")
        namespace_str = typer.prompt("✏️ ", default="", show_default=False)
        namespace = [n.strip() for n in namespace_str.split(',') if n.strip()]

        print("\n[bold]5. Tags[/bold] [dim](optional)[/dim]")
        print(" [italic]Keywords for discovery, comma-separated.[/italic] [dim]e.g., Summarization, Code-Gen[/dim]")
        tags_str = typer.prompt("✏️ ", default="", show_default=False)
        tags = [t.strip() for t in tags_str.split(',') if t.strip()]

        # --- Execution ---
        print("\n[bold]6. Intended Model[/bold] [dim](optional)[/dim]")
        print(" [italic]The specific LLM this prompt is designed for.[/italic] [dim]e.g., openai/gpt-4-turbo[/dim]")
        intended_model = typer.prompt("✏️ ", default="", show_default=False)

        print("\n[bold]7. Notes[/bold] [dim](optional)[/dim]")
        print(" [italic]Any extra comments, warnings, or usage instructions.[/italic]")
        notes = typer.prompt("✏️ ", default="", show_default=False)

        # Now, call the SDK's create_prompt method with all the collected data
        new_prompt = project.create_prompt(
            name=name,
            version=version,
            description=description,
            namespace=namespace,
            tags=tags,
            intended_model=intended_model,
            notes=notes
        )
        
        print("\n" + "-" * 40)
        print(f"✅ [bold green]Success! Created new prompt file at:[/bold green]")
        print(f"   [cyan]{escape(str(new_prompt.path))}[/cyan]")
        print("\n[dim]Next step: Open the file and start editing your new prompt![/dim]")

    except FileExistsError as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError:
        print("❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)


def run(
    identifier: str = typer.Argument(..., help="Name, ID, or path of the prompt to run."),
    vars: List[str] = typer.Option(None, "--var", "-v", help="Pre-set a template variable, e.g., --var name=Alex"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open an editor to fill in variables.", is_flag=True)
):
    """Renders a prompt with provided parameters and displays the output."""
    try:
        project = Project()
        prompt = project.get_prompt(identifier)

        if not prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        is_secure, status = prompt.verify()
        if not is_secure:
            print(f"🚨 [bold red]Warning: Cannot run a tampered prompt (status: {status}).[/bold red]")
            raise typer.Exit(code=1)

        # --- REVISED VARIABLE HANDLING LOGIC ---

        # 1. Start with an empty dictionary for final variables.
        template_vars = {}
        
        # 2. Pre-fill from --var flags. These have the highest precedence.
        if vars:
            for var in vars:
                if "=" in var:
                    key, value = var.split("=", 1)
                    template_vars[key] = value

        # 3. Get all required variables and default values from the SDK.
        required_vars = prompt.required_variables
        default_vars = prompt.data.get("default_inputs", {})

        # 4. Handle interactive input (--edit or prompting).
        if edit:
            # --edit mode is a special case.
            # Pre-fill editor with defaults, then override with --var values.
            vars_for_editor = {**default_vars, **template_vars}
            # Ensure all required vars are present in the editor text
            for key in required_vars:
                if key not in vars_for_editor:
                    vars_for_editor[key] = ""
            
            editor_content_str = yaml.dump(vars_for_editor, sort_keys=False, indent=2)
            edited_content = typer.edit(f"# Please fill in values for the prompt variables.\n{editor_content_str}", extension=".yml")
            if edited_content is None: raise typer.Exit()
            user_provided_vars = yaml.safe_load(edited_content) or {}
            template_vars.update(user_provided_vars)

        else:
            # Normal interactive prompting mode.
            # Figure out which variables we still need to ask about.
            vars_to_ask = sorted(list(required_vars - set(template_vars.keys())))
            
            if vars_to_ask:
                print("📌 [bold]Enter your prompt inputs (press Enter to use default):[/bold]")
                for var_name in vars_to_ask:
                    # Get the default value for this specific variable.
                    default_value = default_vars.get(var_name)
                    
                    # Create a rich prompt that shows the default value.
                    prompt_text = Text(f"{var_name}", style="bright_cyan")
                    if default_value is not None:
                        prompt_text.append(f" [dim](default: {escape(str(default_value))})[/dim]")
                    
                    # Ask the user. The `default` here is what's returned if they hit Enter.
                    user_input = typer.prompt(prompt_text, default=default_value)
                    
                    # Store the result (either user's input or the default they accepted).
                    template_vars[var_name] = user_input
        
        # 5. Render the prompt. Use non-strict mode to handle any remaining missing variables.
        # This will only affect variables that had no default and the user also skipped.
        rendered_prompt = prompt.render(strict=False, **template_vars)
        
        # Optional: Add a warning if placeholders are still present.
        if "<<_input>>" in rendered_prompt.replace("user_input", "_input"): # Avoid <<user_input>> trigger
             if "<<user_input>>" not in rendered_prompt: # Check again
                print("🟡 [yellow]Warning:[/yellow] Some inputs were not filled and have no defaults.")


        print(Panel(
            rendered_prompt,
            title=f"Rendered Prompt: [cyan]{escape(prompt.path.name)}[/cyan]",
            border_style="blue", expand=False
        ))

    except FileNotFoundError:
        print("❌ [bold red]Error:[/bold red] Not a PromptLockbox project.")
        raise typer.Exit(code=1)
    except jinja2.exceptions.TemplateSyntaxError as e:
        print(f"❌ [bold red]Template Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def list_prompts(
    wide: bool = typer.Option(False, "--wide", "-w", help="Display more details in the output.")
):
    """Lists all prompts in a table format."""
    try:
        project = Project()
        prompts = project.list_prompts()

        if not prompts:
            print("✅ [green]The 'prompts' directory is empty. No prompts to list.[/green]")
            raise typer.Exit()
        
        # We delegate the table creation to a UI helper
        # We will create this function next
        prompts_table = display.create_list_table(prompts, wide=wide)
        print(prompts_table)

    except FileNotFoundError:
        print("❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)


def show(
    identifier: str = typer.Argument(
        ...,
        help="The name, ID, or path of the prompt to display."
    )
):
    """Displays the full metadata and template for a single prompt."""
    try:
        project = Project()
        prompt = project.get_prompt(identifier)

        if not prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        # Let UI helpers do the work of creating the panels
        metadata_panel, template_panel = display.create_show_panels(prompt)

        # Check for tampering right before display
        is_secure, status = prompt.verify()
        if not is_secure:
            print(Panel(
                f"🚨 [bold yellow]TAMPERING DETECTED[/bold yellow] 🚨\nThis file's hash does not match the lockfile. Status: {status}",
                style="bold red", border_style="red"
            ))

        print(metadata_panel)
        print(template_panel)

    except FileNotFoundError:
        print("❌ [bold red]Error:[/bold red] Not a PromptLockbox project. Have you run `plb init`?")
        raise typer.Exit(code=1)


def version(
    identifier: str = typer.Argument(..., help="Name or path of the prompt to version."),
    major: bool = typer.Option(False, "--major", "-M", help="Perform a major version bump."),
    patch: bool = typer.Option(False, "--patch", "-P", help="Perform a patch version bump."),
):
    """Creates a new, version-bumped copy of a prompt."""
    if major and patch:
        print("❌ [bold red]Error:[/bold red] Cannot specify --major and --patch at the same time.")
        raise typer.Exit(code=1)

    bump_type = "minor"
    if major: bump_type = "major"
    if patch: bump_type = "patch"
    
    try:
        project = Project()
        source_prompt = project.get_prompt(identifier)

        if not source_prompt:
            print(f"❌ [bold red]Error:[/bold red] Prompt '{identifier}' not found.")
            raise typer.Exit(code=1)

        # Let the SDK do all the work!
        new_prompt = source_prompt.new_version(bump_type=bump_type)

        print(f"✅ [bold green]Success! Created new version at:[/bold green] [cyan]{escape(str(new_prompt.path))}[/cyan]")
        print("   [dim]Status has been reset to 'Draft'. You can now edit the new file.[/dim]")

    except FileExistsError as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except ValueError as e: # Catches invalid version strings from the SDK
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError:
        print("❌ [bold red]Error:[/bold red] Not a PromptLockbox project.")
        raise typer.Exit(code=1)