#
# FILE: prompt_lockbox/ui/display.py
#
"""
This module provides helper functions for creating rich, formatted terminal
output using the `rich` library. It decouples the presentation logic from
the command-line and API logic, making both easier to manage.
"""

from rich.table import Table
from rich.markup import escape
import zoneinfo
from datetime import datetime
from rich.panel import Panel
from rich.text import Text
import textwrap
import json
from rich.tree import Tree
from rich import print


def create_status_table(status_report: dict, project_root) -> Table:
    """Takes status report data from the SDK and returns a Rich Table.

    This function formats the data from `project.get_status_report()` into a
    user-friendly table showing the lock status of all prompts.

    Args:
        status_report: The dictionary report from the SDK.
        project_root: The project's root path, used for display purposes.

    Returns:
        A `rich.table.Table` object ready to be printed.
    """
    table = Table(title="Prompt Lock Status", show_header=True, highlight=True, border_style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Prompt File", style="white", no_wrap=True)
    table.add_column("Locked At", style="yellow")

    try:
        # Get the user's local timezone for displaying timestamps nicely.
        local_tz = datetime.now(zoneinfo.ZoneInfo("UTC")).astimezone().tzinfo
    except zoneinfo.ZoneInfoNotFoundError:
        local_tz = None

    def format_timestamp(ts_str):
        """A nested helper to format ISO 8601 timestamps into a local, readable format."""
        if not ts_str or not local_tz:
            return "[dim]--[/dim]"
        dt_utc = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        dt_local = dt_utc.astimezone(local_tz)
        return dt_local.strftime("%Y-%m-%d %H:%M")

    # Populate the table with rows for each status category.
    for p in status_report.get('locked', []):
        ts = p.data.get('last_update')
        table.add_row("[green]✔ Locked[/]", escape(p.path.name), format_timestamp(ts))

    for p in status_report.get('unlocked', []):
        table.add_row("[dim]● Unlocked[/]", escape(p.path.name), "[dim]--[/dim]")

    for p in status_report.get('tampered', []):
        ts = p.data.get('last_update')
        table.add_row("[bold red]❌ TAMPERED[/]", escape(p.path.name), format_timestamp(ts))

    for p_info in status_report.get('missing', []):
        table.add_row("[bold red]❗ MISSING[/]", escape(p_info['path']), "[dim]Locked but file deleted[/dim]")

    return table


def create_list_table(prompts: list, wide: bool) -> Table:
    """Takes a list of Prompt objects and returns a Rich Table for `plb list`.

    Args:
        prompts: A list of `Prompt` objects from the SDK.
        wide: If True, the table will include additional columns for more details.

    Returns:
        A `rich.table.Table` object ready to be printed.
    """
    table = Table(
        title="Prompt Lockbox Library",
        caption="To see full details, run `plb show <prompt-name>`",
        expand=False, border_style="dim", show_header=True, highlight=True
    )

    # Define the table columns.
    table.add_column("Name", justify="left", style="bright_cyan", no_wrap=True)
    table.add_column("Version", style="magenta")
    table.add_column("Status", justify="center", style="white")

    if wide:
        table.add_column("Intended Model", style="yellow")
        table.add_column("Default Inputs", style="dim")
        table.add_column("Last Update", style="white")

    table.add_column("Description", style="default")

    # Iterate through the prompt objects and add a row for each.
    for p in prompts:
        row_data = [
            p.name or "[n/a]",
            p.version or "[n/a]",
            p.data.get("status", "N/A"),
        ]

        if wide:
            # Format the 'wide' columns for display.
            defaults = p.data.get("default_inputs", {}) or {}
            defaults_display = ", ".join(defaults.keys()) if defaults else ""

            last_update_iso = p.data.get("last_update")
            last_update_display = ""
            if last_update_iso:
                try:
                    dt_obj = datetime.fromisoformat(last_update_iso.replace("Z", "+00:00"))
                    last_update_display = dt_obj.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    last_update_display = "[dim]Invalid[/dim]"

            row_data.extend([
                p.data.get("intended_model", ""),
                defaults_display,
                last_update_display
            ])

        row_data.append(p.description or "")
        table.add_row(*row_data)

    return table


def create_show_panels(prompt) -> tuple[Panel, Panel]:
    """Takes a Prompt object and returns metadata and template Rich Panels.

    This function dynamically builds a panel for all metadata fields found
    in the prompt's data, and a separate panel for its template content.

    Args:
        prompt: A single `Prompt` object from the SDK.

    Returns:
        A tuple containing two `rich.panel.Panel` objects.
    """

    def format_line(label, value, style="default"):
        """A nested helper to format a single key-value line for the metadata panel."""
        text = Text()
        text.append(f"{label:<18}", style="bold dim")
        text.append(value, style=style)
        return text

    metadata_items = []
    # Iterate through all key-value pairs in the prompt's data dictionary.
    for key, value in prompt.data.items():
        # Skip the 'template' key, as it gets its own dedicated panel.
        if key == "template":
            continue

        # Format the value for display based on its type.
        if value is None:
            display_value = "[dim]None[/dim]"
        elif isinstance(value, list):
            display_value = ", ".join(map(str, value))
        elif isinstance(value, dict):
            # Pretty-print dictionaries on new, indented lines for readability.
            pretty_dict = json.dumps(value, indent=2)
            display_value = "\n" + textwrap.indent(pretty_dict, ' ' * 18)
        else:
            display_value = str(value)

        display_style = "bright_cyan" if key == "name" else "default"

        # Convert snake_case key to Title Case for a clean label.
        label = key.replace('_', ' ').title() + ":"
        metadata_items.append(format_line(label, display_value, style=display_style))

    # Join all formatted lines into a single Text object for rendering.
    metadata_renderable = Text("\n").join(metadata_items)

    metadata_panel = Panel(
        metadata_renderable,
        title=f"Metadata: [bold cyan]{escape(prompt.path.name)}[/bold cyan]",
        border_style="blue",
        expand=False
    )

    # The template panel simply displays the raw template content.
    template_panel = Panel(
        Text(prompt.data.get("template", ""), style="default"),
        title="[bold]Prompt Template[/bold]",
        border_style="green",
        expand=False
    )

    return metadata_panel, template_panel


def print_lint_report(report_data: dict):
    """Takes linting data from the SDK and prints a rich, detailed report.

    This function first prints a summary table and then provides detailed
    panels for each category that has errors or warnings.

    Args:
        report_data: The dictionary report from `project.lint()`.
    """
    summary_table = Table(title="Linting Report Summary", show_header=True)
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Checklist", style="bright_white")
    summary_table.add_column("Details")

    total_errors = 0
    total_warnings = 0

    # Populate the summary table.
    for category, results in report_data.items():
        errors = results["errors"]
        warnings = results["warnings"]
        total_errors += len(errors)
        total_warnings += len(warnings)

        status_emoji, style = ("✅", "green")
        if errors: status_emoji, style = ("❌", "red")
        elif warnings: status_emoji, style = ("🟡", "yellow")

        details = f"{len(errors)} errors, {len(warnings)} warnings"
        summary_table.add_row(f"[{style}]{status_emoji}[/]", category, details)

    print(summary_table)

    if total_errors == 0 and total_warnings == 0:
        return

    # Print detailed panels for categories with issues.
    print("\n" + "-"*50 + "\n[bold]Detailed Report:[/bold]")
    for category, results in report_data.items():
        errors, warnings = results["errors"], results["warnings"]
        if not errors and not warnings:
            continue

        panel_content = ""
        if errors:
            panel_content += "[red]Errors:[/red]\n"
            for path, msg in errors:
                panel_content += f"  - [cyan]{escape(path)}[/cyan]: {escape(msg)}\n"
        if warnings:
            panel_content += "\n[yellow]Warnings:[/yellow]\n" if errors else "[yellow]Warnings:[/yellow]\n"
            for path, msg in warnings:
                panel_content += f"  - [cyan]{escape(path)}[/cyan]: {escape(msg)}\n"

        border_style = "red" if errors else "yellow"

        print(Panel(panel_content.strip(), title=f"[bold]{category}[/bold]", border_style=border_style))


def create_tree_view(prompts: list) -> Tree:
    """Takes a list of Prompt objects and returns a Rich Tree based on namespace.

    This function parses the `namespace` field of each prompt to build a
    hierarchical folder-like structure for display.

    Args:
        prompts: A list of `Prompt` objects from the SDK.

    Returns:
        A `rich.tree.Tree` object ready to be printed.
    """
    namespace_map = {}
    no_namespace = []

    # Build a nested dictionary representing the namespace hierarchy.
    for p in prompts:
        namespace = p.data.get("namespace")
        if namespace and isinstance(namespace, list) and namespace[0]:
            current_level = namespace_map
            for part in namespace:
                current_level = current_level.setdefault(part, {})
            # Use a special key to store the list of prompts at this level.
            current_level.setdefault("_prompts_", []).append(p.path.name)
        else:
            no_namespace.append(p.path.name)

    tree = Tree("🥡 [bold]Prompt Library[/bold]", guide_style="cyan")

    def build_tree(branch, data):
        """A recursive helper to build the rich.tree.Tree from the nested dict."""
        # Sort keys so folders appear before prompts.
        sorted_keys = sorted(data.keys(), key=lambda k: (k.startswith('_'), k))
        for key in sorted_keys:
            if key == "_prompts_":
                # Add prompt files to the current branch.
                for name in sorted(data[key]):
                    branch.add(f"📄 {escape(name)}")
            else:
                # Add a new folder (branch) and recurse into it.
                new_branch = branch.add(f"🗂 [bold]{escape(key)}[/bold]")
                build_tree(new_branch, data[key])

    # Build the main tree and then add any prompts that had no namespace.
    build_tree(tree, namespace_map)
    for name in sorted(no_namespace):
        tree.add(f"📄 {escape(name)} [dim](No Namespace)[/dim]")

    return tree