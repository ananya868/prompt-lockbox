#
# FILE: prompt_lockbox/cli/main.py (Updated again)
#

import typer
from .project import init, status, lock, unlock, verify
from .manage import list_prompts, show, create, run, version

app = typer.Typer(
    name="plb",
    help="A framework to secure, manage, and develop prompts.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="markdown"
)

# Attach the command directly to the main app
app.command(rich_help_panel="Project & Integrity")(init)
app.command(rich_help_panel="Project & Integrity")(status) # <--- Add this line
app.command(rich_help_panel="Project & Integrity")(lock)      # <--- Add this
app.command(rich_help_panel="Project & Integrity")(unlock)    # <--- Add this
app.command(rich_help_panel="Project & Integrity")(verify)    # <--- Add this


# Prompt Management Commands
# To get the desired command name `list`, we specify it here
app.command("list", rich_help_panel="Prompt Management")(list_prompts)
app.command(rich_help_panel="Prompt Management")(show)
app.command(rich_help_panel="Prompt Management")(create)  # <--- Add this
app.command(rich_help_panel="Prompt Management")(run)     # <--- Add this
app.command(rich_help_panel="Prompt Management")(version) # <--- Add this



def run():
    """The main function to run the Typer app."""
    app()


