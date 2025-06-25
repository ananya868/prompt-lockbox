#
# FILE: prompt_lockbox/cli/search.py
#
"""
This file defines the CLI commands for building search indexes and searching
for prompts using various methods like fuzzy, hybrid, and SPLADE.
"""

import typer
from rich import print
from rich.table import Table
from rich.markup import escape

from prompt_lockbox.api import Project


def index(
    method: str = typer.Option("hybrid", "--method", "-m", help="Indexing method: 'hybrid' or 'splade'.")
):
    """Builds a search index for all prompts.

    This command processes all prompt files in the project and creates the
    necessary index files in the `.plb/` directory for the specified search
    method. This is a prerequisite for using advanced search commands.

    Args:
        method: The search indexing method to use ('hybrid' or 'splade').
    """
    try:
        project = Project()
        print(f"🚀 [bold]Building '{method}' search index... This may take a moment.[/bold]")
        # Delegate the core indexing logic to the SDK.
        project.index(method=method)
        print(f"\n✅ [bold green]Successfully built the '{method}' search index.[/bold green]")
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# Create a 'search' command group for organizing the different search methods.
search_app = typer.Typer(name="search", help="Search for prompts using different methods.", no_args_is_help=True)


def _print_search_results(results: list, title: str):
    """A helper function to display search results in a consistent Rich table.

    Args:
        results: A list of result dictionaries, each expecting 'score', 'name',
                 'path', and 'description' keys.
        title: The title to display for the results table.
    """
    if not results:
        print("No results found.")
        return

    # Define the table structure and styling.
    table = Table(title=title, show_header=True, highlight=True)
    table.add_column("Score", style="magenta", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Path", style="cyan")
    table.add_column("Description", style="yellow")

    # Populate the table with the search results.
    for res in results:
        table.add_row(f"{res['score']:.3f}", res['name'], res['path'], res['description'])
    print(table)


@search_app.command("hybrid")
def search_hybrid_cli(
    query: str = typer.Argument(..., help="The natural language search query."),
    limit: int = typer.Option(3, "--limit", "-l", help="Number of results to return."),
    alpha: float = typer.Option(0.5, "--alpha", "-a", help="Balance: 1.0=semantic, 0.0=keyword."),
):
    """Search using the Hybrid (TF-IDF + FAISS) engine.

    This method combines traditional keyword search (TF-IDF) with modern
    semantic search (FAISS) for balanced results. Requires a 'hybrid' index.

    Args:
        query: The search query.
        limit: The maximum number of results to return.
        alpha: The weighting factor between semantic (1.0) and keyword (0.0) search.
    """
    try:
        project = Project()
        # Delegate the actual search logic to the SDK.
        results = project.search(query, method="hybrid", limit=limit, alpha=alpha)
        # Use the helper function to display the results.
        _print_search_results(results, f"Hybrid Search Results for \"{escape(query)}\"")
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@search_app.command("splade")
def search_splade_cli(
    query: str = typer.Argument(..., help="The search query."),
    limit: int = typer.Option(3, "--limit", "-l", help="Number of results to return."),
):
    """Search using the powerful SPLADE sparse vector engine.

    This method provides highly relevant results by understanding the context
    and importance of words. Requires a 'splade' index.

    Args:
        query: The search query.
        limit: The maximum number of results to return.
    """
    try:
        project = Project()
        # Delegate the actual search logic to the SDK.
        results = project.search(query, method="splade", limit=limit)
        # Use the helper function to display the results.
        _print_search_results(results, f"SPLADE Search Results for \"{escape(query)}\"")
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@search_app.command("fuzzy")
def search_fuzzy_cli(
    query: str = typer.Argument(..., help="The search query."),
    limit: int = typer.Option(3, "--limit", "-l", help="Number of results to return."),
):
    """(Default) Perform a quick, lightweight fuzzy search on prompt names and metadata.

    This method does not require an index and provides fast results based on
    simple string matching against prompt metadata like names and descriptions.

    Args:
        query: The search query.
        limit: The maximum number of results to return.
    """
    try:
        project = Project()
        # Delegate the actual search logic to the SDK.
        results = project.search(query, method="fuzzy", limit=limit)

        # Use the helper function to display the results.
        _print_search_results(results, f"Fuzzy Search Results for \"{escape(query)}\"")

    except Exception as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)