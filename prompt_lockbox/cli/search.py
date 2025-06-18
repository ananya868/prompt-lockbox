#
# FILE: prompt_lockbox/cli/search.py (Complete Final Version)
#
import typer
from rich import print
from rich.table import Table
from rich.markup import escape

from prompt_lockbox.api import Project

def index(
    method: str = typer.Option("hybrid", "--method", "-m", help="Indexing method: 'hybrid' or 'splade'.")
):
    """Builds a search index for all prompts."""
    try:
        project = Project()
        print(f"🚀 [bold]Building '{method}' search index... This may take a moment.[/bold]")
        project.index(method=method)
        print(f"\n✅ [bold green]Successfully built the '{method}' search index.[/bold green]")
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

search_app = typer.Typer(name="search", help="Search for prompts using different methods.", no_args_is_help=True)

def _print_search_results(results: list, title: str):
    if not results:
        print("No results found.")
        return
    
    table = Table(title=title, show_header=True, highlight=True)
    table.add_column("Score", style="magenta", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Path", style="cyan")
    table.add_column("Description", style="yellow")
    
    for res in results:
        table.add_row(f"{res['score']:.3f}", res['name'], res['path'], res['description'])
    print(table)

@search_app.command("hybrid")
def search_hybrid_cli(
    query: str = typer.Argument(..., help="The natural language search query."),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results to return."),
    alpha: float = typer.Option(0.5, "--alpha", "-a", help="Balance: 1.0=semantic, 0.0=keyword."),
):
    """Search using the Hybrid (TF-IDF + FAISS) engine."""
    try:
        project = Project()
        results = project.search(query, method="hybrid", limit=limit, alpha=alpha)
        _print_search_results(results, f"Hybrid Search Results for \"{escape(query)}\"")
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

@search_app.command("splade")
def search_splade_cli(
    query: str = typer.Argument(..., help="The search query."),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results to return."),
):
    """Search using the powerful SPLADE sparse vector engine."""
    try:
        project = Project()
        results = project.search(query, method="splade", limit=limit)
        _print_search_results(results, f"SPLADE Search Results for \"{escape(query)}\"")
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)