import typer
from rich.console import Console

from tja_ai_chartgen import __version__

app = typer.Typer(help="AI-assisted TJA chart draft generator")
console = Console()


@app.command()
def version() -> None:
    console.print(f"tja-ai-chartgen {__version__}")


if __name__ == "__main__":
    app()
