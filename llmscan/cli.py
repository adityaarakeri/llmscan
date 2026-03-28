from __future__ import annotations

import json
import re

import typer
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import __version__
from .catalog import load_user_catalog, save_user_catalog
from .detector import MachineProfile, detect_machine, profile_json
from .estimator import RATING_ORDER, evaluate_models, load_catalog
from .huggingface import HuggingFaceError, get_model_files, parse_gguf_filename, search_gguf_models
from .vram import BITS_PER_WEIGHT, build_model_entry, estimate_vram

_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _validate_model_id(model_id: str) -> None:
    """Ensure a model ID contains only safe characters."""
    if not _MODEL_ID_PATTERN.match(model_id):
        raise typer.BadParameter(
            f"Invalid model ID '{model_id}'. "
            "Must start with alphanumeric and contain only lowercase letters, digits, hyphens, underscores, and dots."
        )


def _version_callback(value: bool) -> None:
    if value:
        print(f"llmscan {__version__}")
        raise typer.Exit()


app = typer.Typer(
    add_completion=False, no_args_is_help=False, help="Fast local LLM compatibility checker.", rich_markup_mode="rich"
)
console = Console()

_cached_profile: MachineProfile | None = None


def _get_profile() -> MachineProfile:
    """Return the machine profile, detecting only once per process."""
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = detect_machine()
    return _cached_profile


BANNER = """
██╗     ██╗     ███╗   ███╗    ███████╗ ██████╗ █████╗ ███╗   ██╗
██║     ██║     ████╗ ████║    ██╔════╝██╔════╝██╔══██╗████╗  ██║
██║     ██║     ██╔████╔██║    ███████╗██║     ███████║██╔██╗ ██║
██║     ██║     ██║╚██╔╝██║    ╚════██║██║     ██╔══██║██║╚██╗██║
███████╗███████╗██║ ╚═╝ ██║    ███████║╚██████╗██║  ██║██║ ╚████║
╚══════╝╚══════╝╚═╝     ╚═╝    ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝
"""


def _rating_style(rating: str) -> str:
    return {"great": "bold green", "ok": "bold cyan", "tight": "bold yellow", "no": "bold red"}.get(rating, "white")


def _badge(label: str, value: str, style: str) -> Panel:
    text = Text.assemble((label + "\n", "dim"), (value, style))
    return Panel(Align.center(text), border_style=style, padding=(0, 2))


def _render_header() -> None:
    group = Group(
        Text(BANNER, style="bold cyan"),
        Align.center(Text("fast local model fit inspector", style="bright_black")),
        Align.center(Text(f"v{__version__}", style="bold magenta")),
    )
    console.print(Panel(group, border_style="bright_black", padding=(0, 1)))


def _machine_summary_panel(profile: MachineProfile) -> None:
    primary_vram = max((g.vram_gb for g in profile.gpus), default=0)
    cards = [
        _badge("PRIMARY VRAM", f"{primary_vram} GB", "green" if primary_vram else "yellow"),
        _badge("SYSTEM RAM", f"{profile.ram_gb} GB", "cyan"),
        _badge("ARCH", profile.arch or "unknown", "magenta"),
        _badge("OS", profile.os, "blue"),
    ]
    console.print(Columns(cards, equal=True, expand=True))


def _gpu_lines(profile: MachineProfile) -> str:
    if not profile.gpus:
        return "[yellow]No dedicated GPU detected[/yellow]"
    lines = []
    for gpu in profile.gpus:
        count = f" x{gpu.count}" if gpu.count > 1 else ""
        lines.append(f"• {gpu.vendor} {gpu.name}{count} — {gpu.vram_gb} GB [dim]({gpu.source})[/dim]")
    return "\n".join(lines)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    if ctx.invoked_subcommand:
        return
    _render_header()
    profile = _get_profile()
    _machine_summary_panel(profile)
    console.print(
        Panel(
            f"[bold]CPU[/bold]  {profile.cpu or 'unknown'}\n[bold]GPU[/bold]\n{_gpu_lines(profile)}",
            title="System Snapshot",
            border_style="bright_black",
        )
    )
    console.print(Rule(style="bright_black"))
    list_models(min_rating="ok", catalog=None, json_output=False)
    console.print(
        "\n[dim]Try:[/dim]\n"
        "  [bold]llmscan scan[/bold]\n"
        "  [bold]llmscan list --min-rating tight[/bold]\n"
        "  [bold]llmscan explain llama-3.1-8b-instruct[/bold]\n"
        "  [bold]llmscan search llama[/bold]\n"
        "  [bold]llmscan add my-model --params-b 7 --quant Q4_K_M --family Llama[/bold]"
    )


@app.command("scan")
def scan(json_output: bool = typer.Option(False, "--json", help="Print raw machine profile as JSON.")) -> None:
    """Inspect the local machine."""
    profile = _get_profile()
    if json_output:
        console.print_json(profile_json(profile))
        return
    _render_header()
    _machine_summary_panel(profile)
    lines = [
        f"[bold]OS[/bold]             {profile.os}",
        f"[bold]Architecture[/bold]   {profile.arch}",
        f"[bold]CPU[/bold]            {profile.cpu or 'unknown'}",
        f"[bold]RAM[/bold]            {profile.ram_gb} GB",
    ]
    if profile.unified_memory_gb:
        lines.append(f"[bold]Unified memory[/bold] {profile.unified_memory_gb} GB")
    lines.append(f"[bold]GPU[/bold]\n{_gpu_lines(profile)}")
    console.print(Panel("\n".join(lines), title="Machine Profile", border_style="cyan"))


@app.command("list")
def list_models(
    min_rating: str = typer.Option("tight", help="Only show models at or above this rating: great, ok, tight, no."),
    catalog: str | None = typer.Option(None, help="Path to custom model catalog JSON."),
    json_output: bool = typer.Option(False, "--json", help="Print model results as JSON."),
) -> None:
    """List compatible models for this machine."""
    min_rating = min_rating.lower()
    if min_rating not in RATING_ORDER:
        raise typer.BadParameter("min-rating must be one of: great, ok, tight, no")

    profile = _get_profile()
    rows = evaluate_models(profile, load_catalog(catalog))
    threshold = RATING_ORDER[min_rating]
    filtered = [r for r in rows if RATING_ORDER[r["rating"]] >= threshold]

    if json_output:
        console.print_json(json.dumps({"machine": profile.to_dict(), "models": filtered}, indent=2))
        return

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Model", style="bold white")
    table.add_column("Family", style="bright_black")
    table.add_column("Params", justify="right")
    table.add_column("Fit", justify="center")
    table.add_column("Min VRAM", justify="right")
    table.add_column("Rec VRAM", justify="right")
    table.add_column("Rec RAM", justify="right")
    table.add_column("Notes", overflow="fold")

    for row in filtered:
        table.add_row(
            escape(row["id"]),
            escape(row["family"]),
            f"{row['params_b']}B",
            f"[{_rating_style(row['rating'])}]{row['rating']}[/]",
            f"{row['min_vram_gb']} GB",
            f"{row['recommended_vram_gb']} GB",
            f"{row['recommended_ram_gb']} GB",
            escape(row["fit_notes"]),
        )

    console.print(Panel(table, title=f"Model Fit Matrix • showing {min_rating}+", border_style="green", padding=(0, 1)))


@app.command()
def explain(
    model_id: str = typer.Argument(..., help="Model id from the catalog."),
    catalog: str | None = typer.Option(None, help="Path to custom model catalog JSON."),
) -> None:
    """Explain why a specific model is or is not a fit."""
    catalog_data = load_catalog(catalog)
    entry = next((m for m in catalog_data if m["id"].lower() == model_id.lower()), None)
    if not entry:
        raise typer.BadParameter(f"Model '{model_id}' was not found in the catalog")
    profile = _get_profile()
    match = evaluate_models(profile, [entry])[0]

    cards = Columns(
        [
            _badge(
                "FIT",
                match["rating"].upper(),
                "green"
                if match["rating"] == "great"
                else "yellow"
                if match["rating"] == "tight"
                else "cyan"
                if match["rating"] == "ok"
                else "red",
            ),
            _badge("SIZE", f"{match['params_b']}B", "magenta"),
            _badge("QUANT", match["quant"], "blue"),
        ],
        equal=True,
        expand=True,
    )
    console.print(cards)
    text = (
        f"[bold]Model[/bold]            {escape(match['id'])}\n"
        f"[bold]Family[/bold]           {escape(match['family'])}\n"
        f"[bold]Minimum VRAM[/bold]     {match['min_vram_gb']} GB\n"
        f"[bold]Recommended VRAM[/bold] {match['recommended_vram_gb']} GB\n"
        f"[bold]Recommended RAM[/bold]  {match['recommended_ram_gb']} GB\n"
        f"\n[bold]Why it fits[/bold]\n{escape(match['fit_notes'])}\n"
        f"\n[bold]Catalog note[/bold]\n{escape(match['notes'])}"
    )
    console.print(Panel(text, title="Model Explanation", border_style="magenta"))


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query for Hugging Face GGUF models."),
    limit: int = typer.Option(20, help="Maximum number of results."),
    json_output: bool = typer.Option(False, "--json", help="Print results as JSON."),
) -> None:
    """Search Hugging Face for GGUF models."""
    if not 1 <= limit <= 100:
        raise typer.BadParameter("limit must be between 1 and 100")
    try:
        results = search_gguf_models(query, limit=limit)
    except HuggingFaceError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from None

    if json_output:
        data = [
            {
                "repo_id": r.repo_id,
                "author": r.author,
                "model_name": r.model_name,
                "downloads": r.downloads,
                "likes": r.likes,
            }
            for r in results
        ]
        console.print_json(json.dumps(data, indent=2))
        return

    if not results:
        console.print("[yellow]No GGUF models found for that query.[/yellow]")
        return

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Repo ID", style="bold white")
    table.add_column("Author", style="bright_black")
    table.add_column("Downloads", justify="right")
    table.add_column("Likes", justify="right")

    for r in results:
        table.add_row(escape(r.repo_id), escape(r.author), f"{r.downloads:,}", f"{r.likes:,}")

    console.print(Panel(table, title="Hugging Face GGUF Models", border_style="cyan", padding=(0, 1)))
    console.print("[dim]Tip:[/dim] [bold]llmscan add <repo-id>[/bold] to add a model to your local catalog.")


@app.command("add")
def add_model(
    model_spec: str = typer.Argument(..., help="Model name or Hugging Face repo ID (e.g. TheBloke/Llama-2-7B-GGUF)."),
    params_b: float | None = typer.Option(None, "--params-b", help="Parameter count in billions."),
    quant: str | None = typer.Option(
        None, "--quant", help=f"Quantization type ({', '.join(sorted(BITS_PER_WEIGHT))})."
    ),
    family: str = typer.Option("Unknown", "--family", help="Model family name."),
    notes: str = typer.Option("", "--notes", help="Optional notes for this model."),
    force: bool = typer.Option(False, "--force", help="Overwrite if model ID already exists in user catalog."),
    json_output: bool = typer.Option(False, "--json", help="Print added entry as JSON."),
) -> None:
    """Add a model to your local catalog with auto-computed VRAM requirements."""
    model_id = model_spec

    # If it looks like a HF repo ID, try to auto-detect params and quant
    if "/" in model_spec and (params_b is None or quant is None):
        try:
            files = get_model_files(model_spec)
        except HuggingFaceError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from None

        if not files:
            console.print(f"[yellow]No GGUF files found in '{escape(model_spec)}'.[/yellow]")
            if params_b is None or quant is None:
                console.print("Please specify --params-b and --quant manually.")
                raise typer.Exit(1)

        # Try to infer quant from the first GGUF file if not provided
        if quant is None:
            for f in files:
                parsed = parse_gguf_filename(f.filename)
                if parsed:
                    quant = parsed[1]
                    console.print(f"[dim]Auto-detected quant: {quant}[/dim]")
                    break
            if quant is None:
                console.print("[yellow]Could not detect quant from filenames. Please specify --quant.[/yellow]")
                raise typer.Exit(1)

        # Try to infer params from repo name if not provided
        if params_b is None:
            from .huggingface import infer_params_from_name

            params_b = infer_params_from_name(model_spec)
            if params_b is not None:
                console.print(f"[dim]Auto-detected params: {params_b}B[/dim]")
            else:
                console.print("[yellow]Could not detect param count. Please specify --params-b.[/yellow]")
                raise typer.Exit(1)

        model_id = model_spec.split("/")[-1].lower()

    _validate_model_id(model_id)

    if params_b is None:
        console.print("[bold red]Error:[/bold red] --params-b is required for non-HF model specs.")
        raise typer.Exit(1)
    if quant is None:
        console.print("[bold red]Error:[/bold red] --quant is required for non-HF model specs.")
        raise typer.Exit(1)

    # Validate quant type
    try:
        estimate_vram(params_b, quant)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from None

    entry = build_model_entry(id=model_id, family=family, params_b=params_b, quant=quant, notes=notes)

    # Load existing user catalog and check for duplicates
    user_entries = load_user_catalog()
    existing_idx = next((i for i, m in enumerate(user_entries) if m["id"] == model_id), None)
    if existing_idx is not None and not force:
        console.print(
            f"[yellow]Model '{escape(model_id)}' already exists in your catalog. Use --force to overwrite.[/yellow]"
        )
        raise typer.Exit(1)

    if existing_idx is not None:
        user_entries[existing_idx] = entry
    else:
        user_entries.append(entry)

    save_user_catalog(user_entries)

    if json_output:
        console.print_json(json.dumps(entry, indent=2))
        return

    console.print(
        Panel(
            f"[bold]ID[/bold]               {escape(entry['id'])}\n"
            f"[bold]Family[/bold]           {escape(entry['family'])}\n"
            f"[bold]Params[/bold]           {entry['params_b']}B\n"
            f"[bold]Quant[/bold]            {escape(entry['quant'])}\n"
            f"[bold]Min VRAM[/bold]         {entry['min_vram_gb']} GB\n"
            f"[bold]Recommended VRAM[/bold] {entry['recommended_vram_gb']} GB\n"
            f"[bold]Recommended RAM[/bold]  {entry['recommended_ram_gb']} GB",
            title="[green]Model added to local catalog[/green]",
            border_style="green",
        )
    )


@app.command("remove")
def remove_model(
    model_id: str = typer.Argument(..., help="Model ID to remove from your local catalog."),
) -> None:
    """Remove a model from your local catalog."""
    user_entries = load_user_catalog()
    idx = next((i for i, m in enumerate(user_entries) if m["id"] == model_id), None)
    if idx is None:
        console.print(f"[bold red]Error:[/bold red] Model '{escape(model_id)}' not found in your local catalog.")
        console.print(
            "[dim]Note: Only models added via 'llmscan add' can be removed. Bundled models cannot be removed.[/dim]"
        )
        raise typer.Exit(1)

    user_entries.pop(idx)
    save_user_catalog(user_entries)
    console.print(f"[green]Removed '{escape(model_id)}' from your local catalog.[/green]")


if __name__ == "__main__":
    app()
