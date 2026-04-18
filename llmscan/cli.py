from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path

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
from .estimator import RATING_ORDER, VALID_BACKENDS, evaluate_models, load_catalog
from .huggingface import (
    HuggingFaceError,
    get_model_files,
    infer_params_from_name,
    parse_gguf_filename,
    search_gguf_models,
)
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


_COMPLETION_SCRIPTS = {
    "bash": """# llmscan bash completion
_llmscan_completion() {
    COMPREPLY=($(compgen -W "scan list explain search add doctor" -- "${COMP_WORDS[COMP_CWORD]}"))
}
complete -F _llmscan_completion llmscan
complete -F _llmscan_completion llmc
""",
    "zsh": """#compdef llmscan llmc
_llmscan_completion() {
    local -a commands
    commands=(
        'scan:Inspect the local machine'
        'list:List models and fit ratings'
        'explain:Explain model fit'
        'search:Search GGUF models'
        'add:Add a custom catalog entry'
        'doctor:Run environment checks'
    )
    _describe 'command' commands
}
compdef _llmscan_completion llmscan
compdef _llmscan_completion llmc
""",
    "fish": """# llmscan fish completion
complete -c llmscan -f -a "scan list explain search add doctor"
complete -c llmc -f -a "scan list explain search add doctor"
""",
}


def _completion_callback(value: str | None) -> None:
    if value is None:
        return
    script = _COMPLETION_SCRIPTS.get(value.lower())
    if script is None:
        raise typer.BadParameter("Shell must be one of: bash, zsh, fish.")
    typer.echo(script, nl=False)
    raise typer.Exit(code=0)


def _completion_install_path(shell: str) -> Path:
    home = Path.home()
    if shell == "bash":
        return home / ".local" / "share" / "bash-completion" / "completions" / "llmscan"
    if shell == "zsh":
        return home / ".zfunc" / "_llmscan"
    return home / ".config" / "fish" / "completions" / "llmscan.fish"


def _install_completion_callback(value: str | None) -> None:
    if value is None:
        return
    shell = value.lower()
    script = _COMPLETION_SCRIPTS.get(shell)
    if script is None:
        raise typer.BadParameter("Shell must be one of: bash, zsh, fish.")
    target = _completion_install_path(shell)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    if shell == "zsh" and str(target.parent) not in os.environ.get("FPATH", "").split(":"):
        typer.echo(f"Installed completion to {target}. Add {target.parent} to FPATH if needed.")
    else:
        typer.echo(f"Installed completion to {target}")
    raise typer.Exit(code=0)


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
    total_vram = round(sum(g.vram_gb * g.count for g in profile.gpus), 1)
    multi_gpu = total_vram > primary_vram
    cards = [
        _badge("PRIMARY VRAM", f"{primary_vram} GB", "green" if primary_vram else "yellow"),
        _badge("SYSTEM RAM", f"{profile.ram_gb} GB", "cyan"),
        _badge("ARCH", profile.arch or "unknown", "magenta"),
        _badge("OS", profile.os, "blue"),
    ]
    if multi_gpu:
        cards.insert(1, _badge("TOTAL VRAM", f"{total_vram} GB", "green"))
    console.print(Columns(cards, equal=True, expand=True))


def _gpu_lines(profile: MachineProfile) -> str:
    if not profile.gpus:
        return "[yellow]No dedicated GPU detected — scoring based on CPU-only inference[/yellow]"
    lines = []
    for gpu in profile.gpus:
        count = f" x{gpu.count}" if gpu.count > 1 else ""
        lines.append(f"• {gpu.vendor} {gpu.name}{count} — {gpu.vram_gb} GB [dim]({gpu.source})[/dim]")
    return "\n".join(lines)


def _warn_wsl2(profile: MachineProfile) -> None:
    """Print a WSL2 VRAM accuracy warning if running inside WSL2."""
    if not profile.is_wsl2:
        return
    console.print(
        Panel(
            "[bold yellow]WSL2 detected.[/bold yellow] "
            "nvidia-smi VRAM readings may be inaccurate under WSL2. "
            "Verify GPU memory values on the Windows host with [bold]nvidia-smi[/bold] or [bold]GPU-Z[/bold] "
            "before trusting these results.",
            title="[yellow]WSL2 Warning[/yellow]",
            border_style="yellow",
        )
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
    install_completion: str | None = typer.Option(
        None,
        "--install-completion",
        callback=_install_completion_callback,
        is_eager=True,
        help="Install shell completion for bash, zsh, or fish.",
    ),
    show_completion: str | None = typer.Option(
        None,
        "--show-completion",
        callback=_completion_callback,
        is_eager=True,
        help="Print shell completion for bash, zsh, or fish.",
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable color and Rich formatting (for CI / log capture)."
    ),
    plain: bool = typer.Option(False, "--plain", help="Alias for --no-color."),
) -> None:
    global console
    force_terminal = getattr(console, "_force_terminal", None)
    console = Console(
        no_color=bool(no_color or plain),
        highlight=not (no_color or plain),
        force_terminal=force_terminal,
        color_system=None if (no_color or plain) else "auto",
    )
    if ctx.invoked_subcommand:
        return
    _render_header()
    profile = _get_profile()
    _warn_wsl2(profile)
    _machine_summary_panel(profile)
    console.print(
        Panel(
            f"[bold]CPU[/bold]  {profile.cpu or 'unknown'}\n[bold]GPU[/bold]\n{_gpu_lines(profile)}",
            title="System Snapshot",
            border_style="bright_black",
        )
    )
    console.print(Rule(style="bright_black"))
    list_models(
        min_rating="ok", catalog=None, json_output=False, family=None, sort="rating", running=False, backend="llama-cpp"
    )
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
    _warn_wsl2(profile)
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
    total_vram = round(sum(g.vram_gb * g.count for g in profile.gpus), 1)
    primary_vram = max((g.vram_gb for g in profile.gpus), default=0)
    if total_vram > primary_vram:
        lines.append(f"[bold]Total VRAM[/bold]     {total_vram} GB (across all GPUs)")
    console.print(Panel("\n".join(lines), title="Machine Profile", border_style="cyan"))


_SORT_KEYS = {
    "rating": lambda r: (RATING_ORDER[r["rating"]], r["params_b"]),
    "params": lambda r: r["params_b"],
    "vram": lambda r: r["min_vram_gb"],
    "name": lambda r: r["id"],
}


_OLLAMA_API = "http://localhost:11434/api/tags"


def _fetch_ollama_running() -> set[str] | None:
    """Return the set of Ollama model name strings, or None if Ollama is unreachable."""
    import httpx

    try:
        resp = httpx.get(_OLLAMA_API, timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
        return {m["name"] for m in data.get("models", [])}
    except Exception:
        return None


def _is_running_in_ollama(model_id: str, ollama_names: set[str]) -> bool:
    """Return True if model_id appears (case-insensitively) in any Ollama model name."""
    mid = model_id.lower()
    return any(mid in name.lower() for name in ollama_names)


@app.command("list")
def list_models(
    min_rating: str = typer.Option("tight", help="Only show models at or above this rating: great, ok, tight, no."),
    catalog: str | None = typer.Option(None, help="Path to custom model catalog JSON."),
    json_output: bool = typer.Option(False, "--json", help="Print model results as JSON."),
    family: str | None = typer.Option(None, "--family", help="Filter by family name (case-insensitive substring)."),
    sort: str = typer.Option("rating", "--sort", help="Sort by: rating (default), params, vram, name."),
    running: bool = typer.Option(False, "--running", help="Cross-reference with Ollama running models."),
    backend: str = typer.Option(
        "llama-cpp", "--backend", help="Inference backend for scoring: llama-cpp (default), ollama, mlx."
    ),
    csv_output: bool = typer.Option(False, "--csv", help="Print model results as CSV (spreadsheet-friendly)."),
) -> None:
    """List compatible models for this machine."""
    min_rating = min_rating.lower()
    if min_rating not in RATING_ORDER:
        raise typer.BadParameter("min-rating must be one of: great, ok, tight, no")
    sort = sort.lower()
    if sort not in _SORT_KEYS:
        raise typer.BadParameter(f"--sort must be one of: {', '.join(_SORT_KEYS)}")
    backend = backend.lower()
    if backend not in VALID_BACKENDS:
        raise typer.BadParameter(f"--backend must be one of: {', '.join(sorted(VALID_BACKENDS))}")
    if csv_output and json_output:
        raise typer.BadParameter("--csv and --json cannot be used together")

    profile = _get_profile()
    rows = evaluate_models(profile, load_catalog(catalog), backend=backend)
    threshold = RATING_ORDER[min_rating]
    filtered = [r for r in rows if RATING_ORDER[r["rating"]] >= threshold]

    if family:
        filtered = [r for r in filtered if family.lower() in r["family"].lower()]

    reverse = sort != "name"
    filtered = sorted(filtered, key=_SORT_KEYS[sort], reverse=reverse)

    hidden_count = len(rows) - len([r for r in rows if RATING_ORDER[r["rating"]] >= threshold])

    ollama_names: set[str] | None = None
    ollama_unavailable = False
    if running:
        ollama_names = _fetch_ollama_running()
        if ollama_names is None:
            ollama_unavailable = True
            ollama_names = set()
        for row in filtered:
            row["running"] = _is_running_in_ollama(row["id"], ollama_names)

    if json_output:
        console.print_json(json.dumps({"machine": profile.to_dict(), "backend": backend, "models": filtered}, indent=2))
        return

    if csv_output:
        _CSV_COLUMNS = [
            "id",
            "family",
            "params_b",
            "quant",
            "rating",
            "reason_code",
            "min_vram_gb",
            "recommended_vram_gb",
            "recommended_ram_gb",
            "fit_notes",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(filtered)
        typer.echo(buf.getvalue(), nl=False)
        return

    if running and ollama_unavailable:
        console.print("[yellow]Ollama not reachable at localhost:11434 — running status unavailable.[/yellow]")

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Model", style="bold white")
    table.add_column("Family", style="bright_black")
    table.add_column("Params", justify="right")
    table.add_column("Fit", justify="center")
    table.add_column("Min VRAM", justify="right")
    table.add_column("Rec VRAM", justify="right")
    table.add_column("Rec RAM", justify="right")
    table.add_column("Notes", overflow="fold")
    if running:
        table.add_column("Running", justify="center")

    for row in filtered:
        reason = row.get("reason_code", "")
        fit_cell = f"[{_rating_style(row['rating'])}]{row['rating']}[/]"
        if reason:
            fit_cell += f" [dim]({escape(reason)})[/dim]"
        cells = [
            escape(row["id"]),
            escape(row["family"]),
            f"{row['params_b']}B",
            fit_cell,
            f"{row['min_vram_gb']} GB",
            f"{row['recommended_vram_gb']} GB",
            f"{row['recommended_ram_gb']} GB",
            escape(row["fit_notes"]),
        ]
        if running:
            cells.append("[bold green]●[/bold green]" if row.get("running") else "[dim]—[/dim]")
        table.add_row(*cells)

    console.print(Panel(table, title=f"Model Fit Matrix • showing {min_rating}+", border_style="green", padding=(0, 1)))

    if hidden_count > 0:
        console.print(
            f"[dim]+{hidden_count} model{'s' if hidden_count != 1 else ''} hidden "
            f'(rated "no" — insufficient hardware). '
            f"Run with [bold]--min-rating no[/bold] to show all.[/dim]"
        )


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
    min_params: float | None = typer.Option(None, "--min-params", help="Minimum parameter count in billions (e.g. 7)."),
    max_params: float | None = typer.Option(
        None, "--max-params", help="Maximum parameter count in billions (e.g. 70)."
    ),
) -> None:
    """Search Hugging Face for GGUF models."""
    if not 1 <= limit <= 100:
        raise typer.BadParameter("limit must be between 1 and 100")
    try:
        results = search_gguf_models(query, limit=limit)
    except HuggingFaceError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from None

    # Filter by parameter count when --min-params or --max-params is given.
    # Models whose param count cannot be inferred from the name are excluded
    # when any param filter is active (to avoid showing unknown-size models).
    if min_params is not None or max_params is not None:
        filtered_results = []
        for r in results:
            params = infer_params_from_name(r.repo_id)
            if params is None:
                continue  # exclude models with no inferable param count
            if min_params is not None and params < min_params:
                continue
            if max_params is not None and params > max_params:
                continue
            filtered_results.append(r)
        results = filtered_results

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
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview computed entry without saving to catalog."),
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
            detected_quant_file: str | None = None
            for f in files:
                parsed = parse_gguf_filename(f.filename)
                if parsed:
                    quant = parsed[1]
                    detected_quant_file = f.filename
                    break
            if quant is None:
                console.print(
                    "[bold red]Error:[/bold red] Could not detect quantization type from any GGUF filename in "
                    f"'{escape(model_spec)}'. Pass [bold]--quant Q4_K_M[/bold] (or your preferred quant) manually."
                )
                raise typer.Exit(1)
            console.print(
                f"[yellow]Auto-detected quant:[/yellow] [bold]{quant}[/bold] "
                f"[dim](from {escape(detected_quant_file or '')})[/dim] — "
                "use [bold]--quant[/bold] to override if incorrect."
            )

        # Try to infer params from repo name if not provided
        if params_b is None:
            from .huggingface import infer_params_from_name

            params_b = infer_params_from_name(model_spec)
            if params_b is None:
                console.print(
                    f"[bold red]Error:[/bold red] Could not detect parameter count from '{escape(model_spec)}'. "
                    "Pass [bold]--params-b N[/bold] manually (e.g. --params-b 7)."
                )
                raise typer.Exit(1)
            console.print(
                f"[yellow]Auto-detected params:[/yellow] [bold]{params_b}B[/bold] — "
                "use [bold]--params-b[/bold] to override if incorrect."
            )

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

    if dry_run:
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
                title="[yellow]Dry run — preview only, not saved[/yellow]",
                border_style="yellow",
            )
        )
        return

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


_DOCTOR_TOOLS = ["nvidia-smi", "rocm-smi", "xpu-smi", "sysctl", "wmic", "clinfo"]

_REMOTE_CATALOG_URL = "https://raw.githubusercontent.com/adityaarakeri/llmscan/main/llmscan/models.json"
_REMOTE_CATALOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB

catalog_app = typer.Typer(help="Manage the model catalog.", invoke_without_command=True, no_args_is_help=True)
app.add_typer(catalog_app, name="catalog")


def _fetch_remote_catalog(url: str) -> list[dict]:
    import httpx

    from .catalog import CatalogValidationError, validate_catalog

    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f"[bold red]Error:[/bold red] Failed to fetch remote catalog: {exc}")
        raise typer.Exit(1) from None

    raw = resp.content
    if len(raw) > _REMOTE_CATALOG_MAX_BYTES:
        console.print(f"[bold red]Error:[/bold red] Remote catalog response too large ({len(raw)} bytes).")
        raise typer.Exit(1)

    try:
        entries = resp.json()
        if not isinstance(entries, list):
            raise ValueError("Expected a JSON array")
    except Exception:
        console.print("[bold red]Error:[/bold red] Invalid JSON in remote catalog response.")
        raise typer.Exit(1) from None

    try:
        validate_catalog(entries)
    except CatalogValidationError as exc:
        console.print(f"[bold red]Error:[/bold red] Remote catalog failed validation: {exc}")
        raise typer.Exit(1) from None

    return entries


def _compute_diff(remote: list[dict], bundled: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Return (new_ids, updated_ids, removed_ids) comparing remote against bundled catalog."""
    bundled_by_id = {m["id"]: m for m in bundled}
    remote_ids = {m["id"] for m in remote}
    new_ids: list[str] = []
    updated_ids: list[str] = []
    for entry in remote:
        mid = entry["id"]
        if mid not in bundled_by_id:
            new_ids.append(mid)
        elif entry != bundled_by_id[mid]:
            updated_ids.append(mid)
    removed_ids = [mid for mid in bundled_by_id if mid not in remote_ids]
    return new_ids, updated_ids, removed_ids


@catalog_app.command("update")
def catalog_update(
    url: str = typer.Option(_REMOTE_CATALOG_URL, "--url", help="URL of the remote catalog JSON."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes without saving."),
    json_output: bool = typer.Option(False, "--json", help="Print diff as JSON."),
) -> None:
    """Fetch a remote catalog and merge new/updated models into the local catalog."""
    from .catalog import DEFAULT_MODELS

    remote = _fetch_remote_catalog(url)
    new_ids, updated_ids, removed_ids = _compute_diff(remote, DEFAULT_MODELS)

    if json_output:
        if not dry_run:
            user_entries = load_user_catalog()
            _apply_catalog_update(remote, user_entries)
        console.print_json(
            json.dumps({"new": new_ids, "updated": updated_ids, "removed": removed_ids, "dry_run": dry_run}, indent=2)
        )
        return

    # Build diff table
    if not new_ids and not updated_ids and not removed_ids:
        console.print("[green]Catalog is already up to date — no changes.[/green]")
        return

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold", title="Catalog Diff")
    table.add_column("Model ID", style="bold white")
    table.add_column("Change", justify="center")
    for mid in new_ids:
        table.add_row(escape(mid), "[bold green]new[/bold green]")
    for mid in updated_ids:
        table.add_row(escape(mid), "[bold cyan]updated[/bold cyan]")
    for mid in removed_ids:
        table.add_row(escape(mid), "[bold red]removed[/bold red]")
    console.print(Panel(table, border_style="cyan", padding=(0, 1)))

    if dry_run:
        console.print(
            f"[yellow]Dry run — {len(new_ids)} new, {len(updated_ids)} updated, "
            f"{len(removed_ids)} removed. No changes saved.[/yellow]"
        )
        return

    user_entries = load_user_catalog()
    _apply_catalog_update(remote, user_entries)
    console.print(
        f"[green]Catalog updated and saved — {len(new_ids)} new, {len(updated_ids)} updated, "
        f"{len(removed_ids)} removed from bundled catalog.[/green]"
    )


def _apply_catalog_update(remote: list[dict], user_entries: list[dict]) -> None:
    """Merge remote entries into the user catalog and save."""
    merged: dict[str, dict] = {m["id"]: dict(m) for m in user_entries}
    for entry in remote:
        merged[entry["id"]] = dict(entry)
    save_user_catalog(list(merged.values()))


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Print diagnostics as JSON."),
) -> None:
    """Check hardware detection method availability and flag anomalies."""
    import shutil

    tool_results: dict[str, dict] = {}
    for tool in _DOCTOR_TOOLS:
        path = shutil.which(tool)
        tool_results[tool] = {"available": path is not None, "path": path}

    profile = _get_profile()
    anomalies: list[str] = []
    for gpu in profile.gpus:
        if gpu.vram_gb == 0.0:
            anomalies.append(
                f"GPU '{gpu.name}' detected via {gpu.source} but VRAM reported as 0 GB — driver or query issue."
            )

    if json_output:
        data = {"tools": tool_results, "anomalies": anomalies, "gpus": [g.__dict__ for g in profile.gpus]}
        console.print_json(json.dumps(data, indent=2))
        return

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold", title="Detection Tools")
    table.add_column("Tool", style="bold white")
    table.add_column("Status", justify="center")
    table.add_column("Path", style="dim")
    for tool, info in tool_results.items():
        status = "[bold green]found[/bold green]" if info["available"] else "[bold red]missing[/bold red]"
        table.add_row(tool, status, info["path"] or "—")
    console.print(Panel(table, title="[bold]llmscan doctor[/bold]", border_style="cyan", padding=(0, 1)))

    if not profile.gpus:
        console.print("[yellow]No dedicated GPU detected — scoring based on CPU-only inference.[/yellow]")
    else:
        for gpu in profile.gpus:
            console.print(f"• {gpu.vendor} {gpu.name} — {gpu.vram_gb} GB [dim]({gpu.source})[/dim]")

    if anomalies:
        for a in anomalies:
            console.print(Panel(f"[bold yellow]Anomaly:[/bold yellow] {escape(a)}", border_style="yellow"))
    else:
        console.print("[green]No anomalies detected.[/green]")


_PYPI_URL = "https://pypi.org/pypi/llmscan/json"


@app.command("version")
def version_command(
    check: bool = typer.Option(False, "--check", help="Check PyPI for the latest available version."),
) -> None:
    """Show the installed llmscan version and optionally check for updates."""
    console.print(f"llmscan [bold]{__version__}[/bold]")
    if not check:
        return
    import httpx

    try:
        resp = httpx.get(_PYPI_URL, timeout=2.0)
        resp.raise_for_status()
        latest = resp.json()["info"]["version"]
    except Exception:
        console.print("[yellow]Could not reach PyPI to check for updates.[/yellow]")
        return

    if latest == __version__:
        console.print(f"[green]You are up to date.[/green] (latest: {latest})")
    else:
        console.print(
            f"[yellow]Update available:[/yellow] {__version__} → [bold]{latest}[/bold]\n"
            f"  [dim]pip install --upgrade llmscan[/dim]"
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
