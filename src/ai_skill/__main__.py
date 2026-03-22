"""CLI entry point for ai-skill.

Commands:
    workspace begin            Create a new research project workspace (interactive wizard).
    workspace list             List all existing workspaces.
    workspace files [name]     List files available inside a workspace.
    research <topic>           Start a new research pipeline for the given topic.
    status [workspace]         Show the current status of an existing workspace.
    resume [workspace]         Resume a paused research pipeline.
    graph                      Print the Mermaid diagram of the pipeline graph.
    skills list                List all registered skills.
    skills show <name>         Show details for a specific skill.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# LangGraph raises GraphInterrupt when interrupt_before fires (some versions)
try:
    from langgraph.errors import GraphInterrupt as _GRAPH_INTERRUPT  # type: ignore[import-untyped]
except ImportError:
    _GRAPH_INTERRUPT = type("_NeverRaised", (BaseException,), {})  # type: ignore[assignment,misc]

app = typer.Typer(
    name="ai-skill",
    help="AI Academic Research Pipeline Agent",
    add_completion=False,
    no_args_is_help=True,
)
skills_app = typer.Typer(help="Manage and inspect skills.", no_args_is_help=True)
workspace_app = typer.Typer(help="Gerenciar workspaces de pesquisa.", no_args_is_help=True)
app.add_typer(skills_app, name="skills")
app.add_typer(workspace_app, name="workspace")

console = Console()
logger = logging.getLogger(__name__)


def _configure_logging(level: str = "WARNING") -> None:
    """Configure root logging with RichHandler.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=Console(stderr=True))],
    )


def _default_workspace(topic: str) -> Path:
    """Return a workspace path derived from the topic string.

    Args:
        topic: The research topic.

    Returns:
        Path inside AI_SKILL_WORKSPACE_DIR (default: ./research-workspace).
    """
    base = Path(os.environ.get("AI_SKILL_WORKSPACE_DIR", "./research-workspace"))
    slug = topic.lower().replace(" ", "-")[:40]
    return base / slug


# ---------------------------------------------------------------------------
# Workspace commands
# ---------------------------------------------------------------------------


@workspace_app.command("begin")
def workspace_begin() -> None:
    """Iniciar um novo workspace de pesquisa (wizard interativo)."""
    from ai_skill.core.project_workspace import ProjectWorkspace, default_projects_root, slugify

    console.print(Panel("[bold]Novo Workspace de Pesquisa[/bold]", expand=False))
    console.print(
        "\nEscolha um [bold]nome curto[/bold] para identificar esta pesquisa.\n"
        "[dim]Use letras, números e hífens. Máximo 50 caracteres.\n"
        "Caracteres especiais e espaços são convertidos automaticamente.[/dim]\n"
    )

    # --- Prompt for name ---
    while True:
        raw_name = typer.prompt("Nome da pesquisa").strip()
        if not raw_name:
            console.print("[red]Nome não pode ser vazio.[/red]")
            continue
        try:
            slug = slugify(raw_name)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        if len(slug) > 50:
            console.print(
                f"[red]Nome muito longo ({len(slug)} chars após sanitização). "
                "Use no máximo 50 caracteres.[/red]"
            )
            continue
        console.print(f"  [dim]→ pasta: [/dim][cyan]{slug}[/cyan]")
        break

    # --- Prompt for topic ---
    topic = typer.prompt(
        "\nTópico ou questão de pesquisa (pode ser mais descritivo)",
        default=raw_name,
    ).strip()

    # --- Create workspace ---
    root = default_projects_root()
    ws = ProjectWorkspace(raw_name, root=root)

    if ws.exists():
        console.print(f"\n[yellow]Workspace já existe:[/yellow] {ws.path}")
        if not typer.confirm("Deseja abrir o workspace existente?", default=True):
            raise typer.Exit(0)
    else:
        try:
            ws.create(topic=topic)
        except FileExistsError:
            console.print(f"[red]Workspace já existe:[/red] {ws.path}")
            raise typer.Exit(1)

        console.print(f"\n[green]✓[/green] Workspace criado em: [bold]{ws.path}[/bold]")
        console.print(f"  [green]✓[/green] README.md  — Metodologia da Pesquisa")
        console.print(f"  [green]✓[/green] attachments/  — coloque aqui arquivos para o agente")
        console.print(f"  [green]✓[/green] workspace.yaml  — metadados do projeto")

    console.print(
        f"\n[dim]Para iniciar a pesquisa neste workspace:[/dim]\n"
        f'  ai-skill begin-research "{topic}" --workspace "{ws.path}"'
    )

    if typer.confirm("\nIniciar a pesquisa agora?", default=True):
        begin_research(topic=topic, workspace=ws.path, model="")


@workspace_app.command("list")
def workspace_list() -> None:
    """Listar todos os workspaces de pesquisa disponíveis."""
    from ai_skill.core.project_workspace import default_projects_root, list_workspaces

    root = default_projects_root()
    workspaces = list_workspaces(root)

    if not workspaces:
        console.print(f"[dim]Nenhum workspace encontrado em:[/dim] {root}")
        console.print("[dim]Crie um com:[/dim]  ai-skill workspace begin")
        return

    table = Table(title=f"Workspaces de Pesquisa  [dim]({root})[/dim]")
    table.add_column("#", style="dim", width=3)
    table.add_column("Nome", style="cyan")
    table.add_column("Tópico")
    table.add_column("Checkpoint", justify="center")
    table.add_column("Status")
    table.add_column("Criado em", style="dim")

    for i, ws in enumerate(workspaces, start=1):
        meta = ws.load_metadata()
        table.add_row(
            str(i),
            meta.get("name", ws.slug),
            (meta.get("topic", "?"))[:45] + ("…" if len(meta.get("topic", "")) > 45 else ""),
            str(meta.get("current_checkpoint", 0)),
            meta.get("status", "?"),
            (meta.get("created", "?"))[:10],
        )

    console.print(table)


@workspace_app.command("files")
def workspace_files(
    name: Annotated[
        str | None,
        typer.Argument(help="Nome ou slug do workspace. Omita para seleção interativa."),
    ] = None,
) -> None:
    """Listar arquivos disponíveis em um workspace (checkpoints + attachments)."""
    ws = _resolve_workspace(name)
    if ws is None:
        raise typer.Exit(1)

    files = ws.list_all_files()
    if not files:
        console.print(f"[dim]Nenhum arquivo disponível em:[/dim] {ws.path}")
        console.print(
            f"  Outputs de checkpoint aparecem aqui automaticamente.\n"
            f"  Coloque arquivos de entrada em: [cyan]{ws.attachments_path}[/cyan]"
        )
        return

    console.print(f"\n[bold]Arquivos em:[/bold] {ws.path}\n")
    checkpoints = ws.list_checkpoints()
    attachments = ws.list_attachments()

    if checkpoints:
        console.print("[bold]Checkpoints:[/bold]")
        for i, f in enumerate(checkpoints, start=1):
            console.print(f"  [cyan]{i:>2}.[/cyan] {f.name}")

    if attachments:
        offset = len(checkpoints)
        console.print("\n[bold]Attachments:[/bold]")
        for i, f in enumerate(attachments, start=offset + 1):
            rel = f.relative_to(ws.path)
            console.print(f"  [cyan]{i:>2}.[/cyan] {rel}")

    console.print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("begin-research")
def begin_research(
    topic: Annotated[str, typer.Argument(help="Research topic or question.")],
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Custom workspace directory."),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Claude model ID to use."),
    ] = "",
) -> None:
    """Iniciar o Research Charter (Checkpoint 1) para um novo projeto de pesquisa."""
    _configure_logging(os.environ.get("AI_SKILL_LOG_LEVEL", "WARNING"))

    if model:
        os.environ["AI_SKILL_MODEL"] = model

    ws_path = workspace or _default_workspace(topic)

    # If the given path is a ProjectWorkspace, keep research artefacts inside
    # it but store the LangGraph internal state under <project>/.state/
    if (ws_path / "workspace.yaml").exists():
        state_path = ws_path / ".state"
        _warn_overwrite(ws_path, checkpoints=[1])
    else:
        state_path = ws_path

    console.print(f"[bold]Starting research:[/bold] {topic}")
    console.print(f"[dim]Workspace:[/dim] {ws_path}")

    # Late imports keep CLI startup fast
    from ai_skill.core.graph import build_cp1_graph
    from ai_skill.core.state import initial_state
    from ai_skill.core.workspace import ResearchWorkspace

    ws = ResearchWorkspace(state_path)
    ws.initialise(topic)

    graph = build_cp1_graph()
    state = initial_state(state_path, topic=topic)
    current_state: dict = dict(state)

    try:
        for event in graph.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    ws.save_state(current_state)
                    _handle_checkpoint(node_output, current_state, state_path)
                    return
                _print_node_progress(node_name, node_output)
                if isinstance(node_output, dict):
                    current_state.update(node_output)
                    ws.save_state(current_state)  # persist after every node
    except _GRAPH_INTERRUPT as _exc:  # type: ignore[misc]
        # LangGraph raises GraphInterrupt instead of yielding an event in some versions
        ws.save_state(current_state)
        _handle_checkpoint(_exc, current_state, state_path)
        return
    except KeyboardInterrupt:
        ws.save_state(current_state)
        console.print("\n[yellow]Research paused. Resume with:[/yellow]")
        console.print(f"  ai-skill resume --workspace {state_path}")
        sys.exit(0)


@app.command()
def status(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Workspace directory to inspect."),
    ] = Path("./research-workspace"),
) -> None:
    """Show the current status of an existing research workspace."""
    _configure_logging()

    from ai_skill.core.workspace import ResearchWorkspace

    ws = ResearchWorkspace(workspace)
    state = ws.load_state()
    if state is None:
        console.print(f"[red]No research state found in:[/red] {workspace}")
        raise typer.Exit(1)

    objective = state.get("objective", {})
    console.print(f"[bold]Topic:[/bold] {objective.get('topic', '(unknown)')}")
    console.print(f"[bold]Stage:[/bold] {state.get('stage', '?')}")
    console.print(f"[bold]Status:[/bold] {state.get('status', '?')}")
    console.print(f"[bold]Attempt:[/bold] {state.get('attempt', 0)}")
    findings_count = len(state.get("findings", []))
    console.print(f"[bold]Findings:[/bold] {findings_count}")


@app.command()
def resume(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Workspace directory to resume."),
    ] = Path("./research-workspace"),
    correct: Annotated[
        bool,
        typer.Option(
            "--correct", "-c",
            help="Read the last preview docx, extract corrections and generate a new preview.",
            is_flag=True,
        ),
    ] = False,
) -> None:
    """Resume a paused research pipeline.

    Without flags: approve the current checkpoint and advance.\n
    With --correct: read the last preview .docx, apply the researcher's
    edits and generate a new preview for review.
    """
    _configure_logging(os.environ.get("AI_SKILL_LOG_LEVEL", "WARNING"))

    from ai_skill.core.graph import build_cp1_graph, build_cp2_graph
    from ai_skill.core.workspace import ResearchWorkspace

    ws = ResearchWorkspace(workspace)
    state = ws.load_state()
    project_path = workspace.parent
    is_project_workspace = (project_path / "workspace.yaml").exists()

    if state is None:
        if correct and is_project_workspace:
            console.print(
                "[yellow]State file ausente — reconstruindo a partir do workspace...[/yellow]"
            )
            try:
                state = _reconstruct_state_for_correct(project_path, workspace)
            except Exception as _rec_exc:
                logger.debug("State reconstruction failed: %s", _rec_exc)
                state = None
        if state is None:
            console.print(f"[red]No state found in:[/red] {workspace}")
            console.print(
                f"[dim]Re-run:[/dim]  ai-skill begin-research \"<tópico>\" --workspace \"{project_path}\""
            )
            raise typer.Exit(1)

    charter_approved: bool = bool(state.get("charter_approved"))
    active_cp: int = int(state.get("active_checkpoint", 1))

    # --- Approve path (no --correct): directly record approval, no graph re-run ---
    if not correct:
        if not charter_approved:
            # CP1 approval
            state["charter_approved"] = True
            ws.save_state(state)
            _finalize_checkpoint(workspace, 1)
            _handle_cp2_start(state, workspace)
        elif active_cp == 2 and not state.get("literature_approved"):
            # CP2 approval
            state["literature_approved"] = True
            ws.save_state(state)
            _finalize_checkpoint(workspace, 2)
            console.print(
                "\n[bold green]✓ Checkpoint 2 aprovado![/bold green] "
                "Revisão bibliográfica finalizada."
            )
        else:
            console.print(
                "[yellow]Nenhum checkpoint pendente encontrado.[/yellow]\n"
                "[dim]Use --correct para aplicar correções a um preview existente.[/dim]"
            )
        return

    # --- Correct path (--correct): extract feedback, re-run appropriate graph ---
    if not is_project_workspace:
        console.print(
            "[red]--correct requer um ProjectWorkspace. "
            "Use --workspace apontando para o diretório .state/ dentro do workspace.[/red]"
        )
        raise typer.Exit(1)

    from ai_skill.core.project_workspace import ProjectWorkspace
    pw = ProjectWorkspace.from_path(project_path)

    # Guard: if [final].docx already exists, confirm the user wants to revise it
    final_path = pw.checkpoint_final_path(active_cp)
    if final_path.exists():
        console.print(
            f"\n[bold yellow]⚠ Já existe um arquivo aprovado:[/bold yellow]\n"
            f"  {final_path.name}"
        )
        if not typer.confirm(
            "Deseja continuar e gerar um novo preview a partir do arquivo final?",
            default=False,
        ):
            console.print("[dim]Operação cancelada.[/dim]")
            raise typer.Exit(0)

    last_preview = pw.get_last_preview(active_cp)
    if last_preview is None:
        if final_path.exists():
            last_preview = final_path
        else:
            cp_names = {1: "Checkpoint 1 (Research Charter)", 2: "Checkpoint 2 (Literature Review)"}
            console.print(
                f"[red]Nenhum preview encontrado para {cp_names.get(active_cp, f'Checkpoint {active_cp}')}. "
                "Execute a pesquisa primeiro.[/red]"
            )
            raise typer.Exit(1)

    console.print(f"[dim]Analisando preview:[/dim] {last_preview.name}")
    corrections = _extract_docx_corrections(last_preview)

    if corrections is None:
        console.print(
            "\n[green]✓ Sem correções pendentes[/green] — o documento não contém "
            "marcas de revisão, comentários nem realces."
        )
        if typer.confirm("\nDeseja gerar o arquivo final?", default=True):
            _finalize_checkpoint(workspace, active_cp)
        raise typer.Exit(0)

    state["user_feedback"] = _format_corrections_for_llm(corrections)

    # Select graph based on active checkpoint
    graph = build_cp1_graph() if active_cp == 1 else build_cp2_graph()
    console.print(f"[bold]Aplicando correções — CP{active_cp}:[/bold] {state.get('objective', {}).get('topic', '?')}")
    current_state: dict = dict(state)

    try:
        for event in graph.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    ws.save_state(current_state)
                    _handle_checkpoint(node_output, current_state, workspace)
                    return
                _print_node_progress(node_name, node_output)
                if isinstance(node_output, dict):
                    current_state.update(node_output)
                    ws.save_state(current_state)
    except _GRAPH_INTERRUPT as _exc:  # type: ignore[misc]
        ws.save_state(current_state)
        _handle_checkpoint(_exc, current_state, workspace)
        return
    except KeyboardInterrupt:
        ws.save_state(current_state)
        console.print("\n[yellow]Research paused. Resume with:[/yellow]")
        console.print(f"  ai-skill resume --workspace \"{workspace}\"")
        sys.exit(0)


@app.command("begin-literature")
def begin_literature(
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace", "-w",
            help="ProjectWorkspace ou diretório .state/ do projeto.",
        ),
    ] = Path("./research-workspace"),
) -> None:
    """Iniciar a Revisão Bibliográfica (Checkpoint 2).

    Requer que o Checkpoint 1 (Research Charter) já esteja aprovado,
    ou seja, que o arquivo [final].docx do CP1 exista no workspace.
    """
    _configure_logging(os.environ.get("AI_SKILL_LOG_LEVEL", "WARNING"))

    # --- Resolve project path and state path ---
    if (workspace / "workspace.yaml").exists():
        project_path = workspace
        state_path = workspace / ".state"
    elif (workspace.parent / "workspace.yaml").exists():
        project_path = workspace.parent
        state_path = workspace
    else:
        console.print(
            "[red]Workspace não reconhecido.[/red]\n"
            "[dim]Aponte --workspace para o ProjectWorkspace "
            "ou para o diretório .state/ dentro dele.[/dim]"
        )
        raise typer.Exit(1)

    # --- Validate CP1 [final].docx exists ---
    from ai_skill.core.project_workspace import ProjectWorkspace
    pw = ProjectWorkspace.from_path(project_path)
    final_cp1 = pw.checkpoint_final_path(1)

    if not final_cp1.exists():
        console.print(
            f"\n[red]✗ Checkpoint 1 não aprovado.[/red]\n"
            f"  O arquivo [bold]{final_cp1.name}[/bold] não foi encontrado em:\n"
            f"  {project_path}\n"
            f"\n[dim]Complete o Checkpoint 1 antes de iniciar a Revisão Bibliográfica:[/dim]\n"
            f"  ai-skill begin-research \"<tópico>\" --workspace \"{project_path}\"\n"
            f"  ai-skill resume --workspace \"{state_path}\""
        )
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Checkpoint 1 aprovado: [dim]{final_cp1.name}[/dim]")

    # --- Load or reconstruct state ---
    from ai_skill.core.graph import build_cp2_graph
    from ai_skill.core.workspace import ResearchWorkspace

    ws = ResearchWorkspace(state_path)
    state = ws.load_state()

    if state is None:
        state = _reconstruct_state_for_literatura(project_path, state_path)
        if state is None:
            console.print(
                "[red]Não foi possível carregar o estado do projeto.[/red]\n"
                "[dim]Tente re-rodar:[/dim]\n"
                f"  ai-skill begin-research \"<tópico>\" --workspace \"{project_path}\""
            )
            raise typer.Exit(1)

    # Guarantee charter is approved so the graph passes through the gate
    state["charter_approved"] = True
    state["user_feedback"] = None

    # Inject CP1 final.docx text so compile_literature can build on it
    state["charter_document_text"] = _read_docx_text(final_cp1)

    topic = (state.get("objective") or {}).get("topic", "")
    console.print(
        f"\n[bold]Iniciando Revisão Bibliográfica[/bold]"
        + (f": {topic}" if topic else "")
    )

    # --- Stream the CP2 graph ---
    graph_instance = build_cp2_graph()
    current_state: dict = dict(state)

    try:
        for event in graph_instance.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    ws.save_state(current_state)
                    _handle_checkpoint(node_output, current_state, state_path)
                    return
                _print_node_progress(node_name, node_output)
                if isinstance(node_output, dict):
                    current_state.update(node_output)
                    ws.save_state(current_state)
                    if (
                        node_name == "review_literature"
                        and node_output.get("literature_approved")
                    ):
                        _finalize_checkpoint(state_path, 2)
    except _GRAPH_INTERRUPT as _exc:  # type: ignore[misc]
        ws.save_state(current_state)
        _handle_checkpoint(_exc, current_state, state_path)
        return
    except KeyboardInterrupt:
        ws.save_state(current_state)
        console.print("\n[yellow]Pesquisa pausada. Retome com:[/yellow]")
        console.print(f"  ai-skill resume --workspace \"{state_path}\"")
        sys.exit(0)


@app.command()
def graph() -> None:
    """Print the Mermaid diagram of the research pipeline graph."""
    from ai_skill.core.graph import get_graph_mermaid

    mermaid = get_graph_mermaid()
    console.print(mermaid)


# ---------------------------------------------------------------------------
# Skills sub-commands
# ---------------------------------------------------------------------------


@skills_app.command("list")
def skills_list() -> None:
    """List all registered skills."""
    from ai_skill.skills.registry import SkillRegistry

    registry = SkillRegistry()
    registry.auto_discover()

    table = Table(title="Registered Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Tags")
    table.add_column("Description")

    for meta in registry.all():
        table.add_row(
            meta["name"],
            meta["version"],
            ", ".join(meta.get("tags", [])),
            meta["description"][:60] + ("…" if len(meta["description"]) > 60 else ""),
        )

    console.print(table)


@skills_app.command("show")
def skills_show(
    name: Annotated[str, typer.Argument(help="Skill name to inspect.")],
) -> None:
    """Show details for a specific skill."""
    from ai_skill.skills.registry import SkillRegistry

    registry = SkillRegistry()
    registry.auto_discover()

    skill = registry.get(name)
    if skill is None:
        console.print(f"[red]Skill not found:[/red] {name}")
        raise typer.Exit(1)

    meta = skill.SKILL_META
    console.print(f"[bold]{meta.name}[/bold] v{meta.version}")
    console.print(f"[dim]{meta.description}[/dim]")
    console.print(f"Author: {meta.author}  License: {meta.license}")
    console.print(f"Tags: {', '.join(meta.tags)}")
    if meta.dependencies:
        console.print(f"Depends on: {', '.join(meta.dependencies)}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_workspace(name: str | None) -> "ProjectWorkspace | None":  # noqa: F821
    """Resolve a ProjectWorkspace by name or interactive selection.

    Args:
        name: Workspace name/slug, or None for interactive selection.

    Returns:
        The resolved ProjectWorkspace, or None if not found / cancelled.
    """
    from ai_skill.core.project_workspace import ProjectWorkspace, default_projects_root, list_workspaces

    root = default_projects_root()

    if name:
        ws = ProjectWorkspace(name, root=root)
        if not ws.exists():
            console.print(f"[red]Workspace não encontrado:[/red] {name}")
            console.print(f"[dim]Workspaces disponíveis em:[/dim] {root}")
            return None
        return ws

    workspaces = list_workspaces(root)
    if not workspaces:
        console.print(f"[red]Nenhum workspace encontrado em:[/red] {root}")
        console.print("[dim]Crie um com:[/dim]  ai-skill workspace begin")
        return None

    if len(workspaces) == 1:
        return workspaces[0]

    console.print("\n[bold]Workspaces disponíveis:[/bold]")
    for i, ws in enumerate(workspaces, start=1):
        meta = ws.load_metadata()
        console.print(f"  [cyan]{i}.[/cyan] {meta.get('name', ws.slug)}")

    raw = typer.prompt("\nEscolha o número do workspace", default="1")
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(workspaces):
            return workspaces[idx]
        console.print("[red]Número inválido.[/red]")
        return None
    except ValueError:
        console.print("[red]Entrada inválida.[/red]")
        return None


def prompt_file_selection(workspace_path: Path) -> Path | None:
    """Present a numbered list of workspace files and return the user's choice.

    Lists checkpoint ``.docx`` files first, then files in ``attachments/``.
    Returns None if the workspace is empty or the user cancels.

    Args:
        workspace_path: Root directory of the ProjectWorkspace.

    Returns:
        The selected Path, or None.
    """
    from ai_skill.core.project_workspace import ProjectWorkspace

    ws = ProjectWorkspace.from_path(workspace_path)
    files = ws.list_all_files()
    if not files:
        console.print("[dim]Nenhum arquivo disponível no workspace.[/dim]")
        return None

    console.print("\n[bold]Arquivos disponíveis:[/bold]")
    for i, f in enumerate(files, start=1):
        rel = f.relative_to(workspace_path)
        console.print(f"  [cyan]{i:>2}.[/cyan] {rel}")

    raw = typer.prompt("\nNúmero do arquivo (Enter para cancelar)", default="")
    if not raw.strip():
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(files):
            return files[idx]
        console.print("[red]Número inválido.[/red]")
        return None
    except ValueError:
        console.print("[red]Entrada inválida.[/red]")
        return None


def _warn_overwrite(project_path: Path, checkpoints: list[int]) -> None:
    """Warn the user if checkpoint work already exists and ask to confirm.

    Previews auto-increment so they are never overwritten, but if a
    [final].docx already exists the user is re-starting an approved stage.

    Args:
        project_path: Root of the ProjectWorkspace.
        checkpoints: Checkpoint numbers to inspect.
    """
    from ai_skill.core.project_workspace import ProjectWorkspace

    pw = ProjectWorkspace.from_path(project_path)
    finals, previews = [], []
    for n in checkpoints:
        final = pw.checkpoint_final_path(n)
        if final.exists():
            finals.append(final)
        else:
            last_preview = pw.get_last_preview(n)
            if last_preview is not None:
                previews.append(last_preview)

    if finals:
        console.print("\n[bold yellow]⚠ Um checkpoint já foi aprovado:[/bold yellow]")
        for f in finals:
            console.print(f"  [yellow]•[/yellow] {f.name}")
        console.print(
            "[dim]Iniciar uma nova pesquisa gerará um novo preview (o arquivo final não será alterado agora).[/dim]"
        )
        if not typer.confirm("\nDeseja continuar mesmo assim?", default=False):
            console.print("[dim]Operação cancelada.[/dim]")
            raise typer.Exit(0)
    elif previews:
        console.print("\n[bold yellow]⚠ Já existem previews neste workspace:[/bold yellow]")
        for f in previews:
            console.print(f"  [yellow]•[/yellow] {f.name}")
        console.print(
            "[dim]Iniciar uma nova pesquisa gerará um novo preview numerado (os anteriores não serão apagados).[/dim]"
        )
        if not typer.confirm("\nDeseja continuar mesmo assim?", default=False):
            console.print("[dim]Operação cancelada.[/dim]")
            raise typer.Exit(0)


def _read_docx_text(path: Path) -> str:
    """Extract plain text from a .docx file paragraph by paragraph.

    Args:
        path: Path to the .docx file.

    Returns:
        Plain text content of the document, or an error message.
    """
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        return f"(Erro ao ler o documento: {exc})"


def _finalize_checkpoint(state_path: Path, number: int) -> None:
    """Copy the latest preview to [final].docx when a checkpoint is approved.

    If a [final].docx already exists the user is asked to confirm overwrite.
    Skips silently when state_path is not inside a ProjectWorkspace.

    Args:
        state_path: Path to the ResearchWorkspace (.state/) directory.
        number: Checkpoint number to finalize.
    """
    project_path = state_path.parent
    if not (project_path / "workspace.yaml").exists():
        return
    from ai_skill.core.project_workspace import ProjectWorkspace
    pw = ProjectWorkspace.from_path(project_path)
    final_path = pw.checkpoint_final_path(number)

    if final_path.exists():
        console.print(
            f"\n[bold yellow]⚠ Já existe um arquivo final:[/bold yellow]\n"
            f"  {final_path.name}"
        )
        if not typer.confirm("Deseja sobrescrever?", default=False):
            console.print("[dim]Arquivo final mantido sem alteração.[/dim]")
            return

    final = pw.finalize_checkpoint(number)
    if final:
        console.print(f"  [green]✓[/green] Charter aprovado → [bold]{final.name}[/bold]")


def _extract_docx_corrections(path: Path) -> dict | None:
    """Extract actionable corrections from an edited .docx file.

    Scans for three types of user markup:
    - **Track changes** (``w:ins`` / ``w:del``): text the user added or removed
    - **Comments** (``word/comments.xml``): instructions written as Word comments
    - **Yellow highlights**: paragraphs/runs the user wants fully regenerated

    Args:
        path: Path to the edited .docx file.

    Returns:
        Dict with keys ``insertions``, ``deletions``, ``comments``, ``highlights``,
        or ``None`` if the document contains none of these markers.
    """
    try:
        import zipfile
        from docx import Document as _DocxDoc
        from docx.oxml.ns import qn
    except ImportError:
        return None

    try:
        doc = _DocxDoc(str(path))
    except Exception:
        return None

    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    insertions: list[dict] = []
    deletions: list[dict] = []
    highlights: list[dict] = []

    for para in doc.paragraphs:
        para_text = para.text.strip()

        # Yellow highlights on individual runs
        for run in para.runs:
            rpr = run._element.find(qn("w:rPr"))
            if rpr is not None:
                h = rpr.find(qn("w:highlight"))
                if h is not None and h.get(qn("w:val")) in ("yellow", "darkYellow"):
                    if run.text.strip():
                        highlights.append({
                            "highlighted_text": run.text.strip(),
                            "paragraph_context": para_text[:120],
                        })

        # Also flag entire paragraph if ALL text is highlighted yellow
        all_runs = para.runs
        if all_runs and all(
            r._element.find(qn("w:rPr")) is not None
            and r._element.find(qn("w:rPr")).find(qn("w:highlight")) is not None  # type: ignore[union-attr]
            and r._element.find(qn("w:rPr")).find(qn("w:highlight")).get(qn("w:val")) in ("yellow", "darkYellow")  # type: ignore[union-attr]
            for r in all_runs if r.text.strip()
        ) and para_text:
            # Deduplicate: if paragraph already covered by run-level highlights, skip
            if not any(h["paragraph_context"] == para_text[:120] for h in highlights):
                highlights.append({
                    "highlighted_text": para_text,
                    "paragraph_context": para_text[:120],
                })

        # Track changes: insertions (w:ins)
        for ins_elem in para._element.findall(".//" + qn("w:ins")):
            text = "".join(t.text or "" for t in ins_elem.findall(".//" + qn("w:t")))
            if text.strip():
                insertions.append({"text": text.strip(), "context": para_text[:120]})

        # Track changes: deletions (w:del)
        for del_elem in para._element.findall(".//" + qn("w:del")):
            text = "".join(
                t.text or "" for t in del_elem.findall(".//" + qn("w:delText"))
            )
            if text.strip():
                deletions.append({"text": text.strip(), "context": para_text[:120]})

    # Comments from word/comments.xml
    comments: list[dict] = []
    try:
        with zipfile.ZipFile(str(path)) as z:
            if "word/comments.xml" in z.namelist():
                from lxml import etree  # type: ignore[import-untyped]
                root = etree.fromstring(z.read("word/comments.xml"))
                for comment_elem in root.findall(f"{{{W}}}comment"):
                    author = comment_elem.get(f"{{{W}}}author", "Usuário")
                    t_elems = comment_elem.findall(f".//{{{W}}}t")
                    text = "".join(t.text or "" for t in t_elems).strip()
                    if text:
                        comments.append({"author": author, "text": text})
    except Exception:
        pass

    if not insertions and not deletions and not comments and not highlights:
        return None

    return {
        "insertions": insertions,
        "deletions": deletions,
        "comments": comments,
        "highlights": highlights,
    }


def _format_corrections_for_llm(corrections: dict) -> str:
    """Format extracted docx corrections into a structured prompt string.

    Args:
        corrections: Output of :func:`_extract_docx_corrections`.

    Returns:
        Formatted string describing all corrections for the LLM.
    """
    parts: list[str] = []

    if corrections.get("comments"):
        parts.append("## Comentários do pesquisador (instruções a implementar):")
        for c in corrections["comments"]:
            parts.append(f'- [{c["author"]}]: {c["text"]}')

    if corrections.get("insertions"):
        parts.append("\n## Trechos inseridos pelo pesquisador (incorporar):")
        for i in corrections["insertions"]:
            parts.append(f'- Inseriu: "{i["text"]}"  (contexto: {i["context"]})')

    if corrections.get("deletions"):
        parts.append("\n## Trechos removidos pelo pesquisador (excluir):")
        for d in corrections["deletions"]:
            parts.append(f'- Removeu: "{d["text"]}"  (contexto: {d["context"]})')

    if corrections.get("highlights"):
        parts.append(
            "\n## Trechos em destaque amarelo (regerar completamente — "
            "manter o tema mas reescrever o conteúdo):"
        )
        for h in corrections["highlights"]:
            parts.append(f'- "{h["highlighted_text"]}"')

    return "\n".join(parts)


def _reconstruct_state_for_literatura(project_path: Path, state_path: Path) -> dict | None:
    """Build a minimal ResearchState for the ``literatura`` command.

    Used when the state file is missing or corrupt but CP1 [final].docx exists.
    Reads the topic from workspace.yaml and restores the objective from the
    CP1 final document.

    Args:
        project_path: Root of the ProjectWorkspace.
        state_path: Path to the .state/ directory.

    Returns:
        A state dict with ``charter_approved=True``, or None on failure.
    """
    import yaml as _yaml
    from ai_skill.core.project_workspace import ProjectWorkspace
    from ai_skill.core.state import initial_state

    meta_file = project_path / "workspace.yaml"
    if not meta_file.exists():
        return None
    try:
        meta = _yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    topic = meta.get("topic", "")
    state: dict = dict(initial_state(str(state_path), topic=topic))

    pw = ProjectWorkspace.from_path(project_path)
    final_cp1 = pw.checkpoint_final_path(1)
    if final_cp1.exists():
        objective = _parse_objective_from_docx(final_cp1)
        if objective:
            state["objective"] = objective
        state["charter_document_text"] = _read_docx_text(final_cp1)

    state["charter_approved"] = True
    return state


def _reconstruct_state_for_correct(project_path: Path, state_path: Path) -> dict | None:
    """Build a minimal ResearchState when research-state.yaml is absent.

    Reads the workspace metadata for the topic, and tries to restore the
    objective fields by parsing the last preview .docx.  This allows
    ``resume --correct`` to work even when the state file was not saved.

    Args:
        project_path: Root of the ProjectWorkspace.
        state_path: Path to the .state/ directory (ResearchWorkspace).

    Returns:
        A minimal state dict, or None if reconstruction is not possible.
    """
    import yaml as _yaml
    from ai_skill.core.project_workspace import ProjectWorkspace
    from ai_skill.core.state import initial_state

    meta_file = project_path / "workspace.yaml"
    if not meta_file.exists():
        return None
    try:
        meta = _yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    topic = meta.get("topic", "")
    state: dict = dict(initial_state(str(state_path), topic=topic))

    # Restore objective from the last preview docx
    pw = ProjectWorkspace.from_path(project_path)
    last_preview = pw.get_last_preview(1)
    if last_preview is not None:
        objective = _parse_objective_from_docx(last_preview)
        if objective:
            state["objective"] = objective

    return state


def _parse_objective_from_docx(path: Path) -> dict | None:
    """Restore ResearchObjective fields from a charter .docx file.

    Parses the structured headings generated by :func:`_charter_to_docx`
    to reconstruct the objective dict without calling the LLM.

    Args:
        path: Path to a charter ``[preview_N].docx`` file.

    Returns:
        A partial ResearchObjective dict, or None if parsing fails.
    """
    try:
        from docx import Document as _DocxDoc
        doc = _DocxDoc(str(path))
    except Exception:
        return None

    obj: dict = {}
    current_section: str | None = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = getattr(para.style, "name", "") or ""
        if "Heading" in style or "Title" in style:
            lower = text.lower()
            if "tópico" in lower or "topic" in lower:
                current_section = "topic"
            elif "objetivo" in lower or "goal" in lower:
                current_section = "goals"
            elif "métrica" in lower or "metric" in lower or "sucesso" in lower:
                current_section = "success_metrics"
            elif "restrição" in lower or "escopo" in lower or "constraint" in lower:
                current_section = "scope_constraints"
            elif "metodolog" in lower:
                current_section = "methodology_preference"
            else:
                current_section = None
        else:
            if current_section == "topic":
                obj["topic"] = text
            elif current_section in ("goals", "success_metrics", "scope_constraints"):
                item = text.lstrip("•-– ").strip()
                if item:
                    obj.setdefault(current_section, []).append(item)
            elif current_section == "methodology_preference":
                obj["methodology_preference"] = text

    return obj if obj.get("goals") else None


_NODE_LABELS: dict[str, str] = {
    # CP1 — Research Charter
    "initiate":         "Iniciando workspace",
    "align_charter":    "Redigindo Research Charter",
    "review_charter":   "Processando revisão do Charter",
    # CP2 — Literature Review
    "cp2_router":       "Iniciando Revisão Bibliográfica",
    "plan":             "Planejando próximas buscas",
    "execute":          "Executando buscas bibliográficas",
    "evaluate":         "Avaliando qualidade dos resultados",
    "request_support":  "Solicitando orientação adicional",
    "compile_literature":  "Compilando Revisão Bibliográfica",
    "verify_literature":   "Verificando fontes e referências",
    "deliver_literature":  "Gerando documento preview",
    "review_literature":   "Processando revisão da literatura",
    "refine_literature":   "Refinando com base nas correções",
}


def _print_node_progress(node_name: str, output: object) -> None:
    """Print a user-friendly progress line for a completed node.

    Shows the friendly label, attempt number (for retry nodes), and
    a summary of sub-steps when the output provides that information.

    Args:
        node_name: The LangGraph node name.
        output: The node's partial state output dict.
    """
    label = _NODE_LABELS.get(node_name, node_name)
    extra = ""

    if isinstance(output, dict):
        attempt = output.get("attempt")
        findings = output.get("findings")
        evaluation = output.get("evaluation")

        if attempt is not None and node_name == "plan":
            extra = f" [dim](tentativa {attempt + 1})[/dim]"
        elif node_name == "execute" and findings is not None:
            n = len(findings)
            extra = f" [dim]({n} resultado{'s' if n != 1 else ''} acumulado{'s' if n != 1 else ''})[/dim]"
        elif node_name == "evaluate" and evaluation:
            score = evaluation.get("total_score", 0.0)
            converged = evaluation.get("converged", False)
            flag = "[green]convergiu[/green]" if converged else "[yellow]ainda em progresso[/yellow]"
            extra = f" [dim](score {score:.2f} — {flag})[/dim]"
        elif node_name == "align_charter" and not output:
            label = "Research Charter (sem alterações — aguardando revisão)"
        elif node_name == "verify_literature":
            verified = (output.get("literature_review_doc") or {}).get("verified_sources", [])
            if verified:
                ok = sum(1 for v in verified if v.get("content_matches"))
                extra = f" [dim]({ok}/{len(verified)} verificadas)[/dim]"

    console.print(f"  [green]✓[/green] {label}{extra}")


def _handle_checkpoint(
    _interrupt_data: object,
    state: dict,  # type: ignore[type-arg]
    workspace: Path,
) -> None:
    """Dispatch to the correct checkpoint handler.

    Dispatch logic:
    - active_checkpoint == 2              → CP2 review prompt
    - active_checkpoint == 1, approved   → CP2 start prompt (begin_literature gate)
    - active_checkpoint == 1, not yet    → CP1 review prompt

    Args:
        _interrupt_data: Data from the ``__interrupt__`` event (unused).
        state: Accumulated research state at the point of interruption.
        workspace: Path to the ResearchWorkspace (.state/) directory.
    """
    active_cp = int(state.get("active_checkpoint", 1))
    if active_cp == 2:
        _handle_checkpoint_2(state, workspace)
    elif state.get("charter_approved"):
        _handle_cp2_start(state, workspace)
    else:
        _handle_checkpoint_1(state, workspace)


def _handle_checkpoint_1(state: dict, workspace: Path) -> None:  # type: ignore[type-arg]
    """Display Checkpoint 1 (Research Charter) summary and resume instructions.

    Args:
        state: Accumulated research state.
        workspace: Path to the ResearchWorkspace (.state/) directory.
    """
    console.print("\n[bold yellow]━━ CHECKPOINT 1 — Research Charter ━━[/bold yellow]")

    docx_path = state.get("checkpoint_label", "")

    objective = state.get("objective") or {}
    if objective.get("goals"):
        console.print("\n[bold]Proposta do agente:[/bold]")
        console.print(f"  [cyan]Tópico:[/cyan] {objective.get('topic', '')}")
        goals = objective.get("goals", [])
        if goals:
            console.print("  [cyan]Objetivos:[/cyan]")
            for g in goals:
                console.print(f"    • {g}")
        metrics = objective.get("success_metrics", [])
        if metrics:
            console.print("  [cyan]Métricas de sucesso:[/cyan]")
            for m in metrics:
                console.print(f"    • {m}")

    if docx_path:
        console.print("\n[green]✓ Documento salvo em:[/green]")
        console.print(f"  [bold]{docx_path}[/bold]")
        console.print("\n[dim]Abra o arquivo, revise e edite conforme necessário.[/dim]")

    console.print(
        f"\n[bold]Para aprovar[/bold] (sem alterações):\n"
        f"  ai-skill resume --workspace \"{workspace}\"\n"
        f"\n[bold]Para solicitar correções[/bold]:\n"
        f"  1. Abra o preview acima no Word\n"
        f"  2. Use uma das formas para indicar correções:\n"
        f"       [dim]• Comentários[/dim]       → aciona a IA para implementar a instrução\n"
        f"       [dim]• Marcas de revisão[/dim] → a IA incorpora as suas alterações\n"
        f"       [dim]• Realce amarelo[/dim]    → a IA regera o trecho completamente\n"
        f"  3. Salve o arquivo\n"
        f"  4. ai-skill resume --workspace \"{workspace}\" --correct"
    )


def _handle_cp2_start(state: dict, workspace: Path) -> None:  # type: ignore[type-arg]
    """Display the CP1 approval confirmation and CP2 start instructions.

    Shown when the graph pauses at the ``begin_literature`` gate after the
    user approved the Research Charter.

    Args:
        state: Accumulated research state.
        workspace: Path to the ResearchWorkspace (.state/) directory.
    """
    topic = (state.get("objective") or {}).get("topic", "")
    title = f"✓ Checkpoint 1 — Research Charter aprovado"
    if topic:
        title += f": [italic]{topic}[/italic]"
    console.print(f"\n[bold green]{title}[/bold green]")

    project_path = workspace.parent
    if (project_path / "workspace.yaml").exists():
        from ai_skill.core.project_workspace import ProjectWorkspace
        pw = ProjectWorkspace.from_path(project_path)
        final = pw.checkpoint_final_path(1)
        if final.exists():
            console.print(f"  [dim]Arquivo final:[/dim] {final.name}")

    console.print(
        f"\n[bold]Para iniciar a Revisão Bibliográfica (Checkpoint 2):[/bold]\n"
        f"  ai-skill resume --workspace \"{workspace}\"\n"
        f"\n[dim]O agente irá pesquisar, compilar e verificar as referências "
        f"bibliográficas automaticamente.[/dim]"
    )


def _handle_checkpoint_2(state: dict, workspace: Path) -> None:  # type: ignore[type-arg]
    """Display Checkpoint 2 (Literature Review) summary and resume instructions.

    Args:
        state: Accumulated research state.
        workspace: Path to the ResearchWorkspace (.state/) directory.
    """
    console.print("\n[bold yellow]━━ CHECKPOINT 2 — Revisão Bibliográfica ━━[/bold yellow]")

    docx_path = state.get("checkpoint_label", "")
    review_doc = state.get("literature_review_doc") or {}
    sections = review_doc.get("sections", [])
    references = review_doc.get("references", [])
    verified = review_doc.get("verified_sources", [])

    if sections:
        console.print(f"\n[bold]Seções geradas ({len(sections)}):[/bold]")
        for s in sections:
            console.print(f"  [cyan]•[/cyan] {s.get('section_title', '')}")

    if references:
        console.print(f"\n[bold]Referências bibliográficas ({len(references)}):[/bold]")

    if verified:
        ok = sum(1 for v in verified if v.get("content_matches"))
        warn = sum(1 for v in verified if v.get("accessible") and not v.get("content_matches"))
        fail = sum(1 for v in verified if not v.get("accessible"))
        console.print(
            f"\n[bold]Verificação de fontes:[/bold]  "
            f"[green]✓ {ok} verificadas[/green]  "
            f"[yellow]⚠ {warn} questionáveis[/yellow]  "
            f"[red]✗ {fail} inacessíveis[/red]"
        )

    if docx_path:
        console.print("\n[green]✓ Documento salvo em:[/green]")
        console.print(f"  [bold]{docx_path}[/bold]")
        console.print(
            "\n[dim]Abra o arquivo, revise os textos e as referências bibliográficas.[/dim]\n"
            "[dim]Marcações coloridas indicam o status de verificação de cada fonte.[/dim]"
        )

    console.print(
        f"\n[bold]Para aprovar[/bold] (sem alterações):\n"
        f"  ai-skill resume --workspace \"{workspace}\"\n"
        f"\n[bold]Para solicitar correções[/bold]:\n"
        f"  1. Abra o preview acima no Word\n"
        f"  2. Use uma das formas para indicar correções:\n"
        f"       [dim]• Comentários[/dim]       → aciona a IA para implementar a instrução\n"
        f"       [dim]• Marcas de revisão[/dim] → a IA incorpora as suas alterações\n"
        f"       [dim]• Realce amarelo[/dim]    → a IA regera o trecho completamente\n"
        f"  3. Salve o arquivo\n"
        f"  4. ai-skill resume --workspace \"{workspace}\" --correct"
    )


if __name__ == "__main__":
    app()
