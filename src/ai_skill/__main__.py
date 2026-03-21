"""CLI entry point for ai-skill.

Commands:
    research <topic>   Start a new research pipeline for the given topic.
    status [workspace] Show the current status of an existing workspace.
    resume [workspace] Resume a paused research pipeline.
    graph              Print the Mermaid diagram of the pipeline graph.
    skills list        List all registered skills.
    skills show <name> Show details for a specific skill.
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
from rich.table import Table

app = typer.Typer(
    name="ai-skill",
    help="AI Academic Research Pipeline Agent",
    add_completion=False,
    no_args_is_help=True,
)
skills_app = typer.Typer(help="Manage and inspect skills.", no_args_is_help=True)
app.add_typer(skills_app, name="skills")

console = Console()


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
# Commands
# ---------------------------------------------------------------------------


@app.command()
def research(
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
    """Start a new academic research pipeline for TOPIC."""
    _configure_logging(os.environ.get("AI_SKILL_LOG_LEVEL", "WARNING"))

    if model:
        os.environ["AI_SKILL_MODEL"] = model

    ws_path = workspace or _default_workspace(topic)

    console.print(f"[bold]Starting research:[/bold] {topic}")
    console.print(f"[dim]Workspace:[/dim] {ws_path}")

    # Late imports keep CLI startup fast
    from ai_skill.core.graph import build_research_graph
    from ai_skill.core.state import initial_state
    from ai_skill.core.workspace import ResearchWorkspace

    ws = ResearchWorkspace(ws_path)
    ws.initialise(topic)

    graph = build_research_graph()
    state = initial_state(ws_path, topic=topic)

    try:
        for event in graph.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    _handle_checkpoint(node_output, graph, state, ws_path)
                    return
                _print_node_progress(node_name, node_output)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research paused. Resume with:[/yellow]")
        console.print(f"  ai-skill resume --workspace {ws_path}")
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
    feedback: Annotated[
        str,
        typer.Option("--feedback", "-f", help="User feedback to inject before resuming."),
    ] = "",
) -> None:
    """Resume a paused research pipeline."""
    _configure_logging(os.environ.get("AI_SKILL_LOG_LEVEL", "WARNING"))

    from ai_skill.core.graph import build_research_graph
    from ai_skill.core.workspace import ResearchWorkspace

    ws = ResearchWorkspace(workspace)
    state = ws.load_state()
    if state is None:
        console.print(f"[red]No state found in:[/red] {workspace}")
        raise typer.Exit(1)

    if feedback:
        state["user_feedback"] = feedback

    graph = build_research_graph()
    console.print(f"[bold]Resuming research:[/bold] {state.get('objective', {}).get('topic', '?')}")

    try:
        for event in graph.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    _handle_checkpoint(node_output, graph, state, workspace)
                    return
                _print_node_progress(node_name, node_output)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research paused. Resume with:[/yellow]")
        console.print(f"  ai-skill resume --workspace \"{workspace}\"")
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


def _print_node_progress(node_name: str, _output: object) -> None:
    """Print a brief progress line for a completed node.

    Args:
        node_name: The LangGraph node name.
        _output: The node's partial state output (unused; reserved for future logging).
    """
    console.print(f"  [green]✓[/green] {node_name}")


def _handle_checkpoint(
    interrupt_data: object,
    _graph: object,
    _state: dict,  # type: ignore[type-arg]
    workspace: Path,
) -> None:
    """Handle a LangGraph interrupt (human-in-the-loop checkpoint).

    Prints the checkpoint prompt and collects user input. The pipeline
    can be resumed with ``ai-skill resume``.

    Args:
        interrupt_data: Data from the ``__interrupt__`` event.
        graph: The compiled LangGraph graph.
        state: Current research state dict.
        workspace: Workspace path for the save prompt.
    """
    console.print("\n[bold yellow]CHECKPOINT[/bold yellow]")

    if isinstance(interrupt_data, (list, tuple)):
        for item in interrupt_data:
            value = getattr(item, "value", item)
            console.print(value)
    else:
        console.print(str(interrupt_data))

    console.print(
        "\n[dim]State saved. Resume after reviewing with:[/dim]\n"
        f"  ai-skill resume --workspace \"{workspace}\" --feedback \"<your feedback>\""
    )


if __name__ == "__main__":
    app()
