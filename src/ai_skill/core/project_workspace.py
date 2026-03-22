"""Project workspace management.

A ProjectWorkspace is a named folder created at the same level as the plato
project directory.  It stores all research artefacts produced during a
research session:

  <projects_root>/
    <slug>/
      README.md                              — Metodologia da Pesquisa reference
      workspace.yaml                         — Metadata (name, topic, stage, created)
      Checkpoint 1 - Research Charter.docx   — Generated at each checkpoint
      Checkpoint 2 - Literature Review.docx
      ...
      attachments/                           — User-provided input files

The default projects root is the parent directory of the plato project, or the
value of the AI_SKILL_PROJECTS_DIR environment variable.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

CHECKPOINT_NAMES: dict[int, str] = {
    1: "Research Charter",
    2: "Literature Review",
    3: "Research Design",
    4: "Data Collection Guide",
    5: "Analysis Guide",
    6: "Analysis Results",
    7: "Results Interpretation",
    8: "Paper Draft",
}

# ---------------------------------------------------------------------------
# README template (Metodologia da Pesquisa)
# ---------------------------------------------------------------------------

_README_TEMPLATE = """\
# Metodologia da Pesquisa

**Projeto:** {name}
**Tópico:** {topic}
**Iniciado em:** {date}

---

## Pipeline de 8 Estágios com Checkpoints

O princípio central: **o agente orienta, o usuário decide e executa onde os dados são reais**.

---

### ESTÁGIO 1 — Research Charter

Definição do projeto de pesquisa: objetivo, metas, métricas de sucesso e escopo.
Corresponde à Etapa 1 da Pesquisa Operacional (Churchman, Ackoff & Arnoff, 1957)
e ao grupo *Initiating* do PMBOK 8ª Edição.

> **Checkpoint 1** — Usuário revisa e aprova objetivos, metas e métricas de sucesso.
> Output: `Checkpoint 1 - Research Charter.docx`

---

### ESTÁGIO 2 — Revisão Bibliográfica

Busca multicamada (Exa.ai → Google CSA → Firecrawl → arXiv → Semantic Scholar),
leitura (PyMuPDF), resumo via LLM e fact-checking cruzado.

> **Checkpoint 2** — Usuário revisa e cura a bibliografia.
> Output: `Checkpoint 2 - Literature Review.docx`

---

### ESTÁGIO 3 — Design Metodológico

Identificação do método de pesquisa adequado ao objetivo.
Formulação de hipóteses, variáveis e instrumentos de coleta.

> **Checkpoint 3** — Usuário aprova a metodologia.
> Output: `Checkpoint 3 - Research Design.docx`

---

### ESTÁGIO 4 — Guia de Coleta de Dados

Protocolo de coleta, instrumentos e checklist gerados pelo agente.

> **Handoff** — O usuário executa a coleta. O agente **nunca** fabrica dados.
> **Checkpoint 4** — Usuário entrega os dados coletados via pasta `attachments/`.
> Output: `Checkpoint 4 - Data Collection Guide.docx`

---

### ESTÁGIO 5 — Guia de Análise

Passo a passo do método, fórmulas em LaTeX, guia de software e checklist de verificação.

> **Checkpoint 5** — Usuário revisa fórmulas e método.
> **Handoff** — Usuário executa a análise e apura os resultados.
> **Checkpoint 6** — Usuário entrega os resultados via pasta `attachments/`.
> Output: `Checkpoint 5 - Analysis Guide.docx` / `Checkpoint 6 - Analysis Results.docx`

---

### ESTÁGIO 6 — Interpretação dos Resultados

O agente redige a seção de Resultados com base exclusivamente nos dados fornecidos.

> **Checkpoint 7** — Usuário revisa via track changes + comentários Word (skill cowork).
> Output: `Checkpoint 7 - Results Interpretation.docx`

---

### ESTÁGIO 7 — Composição e Revisão do Paper

Abstract, Introdução, Revisão da Literatura, Metodologia, Resultados, Conclusão.
ReviewAgent avalia coerência, cobertura, rigor, clareza e conformidade ABNT.

> **Checkpoint 8** — Revisão seção por seção via cowork.
> Output: `Checkpoint 8 - Paper Draft.docx`

---

### ESTÁGIO 8 — Publicação

Formatação ABNT NBR 6023:2018, validação de ciência aberta (TOP Guidelines + FAIR +
NIH 2025), exportação em .docx, Markdown, PDF e LaTeX.

---

## Princípio Anti-Alucinação

O agente **nunca gera, estima ou interpola** resultados experimentais, dados de pesquisa
ou valores numéricos originais. Nos Estágios 4 e 5, o agente é **metodologista**:
descreve o método, as fórmulas e as ferramentas. A execução e os números são
exclusivamente do usuário.

---

## Como usar esta pasta

- Os **outputs de checkpoint** são salvos automaticamente como `Checkpoint X - NOME.docx`.
- Para **fornecer arquivos ao agente** (dados coletados, resultados de análise, PDFs de
  referência etc.), coloque-os na pasta `attachments/`. O agente listará os arquivos
  disponíveis e você selecionará pelo número.
- Consulte `workspace.yaml` para acompanhar o estado atual da pesquisa.

---

## Referências Metodológicas

| Base | Aplicação |
|---|---|
| Pesquisa Operacional — Churchman, Ackoff & Arnoff (1957) | 7 etapas mapeadas ao pipeline |
| PMBOK 8ª Edição (PMI, 2025) | Grupos de processo: Initiating → Closing |
| ISO 690:2021 | Referências bibliográficas |
| ISO 9001:2015 | Critérios de aceitação dos checkpoints |
| PRISMA 2020 | Revisão sistemática da literatura |
| ABNT NBR 6023:2018 | Formatação de referências |
"""

# ---------------------------------------------------------------------------
# Slug utilities
# ---------------------------------------------------------------------------


def slugify(name: str, max_length: int = 50) -> str:
    """Convert a research name to a filesystem-safe slug.

    Steps:
    1. Normalise unicode to ASCII equivalents (é → e, ç → c …)
    2. Keep only alphanumeric, space, hyphen, underscore
    3. Collapse whitespace / hyphens to a single hyphen
    4. Lowercase and truncate

    Args:
        name: Raw user-supplied research name.
        max_length: Maximum length of the resulting slug (default 50).

    Returns:
        Cleaned, filesystem-safe slug string.

    Raises:
        ValueError: If the result after sanitisation is empty.
    """
    normalised = unicodedata.normalize("NFKD", name)
    ascii_str = normalised.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^\w\s-]", "", ascii_str)
    slug = re.sub(r"[-\s]+", "-", cleaned).strip("-").lower()
    slug = slug[:max_length].rstrip("-")
    if not slug:
        raise ValueError(
            f"O nome '{name}' gerou um slug vazio após sanitização. "
            "Use letras, números ou hífens."
        )
    return slug


# ---------------------------------------------------------------------------
# Default projects root
# ---------------------------------------------------------------------------


def default_projects_root() -> Path:
    """Return the default root directory for project workspaces.

    Priority:
    1. ``AI_SKILL_PROJECTS_DIR`` environment variable (if set).
    2. Parent directory of the plato project (sibling to ``plato/``).

    Returns:
        Absolute Path to the projects root directory.
    """
    env = os.environ.get("AI_SKILL_PROJECTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # project_workspace.py lives at:
    #   plato/ai_skill/src/ai_skill/core/project_workspace.py
    # parents[0] = core/
    # parents[1] = ai_skill/  (package)
    # parents[2] = src/
    # parents[3] = ai_skill/  (project)
    # parents[4] = plato/      ← projects root (workspaces are siblings of ai_skill/)
    return Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# ProjectWorkspace
# ---------------------------------------------------------------------------


class ProjectWorkspace:
    """Named research project workspace on the filesystem.

    Args:
        name: Human-readable project name (will be slugified for the folder).
        root: Root directory where all workspaces live.
              Defaults to :func:`default_projects_root`.
    """

    def __init__(self, name: str, root: Path | None = None) -> None:
        self.name = name
        self.slug = slugify(name)
        self.root: Path = root or default_projects_root()
        self.path: Path = self.root / self.slug
        self._attachments_dir: Path = self.path / "attachments"
        self._metadata_file: Path = self.path / "workspace.yaml"
        self._readme_file: Path = self.path / "README.md"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def create(self, topic: str = "") -> None:
        """Create the workspace directory structure and initial files.

        Args:
            topic: The research topic (used in README and metadata).

        Raises:
            FileExistsError: If a workspace with this slug already exists.
        """
        if self.path.exists():
            raise FileExistsError(f"Workspace já existe: {self.path}")

        self.path.mkdir(parents=True)
        self._attachments_dir.mkdir()

        now = _utcnow()
        self._readme_file.write_text(
            _README_TEMPLATE.format(
                name=self.name,
                date=now[:10],
                topic=topic or self.name,
            ),
            encoding="utf-8",
        )

        metadata: dict[str, Any] = {
            "name": self.name,
            "slug": self.slug,
            "topic": topic or self.name,
            "created": now,
            "current_stage": 0,
            "current_checkpoint": 0,
            "status": "created",
        }
        self._metadata_file.write_text(
            yaml.dump(metadata, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def exists(self) -> bool:
        """Return True if the workspace directory exists."""
        return self.path.exists()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def load_metadata(self) -> dict[str, Any]:
        """Load workspace metadata from workspace.yaml.

        Returns:
            Metadata dict, or empty dict if not found.
        """
        if not self._metadata_file.exists():
            return {}
        try:
            return yaml.safe_load(self._metadata_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def update_metadata(self, updates: dict[str, Any]) -> None:
        """Merge *updates* into workspace.yaml.

        Args:
            updates: Key-value pairs to update.
        """
        meta = self.load_metadata()
        meta.update(updates)
        meta["last_updated"] = _utcnow()
        self._metadata_file.write_text(
            yaml.dump(meta, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Checkpoint output — preview / final naming
    #
    # During the review loop each agent pass produces a numbered preview:
    #   Checkpoint 1 - Research Charter [preview_1].docx
    #   Checkpoint 1 - Research Charter [preview_2].docx
    #   …
    # When the user approves the charter the latest preview is copied to:
    #   Checkpoint 1 - Research Charter [final].docx
    # ------------------------------------------------------------------

    def checkpoint_preview_path(self, number: int, preview_num: int) -> Path:
        """Return the path for a specific preview file.

        Args:
            number: Checkpoint number (1–8).
            preview_num: Preview counter (1-based).

        Returns:
            Path of the form ``Checkpoint N - Stage Name [preview_N].docx``.
        """
        stage_name = CHECKPOINT_NAMES.get(number, f"Stage {number}")
        return self.path / f"Checkpoint {number} - {stage_name} [preview_{preview_num}].docx"

    def checkpoint_final_path(self, number: int) -> Path:
        """Return the path for the final approved checkpoint file.

        Args:
            number: Checkpoint number (1–8).

        Returns:
            Path of the form ``Checkpoint N - Stage Name [final].docx``.
        """
        stage_name = CHECKPOINT_NAMES.get(number, f"Stage {number}")
        return self.path / f"Checkpoint {number} - {stage_name} [final].docx"

    # Keep legacy name as alias so older callers don't break
    def checkpoint_path(self, number: int) -> Path:
        """Return the final checkpoint path (alias for :meth:`checkpoint_final_path`)."""
        return self.checkpoint_final_path(number)

    def _count_previews(self, number: int) -> int:
        """Count existing preview files for a checkpoint."""
        if not self.path.exists():
            return 0
        stage_name = CHECKPOINT_NAMES.get(number, f"Stage {number}")
        prefix = f"Checkpoint {number} - {stage_name} [preview_"
        return sum(
            1
            for f in self.path.iterdir()
            if f.is_file() and f.name.startswith(prefix) and f.name.endswith("].docx")
        )

    def save_checkpoint_preview(self, number: int, content_bytes: bytes) -> Path:
        """Save a new numbered preview for a checkpoint.

        Auto-increments the preview counter on each call.

        Args:
            number: Checkpoint number (1–8).
            content_bytes: Raw .docx file content.

        Returns:
            Path where the preview was written, e.g.
            ``Checkpoint 1 - Research Charter [preview_2].docx``.
        """
        next_num = self._count_previews(number) + 1
        target = self.checkpoint_preview_path(number, next_num)
        target.write_bytes(content_bytes)
        self.update_metadata({
            "current_checkpoint": number,
            "status": f"checkpoint_{number}_preview_{next_num}",
        })
        return target

    def get_last_preview(self, number: int) -> Path | None:
        """Return the most recent preview file for a checkpoint, or None.

        Args:
            number: Checkpoint number (1–8).

        Returns:
            Path to the highest-numbered preview, or None if none exist.
        """
        if not self.path.exists():
            return None
        stage_name = CHECKPOINT_NAMES.get(number, f"Stage {number}")
        prefix = f"Checkpoint {number} - {stage_name} [preview_"
        previews = [
            f for f in self.path.iterdir()
            if f.is_file() and f.name.startswith(prefix) and f.name.endswith("].docx")
        ]
        if not previews:
            return None

        def _num(p: Path) -> int:
            try:
                return int(p.stem.rsplit("_", 1)[-1].rstrip("]"))
            except ValueError:
                return 0

        return max(previews, key=_num)

    def finalize_checkpoint(self, number: int) -> Path | None:
        """Copy the latest preview to the ``[final].docx`` file.

        Args:
            number: Checkpoint number (1–8).

        Returns:
            Path of the final file, or None if no preview exists.
        """
        last = self.get_last_preview(number)
        if last is None:
            return None
        final = self.checkpoint_final_path(number)
        final.write_bytes(last.read_bytes())
        self.update_metadata({
            "current_checkpoint": number,
            "status": f"checkpoint_{number}_final",
        })
        return final

    def list_checkpoints(self) -> list[Path]:
        """Return the most relevant checkpoint file for each stage, sorted.

        Prefers ``[final].docx``; falls back to the latest ``[preview_N].docx``.

        Returns:
            List of Path objects for checkpoints that actually exist on disk.
        """
        result = []
        for n in range(1, 9):
            final = self.checkpoint_final_path(n)
            if final.exists():
                result.append(final)
                continue
            last_preview = self.get_last_preview(n)
            if last_preview is not None:
                result.append(last_preview)
        return result

    # ------------------------------------------------------------------
    # File selection helpers (used at HANDOFF stages)
    # ------------------------------------------------------------------

    def list_attachments(self) -> list[Path]:
        """Return a sorted list of files inside ``attachments/``.

        Returns:
            List of Path objects sorted by filename.
        """
        if not self._attachments_dir.exists():
            return []
        return sorted(f for f in self._attachments_dir.iterdir() if f.is_file())

    def list_all_files(self) -> list[Path]:
        """Return all selectable files: existing checkpoints then attachments.

        Returns:
            Ordered list — checkpoints first, attachments second.
        """
        return self.list_checkpoints() + self.list_attachments()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def attachments_path(self) -> Path:
        """Absolute path to the ``attachments/`` directory."""
        return self._attachments_dir

    @property
    def readme_path(self) -> Path:
        """Absolute path to ``README.md``."""
        return self._readme_file

    @classmethod
    def from_path(cls, path: Path) -> "ProjectWorkspace":
        """Reconstruct a ProjectWorkspace from an existing directory path.

        Reads the workspace name from ``workspace.yaml`` if present; falls back
        to the directory name.

        Args:
            path: Absolute path to an existing workspace directory.

        Returns:
            A :class:`ProjectWorkspace` instance pointing at *path*.
        """
        path = path.resolve()
        meta_file = path / "workspace.yaml"
        name = path.name
        if meta_file.exists():
            try:
                meta = yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
                name = meta.get("name", name)
            except Exception:
                pass
        return cls(name, root=path.parent)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def list_workspaces(root: Path | None = None) -> list[ProjectWorkspace]:
    """Discover all project workspaces under *root*.

    A directory is recognised as a workspace when it contains
    ``workspace.yaml``.

    Args:
        root: Root directory to search. Defaults to :func:`default_projects_root`.

    Returns:
        Sorted list of :class:`ProjectWorkspace` instances (alphabetically by slug).
    """
    base = root or default_projects_root()
    if not base.exists():
        return []
    workspaces: list[ProjectWorkspace] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        meta_file = entry / "workspace.yaml"
        if not meta_file.exists():
            continue
        try:
            meta = yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        ws = ProjectWorkspace(meta.get("name", entry.name), root=base)
        workspaces.append(ws)
    return workspaces


def _utcnow() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
