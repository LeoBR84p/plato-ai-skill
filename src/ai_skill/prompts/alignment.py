"""Prompt templates for the user alignment (Research Charter) node."""

from __future__ import annotations

import json

from ai_skill.core.state import ResearchObjective


CHARTER_DRAFT_SYSTEM = """\
Você é um orientador de pesquisa acadêmica. Dado um tópico de pesquisa, elabore um
Research Charter estruturado que guiará um pipeline de pesquisa assistido por IA em 8 estágios.

O charter deve conter:
1. **3-5 metas de pesquisa específicas e mensuráveis** para o projeto completo.
2. **3-5 métricas de sucesso concretas** para o projeto GLOBAL (quantificáveis quando possível).
3. **Restrições de escopo** implícitas no tópico (janela temporal, geografia, domínio, etc.).
4. **Preferência de metodologia** (se inferível do tópico).
5. **Estilo bibliográfico**: padrão "abnt".
6. **Idioma**: padrão "pt-BR".
7. **Diretrizes por estágio** (stage_guidelines): um dict por nome de estágio com 4–8
   diretivas acionáveis cada. Estas orientam planejamento e avaliação DAQUELE estágio —
   independentemente das success_metrics globais. Seja específico ao tópico de pesquisa.

Chaves de estágio e seus focos:

  "literature_review"
    — Diretivas para a fase de pesquisa bibliográfica (CP2). Estas diretrizes
      devem descrever APENAS o que pode ser realizado buscando e lendo artigos
      (skills article_search / web_search / content_summarizer). Exemplos:
      sub-áreas temáticas a cobrir, número de queries distintas necessárias,
      mínimo de total_found em todas as buscas, janela temporal alvo,
      bases prioritárias (arXiv, Semantic Scholar, Web of Science, etc.),
      termos de busca e combinações booleanas.
      PROIBIDO: desenvolvimento de frameworks, redação ou submissão de artigos,
      fluxograma PRISMA formal, execução de experimentos, coleta de datasets, ou
      qualquer entrega que exija trabalho além de busca e sumarização bibliográfica.

  "research_design"
    — Diretivas para a fase de metodologia e design (CP3). Exemplos:
      tipo de estudo (experimental/observacional/misto), instrumentos ou datasets necessários,
      hipóteses a testar, critérios de validação, considerações éticas.

  "data_collection_guide"
    — Diretivas para o protocolo de coleta de dados (CP4). Exemplos:
      datasets ou populações alvo, requisitos de tamanho amostral, critérios de qualidade,
      ferramentas de coleta, padrões de reprodutibilidade.

  "analysis_guide"
    — Diretivas para a fase de análise (CP5). Exemplos:
      técnicas estatísticas ou computacionais, software/bibliotecas, limiares de significância,
      requisitos de estudo de ablação, comparações com baseline.

  "results_interpretation"
    — Diretivas para interpretação e reporte de resultados (CP6/7). Exemplos:
      comparação com literatura prévia, padrões de reporte de tamanho de efeito,
      requisitos de intervalo de confiança, análise de modos de falha.

  "paper_composition"
    — Diretivas para redação do artigo final (CP8). Exemplos:
      periódico ou conferência alvo, limites de palavras, seções obrigatórias,
      especificações de figuras e tabelas, política de coautoria.

  "publication"
    — Diretivas para a fase de publicação (CP8+). Exemplos:
      nível Qualis/Scopus do veículo alvo, requisitos de acesso aberto, declaração
      de disponibilidade de dados, checklist de reprodutibilidade, política de pré-print.

Seja específico e acadêmico no tom. Evite declarações vagas.
"""

CHARTER_DRAFT_USER = """\
Research topic: {topic}

Draft a Research Charter for this topic.
"""

CHARTER_REFINE_SYSTEM = """\
Você é um orientador de pesquisa acadêmica aplicando correções cirúrgicas a um
Research Charter que o pesquisador já revisou e parcialmente aprovou. O pesquisador
marcou apenas as partes que deseja alterar; todo o restante deve ser preservado
LITERALMENTE — sem reformular, sem melhorar, sem reorganizar.

REGRA CRÍTICA — padrão é PRESERVAR:
  Copie cada campo, frase e item de lista do original exatamente como está,
  A MENOS QUE seja diretamente alvo de uma das correções abaixo.
  NÃO use isto como oportunidade para reescrever, melhorar ou limpar conteúdo
  não marcado. Fidelidade caractere-por-caractere ao original é exigida para
  todas as seções não marcadas.

  Campos estruturados (goals, success_metrics, stage_guidelines, etc.) devem
  manter seus valores originais intactos, a menos que explicitamente corrigidos.

Como lidar com cada tipo de correção:
- **Comentários** ("Comentários"): localize a passagem que o comentário referencia
  e aplique a instrução declarada apenas àquela passagem.
- **Track changes — texto inserido** ("Trechos inseridos"): insira o texto no
  local exato indicado, sem alterar nada ao redor.
- **Track changes — texto removido** ("Trechos removidos"): remova apenas essas
  palavras; mantenha o conteúdo ao redor intacto.
- **Destaque amarelo** ("Trechos em destaque amarelo"): reescreva APENAS o
  trecho destacado; preserve tudo antes e depois sem alteração.

Ao finalizar, o output deve estar limpo (sem marcas, comentários ou destaques) e
diferir do original apenas onde as correções explicitamente exigiram uma mudança.
"""

CHARTER_REFINE_USER = """\
Research Charter approved by the researcher (treat as authoritative — do NOT
rewrite any part that is not explicitly targeted by a correction):
{charter_json}

Corrections to apply (touch only what is listed here):
---
{feedback}
---

Return the complete Research Charter with ONLY the listed corrections applied.
All unmarked content must be identical to the original.
"""


def build_charter_draft_messages(topic: str) -> tuple[str, list[dict[str, str]]]:
    """Build messages to draft an initial Research Charter from a topic.

    Args:
        topic: Free-form description of the research topic.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = CHARTER_DRAFT_USER.format(topic=topic)
    return CHARTER_DRAFT_SYSTEM, [{"role": "user", "content": user_content}]


def build_charter_refine_messages(
    charter: ResearchObjective, feedback: str
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to refine a charter based on user feedback.

    Args:
        charter: The current Research Charter draft.
        feedback: User's free-form feedback or edit instructions.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = CHARTER_REFINE_USER.format(
        charter_json=json.dumps(dict(charter), ensure_ascii=False, indent=2),
        feedback=feedback,
    )
    return CHARTER_REFINE_SYSTEM, [{"role": "user", "content": user_content}]
