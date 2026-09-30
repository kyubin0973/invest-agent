"""Investment Judge 단계별 콘솔 로그.

애플리케이션 또는 데모 진입점에서 ``configure_judge_logging``을 호출하면
Judge가 기업별 입력, LLM 평가, 보정, 최종 검증 결과를 Rich 표로 출력한다.
"""

import logging
import os
from typing import Any

from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from evaluation.criteria import CRITERIA_BY_ID

LOGGER_NAME = "investment_judge"
logger = logging.getLogger(LOGGER_NAME)
_console = Console(stderr=True, width=int(os.getenv("JUDGE_LOG_WIDTH", "180")))


def configure_judge_logging(level: str | None = None) -> None:
    """RichHandler를 한 번만 설치한다. JUDGE_LOG_LEVEL로 출력 수준을 조절할 수 있다."""
    configured_level = (level or os.getenv("JUDGE_LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, configured_level, logging.INFO)
    if not logger.handlers:
        logger.addHandler(
            RichHandler(
                console=_console,
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
                markup=False,
            )
        )
    logger.setLevel(numeric_level)
    logger.propagate = False


def log_stage(company: str, stage: str, message: str) -> None:
    logger.info("[%s] %s | %s", company, stage, message)


def _criterion_label(criterion_id: str) -> str:
    criterion = CRITERIA_BY_ID.get(criterion_id)
    return f"{criterion.q}\n{criterion_id}" if criterion else criterion_id or "?"


def _format_evidence(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return "-"
    rendered = []
    for item in items[:2]:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id", "?")
        source_id = item.get("source_id", "?")
        level = item.get("evidence_level")
        rendered.append(f"{chunk_id} | {source_id}" + (f" | {level}" if level else ""))
    if len(items) > 2:
        rendered.append(f"+{len(items) - 2} more")
    return "\n".join(rendered) or "-"


def _compact(value: Any, limit: int = 150) -> str:
    text = " ".join(str(value or "-").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def log_criteria(company: str, title: str, criteria: list[dict]) -> None:
    """문항별 점수·근거·판단을 한 표로 출력한다."""
    if not logger.isEnabledFor(logging.INFO):
        return

    table = Table(
        title=f"{company} — {title}",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_lines=False,
        expand=True,
    )
    table.add_column("문항", width=8, no_wrap=True)
    table.add_column("점수", width=5, justify="center")
    table.add_column("상태", width=8)
    table.add_column("신뢰도", width=8, justify="center")
    table.add_column("Evidence (chunk | source | level)", min_width=24)
    table.add_column("판단 근거 / 미확인 정보", min_width=72, ratio=2)

    for item in criteria:
        score = item.get("score")
        reasoning = _compact(item.get("reasoning"))
        missing = item.get("missing_information") or []
        if missing:
            reasoning += "\n[미확인] " + _compact("; ".join(map(str, missing)), 100)
        table.add_row(
            _criterion_label(str(item.get("criterion_id") or "")),
            "N/A" if score is None else str(score),
            "SCORED" if item.get("status") == "SCORED" else "N/A",
            str(item.get("confidence") or "?"),
            _format_evidence(item.get("evidence")),
            reasoning,
        )
    _console.print(table)


def log_repair_issues(company: str, issues: dict[str, list[str]]) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    table = Table(title=f"{company} — 재평가 대상", box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("문항", width=10)
    table.add_column("검증 실패 사유", min_width=50)
    for criterion_id in sorted(issues, key=lambda value: list(CRITERIA_BY_ID).index(value)):
        table.add_row(_criterion_label(criterion_id), "\n".join(issues[criterion_id]))
    _console.print(table)


def log_decision(company: str, result: dict) -> None:
    log_stage(
        company,
        "FINAL",
        (
            f"decision={result['decision']} | score={result['final_score']} / 5 | "
            f"coverage={result['evidence_coverage']:.1%} | {result['decision_reason']}"
        ),
    )
