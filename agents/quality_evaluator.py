"""5번 Hybrid QA node. Routing, counters and final status belong to Supervisor.

Extra nested result fields are QA-local pending State-owner agreement. Original
documents are injected into the standalone evaluator, not read from an invented
shared State field. The current placeholder fails without calling a paid API.
"""

from langsmith import traceable

from core.state import InvestmentState
from evaluation.report_quality import evaluate, redact_secrets, report_identity, result_is_current


def _trace_input(inputs: dict) -> dict:
    state = inputs.get("state") or {}
    version, digest = report_identity(state)
    ids = set()
    for key in ("technology_results", "market_traction_results"):
        for analysis in (state.get(key) or {}).values():
            ids.update((e.get("source_id"), e.get("chunk_id")) for e in analysis.get("evidence", []))
    for result in (state.get("investment_results") or {}).values():
        for criterion in result.get("criteria", []):
            ids.update((e.get("source_id"), e.get("chunk_id")) for e in criterion.get("evidence", []))
    return redact_secrets({"trace_id": state.get("trace_id"), "report_version": version, "report_hash": digest,
                           "evidence_ids": [{"source_id": s, "chunk_id": c} for s, c in sorted(ids, key=str)]})


def _trace_output(output: dict) -> dict:
    # Do not duplicate the report or originals in QA traces.
    if not isinstance(output, dict):
        return {"error": "QA evaluation failed"}
    return redact_secrets({**output, "feedback": [], "checks": [
        {**check, "issues": [{k: v for k, v in issue.items() if k != "statement"}
                             for issue in check["issues"]]} for check in output["checks"]]})


@traceable(name="report_quality_evaluation", process_inputs=_trace_input, process_outputs=_trace_output)
def evaluate_report(state: InvestmentState, *, original_documents: dict | None = None, judge=None) -> dict:
    return evaluate(state, original_documents=original_documents, judge=judge)


def quality_evaluator_node(state: InvestmentState) -> dict:
    # Exceptions reach the existing worker boundary; no failed Judge becomes a pass.
    return {"quality_result": evaluate_report(state)}


__all__ = ["evaluate_report", "quality_evaluator_node", "result_is_current"]
