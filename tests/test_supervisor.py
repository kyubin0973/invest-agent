"""Supervisor 라우팅·재작업·종료 보장 (LLM·검색 없이 하위 Agent를 대역으로 바꿔 검증)"""

import logging
import unittest
from unittest.mock import patch

from agents import supervisor
from agents.supervisor import COMPETITION, FINISH, JUDGE, QUALITY, REPORT
from core import graph as graph_module
from evaluation.criteria import CRITERIA, MARKET, TECH

COMPANIES = ["A", "B"]
PROFILES = {c: {"name": c} for c in COMPANIES}


def analysis(tag: str) -> dict:
    return {"summary": tag, "findings": [], "evidence": [], "risks": [], "missing_information": []}


def judged(missing: tuple[str, ...]) -> dict:
    criteria = [
        {"criterion_id": c.id, "status": "INSUFFICIENT_EVIDENCE" if c.id in missing else "SCORED"}
        for c in CRITERIA
    ]
    return {"criteria": criteria, "decision": "HOLD"}


class Workers:
    """호출 순서를 기록하는 하위 Agent 대역. 재작업을 받은 기업은 다음 평가에서 근거가 충분해진다."""

    def __init__(self, missing: dict[str, tuple[str, ...]] | None = None, quality: list[bool] | None = None,
                 fail: dict[str, int] | None = None, fixed_by_rework: bool = True,
                 issue_owner: tuple[str, str | None] = (REPORT, None)):
        self.issue_owner = issue_owner
        self.calls: list[tuple] = []
        self.missing = missing or {}
        self.quality = quality or [True]
        self.fail = dict(fail or {})
        self.fixed_by_rework = fixed_by_rework
        self.reworked: set[str] = set()

    def _run(self, name: str, state: dict) -> None:
        company = state["current_company"] if name in supervisor.COMPANY_WORKERS else None
        self.calls.append((name, company, bool(state["rework_request"])))
        key = supervisor.status_key(name, company)
        if self.fail.get(key, 0) > 0:
            self.fail[key] -= 1
            raise RuntimeError(f"{key} boom")

    def analysis_node(self, name: str, result_key: str):
        def node(state):
            self._run(name, state)
            request = state["rework_request"]
            if request:
                assert request["agent"] == name and request["company"] == state["current_company"]
                self.reworked.add(request["company"])
            return {result_key: {state["current_company"]: analysis(name)}}
        return node

    def competition(self, state):
        self._run(COMPETITION, state)
        return {"competition_result": {"comparisons": []}}

    def judge(self, state):
        self._run(JUDGE, state)
        company = state["current_company"]
        fixed = self.fixed_by_rework and company in self.reworked
        return {"investment_results": {company: judged(() if fixed else self.missing.get(company, ()))}}

    def report(self, state):
        self._run(REPORT, state)
        return {"final_report": f"report v{state['report_attempts']}", "report_path": "x.pdf", "references": []}

    def quality_node(self, state):
        self._run(QUALITY, state)
        passed = self.quality[min(state["report_attempts"], len(self.quality)) - 1]
        owner, company = self.issue_owner
        issue = {"owner": owner, "company": company, "statement": "출처 없는 문장", "reason": "REFERENCE에 없음"}
        checks = [{"item": "groundedness", "passed": passed, "issues": [] if passed else [issue]}]
        return {"quality_result": {"passed": passed, "checks": checks, "feedback": []}}

    def nodes(self) -> dict:
        return {
            TECH: self.analysis_node(TECH, "technology_results"),
            MARKET: self.analysis_node(MARKET, "market_traction_results"),
            COMPETITION: self.competition,
            JUDGE: self.judge,
            REPORT: self.report,
            QUALITY: self.quality_node,
        }


def run(workers: Workers, **inputs) -> dict:
    with patch.dict(graph_module.WORKER_NODES, workers.nodes()):
        return graph_module.build_graph().invoke(
            {"candidate_companies": COMPANIES, "company_profiles": PROFILES, **inputs},
            config={"recursion_limit": graph_module.recursion_limit(len(COMPANIES))},
        )


class SupervisorGraphTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_sufficient_evidence_goes_straight_to_report(self):
        workers = Workers()
        state = run(workers)
        names = [c[0] for c in workers.calls]
        self.assertEqual(names, [TECH, MARKET, TECH, MARKET, COMPETITION, JUDGE, JUDGE, REPORT, QUALITY])
        self.assertEqual(state["status"], "SUCCESS")
        self.assertEqual(state["next"], FINISH)
        self.assertEqual(state["rework_counts"], {})
        self.assertEqual(state["step_count"], len(workers.calls))

    def test_insufficient_evidence_requests_rework_from_owning_agent_only(self):
        # B01(Market Size)·B03·B05·B07은 Market 담당 → A사의 Market Agent만 재작업
        workers = Workers(missing={"A": ("B01", "B03", "B05", "B07")})
        state = run(workers)
        reworks = [c for c in workers.calls if c[2]]
        self.assertEqual(reworks, [(MARKET, "A", True)])
        self.assertEqual(state["rework_counts"], {f"A:{MARKET}": 1})
        # 재작업 뒤 비교와 A사 평가만 다시 하고, 그 뒤에 보고서를 쓴다
        after = [c[:2] for c in workers.calls[workers.calls.index(reworks[0]) + 1:]]
        self.assertEqual(after, [(COMPETITION, None), (JUDGE, "A"), (REPORT, None), (QUALITY, None)])
        self.assertEqual(state["status"], "SUCCESS")

    def test_rework_is_capped_when_evidence_stays_insufficient(self):
        missing = tuple(c.id for c in CRITERIA[:6])
        workers = Workers(missing={"A": missing, "B": missing}, fixed_by_rework=False)
        state = run(workers)
        reworks = [c for c in workers.calls if c[2]]
        self.assertLessEqual(len(reworks), 2 * len(COMPANIES) * supervisor.MAX_REWORK_PER_AGENT)
        self.assertTrue(all(v == supervisor.MAX_REWORK_PER_AGENT for v in state["rework_counts"].values()))
        self.assertEqual(state["status"], "SUCCESS")
        self.assertLessEqual(state["step_count"], state["max_steps"])

    def test_quality_failure_loops_back_to_report_until_cap(self):
        workers = Workers(quality=[False, True])
        state = run(workers)
        self.assertEqual([c[0] for c in workers.calls][-4:], [REPORT, QUALITY, REPORT, QUALITY])
        self.assertEqual(state["report_attempts"], 2)
        self.assertTrue(state["quality_result"]["passed"])

        workers = Workers(quality=[False])
        state = run(workers)
        self.assertEqual(state["report_attempts"], supervisor.MAX_REPORT_ATTEMPTS)
        self.assertFalse(state["quality_result"]["passed"])
        self.assertEqual(state["next"], FINISH)

    def test_quality_issue_owned_by_analysis_agent_is_reworked_before_rewrite(self):
        workers = Workers(quality=[False, True], issue_owner=(MARKET, "A"))
        state = run(workers)
        tail = workers.calls[workers.calls.index((REPORT, None, False)):]
        self.assertEqual(
            [c[:2] for c in tail],
            [(REPORT, None), (QUALITY, None), (MARKET, "A"), (COMPETITION, None), (JUDGE, "A"), (REPORT, None), (QUALITY, None)],
        )
        self.assertTrue(tail[2][2])  # Market은 재작업 요청으로 호출된다
        self.assertEqual(state["rework_counts"], {f"A:{MARKET}": 1})
        self.assertEqual(state["status"], "SUCCESS")

    def test_quality_issue_goes_to_rewrite_when_agent_rework_is_used_up(self):
        # A사 Market은 근거 부족으로 이미 재작업했으므로, 품질 지적은 보고서 재작성으로만 처리한다
        workers = Workers(missing={"A": ("B01", "B03", "B05", "B07")}, quality=[False, True], issue_owner=(MARKET, "A"))
        state = run(workers)
        self.assertEqual([c[0] for c in workers.calls][-4:], [REPORT, QUALITY, REPORT, QUALITY])
        self.assertEqual(state["rework_counts"], {f"A:{MARKET}": 1})

    def test_no_rework_after_last_report(self):
        workers = Workers(quality=[False], issue_owner=(TECH, "B"))
        state = run(workers)
        self.assertEqual(sum(1 for c in workers.calls if c[2]), 1)
        self.assertEqual(state["report_attempts"], supervisor.MAX_REPORT_ATTEMPTS)
        self.assertEqual(state["next"], FINISH)

    def test_failed_worker_is_retried_then_excluded(self):
        workers = Workers(fail={f"{TECH}:A": 1})
        state = run(workers)
        self.assertEqual([c[:2] for c in workers.calls[:2]], [(TECH, "A"), (TECH, "A")])
        self.assertEqual(len(state["errors"]), 1)
        self.assertEqual(state["node_status"][f"{TECH}:A"], "DONE")

        workers = Workers(fail={f"{TECH}:A": 5})
        state = run(workers)
        self.assertEqual(sum(1 for c in workers.calls if c[:2] == (TECH, "A")), 1 + supervisor.MAX_NODE_RETRIES)
        self.assertEqual(state["technology_results"]["A"]["findings"], [])
        self.assertEqual(state["status"], "SUCCESS")

    def test_failed_report_stops_with_failed_status(self):
        workers = Workers(fail={REPORT: 5})
        state = run(workers)
        self.assertEqual(state["status"], "FAILED")
        self.assertIn("boom", state["last_error"])

    def test_step_limit_ends_the_run(self):
        workers = Workers()
        state = run(workers, max_steps=4)
        self.assertEqual(len(workers.calls), 4)
        self.assertEqual(state["status"], "FAILED")
        self.assertEqual(state["next"], FINISH)


if __name__ == "__main__":
    unittest.main()
