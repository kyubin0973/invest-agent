import unittest
from unittest.mock import patch

import agents.investment_judge as judge
from evaluation.criteria import CRITERIA, decide, validate_criteria


def make_evidence(level: str = "E3", chunk_id: str = "TEST_C01", source_id: str = "TEST") -> dict:
    return {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "page": 1,
        "title": "Test Evidence",
        "publisher": "Test Publisher",
        "published_date": "2026-01-01",
        "source_type": "partner" if level in ("E3", "E4", "E5") else "company",
        "reference_metadata": {"reference_type": "web", "url": "https://example.com"},
        "evidence_level": level,
        "fact": "테스트 근거",
    }


def make_assessment(
    criterion_id: str,
    *,
    score: int | None = 3,
    status: str = "SCORED",
    chunk_id: str = "TEST_C01",
    source_id: str = "TEST",
    confidence: str = "Medium",
) -> dict:
    evidence = [] if status == "INSUFFICIENT_EVIDENCE" else [{"chunk_id": chunk_id, "source_id": source_id}]
    return {
        "criterion_id": criterion_id,
        "score": score,
        "status": status,
        "evidence": evidence,
        "reasoning": f"{criterion_id} 테스트 판단",
        "confidence": confidence,
        "missing_information": [],
    }


def make_state(evidence: dict | None = None) -> dict:
    items = [] if evidence is None else [evidence]
    return {
        "candidate_companies": ["Test Robotics"],
        "company_profiles": {
            "Test Robotics": {
                "name": "Test Robotics",
                "target_market": "산업용",
                "funding_stage": None,
                "eligibility": {"is_private": True, "exit_completed": False, "evaluation_date": "2026-01-01"},
                "document_scope": {"source_ids": ["TEST"], "document_types": ["deployment"]},
                "source_refs": {},
            }
        },
        "current_company": "Test Robotics",
        "current_company_index": 0,
        "technology_results": {
            "Test Robotics": {
                "summary": "기술 분석",
                "findings": [],
                "evidence": items,
                "risks": [],
                "missing_information": [],
            }
        },
        "market_traction_results": {
            "Test Robotics": {
                "summary": "시장 분석",
                "findings": [],
                "evidence": [],
                "risks": [],
                "missing_information": [],
            }
        },
        "competition_result": {
            "target_market_context": {"Test Robotics": "산업용"},
            "comparisons": [],
            "differentiation": {"Test Robotics": "테스트 차별점"},
            "relative_risks": {"Test Robotics": "테스트 리스크"},
        },
        "investment_results": {},
        "completed_companies": [],
    }


class FakeStructuredJudge:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


class InvestmentJudgeNodeTests(unittest.TestCase):
    def test_valid_batch_uses_one_call_and_keeps_grounded_summary(self):
        evidence = make_evidence("E3")
        response = {
            "criteria": [make_assessment(criterion.id, score=5, confidence="High") for criterion in CRITERIA],
            "key_strengths": [
                {
                    "text": "외부 확인된 강점",
                    "criterion_ids": ["B01"],
                    "evidence": [{"chunk_id": "TEST_C01", "source_id": "TEST"}],
                },
                {
                    "text": "조작된 강점",
                    "criterion_ids": ["B01"],
                    "evidence": [{"chunk_id": "FAKE", "source_id": "FAKE"}],
                },
            ],
            "key_risks": [],
        }
        fake = FakeStructuredJudge([response])

        with patch.object(judge, "get_structured_llm", return_value=fake):
            update = judge.investment_judge_node(make_state(evidence))

        result = update["investment_results"]["Test Robotics"]
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result["decision"], "INVEST")
        self.assertEqual(result["final_score"], 5.0)
        self.assertEqual(result["key_strengths"], ["외부 확인된 강점"])
        self.assertEqual(update["completed_companies"], ["Test Robotics"])

    def test_only_invalid_criterion_is_retried_once(self):
        evidence = make_evidence("E3")
        initial = [make_assessment(criterion.id) for criterion in CRITERIA]
        initial[0] = make_assessment("B01", score=4)
        repair = make_assessment("B01", score=3)
        fake = FakeStructuredJudge(
            [
                {"criteria": initial, "key_strengths": [], "key_risks": []},
                {"criteria": [repair], "key_strengths": [], "key_risks": []},
            ]
        )

        with patch.object(judge, "get_structured_llm", return_value=fake):
            update = judge.investment_judge_node(make_state(evidence))

        result = update["investment_results"]["Test Robotics"]
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(len(result["criteria"]), 12)
        self.assertTrue(all(item["score"] == 3 for item in result["criteria"]))
        self.assertIn("실패한 문항만", fake.calls[1][1][1])

    def test_no_evidence_is_valid_na_without_retry(self):
        response = {
            "criteria": [
                make_assessment(criterion.id, score=None, status="INSUFFICIENT_EVIDENCE", confidence="Low")
                for criterion in CRITERIA
            ],
            "key_strengths": [],
            "key_risks": [],
        }
        fake = FakeStructuredJudge([response])

        with patch.object(judge, "get_structured_llm", return_value=fake):
            update = judge.investment_judge_node(make_state())

        result = update["investment_results"]["Test Robotics"]
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(result["decision"], "HOLD_INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["evidence_coverage"], 0.0)


class CriteriaValidationTests(unittest.TestCase):
    def test_score_five_without_external_evidence_is_capped_at_three(self):
        evidence = make_evidence("E2")
        available = {evidence["chunk_id"]: evidence}
        raw = [make_assessment(criterion.id, score=5, confidence="High") for criterion in CRITERIA]

        validated = validate_criteria(raw, available)

        self.assertTrue(all(item["score"] == 3 for item in validated))
        self.assertTrue(
            all("5점을 3점으로 조정" in item["reasoning"] for item in validated)
        )

    def test_duplicate_id_is_na_and_invalid_confidence_is_low(self):
        evidence = make_evidence("E3")
        available = {evidence["chunk_id"]: evidence}
        raw = [make_assessment(criterion.id) for criterion in CRITERIA]
        raw.append(make_assessment("B01"))
        raw[1] = make_assessment("B02", confidence="Certain")

        validated = validate_criteria(raw, available)
        by_id = {item["criterion_id"]: item for item in validated}

        self.assertIsNone(by_id["B01"]["score"])
        self.assertIn("중복", by_id["B01"]["missing_information"][0])
        self.assertEqual(by_id["B02"]["score"], 3)
        self.assertEqual(by_id["B02"]["confidence"], "Low")

    def test_fabricated_evidence_is_removed_and_score_becomes_na(self):
        evidence = make_evidence("E3")
        available = {evidence["chunk_id"]: evidence}
        raw = [make_assessment(criterion.id) for criterion in CRITERIA]
        raw[0] = make_assessment("B01", chunk_id="FAKE", source_id="FAKE")

        validated = validate_criteria(raw, available)

        self.assertIsNone(validated[0]["score"])
        self.assertEqual(validated[0]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(validated[0]["evidence"], [])

    def test_decision_thresholds(self):
        ids = [criterion.id for criterion in CRITERIA]
        insufficient = {criterion_id: (3 if index < 8 else None) for index, criterion_id in enumerate(ids)}
        hold = {criterion_id: (3 if index < 9 else None) for index, criterion_id in enumerate(ids)}
        invest = {criterion_id: (5 if index < 3 else 3 if index < 9 else None) for index, criterion_id in enumerate(ids)}

        self.assertEqual(decide(insufficient)["decision"], "HOLD_INSUFFICIENT_EVIDENCE")
        self.assertEqual(decide(hold)["decision"], "HOLD")
        self.assertEqual(decide(invest)["decision"], "INVEST")


if __name__ == "__main__":
    unittest.main()
