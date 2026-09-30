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


def add_competitor_evidence(state: dict, criterion_ids: list[str] | None = None) -> dict:
    competitor = "Competitor Robotics"
    competitor_evidence = make_evidence("E3", "COMP_C01", "COMP")
    state["candidate_companies"].append(competitor)
    state["company_profiles"][competitor] = {
        "name": competitor,
        "target_market": "산업용",
        "funding_stage": None,
        "eligibility": {"is_private": True, "exit_completed": False, "evaluation_date": "2026-01-01"},
        "document_scope": {"source_ids": ["COMP"], "document_types": ["deployment"]},
        "source_refs": {},
    }
    state["technology_results"][competitor] = {
        "summary": "경쟁사 기술 분석",
        "findings": [],
        "evidence": [competitor_evidence],
        "risks": [],
        "missing_information": [],
    }
    state["market_traction_results"][competitor] = {
        "summary": "경쟁사 시장 분석",
        "findings": [],
        "evidence": [],
        "risks": [],
        "missing_information": [],
    }
    state["competition_result"]["target_market_context"][competitor] = "산업용"
    state["competition_result"]["differentiation"][competitor] = "경쟁사 차별점"
    state["competition_result"]["relative_risks"][competitor] = "경쟁사 리스크"
    state["competition_result"]["comparisons"] = [
        {
            "dimension": "차별성",
            "criterion_ids": criterion_ids or ["B04"],
            "company_findings": {
                "Test Robotics": "현재 기업 차별성",
                competitor: "경쟁사 비교 결과",
            },
            "evidence_refs": [{"chunk_id": "COMP_C01", "source_id": "COMP"}],
        }
    ]
    return state


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

    def test_competitor_evidence_is_allowed_for_differentiation(self):
        own_evidence = make_evidence("E3")
        state = add_competitor_evidence(make_state(own_evidence))
        criteria = [
            make_assessment(criterion.id, score=None, status="INSUFFICIENT_EVIDENCE", confidence="Low")
            for criterion in CRITERIA
        ]
        criteria[3] = make_assessment(
            "B04",
            score=3,
            chunk_id="COMP_C01",
            source_id="COMP",
        )
        fake = FakeStructuredJudge(
            [{"criteria": criteria, "key_strengths": [], "key_risks": []}]
        )

        with patch.object(judge, "get_structured_llm", return_value=fake):
            update = judge.investment_judge_node(state)

        result = update["investment_results"]["Test Robotics"]
        b04 = next(item for item in result["criteria"] if item["criterion_id"] == "B04")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(b04["score"], 3)
        self.assertEqual(b04["evidence"][0]["chunk_id"], "COMP_C01")
        self.assertIn("Competition Evidence", fake.calls[0][1][1])
        self.assertIn("Competitor Robotics", fake.calls[0][1][1])
        self.assertIn("<allowed_criteria>B04</allowed_criteria>", fake.calls[0][1][1])

    def test_q4_scoped_competitor_evidence_is_rejected_for_q9(self):
        state = add_competitor_evidence(make_state(make_evidence("E3")), ["B04"])
        initial = [
            make_assessment(criterion.id, score=None, status="INSUFFICIENT_EVIDENCE", confidence="Low")
            for criterion in CRITERIA
        ]
        initial[8] = make_assessment("B09", score=3, chunk_id="COMP_C01", source_id="COMP")
        repaired = make_assessment(
            "B09", score=None, status="INSUFFICIENT_EVIDENCE", confidence="Low"
        )
        fake = FakeStructuredJudge(
            [
                {"criteria": initial, "key_strengths": [], "key_risks": []},
                {"criteria": [repaired], "key_strengths": [], "key_risks": []},
            ]
        )

        with patch.object(judge, "get_structured_llm", return_value=fake):
            update = judge.investment_judge_node(state)

        b09 = next(
            item
            for item in update["investment_results"]["Test Robotics"]["criteria"]
            if item["criterion_id"] == "B09"
        )
        self.assertEqual(len(fake.calls), 2)
        self.assertIsNone(b09["score"])

    def test_competitor_evidence_is_rejected_outside_q4_q9(self):
        own_evidence = make_evidence("E3")
        state = add_competitor_evidence(make_state(own_evidence))
        initial = [
            make_assessment(criterion.id, score=None, status="INSUFFICIENT_EVIDENCE", confidence="Low")
            for criterion in CRITERIA
        ]
        initial[0] = make_assessment(
            "B01",
            score=3,
            chunk_id="COMP_C01",
            source_id="COMP",
        )
        repaired_b01 = make_assessment(
            "B01",
            score=None,
            status="INSUFFICIENT_EVIDENCE",
            confidence="Low",
        )
        fake = FakeStructuredJudge(
            [
                {"criteria": initial, "key_strengths": [], "key_risks": []},
                {"criteria": [repaired_b01], "key_strengths": [], "key_risks": []},
            ]
        )

        with patch.object(judge, "get_structured_llm", return_value=fake):
            update = judge.investment_judge_node(state)

        result = update["investment_results"]["Test Robotics"]
        b01 = next(item for item in result["criteria"] if item["criterion_id"] == "B01")
        self.assertEqual(len(fake.calls), 2)
        self.assertIsNone(b01["score"])
        self.assertEqual(b01["evidence"], [])


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

    def test_confidence_is_capped_by_evidence_level_and_source_diversity(self):
        one_external = make_evidence("E3")
        available = {one_external["chunk_id"]: one_external}
        raw = [make_assessment(criterion.id, confidence="High") for criterion in CRITERIA]

        validated = validate_criteria(raw, available)

        self.assertTrue(all(item["confidence"] == "Medium" for item in validated))
        self.assertTrue(all("confidence 상한을 Medium" in item["reasoning"] for item in validated))

    def test_high_confidence_requires_two_external_sources(self):
        first = make_evidence("E3", "FIRST_C01", "FIRST")
        second = make_evidence("E4", "SECOND_C01", "SECOND")
        available = {item["chunk_id"]: item for item in (first, second)}
        raw = [make_assessment(criterion.id, confidence="High") for criterion in CRITERIA]
        for item in raw:
            item["evidence"] = [
                {"chunk_id": "FIRST_C01", "source_id": "FIRST"},
                {"chunk_id": "SECOND_C01", "source_id": "SECOND"},
            ]

        validated = validate_criteria(raw, available)

        self.assertTrue(all(item["confidence"] == "High" for item in validated))

    def test_single_company_source_caps_confidence_at_low(self):
        one_company_source = make_evidence("E1")
        available = {one_company_source["chunk_id"]: one_company_source}
        raw = [make_assessment(criterion.id, confidence="Medium") for criterion in CRITERIA]

        validated = validate_criteria(raw, available)

        self.assertTrue(all(item["confidence"] == "Low" for item in validated))

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
