"""실제 API/임베딩 다운로드 없이 실행: python -m unittest discover -s tests -v"""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from pydantic import ValidationError

from agents.technology import (
    FindingDraft,
    QuestionReview,
    RetrievalReview,
    TechnologyAnalysis,
    _assemble,
    _review,
    technology_node,
)
from core.state import AnalysisResult, EvidenceItem, Finding
from prompts.technology import QUESTIONS


TEXT = "The robot demonstrated sorting parts in a controlled test. Human supervision was required."


def document(source="F1", *, company="figure", source_type="company", text=TEXT, **metadata):
    return Document(page_content=text, metadata={
        "chunk_id": f"{source}_P02_C01", "source_id": source, "company": company,
        "domain": "technology", "document_type": "hardware", "source_type": source_type,
        "title": f"{source} original title", "publisher": "Original publisher",
        "published_date": "2026-01-02", "page": 2,
        "reference_metadata": json.dumps({"reference_type": "web", "url": "https://example.org/source"}),
        "score": 0.99, **metadata,
    })


def ref(doc):
    return {key: doc.metadata[key] for key in ("chunk_id", "source_id")}


def review(doc, *, insufficient=(), unrelated=False):
    return {"reviews": [{
        "question_id": q.id,
        "relevant_refs": [] if unrelated else [ref(doc)],
        "sufficient": q.id not in insufficient and not unrelated,
        "missing_information": [f"{q.dimension} 추가 확인 필요"] if q.id in insufficient else [],
        "retry_query": f"targeted {q.id} measurements" if q.id in insufficient else "",
    } for q in QUESTIONS]}


def analysis(doc, *, level="E2", all_questions=True):
    return {
        "evidence": [{**ref(doc), "fact": "제한 환경에서 부품 분류를 시연했다고 발표했다.",
                      "supporting_quote": "The robot demonstrated sorting parts in a controlled test.",
                      "evidence_level": level}],
        "findings": [{"question_id": q.id, "statement": "제한 환경에서의 시연 근거가 있다.",
                      "evidence_refs": [ref(doc)]} for q in (QUESTIONS if all_questions else QUESTIONS[:1])],
        "risks": [], "missing_information": [],
    }


class TechnologyQuestionIdTests(unittest.TestCase):
    def test_all_internal_ids_pass_and_are_enumerated_in_llm_schema(self):
        raw = review(document())
        parsed = RetrievalReview.model_validate(raw)
        expected = [q.id for q in QUESTIONS]
        self.assertEqual([item.question_id for item in parsed.reviews], expected)
        parsed_findings = TechnologyAnalysis.model_validate(analysis(document())).findings
        self.assertEqual([item.question_id for item in parsed_findings], expected)
        for schema_type, definition in ((RetrievalReview, "QuestionReview"), (TechnologyAnalysis, "FindingDraft")):
            schema = schema_type.model_json_schema()
            self.assertEqual(
                set(schema["$defs"][definition]["properties"]["question_id"]["enum"]),
                set(expected),
            )

    def test_criterion_ids_are_rejected_without_implicit_mapping(self):
        for criterion_id in ("B02", "B04", "B08", "B09", "H11", "H12"):
            with self.subTest(criterion_id=criterion_id):
                for model, raw in ((QuestionReview, review(document())["reviews"][0]),
                                   (FindingDraft, analysis(document())["findings"][0])):
                    raw["question_id"] = criterion_id
                    with self.assertRaises(ValidationError) as error:
                        model.model_validate(raw)
                    self.assertEqual(error.exception.errors()[0]["type"], "literal_error")
                    self.assertEqual(raw["question_id"], criterion_id)


class TechnologyNodeTests(unittest.TestCase):
    def setUp(self):
        self.doc = document()
        self.state = {"current_company": "Figure AI", "technology_results": {"Existing": {"summary": "keep"}},
                      "market_traction_results": {"Figure AI": {"summary": "market"}}}
        self.search = self.enterContext(patch("agents.technology.search_technology", return_value=[self.doc]))
        self.factory = self.enterContext(patch("agents.technology.get_structured_llm"))
        self.llm = self.factory.return_value

    def run_node(self, responses, state=None):
        self.llm.invoke.side_effect = responses
        return technology_node(self.state if state is None else state)

    def result(self, responses):
        return self.run_node(responses)["technology_results"]["Figure AI"]

    def test_sufficient_evidence_preserves_contract_metadata_and_input(self):
        before = deepcopy(self.state)
        update = self.run_node([review(self.doc), analysis(self.doc)])
        self.assertEqual(self.state, before)
        self.assertEqual(set(update), {"technology_results", "technology_done"})
        self.assertIs(update["technology_done"], True)
        self.assertEqual(set(update["technology_results"]), {"Figure AI"})
        result = update["technology_results"]["Figure AI"]
        self.assertEqual(set(result), set(AnalysisResult.__annotations__))
        self.assertEqual(set(result["findings"][0]), set(Finding.__annotations__))
        self.assertEqual(set(result["evidence"][0]), set(EvidenceItem.__annotations__))
        evidence = result["evidence"][0]
        for key in ("source_id", "chunk_id", "page", "title", "publisher", "published_date", "source_type"):
            self.assertEqual(evidence[key], self.doc.metadata[key])
        self.assertEqual(evidence["reference_metadata"], json.loads(self.doc.metadata["reference_metadata"]))
        self.assertIn("자체 발표", evidence["fact"])
        self.assertEqual(result["missing_information"], [])
        self.assertEqual(len(result["evidence"]), 1)  # 모든 검색에서 같은 청크가 나와도 한 번만
        self.assertEqual(self.search.call_count, len(QUESTIONS))
        self.assertEqual([c.args[0] for c in self.factory.call_args_list], [RetrievalReview, TechnologyAnalysis])
        for call in self.search.call_args_list:
            self.assertEqual(call.args[1], "Figure AI")
            self.assertEqual(call.kwargs, {"k": 5})

    def test_only_missing_question_is_retried_and_resolved_missing_is_removed(self):
        result = self.result([review(self.doc, insufficient={"reliability"}), review(self.doc), analysis(self.doc)])
        self.assertEqual(self.search.call_count, len(QUESTIONS) + 1)
        self.assertEqual(self.search.call_args.args[0], "Figure AI targeted reliability measurements")
        self.assertEqual(result["missing_information"], [])
        self.assertIn("재검색 가능: False", self.llm.invoke.call_args_list[1].args[0][1][1])

    def test_retry_is_one_round_even_when_all_questions_remain_insufficient(self):
        insufficient = {q.id for q in QUESTIONS}
        result = self.result([review(self.doc, insufficient=insufficient),
                              review(self.doc, insufficient=insufficient), analysis(self.doc)])
        self.assertEqual(self.search.call_count, 2 * len(QUESTIONS))
        self.assertEqual(self.llm.invoke.call_count, 3)
        self.assertEqual(len(result["missing_information"]), len(QUESTIONS))
        self.assertTrue(result["findings"])  # 부족해도 확인된 부분 근거는 보존

    def test_empty_search_returns_missing_without_llm_or_fabricated_risk(self):
        self.search.return_value = []
        result = self.result([])
        self.assertEqual(self.search.call_count, 2 * len(QUESTIONS))
        self.factory.assert_not_called()
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["risks"], [])
        self.assertIn("부족", result["summary"])
        for q in QUESTIONS:
            self.assertTrue(any(q.dimension in m for m in result["missing_information"]))

    def test_irrelevant_high_score_documents_do_not_reach_analysis(self):
        result = self.result([review(self.doc, unrelated=True), review(self.doc, unrelated=True)])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(self.llm.invoke.call_count, 2)
        self.assertTrue(all(c.args[0] is RetrievalReview for c in self.factory.call_args_list))

    def test_wrong_company_market_and_profile_disallowed_documents_are_excluded(self):
        bad_company = document("X1", company="1x", text="OTHER_COMPANY_SECRET")
        market = document("M1", domain="market", text="MARKET_ONLY")
        disallowed = document("F2", text="NOT_ALLOWED")
        wrong_type = document("F3", document_type="ai", text="WRONG_TYPE")
        self.search.return_value = [bad_company, market, disallowed, wrong_type, self.doc]
        state = {**self.state, "company_profiles": {"Figure AI": {"document_scope": {
            "source_ids": ["F1", "F3"], "document_types": ["hardware"],
        }}}}
        update = self.run_node([review(self.doc), analysis(self.doc)], state)
        result = update["technology_results"]["Figure AI"]
        self.assertEqual([e["source_id"] for e in result["evidence"]], ["F1"])
        for call in self.llm.invoke.call_args_list:
            prompt = call.args[0][1][1]
            for marker in ("OTHER_COMPANY_SECRET", "MARKET_ONLY", "NOT_ALLOWED", "WRONG_TYPE"):
                self.assertNotIn(marker, prompt)

    def test_explicit_empty_document_scope_does_not_allow_any_source(self):
        state = {**self.state, "company_profiles": {"Figure AI": {"document_scope": {
            "source_ids": [], "document_types": [],
        }}}}
        result = self.run_node([], state)["technology_results"]["Figure AI"]
        self.assertEqual(result["evidence"], [])
        self.factory.assert_not_called()

    def test_missing_review_question_is_retried(self):
        first = review(self.doc)
        first["reviews"] = first["reviews"][:-1]
        self.result([first, review(self.doc), analysis(self.doc)])
        self.assertEqual(self.search.call_count, len(QUESTIONS) + 1)
        self.assertIn("supply chain", self.search.call_args.args[0])

    def test_invalid_review_reference_cannot_mark_question_sufficient(self):
        first = review(self.doc)
        first["reviews"][0]["relevant_refs"][0]["source_id"] = "FAKE"
        self.result([first, review(self.doc), analysis(self.doc)])
        self.assertEqual(self.search.call_count, len(QUESTIONS) + 1)
        self.assertIn("use case", self.search.call_args.args[0])

    def test_invalid_evidence_removes_dependent_claim_from_all_output_text(self):
        for change in ({"source_id": "FAKE"}, {"chunk_id": "FAKE"},
                       {"supporting_quote": "This text was invented."},
                       {"evidence_level": "E3"}, {"evidence_level": "E4"}, {"evidence_level": "E5"}):
            with self.subTest(change=change):
                draft = analysis(self.doc, all_questions=False)
                draft["evidence"][0].update(change)
                if "chunk_id" in change:
                    # 제외 대상은 가짜 청크를 참조한 문장이다. 실제 검색 ID를
                    # 참조하면서 별도 분석만 누락된 경우는 위 복원 규칙의 대상이다.
                    draft["findings"][0]["evidence_refs"][0]["chunk_id"] = change["chunk_id"]
                draft["findings"][0]["statement"] = "유출되면 안 되는 잘못된 주장"
                draft["risks"] = deepcopy(draft["findings"])
                result = self.result([review(self.doc), draft])
                self.assertEqual(result["evidence"], [])
                self.assertEqual(result["findings"], [])
                self.assertEqual(result["risks"], [])
                self.assertNotIn("유출되면", json.dumps(result, ensure_ascii=False))
                self.assertTrue(any("검증 실패" in m for m in result["missing_information"]))

    def test_partial_invalid_references_drop_entire_compound_finding(self):
        draft = analysis(self.doc, all_questions=False)
        draft["findings"][0]["evidence_refs"].append({"chunk_id": "FAKE", "source_id": "FAKE"})
        result = self.result([review(self.doc), draft])
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["evidence"], [])

    def test_missing_evidence_analysis_keeps_valid_reference_without_inventing_fact_or_level(self):
        partner = document("A4", source_type="partner")
        self.search.return_value = [self.doc, partner]
        checked = review(self.doc)
        checked["reviews"][0]["relevant_refs"].append(ref(partner))
        draft = analysis(self.doc, all_questions=False)
        draft["findings"][0]["evidence_refs"].append(ref(partner))
        result = self.result([checked, draft])
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(len(result["evidence"]), 2)
        restored = next(e for e in result["evidence"] if e["source_id"] == "A4")
        self.assertIsNone(restored["fact"])
        self.assertIsNone(restored["evidence_level"])
        for key in ("source_id", "chunk_id", "title", "page", "source_type"):
            self.assertEqual(restored[key], partner.metadata[key])
        self.assertTrue(any("LLM 근거 분석 누락" in m for m in result["missing_information"]))

    def test_valid_reference_uses_all_retrieved_documents_not_only_final_review_selection(self):
        partner = document("A4", source_type="partner")
        self.search.return_value = [self.doc, partner]
        draft = analysis(partner, all_questions=False)
        draft["evidence"] = []
        # 최종 점검은 self.doc만 선택해도 실제 검색된 partner 참조는 존재한다.
        result = self.result([review(self.doc), draft])
        self.assertEqual(result["findings"][0]["evidence_refs"], [ref(partner)])
        self.assertEqual(result["evidence"][0]["chunk_id"], partner.metadata["chunk_id"])
        self.assertIsNone(result["evidence"][0]["fact"])
        self.assertIsNone(result["evidence"][0]["evidence_level"])

    def test_duplicate_evidence_is_not_arbitrarily_selected(self):
        draft = analysis(self.doc)
        draft["evidence"].append(deepcopy(draft["evidence"][0]))
        result = self.result([review(self.doc), draft])
        self.assertEqual(result["evidence"], [])
        self.assertTrue(any("중복" in m for m in result["missing_information"]))

    def test_independent_sources_with_same_text_are_preserved_and_unused_source_removed(self):
        partner = document("A3", source_type="partner")
        unused = document("F2")
        self.search.return_value = [self.doc, partner, unused]
        checked = review(self.doc)
        for item in checked["reviews"]:
            item["relevant_refs"] += [ref(partner), ref(unused)]
        draft = analysis(self.doc)
        draft["evidence"] += analysis(partner, level="E3")["evidence"] + analysis(unused)["evidence"]
        draft["findings"][0]["evidence_refs"].append(ref(partner))
        result = self.result([checked, draft])
        self.assertEqual({e["source_id"] for e in result["evidence"]}, {"F1", "A3"})
        self.assertEqual(result["evidence"][1]["evidence_level"], "E3")

    def test_external_operating_evidence_keeps_level_and_cited_risk(self):
        self.doc = document("A5", source_type="partner",
                            text="We operated the robot at our factory. Human intervention was required after a fault.")
        self.search.return_value = [self.doc]
        draft = analysis(self.doc, level="E4")
        draft["evidence"][0].update(fact="공장에서 운영했고 고장 후 사람 개입이 필요했다.",
                                     supporting_quote=self.doc.page_content)
        draft["risks"] = [{"question_id": "reliability", "statement": "고장 후 사람의 개입이 필요했다.",
                           "evidence_refs": [ref(self.doc)]}]
        result = self.result([review(self.doc), draft])
        self.assertEqual(result["evidence"][0]["evidence_level"], "E4")
        self.assertIn("[A5_P02_C01 | A5]", result["risks"][0])
        self.assertNotIn("자체 발표", result["evidence"][0]["fact"])

    def test_external_current_application_allows_e3_but_not_demo_or_operating_levels(self):
        doc = document("A4", source_type="partner",
                       text="Apptronik incorporated TI's motor-control technologies.")
        for level in ("E2", "E3", "E4", "E5"):
            with self.subTest(level=level):
                draft = analysis(doc, level=level, all_questions=False)
                draft["evidence"][0].update(fact="TI 모터 제어 기술이 적용되어 있다.", supporting_quote=doc.page_content)
                draft["findings"][0]["statement"] = "TI 모터 제어 기술이 적용되어 있다."
                result = _assemble("Apptronik", [doc], TechnologyAnalysis.model_validate(draft), [])
                self.assertEqual(len(result["evidence"]), int(level == "E3"))
                self.assertEqual(len(result["findings"]), int(level == "E3"))
                if level == "E3":
                    self.assertEqual(result["evidence"][0]["evidence_level"], "E3")
        # 출처만 외부이거나 시연이라는 단어만 나오는 계획/부정문은 실적이 아니다.
        for quote in ("TI will demonstrate a prototype next year.", "TI has not demonstrated a prototype."):
            doc.page_content = quote
            for level in ("E2", "E3"):
                draft["evidence"][0].update(supporting_quote=quote, evidence_level=level)
                self.assertEqual(_assemble("Apptronik", [doc], TechnologyAnalysis.model_validate(draft), [])["evidence"], [])

    def test_component_safety_keeps_application_but_excludes_system_claim_in_fact_or_finding(self):
        doc = document("A4", source_type="partner", text=(
            "Apollo uses TI technology that has been functional safety-certified. "
            "TI provides parts to prepare for upcoming system-level functional safety certifications."
        ))
        good = "TI의 안전 인증 기술이 Apollo의 구동 시스템에 적용되어 있다."
        draft = analysis(doc, level="E3", all_questions=False)
        draft["evidence"][0].update(fact=good, supporting_quote=doc.page_content)
        draft["findings"][0]["statement"] = good
        result = _assemble("Apptronik", [doc], TechnologyAnalysis.model_validate(draft), [])
        self.assertEqual(len(result["findings"]), 1)
        for bad in ("Apollo는 안전성을 확보했다.", "TI 기술로 Apollo의 전체 시스템 안전성이 인증되었다.",
                    "TI 기술을 사용해 공장에서 안전하게 작업할 수 있다.", "장기 신뢰성과 반복 작업 성과가 검증됐다."):
            for field in ("fact", "statement"):
                with self.subTest(bad=bad, field=field):
                    changed = deepcopy(draft)
                    changed["evidence" if field == "fact" else "findings"][0][field] = bad
                    result = _assemble("Apptronik", [doc], TechnologyAnalysis.model_validate(changed), [])
                    self.assertEqual(result["findings"], [])
                    self.assertEqual(result["evidence"], [])
                    self.assertNotIn(bad, result["summary"])
        # 별도 evidence[]가 없어도 메타데이터 복원 경로가 범위 검증을 우회할 수 없다.
        draft["evidence"] = []
        draft["findings"][0]["statement"] = "Apollo는 안전성을 확보했다."
        self.assertEqual(_assemble("Apptronik", [doc], TechnologyAnalysis.model_validate(draft), [])["findings"], [])

    def test_component_only_safety_forces_insufficient_and_one_targeted_retry(self):
        self.doc = document("A4", source_type="partner", text=(
            "Apollo uses safety-certified TI components. "
            "System-level safety certification will be completed in the future."
        ))
        self.search.return_value = [self.doc]
        self.llm.invoke.return_value = review(self.doc)
        checked = _review("Apptronik", [self.doc], can_retry=True)
        safety = next(r for r in checked if r.question_id == "safety")
        self.assertFalse(safety.sufficient)
        self.assertIn("전체 시스템", safety.missing_information[0])
        self.assertTrue(safety.retry_query)
        draft = analysis(self.doc, level="E3", all_questions=False)
        draft["evidence"][0].update(fact="TI의 안전 인증 부품을 사용한다.", supporting_quote=self.doc.page_content)
        draft["findings"][0]["statement"] = "TI의 안전 인증 부품을 사용한다."
        result = self.result([review(self.doc), review(self.doc), draft])
        self.assertEqual(self.search.call_count, len(QUESTIONS) + 1)
        self.assertTrue(any("전체 시스템" in m for m in result["missing_information"]))
        # 현재 완료된 시스템 수준 검증은 부품 인증과 구분하여 허용한다.
        self.llm.invoke.side_effect = None
        self.doc.page_content += " System-level safety testing was completed."
        self.assertTrue(next(r for r in _review("Apptronik", [self.doc], can_retry=False) if r.question_id == "safety").sufficient)

    def test_quote_whitespace_normalization_preserves_pdf_line_breaks(self):
        self.doc.page_content = TEXT.replace("sorting parts", "sorting\n   parts")
        result = self.result([review(self.doc), analysis(self.doc)])
        self.assertEqual(len(result["evidence"]), 1)

    def test_verbatim_english_quote_passes_but_translation_and_paraphrase_are_rejected(self):
        for quote, accepted in (
            ("The robot demonstrated sorting parts in a controlled test.", True),
            ("로봇은 통제된 시험에서 부품을 분류하는 모습을 시연했다.", False),
            ("The robot showed that it could sort components during a controlled trial.", False),
        ):
            with self.subTest(quote=quote):
                draft = analysis(self.doc, all_questions=False)
                draft["evidence"][0]["supporting_quote"] = quote
                result = self.result([review(self.doc), draft])
                self.assertEqual(len(result["evidence"]), int(accepted))
                self.assertEqual(len(result["findings"]), int(accepted))
                if accepted:
                    self.assertIn("제한 환경", result["evidence"][0]["fact"])
                    self.assertIn("시연 근거", result["findings"][0]["statement"])
                else:
                    self.assertTrue(any("인용문이 원문에 없음" in m for m in result["missing_information"]))

    def test_pdf_controls_and_unicode_are_normalized_only_for_quote_comparison(self):
        original = "The ro\x00bot\u200b performed a ﬁeld\n\t test at the cafe\u0301."
        self.doc.page_content = original
        draft = analysis(self.doc)
        draft["evidence"][0]["supporting_quote"] = "The robot performed a field test at the café."
        result = self.result([review(self.doc), draft])
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(self.doc.page_content, original)

    def test_normalization_does_not_restore_missing_letters_or_accept_empty_quote(self):
        self.doc.page_content = "The robot performs speci\x00c tasks."
        for quote in ("The robot performs specific tasks.", "\x00\u200b"):
            with self.subTest(quote=quote):
                draft = analysis(self.doc)
                draft["evidence"][0]["supporting_quote"] = quote
                result = self.result([review(self.doc), draft])
                self.assertEqual(result["evidence"], [])
                self.assertEqual(result["findings"], [])

    def test_api_and_search_errors_propagate_instead_of_returning_done(self):
        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            self.result([RuntimeError("API unavailable")])
        self.search.side_effect = RuntimeError("retriever unavailable")
        with self.assertRaisesRegex(RuntimeError, "retriever unavailable"):
            technology_node(self.state)

    def test_bad_schema_is_error_not_missing_evidence(self):
        with self.assertRaises(ValidationError):
            self.result([{"unexpected": "bad response"}])
        draft = analysis(self.doc)
        draft["evidence"][0]["evidence_level"] = "E6"
        with self.assertRaises(ValidationError):
            self.result([review(self.doc), draft])

    def test_empty_or_unknown_company_is_rejected_before_search(self):
        for company in (None, "", "   ", "Unknown company"):
            with self.subTest(company=company), self.assertRaises(ValueError):
                technology_node({"current_company": company})
        self.search.assert_not_called()
        self.factory.assert_not_called()

    def test_unknown_or_duplicate_review_question_is_rejected(self):
        for duplicate in (False, True):
            first = review(self.doc)
            if duplicate:
                first["reviews"].append(deepcopy(first["reviews"][0]))
            else:
                first["reviews"][0]["question_id"] = "market_size"
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                self.result([first])

    def test_out_of_scope_finding_is_not_exported(self):
        draft = analysis(self.doc, all_questions=False)
        draft["findings"][0]["question_id"] = "market_size"
        # 필터링 후 빈 결과를 내보내기 전에 구조화 출력 검증에서 차단한다.
        with self.assertRaises(ValidationError):
            self.result([review(self.doc), draft])


class TechnologyGraphTests(unittest.TestCase):
    def test_real_graph_keeps_two_company_results_and_parallel_market_results(self):
        from core.graph import build_graph, recursion_limit
        from documents.profiles import load_company_profiles

        figure = document()
        apptronik = document("A1", company="apptronik")
        companies = ["Figure AI", "Apptronik"]
        docs = dict(zip(companies, (figure, apptronik)))
        llm = Mock()
        llm.invoke.side_effect = [review(figure), analysis(figure), review(apptronik), analysis(apptronik)]
        with patch("agents.technology.search_technology", side_effect=lambda query, company, k: [docs[company]]), \
                patch("agents.technology.get_structured_llm", return_value=llm):
            state = build_graph().invoke(
                {"candidate_companies": companies, "company_profiles": load_company_profiles()},
                config={"recursion_limit": recursion_limit(2)},
            )
        self.assertEqual(set(state["technology_results"]), set(companies))
        self.assertEqual(set(state["market_traction_results"]), set(companies))
        self.assertEqual(state["completed_companies"], companies)
        self.assertTrue(state["analysis_join_ready"])
        self.assertTrue(state["technology_done"])
        for company, doc in docs.items():
            result = state["technology_results"][company]
            self.assertEqual(result["evidence"][0]["source_id"], doc.metadata["source_id"])
            self.assertNotIn("STUB", result["summary"])


if __name__ == "__main__":
    unittest.main()
