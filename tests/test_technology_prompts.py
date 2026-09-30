import json
import unittest

from agents.technology import EvidenceDraft
from evaluation.criteria import criteria_for
from prompts.common import CITATION_RULES, EVIDENCE_RULES
from prompts.technology import QUESTIONS, build_analysis_messages, build_review_messages


class TechnologyPromptTests(unittest.TestCase):
    def test_quote_language_exception_is_explicit_in_system_request_and_schema(self):
        messages = build_analysis_messages("Apptronik", [], [])
        for _, prompt in messages:
            self.assertIn("All explanatory fields must be written in Korean.", prompt)
            self.assertIn("supporting_quote is an exception", prompt)
            self.assertIn("copy it verbatim", prompt)
            self.assertIn("Do not translate, paraphrase, correct, or summarize", prompt)
        prompt = messages[1][1]
        self.assertIn('올바른 출력:\nsupporting_quote: "Apollo 2 enables continuous learning through deployment across Robot Park locations."', prompt)
        self.assertIn('잘못된 출력:\nsupporting_quote: "Apollo 2는', prompt)
        description = EvidenceDraft.model_json_schema()["properties"]["supporting_quote"]["description"]
        self.assertIn("Exception to Korean output", description)
        self.assertIn("original language", description)
        self.assertIn("Do not translate, paraphrase", description)

    def test_question_id_contract_separates_criterion_ids_with_one_review_example(self):
        prompt = build_review_messages("Apptronik", [], can_retry=True)[1][1]
        self.assertIn("내부 분석 질문 ID와 투자평가 Criterion ID는 서로 다른 개념", prompt)
        self.assertIn("B02/B04/B08/B09/H11/H12", prompt)
        self.assertIn("question_id로 사용 금지", prompt)
        self.assertIn("다음 8개 내부 하위 질문 ID를 각각 한 번씩", prompt)
        examples = [json.loads(line) for line in prompt.splitlines() if line.startswith('{"question_id"')]
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["question_id"], "product")

    def test_all_calls_include_shared_rules_before_request_and_korean_output(self):
        calls = [build_review_messages("Figure AI", [], can_retry=True),
                 build_review_messages("Figure AI", [], can_retry=False),
                 build_analysis_messages("Figure AI", [], ["장기운영 확인 필요"])]
        for messages in calls:
            with self.subTest(messages=messages[1][1][-80:]):
                self.assertEqual([role for role, _ in messages], ["system", "user"])
                self.assertIn("한국어", messages[0][1])
                user = messages[1][1]
                for rule in (CITATION_RULES, EVIDENCE_RULES):
                    self.assertIn(rule, user)
                    self.assertLess(user.index(rule), user.index("[요청]"))
                self.assertLess(user.index("[근거]"), user.index(EVIDENCE_RULES))
                for criterion in criteria_for("technology"):
                    self.assertIn(f"{criterion.id}({criterion.q})", user)
                for question in QUESTIONS:
                    self.assertIn(question.id, user)

    def test_role_boundaries_and_evidence_levels_are_explicit(self):
        prompt = build_analysis_messages("Apptronik", [], [])[1][1]
        for text in ("Market 담당", "Competition", "Judge 담당", "E1", "E2", "E3", "E4", "E5",
                     "기업 자체 자료는 E3~E5로 올리지 않는다", "계획·인용된 기업 주장만 있으면 E1",
                     "정보가 없다는 이유로 risks를 만들지 않는다", "supporting_quote", "연속 구절",
                     "해당 fact의 검증 수준", "실제 운영·계약 또는 반복·규모화",
                     "E2는 실제 Prototype/Demo/제한 환경 테스트를 설명하는 근거가 있을 때만",
                     "일반 기술 설명이나 판단이 애매하다는 이유로 E2를 부여하지 않는다",
                     "그 기술 적용 사실은 E3 가능", "partner라는 이유만으로 E3를 주지 않으며",
                     "component-level safety만 있으면 safety sufficient=False",
                     "실환경 안전 검증은 missing_information에 남긴다",
                     "금지: \"Apollo는 안전성을 확보했다.\"",
                     "같은 문서의 다른 청크 문장을 섞지 마라", "깨진 단어를 추측 복원하지 말고"):
            self.assertIn(text, prompt)

    def test_manufacturing_plans_alone_are_insufficient_in_review_and_analysis(self):
        for messages in (build_review_messages("Apptronik", [], can_retry=True),
                         build_analysis_messages("Apptronik", [], [])):
            prompt = messages[1][1]
            for rule in ("생산 계획·목표·Pilot만으로 sufficient=True로 판단하지 않는다",
                         "실제 생산능력·생산시설", "검증된 제조 파트너 수행", "현재 양산 근거",
                         "계획만 있으면 fact에는 계획임을 명시하고 sufficient=False",
                         "확인되지 않은 부분을 missing_information에 남긴다"):
                self.assertIn(rule, prompt)

    def test_final_review_forbids_more_queries(self):
        prompt = build_review_messages("Figure AI", [], can_retry=False)[1][1]
        self.assertIn("재검색 가능: False", prompt)
        self.assertIn("재검색 불가 상태에서는 retry_query를 빈 문자열", prompt)
        self.assertIn("검색 점수는 관련성 근거가 아니다", prompt)


if __name__ == "__main__":
    unittest.main()
