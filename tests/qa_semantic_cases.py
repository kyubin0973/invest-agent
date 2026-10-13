"""5번 의미 평가용 합성 사례. 실제 기업·제출 보고서·trace 증빙이 아니다."""
from copy import deepcopy
from evaluation.criteria import CRITERIA, decide
from tests.qa_fixtures import fixture

CITE = "[S1_P01_C01 | S1]"
LAB = "[S2_P01_C01 | S2]"
OBSERVATION = f"독립 연구소의 동일 실험실 조건에서 A는 80%, 비교 장비는 70%의 성공률을 보였다. 현장 일반화와 상용 운영은 확인되지 않았으며 안전성 검증이 추가로 필요하다. {LAB}"
CASES = {
    "normal": (None, None),
    "observation": (OBSERVATION, None),
    "supported_recommendation": (OBSERVATION + f" 따라서 A 기술을 반드시 채택해야 한다. {LAB}", "neutrality"),
    "demo_exaggeration": (f"A는 상용 배치를 완료했고 현장 성과가 입증됐다. {CITE}", "groundedness"),
    "number_mismatch": (f"독립 실험에서 A의 성공률은 99%였다. {LAB}", "groundedness"),
    "absence_as_negative": (f"상용 자료가 없으므로 A는 상용 운영에 실패했다. {CITE}", "groundedness"),
    "risk_omission": (f"두 출처가 A의 데모를 소개한다. 위험은 없고 추가 검증도 필요 없다. {CITE} {LAB}", "bias_control"),
    "investment_consistent": ("투자 판단은 State와 동일한 HOLD_INSUFFICIENT_EVIDENCE이며 추가 근거 확인이 필요하다.", None),
    "investment_conflict": ("투자 판단은 INVEST로 확정한다.", "neutrality"),
}


def semantic_fixture(case_id):
    """정답은 평가 기대값이며 Judge 입력에는 전달하지 않는다."""
    statement, expected = CASES[case_id]
    state, originals = fixture()
    evidence = deepcopy(state["technology_results"]["A"]["evidence"][0])
    evidence.update(source_id="S2", chunk_id="S2_P01_C01", title="Independent lab limits",
                    publisher="Independent Lab", source_type="research",
                    fact="동일 실험실 성공률 A 80%, 비교 장비 70%; 현장·안전 일반화 불가")
    evidence["reference_metadata"] = {"reference_type": "web", "url": "https://example.test/lab",
                                      "source_family_id": "independent:lab"}
    if case_id == "risk_omission":
        evidence.update(publisher="Announcement Mirror", source_type="news")
        evidence["reference_metadata"]["source_family_id"] = "announcement:demo"
    state["technology_results"]["A"]["evidence"].append(evidence)
    state["references"].append({**{k: deepcopy(evidence[k]) for k in (
        "source_id", "title", "publisher", "published_date", "reference_metadata")}, "pages": [1]})
    originals["S2_P01_C01"] = {"source_id": "S2", "chunk_id": "S2_P01_C01", "page": 1,
        "text": "Independent lab: in identical laboratory conditions A achieved 80% success and comparator 70%. No field or commercial validation. Safety requires further testing."}
    if case_id == "risk_omission":
        originals["S2_P01_C01"]["text"] = "Reprint of A's announcement: laboratory demo only; commercial deployment unverified. Safety and operational risks require further verification."
    state["final_report"] = state["final_report"].replace("# REFERENCE", OBSERVATION + "\n# REFERENCE")
    if statement is not None:
        state["final_report"] = f"# SUMMARY\n{statement}\n# REFERENCE\n[S1] Demo announcement, A, p. 1\n"
    state["final_report"] += f"[S2] Independent lab limits, {evidence['publisher']}, p. 1\n"
    if case_id.startswith("investment_"):
        state["final_report"] = state["final_report"].replace("# REFERENCE", OBSERVATION + "\n# REFERENCE")
        state["investment_results"]["A"] = {**decide({c.id: None for c in CRITERIA}), "criteria": []}
    state["trace_id"] = "synthetic-qa-semantic-" + case_id
    return state, originals, expected
