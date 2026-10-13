"""5번 독립 QA 실행 도구. 기본은 입력 점검이며 --live만 유료 Judge를 호출한다.

실제 보고서·원문 JSON을 전달한다. Supervisor/State/보고서 담당 파일을 수정하거나
합성 입력을 제출 증빙으로 만드는 기능은 없다. 출력은 outputs/qa에만 저장한다.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

from agents.quality_evaluator import evaluate_report_details as evaluate_report
from evaluation.report_quality import QualityJudgeError, prepare_input, redact_secrets

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "qa"


def load_object(path: Path) -> dict:
    """JSON 객체만 허용하고 파일·JSON 오류의 원문을 출력하지 않는다."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("object required")
        return value
    except (OSError, ValueError):
        raise ValueError("QA 입력 JSON을 읽을 수 없거나 객체 형식이 아닙니다.") from None


def run_quality(state: dict, originals: dict, *, live: bool = False) -> dict:
    """입력 점검 결과를 QA 합격이나 실제 trace 확보로 표시하지 않는다."""
    state = {**state, "trace_id": state.get("trace_id") or uuid.uuid4().hex}
    try:
        if live:
            # 기존 설정을 재사용한다. 키는 출력하지 않고 환경에서만 사용한다.
            from core import tracing
            tracing.setup()
        payload, issues, limitations = prepare_input(state, originals)
        if not live:
            return {"mode": "input_check", "input_ready": not issues, "passed": None,
                    "judge_used": False, "report_version": payload["report_version"],
                    "report_hash": payload["report_hash"], "trace_id": state["trace_id"],
                    "issues": issues, "missing_information": limitations,
                    "next_step": "입력 점검 완료. 실제 의미 평가는 --live로 별도 실행하세요."
                                 if not issues else "보고서·원문 입력을 먼저 보완하세요."}
        # 입력 규칙 실패는 evaluator가 불합격으로 반환하며 API를 호출하지 않는다.
        return {"mode": "live", **evaluate_report(state, original_documents=originals)}
    except QualityJudgeError as exc:
        return {"mode": "live" if live else "input_check", "passed": False,
                "status": "error", "trace_id": state["trace_id"], "error": str(exc)}
    except Exception:
        return {"mode": "live" if live else "input_check", "passed": False,
                "status": "error", "trace_id": state["trace_id"],
                "error": "QA 입력 계약 또는 실행 오류; 담당별 데이터 구조를 확인하세요."}


def save_result(result: dict, name: str, *, input_paths: tuple[Path, ...] = ()) -> Path:
    """QA 전용 출력 폴더 밖 쓰기·입력 덮어쓰기·기존 결과 덮어쓰기를 막는다."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", name):
        raise ValueError("출력 이름은 경로 없는 .json 파일명이어야 합니다.")
    target = OUTPUT_DIR / name
    if OUTPUT_DIR.is_symlink() or OUTPUT_DIR.parent.is_symlink():
        raise ValueError("QA 출력 폴더를 다른 위치로 연결한 경로는 사용할 수 없습니다.")
    if target.resolve().parent != OUTPUT_DIR.resolve():
        raise ValueError("QA 전용 출력 폴더 밖에는 저장할 수 없습니다.")
    if any(target.resolve() == path.resolve() for path in input_paths):
        raise ValueError("입력 파일을 덮어쓸 수 없습니다.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 산출물에도 알려진 비밀값을 남기지 않는다.
    text = json.dumps(redact_secrets(result), ensure_ascii=False, indent=2)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5번 독립 QA 입력 점검 / 실제 Judge 실행")
    parser.add_argument("--state", type=Path, required=True, help="실제 보고서와 분석·REFERENCE State JSON")
    parser.add_argument("--originals", type=Path, required=True, help="chunk_id별 실제 원문 JSON")
    parser.add_argument("--live", action="store_true", help="기존 공통 LLM·LangSmith 설정으로 실제 Judge 실행")
    parser.add_argument("--output-name", default=None, help="outputs/qa 내부 .json 파일명; 기존 파일 덮어쓰기 금지")
    args = parser.parse_args(argv)
    name = args.output_name or f"qa_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.json"
    try:
        state, originals = load_object(args.state), load_object(args.originals)
        result = run_quality(state, originals, live=args.live)
        output = save_result(result, name, input_paths=(args.state, args.originals))
    except (OSError, ValueError):
        print("QA 입력 또는 전용 출력 경로 오류. 입력 JSON과 출력 파일명을 확인하세요.")
        return 2
    print(f"mode={result['mode']} passed={result['passed']} output={output}")
    if result.get("status") == "error":
        return 2
    if result["mode"] == "input_check":
        return 0 if result["input_ready"] else 1
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
