# AI Startup Investment Evaluation Agent
본 프로젝트는 General-purpose Humanoid Robotics 스타트업(Figure AI, Apptronik, 1X Technologies)의 투자 가능성을 기술·시장·경쟁 관점에서 평가하는 에이전트를 Supervisor 패턴 기반으로 설계/개발한 프로젝트입니다.


## Overview
- Objective : 휴머노이드 스타트업을 기술·제품 성숙도, 시장·고객 Traction, 경쟁 차별성, 제조·배치 확장성 관점에서 평가하고 근거가 연결된 투자 보고서 생성
- Pattern : Supervisor - 근거가 부족한 문항이 기업마다 다르게 나오므로, 결과를 보고 필요한 Agent만 다시 부르는 구조가 필요함. 분석 → 비교 → 평가 사이에 순서 의존성이 있어 한 번에 병렬로 나누는 방식보다 매 단계 State를 보고 다음 담당을 정하는 방식이 적합함
- 동적 처리 : 고정 순서 대신 Supervisor가 매 단계 State를 읽어 다음 Agent를 정함
  - 근거 충분성 : 평가 가능 문항이 9개 미만인 기업은 N/A 문항을 맡은 분석 Agent에만 재작업 요청 → 비교 갱신 → 해당 기업만 재평가
  - 보고서 품질 : 평가 미달 시 지적 내용에 따라 분석 Agent 재작업 또는 보고서 재작성으로 분기
  - 보고서 작성 시점은 단계 수가 아니라 근거 충분성 판단으로 결정 (실행마다 단계 수가 달라짐)


## Evaluation Targets
- Figure AI : 산업용 (자동차 제조·물류 공정), 가정용 확장 준비
- Apptronik : 산업용 (제조·물류 현장 물품 이송, 반복 작업)
- 1X Technologies : 가정용 (가사 보조·일상 지원)
- 평가 기준 : Bessemer Checklist 10문항 + 휴머노이드 특화 2문항 (Q1~Q12)


## Features
- PDF 자료 기반 정보 추출 : 기업 자료 15개 + 공통 시장 보고서 7개 (113쪽), 페이지·청크 단위 출처 추적
- Agentic RAG : 하위 질문 검색 → 근거 충분성 점검 → 부족 질문만 재검색 → 근거 대조
- Supervisor 재작업 요청 : 근거 부족 문항 또는 품질 평가 지적을 담당 Agent에 전달
- LLM이 아닌 Python 규칙으로 점수 검증·투자 판정·라우팅 수행 (INVEST / HOLD / HOLD_INSUFFICIENT_EVIDENCE)
- 확증 편향 방지 전략 :
  - 기업 주장 ≠ 달성 성과, Demo ≠ 상용 배치, 근거 없음 ≠ 부정 근거
  - 기업 자체 자료는 Evidence Level 상한(E2) 적용, 5점은 외부 검증 근거(E3 이상)가 있어야 부여
  - 시장 근거와 기업 근거를 분리 검색하고, 현재 평가 기업의 자료만 사용
  - 근거가 없는 문항은 추정하지 않고 N/A 처리, 부족 정보를 보고서에 명시
- 보고서 품질 평가 : 보고서 생성 후 Groundedness·중립성·편향 통제를 평가하고, 미달 시 재작업·재작성 Loop
- LangSmith 트레이싱 : Supervisor의 결정과 사유를 단계별로 기록


## Tech Stack
| Category | Details |
|---|---|
| Framework | LangGraph, LangChain, Python 3.11 |
| LLM/Generator | gpt-4o-mini (OpenAI API, temperature 0, seed 42) |
| LLM/Judge | gpt-4o-mini (구조화 출력 + Python 검증) |
| Retrieval | Chroma - Hit Rate@5 0.906, MRR@5 0.667 (문서 기반 질문 64개, Agent별 검색 범위 적용, 한국어 질의 자동 번역) |
| Embedding | intfloat/multilingual-e5-base |
| Report | reportlab (PDF) |
| Tracing | LangSmith |


## Agents
- Supervisor : State를 읽어 다음에 부를 Agent 하나를 결정. 근거 충분성 평가, 재작업 요청, 실패 재시도·제외, 종료 판단을 담당. LLM을 쓰지 않는 규칙 기반 판단이라 같은 State면 같은 결정을 냄
- Technology & Product Agent : 기업 기술자료를 검색해 제품·AI·자율성·신뢰성·안전 등 기술 성숙도 분석
- Market & Traction Agent : 공통 시장 보고서와 기업 자료를 나눠 검색해 시장 규모, 고객 배치·계약, 사업모델, 팀, 제조·Fleet 운영 분석
- Competition Agent : 모든 후보의 분석 결과를 비교해 Target Market 맥락, 차별성, 상대 Risk 정리
- Investment Judge Agent : 분석 근거로 12개 문항을 채점하고, Python 규칙으로 최종 점수와 투자 판정 확정
- Report Generator Agent : 확정된 결과로 SUMMARY, 기술·시장 분석, 경쟁 비교·점수표, REFERENCE를 담은 PDF 보고서 생성
- Report Quality Evaluator : 보고서의 Groundedness·중립성·편향 통제를 평가하고 미달 문장과 이유를 반환

하위 Agent는 Supervisor하고만 연결됩니다. 하위 Agent는 자기 결과만 State에 기록하고 Supervisor로 돌아오며, 재작업 여부와 대상은 Supervisor가 정합니다.

### 평가 기준 (Q1~Q12)
| 구분 | 문항 |
|---|---|
| Bessemer Checklist | Q1 Market Size · Q2 Problem/Product Fit · Q3 Willingness to Pay · Q4 Differentiation · Q5 Team · Q6 Early Customer Response · Q7 Business Model · Q8 Upside Potential · Q9 Risk · Q10 Founder Commitment |
| 휴머노이드 특화 | Q11 Technology Maturity · Q12 Manufacturing & Deployment |

- 점수 : 1·3·5점 (5점은 외부 검증 근거 E3 이상 필요), 판단 불가 시 N/A
- 판정 : 평가 가능 문항 9개 미만 → HOLD_INSUFFICIENT_EVIDENCE / 평균 3.2 이상 → INVEST / 미만 → HOLD


## State Schema
State는 "작업 페이로드"와 "제어 메타데이터"로 나눠 `core/state.py`에 정의했습니다.

- 제어 vs 페이로드 분리 : 분석·비교·평가 결과와 보고서는 페이로드, 라우팅(`next`, `current_company`, `rework_request`)·작업 추적(`pending_judge`, `rework_counts`, `report_attempts`)·종료(`step_count`, `max_steps`, `status`)는 제어 메타데이터로 구분. 제어 필드는 Supervisor만 쓰고 하위 Agent는 자기 결과 필드만 씀
- 관측성 위치 : 결정 로그(사유 포함)는 State에 쌓지 않고 외부로 분리. Supervisor가 결정마다 `{trace_id, step, next, company, reason}`을 logger와 LangSmith(`supervisor_decision`)에 기록. State에는 현재 결정(`next`)만 남김
- 지속성 비용 : 분석 결과는 기업명을 Key로 덮어쓰므로 재작업해도 늘어나지 않음. 누적되는 필드는 실패 시에만 한 건씩 추가되는 `errors`뿐. PDF 본문은 파일로 저장하고 State에는 경로(`report_path`)만 둠
- 상관 : `trace_id` 하나를 State, 로그, LangSmith 메타데이터, Checkpoint의 `thread_id`에 동일하게 사용
- 재개/복구 : 노드·기업 단위 마지막 상태(`node_status`)와 오류(`errors`, `last_error`)를 기록. 하위 Agent의 예외는 State에 남기고 Supervisor로 복귀하며, Supervisor가 재시도 1회 → 제외(빈 결과) 또는 중단을 결정. 다음 할 일은 State만으로 다시 계산되므로 Checkpoint에서 이어서 실행 가능
- 동시 처리 : Supervisor가 한 번에 한 Agent만 호출하므로 동시 쓰기가 없음. 기업별 결과와 `node_status`는 병합 Reducer(`merge_dict`), `errors`는 `operator.add`로 선언해 서로 덮어쓰지 않도록 함
- 종료 보장 : 반복 상한을 세 겹으로 관리. Agent별 재작업 1회(`rework_counts`), 보고서 작성 2회(`report_attempts`), 전체 호출 수(`step_count` ≤ `max_steps`). 상한은 보고서 작성 시점을 정하는 값이 아니라 끝나지 않는 Loop에서 빠져나오기 위한 장치이며, 재작업 요청 전에 보고서까지 마칠 단계가 남았는지 먼저 확인함


## Architecture
```mermaid
graph TD;
	__start__([START]):::first
	initialize_state(initialize_state)
	supervisor(supervisor)
	technology(technology)
	market_traction(market_traction)
	competition(competition)
	investment_judge(investment_judge)
	report_generator(report_generator)
	quality_evaluator(quality_evaluator)
	__end__([END]):::last
	__start__ --> initialize_state;
	initialize_state --> supervisor;
	supervisor -.-> technology;
	supervisor -.-> market_traction;
	supervisor -.-> competition;
	supervisor -.-> investment_judge;
	supervisor -.-> report_generator;
	supervisor -.-> quality_evaluator;
	supervisor -. FINISH .-> __end__;
	technology --> supervisor;
	market_traction --> supervisor;
	competition --> supervisor;
	investment_judge --> supervisor;
	report_generator --> supervisor;
	quality_evaluator --> supervisor;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

- 점선은 `add_conditional_edges`로 연결한 동적 분기, 실선은 작업 후 Supervisor로의 복귀
- Supervisor의 판단 순서 : 분석이 없는 기업·관점 → 근거 부족·품질 미달에 따른 재작업 → 비교(없거나 분석이 바뀐 경우) → 평가가 필요한 기업 → 보고서 → 품질 평가 → 종료
- 실행 예 (기업 3개, 16단계) : 분석 6 → 비교 1 → 평가 3 → 근거 부족 기업의 재작업 2 → 비교 1 → 재평가 1 → 보고서 1 → 품질 평가 1


## Directory Structure
```
├── data/                  # 문서 풀 (manifest.csv, 기업 자료, 시장 보고서, company_profiles.json)
├── agents/                # Agent 모듈
│   ├── supervisor.py      #   조정 계층 (판단, 근거 충분성 평가, 라우팅 Gate)
│   ├── technology.py, market_traction.py, competition.py
│   ├── investment_judge.py, report_generator.py
│   └── quality_evaluator.py
├── prompts/               # 프롬프트 템플릿 (common.py: 전체 공통 근거·인용 규칙)
├── evaluation/            # 12문항 평가 기준, 점수 검증, 투자 판정 규칙
├── rag/                   # 인덱싱, 검색, 임베딩, 검색 성능 평가
├── documents/             # PDF 로딩·정리, 메타데이터, 문서 구성 점검
├── core/                  # LangGraph 그래프, State, LLM, 트레이싱 설정
├── tests/                 # 단위 테스트
├── outputs/               # 실행 결과 (Markdown 보고서)
├── reports/               # 실행 결과 (PDF 투자 보고서)
├── app.py                 # 실행 스크립트
└── README.md
```


## Usage
[uv](https://docs.astral.sh/uv/)를 설치한 뒤, 프로젝트 폴더에 `.env` 파일을 만들고 아래 값을 입력합니다.
```
OPENAI_API_KEY=발급받은_키
LANGCHAIN_TRACING_V2=false          # LangSmith를 쓰려면 true + LANGCHAIN_API_KEY 입력
LANGCHAIN_API_KEY=
HF_TOKEN=                           # 선택: 임베딩 모델 다운로드 속도 제한 완화
```

```bash
uv sync                              # 패키지 설치
uv run python -m rag.build_index     # 벡터DB 생성 (최초 1회 또는 문서 변경 시)
uv run python app.py                 # 전체 실행 → reports/investment_report.pdf
```

- 일부 기업만 평가 : `uv run python app.py --companies "Apptronik"`
- 단위 테스트 : `uv run python -m unittest discover -s tests`
- 실행이 끝나면 판정 결과와 함께 `trace_id`, 단계 수, 재작업 내역, 품질 평가 결과를 출력합니다.


## Contributors
- 김규빈 : Supervisor 설계, LangGraph 그래프, State Schema
- 박진근 :
- 윤서진 :
- 정지우 :
- 강지훈 :
