# AI Startup Investment Evaluation Agent

General-purpose Humanoid Robotics 스타트업(Figure AI, Apptronik, 1X Technologies)의 투자 가능성을
LangGraph 기반 Multi-Agent + Agentic RAG로 평가하고 투자 보고서를 생성하는 프로젝트입니다.

> **현재 상태: 뼈대 완성, Agent 개발 전**
> 그래프, State, RAG(인덱싱·검색), 평가 기준, 판정 규칙은 동작합니다.
> Agent 5개는 더미 값을 돌려주는 stub이라 `app.py`를 실행하면 흐름만 끝까지 돌고 결과는 의미가 없습니다.

---

## 1. 처음 실행하기

**준비물:** [uv](https://docs.astral.sh/uv/), 인터넷, OpenAI API 키

```bash
git clone <저장소 주소>
cd invest-agent
uv sync                              # 패키지 설치 (Python 3.11 자동 설치)
cp .env.example .env                 # .env에 OPENAI_API_KEY 입력
uv run python -m rag.build_index     # 벡터DB 생성 (최초 1회, 임베딩 모델 약 1GB 다운로드)
uv run python app.py                 # 실행 → outputs/report_*.md
```

- LangSmith 키가 없으면 `.env`에서 `LANGCHAIN_TRACING_V2=false`로 바꾸세요.
- `unauthenticated requests to the HF Hub` 경고는 무시해도 됩니다.
- 실행 기록은 LangSmith **`U_2_4`** 프로젝트에 남습니다 (같은 워크스페이스 키를 써야 한 곳에 모임).

---

## 2. 폴더 구조

```
invest-agent/
├── app.py                  # 실행 스크립트
├── core/
│   ├── graph.py            # LangGraph 워크플로 (설계서 7장)
│   ├── state.py            # State 스키마 (설계서 6장)
│   ├── llm.py              # 공통 LLM: gpt-4o-mini, temperature 0
│   └── tracing.py          # .env 로드 + LangSmith 프로젝트 U_2_4
├── agents/                 # Agent 5개 ← 여기를 개발
│   ├── technology.py
│   ├── market_traction.py
│   ├── competition.py
│   ├── investment_judge.py
│   └── report_generator.py
├── evaluation/criteria.py  # 12문항, Rubric, Python 검증, INVEST/HOLD 판정 규칙
├── prompts/
│   └── common.py           # 모든 Agent 공통: 근거 안전 규칙, 인용 규칙, 근거 표기 형식
├── rag/
│   ├── build_index.py      # 인덱싱 (PDF → 청크 → 임베딩 → Chroma)
│   ├── retriever.py        # Agent용 검색 함수
│   ├── embeddings.py       # multilingual-e5-base
│   ├── config.py           # 청크 450 토큰, overlap 60, Top-K 5
│   ├── evaluate.py         # 검색 성능 측정 (Hit Rate, MRR), 임베딩 비교
│   └── eval/               # 질문셋과 측정 결과
├── documents/
│   ├── config.py           # 기업 목록, 문서 분류, 제외 페이지, 근거 체크리스트
│   ├── loader.py           # PDF 로딩·정리·메타데이터
│   ├── profiles.py         # company_profiles 구성
│   ├── company.py          # (옵션) 기업 문서 전체 읽기
│   └── check_corpus.py     # 문서 구성 점검
└── data/
    ├── manifest.csv        # 문서 목록과 서지정보 (여기 있는 PDF만 사용)
    ├── company_profiles.json  # Target Market·투자 단계·자격 사전 입력란 (현재 비어 있음)
    ├── technology/{figure,apptronik,1x}/   # 기업 자료 15개
    └── market/                              # 공통 시장 자료 7개
```

`vectorstore/`, `outputs/`, `.env`는 Git에 올리지 않습니다.

---

## 3. 그래프 흐름

```
initialize_state
  → [candidate_selection → technology ∥ market_traction → analysis_join → analysis_router] × 기업 수
  → competition (1회)
  → evaluation_init → [investment_judge → evaluation_router] × 기업 수
  → final_decision_router (INVEST_FOUND / NO_INVEST)
  → report_generator → END
```

- 기업 하나 안에서는 Technology와 Market을 **병렬 실행**합니다.
- 첫 INVEST가 나와도 멈추지 않고 **모든 후보를 끝까지 평가**합니다.

---

## 4. Agent 개발 가이드

### 담당 파일

| Agent | 파일 | 입력 → 출력 |
|---|---|---|
| Technology & Product | `agents/technology.py` | 현재 기업 기술자료 검색 → `technology_results[기업]` |
| Market & Traction | `agents/market_traction.py` | 시장자료 + 기업 자료 검색 → `market_traction_results[기업]` |
| Competition | `agents/competition.py` | 두 분석 결과 비교 (재검색 없음) → `competition_result` |
| Investment Judge | `agents/investment_judge.py` | 분석 결과의 Evidence로 12개 항목 채점 → `investment_results[기업]` |
| Report Generator | `agents/report_generator.py` | 확정된 State로 5페이지 보고서 → `final_report` |

각 파일 맨 위 docstring에 입력, 출력, 써야 할 함수, 담당 평가 항목이 적혀 있습니다.
**stub 함수 안쪽만 바꾸면** 그래프는 그대로 동작합니다.

### 꼭 지킬 규칙

1. **LLM은 `core.llm`만 사용** (`ChatOpenAI`를 직접 만들지 않기)
   ```python
   from core.llm import get_llm, get_structured_llm
   ```
2. **자기 파일만 수정.** `core/state.py`, `evaluation/criteria.py` 같은 공통 파일은 팀에 말하고 수정
3. **출력은 `core/state.py`의 형식 그대로** (`AnalysisResult`, `InvestmentResult` 등)
4. **검색 질의는 한국어·영어 모두 가능.** 한국어 질의는 검색 함수가 자동으로 영어로 번역합니다 (한국어 질의 Hit@5 0.719 → 0.938)
5. **LLM 결과 검증과 판정은 Python**: Judge는 `validate_criteria()`로 Evidence ID·점수를 검증한 뒤 `decide()`로 판정 (LLM이 평균·Decision을 계산하지 않음)
6. **프롬프트는 `prompts/`에 작성하고 공통 규칙은 `prompts/common.py`에서 가져오기** (아래 "프롬프트 작성" 참고)

### 검색 함수 (`rag/retriever.py`)

```python
from rag.retriever import search_technology, search_market, search_industry, to_evidence

docs = search_technology("battery safety certification", "Figure AI")   # 현재 기업 기술자료만
groups = search_market("humanoid market size 2035", "Figure AI")        # {"market": [...], "company": [...]}
docs = search_industry("manipulation capability level", document_type="capability_report")
evidence = [to_evidence(d) for d in docs]                               # → EvidenceItem
```

- 결과마다 `d.metadata["chunk_id"]`(예: `F4_P02_C01`), `page`, `title`, `score`, `search_query`(실제 검색어)가 들어 있습니다.
- 한국어 질의는 `gpt-4o-mini`로 번역한 뒤 검색합니다 (질의당 약 120 토큰, 1초, 같은 질의는 캐싱). 끄려면 `translate=False`.
- **유사도 점수로 관련성을 판단하지 마세요.** e5 모델은 관련 없는 문장도 0.7 이상이 나옵니다. 관련성은 LLM으로 평가합니다.

### 프롬프트 작성 (`prompts/common.py`)

```python
from prompts.common import EVIDENCE_RULES, CITATION_RULES, build_system_prompt, build_user_prompt, format_documents

system = build_system_prompt("휴머노이드 스타트업의 기술·제품 역량을 분석하는 애널리스트")
user = build_user_prompt(format_documents(docs), "실제 고객 현장 배치 수준을 평가하라.", EVIDENCE_RULES, CITATION_RULES)
get_llm().invoke([("system", system), ("user", user)])
```

- **규칙은 system이 아니라 user 메시지의 요청 바로 앞에** 넣습니다. gpt-4o-mini는 system에만 둔 규칙을 잘 지키지 않아 기업 자체 발표를 확인된 사실처럼 썼고, 요청 앞에 두니 기업 주장과 외부 확인을 구분했습니다.
- Competition·Judge·Report는 `NO_NEW_EVIDENCE_RULE`, Competition·Judge는 `TARGET_MARKET_RULE`도 함께 넣습니다.
- 검색 결과는 `format_documents(docs)`, EvidenceItem은 `format_evidence_items(items)`로 넣으면 LLM이 `chunk_id`로 인용합니다.
- 프롬프트만으로는 규칙이 완전히 지켜지지 않으니, Evidence Level·점수는 구조화 출력 + Python 검증을 함께 씁니다.

### 참고할 교수님 노트북 (`langgraph-v1/20-RAG`)

| 기능 | 노트북 |
|---|---|
| 검색 결과 관련성 평가 (yes/no) | `13-AgenticRAG` |
| 재검색 횟수 제한 | `02-RelevanceCheck` |
| 질의 재작성 | `04-QueryRewrite` |
| 검색 결과를 프롬프트에 넣는 형식 (`format_docs`) | `rag/utils.py` |

### 테스트

```bash
uv run python -m rag.retriever "factory deployment" --scope technology --company apptronik   # 검색 확인
uv run python app.py --companies "Apptronik"                                                 # 기업 1곳만 실행 (토큰 절약)
```

---

## 5. 평가 기준 요약

- 평가 문항: Bessemer 10개(B01~B10)를 휴머노이드용으로 구체화 + **기술 성숙도(H11), 제조·배치 확장성(H12)** → Q1~Q12 (`evaluation/criteria.py`)
- 점수: 1~5점 척도 중 **1·3·5점만 사용** (2·4점은 경계가 모호해 쓰지 않음), 판단 불가 시 **N/A** (0점이나 1점으로 계산하지 않음)
  - **5점**: 외부 검증 근거(E3 이상)가 있음 / **3점**: 기업 자료 중심이거나 긍정·부정 혼재 / **1점**: 실제 근거에서 실질적 약점 확인
  - 근거가 없으면 1점이 아니라 N/A. 2·4점이 나오면 검증 단계에서 '형식 오류'로 기록 (근거 부족과 구분)
- 판정: 평가 가능 문항 **9개 미만 → HOLD_INSUFFICIENT_EVIDENCE**, 9개 이상이면서 **평균 3.5 이상 → INVEST**, 미만 → **HOLD**
- Evidence Level: E0(없음) · E1(기업 주장) · E2(데모) · E3(외부 확인) · E4(실제 운영) · E5(반복·규모화)
  - 기업 자료(E1~E2)만 있으면 최대 3점, 외부 검증(E3) 이상이 있어야 5점
  - 1·3·5 척도에서 평균 3.5 이상이려면 5점이 충분해야 함 (예: 12문항 중 5점 3개 + 3점 9개 = 3.5 → INVEST)

---

## 6. 문서와 RAG

- 문서 22개: 기업 자료 15개(기업별 5개) + 공통 시장자료 7개, 사용 113쪽 / 한도 200쪽
- 잡음 페이지 제외: OECD 보고서는 필요한 13쪽만, Goldman 연락처·면책 조항, IFR 발행정보, DeepMind 저자 명단
- 문서를 추가하려면: PDF를 폴더에 넣고 `data/manifest.csv`에 한 줄 추가 → `uv run python -m rag.build_index`

```bash
uv run python -m documents.check_corpus   # 페이지 예산, 기업별 근거 종류 점검
uv run python -m rag.evaluate run         # 검색 성능 재측정
```

---

## 7. 결정해야 할 것 / 남은 작업

**결정 사항**
- [ ] `data/company_profiles.json`에 Target Market·투자 단계·자격(비상장, Exit 여부, 평가 기준일)과 근거 출처 입력
- [ ] Team·Founder(Q5, Q10)는 공개 근거가 없으면 N/A → 다른 문항이 2개 더 N/A면 HOLD_INSUFFICIENT_EVIDENCE가 됨을 팀이 인지
- [ ] 기업 자료 처리 방식: RAG 유지 / 전체 읽기(`documents/company.py`)
- [ ] Evidence Level 매기는 기준 합의

**개발**
- [ ] Agent 5개 구현
- [ ] 보고서 5페이지 구성 + PDF 출력
- [ ] 세 기업 전체 실행 → 품질 점검

**제출 (DAY 3, 15시)**
- [ ] 투자 보고서 PDF
- [ ] 제출용 README (아래 Tech Stack 포함)
- [ ] 새로 clone해서 실행 테스트

---

## Tech Stack

- Framework : LangGraph
- LLM/Generator : gpt-4o-mini
- LLM/Judge : gpt-4o-mini
- Retrieval : Chroma - Hit Rate@5 0.906, MRR@5 0.667 (문서 기반 질문 64개, Agent별 검색 범위 적용, 한국어 질의 자동 번역)
- Embedding : intfloat/multilingual-e5-base (gte-multilingual-base와 비교 후 선정, `rag/eval/embedding_comparison.json`)
