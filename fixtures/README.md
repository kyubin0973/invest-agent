# fixtures (개발용, 제출 전 삭제)

앞 단계 Agent의 결과를 저장한 테스트 데이터입니다. 앞 단계를 다시 실행하지 않고 다음 단계 Agent를 개발할 때 씁니다.
실제 실행(`app.py`)은 이 파일을 읽지 않고 LangGraph State로 결과를 전달합니다.

| 폴더 | 내용 | State 키 |
|---|---|---|
| `market_traction/{figure,apptronik,1x}.json` | Market & Traction Agent 결과 (`AnalysisResult`) | `market_traction_results[기업명]` |

## 사용 예 (Judge·Competition·Report 담당)

```python
import json
from documents.company import company_key

def load_market(company: str) -> dict:
    return json.load(open(f"fixtures/market_traction/{company_key(company)}.json", encoding="utf-8"))

state["market_traction_results"] = {c: load_market(c) for c in ["Figure AI", "Apptronik", "1X Technologies"]}
```

파일명은 기업 키입니다: Figure AI → `figure.json`, Apptronik → `apptronik.json`, 1X Technologies → `1x.json`

## 다시 만들기

```bash
uv run python -m scripts.save_fixture market_traction
```
