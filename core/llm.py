"""공통 LLM 설정: 모든 Agent가 같은 모델과 설정을 쓴다.

    from core.llm import get_llm, get_structured_llm

    llm = get_llm()                                   # 일반 텍스트 응답
    judge = get_structured_llm(CriterionScoreModel)   # Pydantic 모델로 구조화 출력
"""

from functools import lru_cache
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from core import tracing

MODEL = "gpt-4o-mini"
MODEL_PROVIDER = "openai"
TEMPERATURE = 0  # 같은 입력이면 최대한 같은 결과 (재현성)
SEED = 42  # OpenAI 호출의 best-effort 재현성. 모델 백엔드 변경까지 동일 출력을 보장하지는 않는다.


class LLMSettings(TypedDict):
    model_provider: str
    model: str
    temperature: int | float
    seed: int


def get_llm_settings() -> LLMSettings:
    """실행 결과와 Trace에 남길 비밀정보 없는 공통 LLM 설정."""
    return {
        "model_provider": MODEL_PROVIDER,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
    }


@lru_cache
def get_llm() -> BaseChatModel:
    tracing.setup()
    return init_chat_model(
        MODEL,
        model_provider=MODEL_PROVIDER,
        temperature=TEMPERATURE,
        seed=SEED,
    )


def get_structured_llm(schema: type[BaseModel]):
    """Pydantic 모델 형식으로 응답을 받는 LLM."""
    return get_llm().with_structured_output(schema)
