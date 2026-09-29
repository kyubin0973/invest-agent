"""환경 변수 로드 + LangSmith 트레이싱 설정

교수님 예제의 `logging.langsmith("프로젝트명")`과 같은 역할이다.
.env의 프로젝트명과 관계없이 실행 기록을 팀 프로젝트(U_2_4)에 남긴다.
"""

import os

from dotenv import find_dotenv, load_dotenv

LANGSMITH_PROJECT = "U_2_4"


def setup(project: str = LANGSMITH_PROJECT) -> None:
    load_dotenv(find_dotenv(usecwd=True), override=True)
    # 구버전(LANGCHAIN_*)과 신버전(LANGSMITH_*) 변수명을 모두 맞춘다
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_PROJECT"] = project
