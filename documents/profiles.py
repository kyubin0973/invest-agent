"""company_profiles 입력 구성

기업 기본정보와 문서 범위(manifest의 source_id)는 자동으로 채우고,
팀·창업자·투자 정보는 data/company_profiles.json에서 읽는다 (출처가 있는 사실만 입력).
"""

import json

from documents.config import COMPANIES, DATA_DIR
from documents.loader import company_of, load_manifest

PROFILES_PATH = DATA_DIR / "company_profiles.json"


def load_company_profiles() -> dict[str, dict]:
    extra = json.loads(PROFILES_PATH.read_text(encoding="utf-8")) if PROFILES_PATH.exists() else {}
    rows = load_manifest()

    profiles = {}
    for key, name in COMPANIES.items():
        info = extra.get(name, {})
        profiles[name] = {
            "name": name,
            "company_key": key,
            "document_ids": [r["id"] for r in rows if company_of(r) == key],
            "team": info.get("team", []),
            "funding": info.get("funding", []),
        }
    return profiles
