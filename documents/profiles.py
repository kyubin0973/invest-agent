"""company_profiles 입력 구성 (설계 CompanyProfile)

- document_scope(허용 source_ids·document_types)는 manifest에서 자동으로 채운다.
- target_market, funding_stage, eligibility, source_refs는 data/company_profiles.json에 사전 입력한다.
  평가 기준일의 자격 확인은 입력 전에 수행하며, 확인하지 못한 값은 null로 둔다 (LLM이 추정하지 않음).
"""

import json

from documents.config import COMPANIES, DATA_DIR, DOCUMENT_TYPES
from documents.loader import company_of, load_manifest

PROFILES_PATH = DATA_DIR / "company_profiles.json"


def load_company_profiles() -> dict[str, dict]:
    inputs = json.loads(PROFILES_PATH.read_text(encoding="utf-8")) if PROFILES_PATH.exists() else {}
    rows = load_manifest()

    profiles = {}
    for key, name in COMPANIES.items():
        info = inputs.get(name, {})
        source_ids = [r["id"] for r in rows if company_of(r) == key]
        eligibility = info.get("eligibility") or {}
        profiles[name] = {
            "name": name,
            "target_market": info.get("target_market"),
            "funding_stage": info.get("funding_stage"),
            "eligibility": {
                "is_private": eligibility.get("is_private"),
                "exit_completed": eligibility.get("exit_completed"),
                "evaluation_date": eligibility.get("evaluation_date"),
            },
            "document_scope": {
                "source_ids": source_ids,
                "document_types": sorted({DOCUMENT_TYPES.get(s, "unknown") for s in source_ids}),
            },
            "source_refs": info.get("source_refs") or {},
        }
    return profiles
