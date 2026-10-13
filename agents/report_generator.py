"""Report Generator (설계 산출물 2.1, 5장 준수 - 전문 투자 심사 보고서)

역할: 확정된 InvestmentState를 기반으로 전문 VC 투자 보고서 양식의 PDF(5~10장) 및 Markdown 생성
입력: 확정된 InvestmentState (재작성이면 quality_result 포함)
출력: references, final_report(Markdown), report_path(PDF 파일 경로)

구조: State → build_report_model → (재작성이면) apply_quality_feedback → render_markdown / generate_pdf_report
- ReportModel은 문장(Statement) 단위다. 문장마다 id, 본문, 근거 source_id를 가진다.
- PDF와 Markdown은 같은 ReportModel만 읽는다. 품질 평가가 읽는 final_report의 문장 본문이 PDF 문장과 같다.
- Markdown의 문장은 한 줄에 하나이며 `- **라벨:** 본문 [chunk_id | source_id]` 형식이다 (품질 평가 인용 형식). PDF는 REFERENCE 번호 [n]을 쓴다.
  품질 평가의 issue.statement는 이 본문(인용 표기 제외)과 같거나 일부 문자열이면 문장을 찾을 수 있다.
- 재작성(report_attempts > 1)에서는 owner == "report_generator"인 지적의 문장을 제외한다.
  분석 Agent 담당 지적은 Supervisor가 재작업시킨 뒤 갱신된 State로 다시 조립하면 반영된다.
- State에는 새 키를 추가하지 않는다 (final_report, report_path, references만 반환).
"""
import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import CondPageBreak, HRFlowable, KeepTogether, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from core.state import InvestmentState, ReferenceItem
from evaluation.criteria import CRITERIA, INVEST_SCORE_THRESHOLD, MIN_SCORED_CRITERIA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 단일 TrueType 한글 폰트 안전 등록 (글리프 누락/공백 현상 차단)
# ---------------------------------------------------------------------------
FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

# (보통체, 굵은체) 후보. 굵은체가 없으면 보통체로 대신한다.
# 레포에 포함한 Nanum Gothic(OFL)을 가장 먼저 쓰므로 OS와 상관없이 같은 모양으로 나온다.
_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
korean_font_candidates = [
    (_FONT_DIR / "NanumGothic-Regular.ttf", _FONT_DIR / "NanumGothic-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/AppleGothic.ttf", None),
    ("/Library/Fonts/NanumGothic.ttf", "/Library/Fonts/NanumGothicBold.ttf"),
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
]


def _register_fonts() -> tuple[str, str]:
    for regular, bold in korean_font_candidates:
        if not os.path.exists(str(regular)):
            continue
        try:
            pdfmetrics.registerFont(TTFont("KoFont", str(regular)))
            bold_name = "KoFont"
            if bold and os.path.exists(str(bold)):
                pdfmetrics.registerFont(TTFont("KoFont-Bold", str(bold)))
                bold_name = "KoFont-Bold"
            # <b>, <i> 태그가 Helvetica로 빠져 한글이 깨지지 않도록 같은 계열로 묶는다 (이탤릭은 보통체로 대신)
            pdfmetrics.registerFontFamily("KoFont", normal="KoFont", bold=bold_name, italic="KoFont", boldItalic=bold_name)
            return "KoFont", bold_name
        except Exception:
            continue
    logger.warning("한글 폰트를 찾지 못해 Helvetica로 대체합니다. PDF의 한글이 깨집니다 (assets/fonts/NanumGothic-Regular.ttf 확인).")
    return "Helvetica", "Helvetica-Bold"


FONT_NAME, FONT_BOLD = _register_fonts()


# ---------------------------------------------------------------------------
# 상단 러닝 헤더 및 하단 페이지 번호 캔버스 (전체 5쪽 엄수)
# ---------------------------------------------------------------------------
class ReportTooLong(ValueError):
    """PDF가 MAX_REPORT_PAGES를 넘을 때. 압축 레이아웃으로 다시 만드는 신호로 쓴다."""


class InvestmentReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        if num_pages > MAX_REPORT_PAGES:
            raise ReportTooLong(f"보고서가 {num_pages}쪽입니다. 최대 {MAX_REPORT_PAGES}장을 넘을 수 없습니다.")
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont(FONT_NAME, 7.5)
        self.setFillColor(colors.HexColor("#6B7280"))

        # 상단 Header Line: y=825에 배치하여 본문과 완전 분리
        self.drawString(36, 825, "AI STARTUP INVESTMENT REVIEW")
        self.drawRightString(559, 825, "CONFIDENTIAL (대외비)")
        self.setStrokeColor(colors.HexColor("#D1D5DB"))
        self.setLineWidth(0.6)
        self.line(36, 819, 559, 819)

        # 하단 Footer Line
        self.line(36, 34, 559, 34)
        self.drawString(36, 23, "AI STARTUP INVESTMENT EVALUATION AGENT")
        self.drawRightString(559, 23, f"{self._pageNumber} / {total_pages}")
        self.restoreState()


# ---------------------------------------------------------------------------
# 출처 집계 및 포맷팅 (설계 5.3)
# ---------------------------------------------------------------------------
def _add_reference(refs: dict[str, ReferenceItem], ev: dict) -> None:
    s_id = ev.get("source_id")
    if not s_id:
        return
    ref = refs.setdefault(
        s_id,
        {
            "source_id": s_id,
            "title": ev.get("title", "제목 없음"),
            "publisher": ev.get("publisher", "발행기관 미상"),
            "published_date": ev.get("published_date", "YYYY"),
            "reference_metadata": ev.get("reference_metadata", {}),
            "pages": [],
        },
    )
    if ev.get("page") and ev["page"] not in ref["pages"]:
        ref["pages"].append(ev["page"])


def collect_references(state: InvestmentState) -> list[ReferenceItem]:
    refs: dict[str, ReferenceItem] = {}

    for result in (state.get("investment_results") or {}).values():
        for criterion in result.get("criteria", []):
            for ev in criterion.get("evidence", []):
                _add_reference(refs, ev)

    for res_dict in [state.get("technology_results") or {}, state.get("market_traction_results") or {}]:
        for comp_res in res_dict.values():
            for ev in comp_res.get("evidence", []):
                _add_reference(refs, ev)

    for ref in refs.values():
        ref["pages"].sort()
    return sorted(refs.values(), key=lambda r: r["source_id"])


def format_reference(ref: ReferenceItem, markup: bool = True) -> str:
    """REFERENCE 한 줄. markup=True면 PDF(Paragraph)용 태그·이스케이프, False면 Markdown용 문자열이다."""
    e = escape if markup else str
    i1, i2 = ("<i>", "</i>") if markup else ("*", "*")
    meta = ref.get("reference_metadata", {})
    pages = f" (사용 페이지: p. {', '.join(map(str, ref['pages']))})" if ref.get("pages") else ""
    url = e(meta.get("url") or "")
    publisher, title = e(ref["publisher"]), e(ref["title"])
    date = ref["published_date"]
    ref_type = meta.get("reference_type", "report")

    if ref_type == "web":
        site = e(meta.get("site_name") or ref["publisher"])
        return f"{publisher}({e(date)}). {i1}{title}{i2}. {site}, {url}{pages}"

    if ref_type == "paper":
        journal = e(meta.get("journal", "학술지 미상"))
        volume = meta.get("volume", "")
        vol_text = f", {e(volume)}" if volume else ""
        return f"{publisher}({e(date[:4])}). {title}. {i1}{journal}{i2}{vol_text}{pages}."

    year = date[:4] if len(date) >= 4 else "YYYY"
    return f"{publisher}({year}). {i1}{title}{i2}. {url}{pages}"


def _coverage(result: dict) -> str:
    n_scored = sum(c.get("status") == "SCORED" for c in result.get("criteria", []))
    cov = result.get("evidence_coverage", 0.0)
    return f"{n_scored}/12 ({cov:.0%})"


def _coverage_md(result: dict) -> str:
    """Markdown용 커버리지. 품질 평가의 수치 인용 검사가 표의 '%'를 근거 없는 수치 주장으로 오인하지 않게 비율로 적는다."""
    n_scored = sum(c.get("status") == "SCORED" for c in result.get("criteria", []))
    return f"{n_scored}/12 (비율 {result.get('evidence_coverage', 0.0):.2f})"


# ---------------------------------------------------------------------------
# 쪽 수: 섹션은 이어서 흐르고(남은 공간이 모자랄 때만 새 쪽), 합계는 MAX_REPORT_PAGES 이내. 길면 글자 → 문장 순으로 줄인다
# ---------------------------------------------------------------------------
SECTION_BREAK_MIN = 280  # 한 쪽에 남은 공간이 이보다 작을 때만 다음 섹션을 새 쪽에서 시작한다 (pt). 빈 쪽이 생기지 않게 이어서 흐른다.
MAX_REPORT_PAGES = 10  # 제출 규정: 평가 보고서는 최대 10장. 넘으면 먼저 글자 크기를 줄이고, 그래도 넘을 때만 분석 문장 수·길이를 줄인다.
FONT_SCALES = (1.0, 0.95, 0.9, 0.85, 0.8)  # 10장을 넘을 때 차례로 시도하는 글자 크기 배율 (본문 최소 약 6pt)
# 글자를 가장 줄여도 10장을 넘을 때만 다음 단계로 간다: (기업별 분석 문장 수, 문장 길이). 앞 단계일수록 내용이 많다.
CONTENT_TIERS = ((8, 320), (6, 260), (5, 200), (4, 160))
MAX_FINDINGS = CONTENT_TIERS[0][0]  # 기업별 분석 문장 수 (기본 단계)
MAX_LIST_ITEMS = 6  # 리스크·실사 항목 수
MAX_TEXT_CHARS = CONTENT_TIERS[0][1]  # 문장 하나의 최대 길이 (기본 단계)


def _short(text, limit: int = MAX_TEXT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _join(items, empty: str, limit: int = MAX_LIST_ITEMS) -> str:
    items = [_short(i, 160) for i in (items or []) if i]
    if not items:
        return empty
    more = f" 외 {len(items) - limit}건" if len(items) > limit else ""
    return "; ".join(items[:limit]) + more


# ---------------------------------------------------------------------------
# 문장 단위 보고서 모델
# ---------------------------------------------------------------------------
ANALYSIS = "analysis"  # 분석 Agent의 근거 기반 서술 → 출처(source_ids)가 있어야 한다
LIMITATION = "limitation"  # 리스크·실사 필요 사항 등 한계 서술 → 출처 불필요
JUDGEMENT = "judgement"  # Judge의 판정·강점·리스크 서술 (판정과 점수는 Python 규칙으로 계산됨)

TECH_SECTION = "technology"
MARKET_SECTION = "market"
SUMMARY_SECTION = "summary"
COMPETITION_SECTION = "competition"

_AGENT_LABELS = {
    TECH_SECTION: {
        "summary": "기술·제품 핵심 컨셉",
        "risks": "기술 리스크 및 한계점",
        "missing": "실사 필요 사항 (Due Diligence)",
        "risks_empty": "특이 리스크 없음",
        "missing_empty": "추가 실사 항목 없음",
    },
    MARKET_SECTION: {
        "summary": "시장 견인력 및 사업화 실적",
        "risks": "시장·사업화 리스크",
        "missing": "팀 구성 및 비공개 정보 한계",
        "risks_empty": "특이 리스크 없음",
        "missing_empty": "정보 완전",
    },
}


@dataclass(frozen=True)
class Statement:
    id: str  # 예: technology:Apptronik:f2
    section: str
    company: str | None
    label: str
    text: str  # 인용 표기를 뺀 문장 본문 (PDF·Markdown 동일)
    source_ids: tuple[str, ...] = ()
    kind: str = ANALYSIS
    chunk_refs: tuple[tuple[str, str], ...] = ()  # (chunk_id, source_id) 쌍: Markdown 인용 [chunk_id | source_id]용

    @property
    def role(self) -> str:
        return self.id.rsplit(":", 1)[-1]

    @property
    def protected(self) -> bool:
        """Python 규칙으로 계산한 결과(판정 사유)는 품질 평가 지적으로 지우지 않는다."""
        return self.section == SUMMARY_SECTION and self.role == "decision"


@dataclass(frozen=True)
class ReportModel:
    companies: tuple[str, ...]
    statements: tuple[Statement, ...]
    omitted: dict = field(default_factory=dict)  # statement id → 제외 사유(품질 평가 항목)

    def of(self, section: str, company: str | None = None) -> list[Statement]:
        return [
            s for s in self.statements
            if s.section == section and (company is None or s.company == company) and s.id not in self.omitted
        ]


def _source_ids(evidence_refs) -> tuple[str, ...]:
    ids: list[str] = []
    for ref in evidence_refs or []:
        sid = ref.get("source_id")
        if sid and sid not in ids:
            ids.append(sid)
    return tuple(ids)


def _chunk_refs(evidence_refs) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for ref in evidence_refs or []:
        pair = (ref.get("chunk_id"), ref.get("source_id"))
        if all(pair) and pair not in pairs:
            pairs.append(pair)
    return tuple(pairs)


def _agent_statements(section: str, company: str, result: dict, max_findings: int, max_chars: int) -> list[Statement]:
    labels = _AGENT_LABELS[section]
    findings = (result.get("findings") or [])[:max_findings]
    # 요약은 개별 finding의 종합이므로 finding들의 출처를 합쳐서 인용한다
    summary_sources = tuple(dict.fromkeys(sid for f in findings for sid in _source_ids(f.get("evidence_refs"))))
    summary_chunks = tuple(dict.fromkeys(p for f in findings for p in _chunk_refs(f.get("evidence_refs"))))
    has_summary = bool(result.get("summary"))
    out = [
        Statement(
            f"{section}:{company}:summary", section, company, labels["summary"],
            _short(result.get("summary") or "분석 데이터 없음", max_chars),
            summary_sources if has_summary else (), ANALYSIS if has_summary else LIMITATION,
            summary_chunks if has_summary else (),
        )
    ]
    for i, f in enumerate(findings):
        out.append(Statement(
            f"{section}:{company}:f{i}", section, company, str(f.get("dimension") or "분석"),
            _short(f.get("statement"), max_chars), _source_ids(f.get("evidence_refs")), ANALYSIS,
            _chunk_refs(f.get("evidence_refs")),
        ))
    out.append(Statement(
        f"{section}:{company}:risks", section, company, labels["risks"],
        _join(result.get("risks"), labels["risks_empty"]), (), LIMITATION,
    ))
    out.append(Statement(
        f"{section}:{company}:missing", section, company, labels["missing"],
        _join(result.get("missing_information"), labels["missing_empty"]), (), LIMITATION,
    ))
    return out


def build_report_model(state: InvestmentState, max_findings: int = MAX_FINDINGS, max_chars: int = MAX_TEXT_CHARS) -> ReportModel:
    """State에서 문장 단위 보고서를 조립한다. 같은 State와 상한이면 항상 같은 모델이다 (LLM 없음)."""
    companies = tuple(state.get("candidate_companies") or [])
    results = state.get("investment_results") or {}
    tech = state.get("technology_results") or {}
    market = state.get("market_traction_results") or {}
    comp = state.get("competition_result") or {}

    statements: list[Statement] = []
    for c in companies:
        r = results.get(c) or {}
        statements += [
            Statement(f"{SUMMARY_SECTION}:{c}:decision", SUMMARY_SECTION, c, "판정 사유",
                      _short(r.get("decision_reason") or "판단 사유 미기재", 400), (), JUDGEMENT),
            Statement(f"{SUMMARY_SECTION}:{c}:strengths", SUMMARY_SECTION, c, "핵심 강점",
                      _join(r.get("key_strengths"), "-", 3), (), JUDGEMENT),
            Statement(f"{SUMMARY_SECTION}:{c}:risks", SUMMARY_SECTION, c, "관리 리스크",
                      _join(r.get("key_risks"), "-", 3), (), JUDGEMENT),
        ]
    for c in companies:
        statements += _agent_statements(TECH_SECTION, c, tech.get(c) or {}, max_findings, max_chars)
    for c in companies:
        statements += _agent_statements(MARKET_SECTION, c, market.get(c) or {}, max_findings, max_chars)
    for i, item in enumerate(comp.get("comparisons") or []):
        sources = _source_ids(item.get("evidence_refs"))
        for c, text in (item.get("company_findings") or {}).items():
            if text:
                statements.append(Statement(
                    f"{COMPETITION_SECTION}:{i}:{c}", COMPETITION_SECTION, c, str(item.get("dimension") or ""),
                    _short(text, max_chars), sources, ANALYSIS, _chunk_refs(item.get("evidence_refs")),
                ))
    return ReportModel(companies, tuple(statements))


# ---------------------------------------------------------------------------
# 재작성: 품질 평가 지적 반영 (Report Generator 담당 지적만)
# ---------------------------------------------------------------------------
# 지울 인용 표기: PDF의 [3], [3, 4]와 Markdown의 [chunk_id | source_id]
_CITE_MARK = re.compile(r"\s*(?:\[\d+(?:\s*,\s*\d+)*\]|\[[A-Za-z0-9_.:-]+\s*\|\s*[A-Za-z0-9_.:-]+\])")


def _normalize(text) -> str:
    return " ".join(_CITE_MARK.sub("", str(text or "")).replace("…", "").split())


def _same_statement(statement_text: str, issue_text) -> bool:
    a, b = _normalize(statement_text), _normalize(issue_text)
    if not a or not b:
        return False
    if len(a) < 8 or len(b) < 8:
        return a == b
    return a in b or b in a


def apply_quality_feedback(model: ReportModel, quality: dict | None) -> tuple[ReportModel, list[str]]:
    """owner가 report_generator인 지적의 문장을 보고서에서 제외한다 (결정적 처리, LLM 없음).

    판정 사유와 점수표는 Python 규칙으로 계산한 결과이므로 지적을 받아도 지우지 않는다 (점수표는 문장이 아니라 항상 출력).

    반환: (수정된 모델, 문장을 찾지 못한 지적 목록). 제외한 문장은 omitted에 평가 항목과 함께 남는다.
    """
    omitted = dict(model.omitted)
    unmatched: list[str] = []
    for check in (quality or {}).get("checks", []):
        if check.get("passed"):
            continue
        for issue in check.get("issues", []):
            if issue.get("owner") != "report_generator":
                continue
            company = issue.get("company")  # 지적에 기업이 있으면 그 기업의 문장만 대상 (다른 기업의 같은 문구는 건드리지 않는다)
            hits = [s for s in model.statements
                    if _same_statement(s.text, issue.get("statement")) and (not company or s.company in (None, company))]
            if not hits:
                unmatched.append(f"[{check.get('item')}] {issue.get('statement')}")
                continue
            reason = str(issue.get("reason") or "")
            # 판정 사유는 계산 결과, 강점·리스크 같은 서술은 애초에 인용을 붙일 수 없다 → '인용 누락' 지적만으로는 지우지 않는다
            keep = [s for s in hits if s.protected or (s.kind != ANALYSIS and "인용" in reason)]
            removable = [s for s in hits if s not in keep]
            for s in keep:
                unmatched.append(f"[{check.get('item')}] 계산 결과·인용 불가 서술이라 제외하지 않음({s.id}): {reason}")
            for s in removable:
                omitted[s.id] = check.get("item", "")
                logger.info("품질 평가 지적으로 제외: %s [%s] 사유=%s / %s", s.id, check.get("item"), reason, _short(s.text, 60))
    return replace(model, omitted=omitted), unmatched


# ---------------------------------------------------------------------------
# 인용 표기
# ---------------------------------------------------------------------------
def _citations(stmt: Statement, ref_index: dict[str, int]) -> list[int]:
    return sorted({ref_index[s] for s in stmt.source_ids if s in ref_index})


def _cite_html(stmt: Statement, ref_index: dict[str, int]) -> str:
    nums = _citations(stmt, ref_index)
    return f" <font color='#64748B'>[{', '.join(map(str, nums))}]</font>" if nums else ""


def _cite_md(stmt: Statement, ref_index: dict[str, int]) -> str:
    """Markdown(final_report)은 품질 평가가 읽는 [chunk_id | source_id] 형식으로 인용한다. 번호는 PDF에서만 쓴다."""
    return "".join(f" [{chunk} | {source}]" for chunk, source in stmt.chunk_refs if source in ref_index)


def _md_reference(ref: ReferenceItem) -> str:
    return format_reference(ref, markup=False)


def _fmt(v) -> str:
    return "N/A" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------
# Markdown 보고서 (품질 평가가 읽는 final_report)
# ---------------------------------------------------------------------------
def render_markdown(model: ReportModel, state: InvestmentState, references: list[ReferenceItem]) -> str:
    ref_index = {r["source_id"]: i for i, r in enumerate(references, 1)}
    results = state.get("investment_results") or {}
    companies = list(model.companies)
    comp = state.get("competition_result") or {}

    def bullet(s: Statement) -> str:
        return f"- **{s.label}:** {s.text}{_cite_md(s, ref_index)}"

    lines = ["# AI Startup Investment Evaluation Report", ""]

    lines += ["## 1. 핵심 요약 (Executive Summary)", "",
              "| Target Company | Decision | Final Score | Coverage |", "|---|---|---|---|"]
    for c in companies:
        r = results.get(c) or {}
        lines.append(f"| {c} | {r.get('decision', 'HOLD')} | {_fmt(r.get('final_score'))} / 5.0 | {_coverage_md(r)} |")
    lines += ["", "### 1.1 판정 요약 (판정은 Python 규칙으로 계산한 결과)", ""]
    for c in companies:
        stmts = model.of(SUMMARY_SECTION, c)
        lines.append(f"**{c}**")
        lines += [bullet(s) for s in stmts] or ["- (품질 평가 지적으로 제외됨)"]
        lines.append("")
    if model.omitted:
        lines += [f"> 품질 평가 지적에 따라 서술 {len(model.omitted)}건을 제외했습니다.", ""]

    lines += ["## 2. 평가 개요", ""]
    lines += [f"- **{label}:** {text}" for label, text in _method_rows(plain=True)]
    lines.append("")

    for title, section in (("## 3. 기술·제품 분석 (Technology & Product)", TECH_SECTION), ("## 4. 시장·고객 분석 (Market & Traction)", MARKET_SECTION)):
        lines += [title, ""]
        for c in companies:
            lines.append(f"### {c}")
            lines += [bullet(s) for s in model.of(section, c)]
            lines.append("")

    lines += ["## 5. 경쟁 비교 및 종합 평가", "", "### 5.1 비교 축별 분석", ""]
    for i, item in enumerate(comp.get("comparisons") or []):
        stmts = [s for s in model.of(COMPETITION_SECTION) if s.id.startswith(f"{COMPETITION_SECTION}:{i}:")]
        if stmts:
            lines.append(f"**{item.get('dimension', '')}**")
            lines += [f"- {s.company}: {s.text}{_cite_md(s, ref_index)}" for s in stmts]
            lines.append("")

    lines += ["### 5.2 12문항 스코어카드", "",
              "| No | Evaluation Criteria | " + " | ".join(companies) + " |",
              "|---|---|" + "---|" * len(companies)]
    for crit in CRITERIA:
        row = [crit.q, crit.name]
        for c in companies:
            item = next((x for x in (results.get(c) or {}).get("criteria", []) if x.get("criterion_id") == crit.id), None)
            row.append(_fmt(item.get("score")) if item else "N/A")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| - | Final Score | " + " | ".join(_fmt((results.get(c) or {}).get("final_score")) for c in companies) + " |")
    lines.append("| - | Evidence Coverage | " + " | ".join(_coverage_md(results.get(c) or {}) for c in companies) + " |")
    lines.append("| - | Final Decision | " + " | ".join(str((results.get(c) or {}).get("decision", "N/A")) for c in companies) + " |")
    lines.append("")

    lines += ["## 부록. 참고문헌 (REFERENCE)", ""]
    lines += [f"[{i}] {_md_reference(r)}" for i, r in enumerate(references, 1)] or ["- 평가에 사용된 검증 Reference가 없습니다."]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PDF 디자인 토큰 (색·너비): 회사 보고서처럼 절제된 남색 + 회색 계열
# ---------------------------------------------------------------------------
CONTENT_W = 511  # 프레임 안쪽 너비 (A4 − 좌우 여백 72 − 프레임 패딩 12)
INK, BODY, MUTED, RULE = "#111827", "#374151", "#6B7280", "#D1D5DB"
NAVY, LIGHT = "#1F3A5F", "#F3F4F6"
GREEN, RED, AMBER, SLATE = "#166534", "#991B1B", "#92400E", "#374151"
DECISION_COLORS = {  # 판정 → (글자색, 배경색)
    "INVEST": (GREEN, "#F0FDF4"),
    "HOLD": (SLATE, "#F3F4F6"),
    "HOLD_INSUFFICIENT_EVIDENCE": (AMBER, "#FFFBEB"),
}
SCORE_BG = {5: "#DCFCE7", 3: "#F3F4F6", 1: "#FEE2E2"}  # 1·3·5점 척도
NA_BG = "#FEF3C7"


def _hex(value: str):
    return colors.HexColor(value)


def _decision_colors(decision: str) -> tuple[str, str]:
    return DECISION_COLORS.get(decision, (SLATE, "#F3F4F6"))


def _decision_label(decision) -> str:
    return str(decision or "N/A").replace("_", " ")


class _ReportDoc(SimpleDocTemplate):
    """제목(_toc_level이 있는 Paragraph)이 놓인 쪽을 목차(TableOfContents)에 알린다."""

    def afterFlowable(self, flowable):
        level = getattr(flowable, "_toc_level", None)
        if level is not None:
            self.notify("TOCEntry", (level, flowable._toc_text, self.page))


def _pdf_styles(scale: float = 1.0) -> dict[str, ParagraphStyle]:
    """scale로 모든 글자 크기·줄 간격을 함께 줄인다 (10장을 넘을 때만 1.0 미만)."""
    base = ParagraphStyle("base", fontName=FONT_NAME, fontSize=7.8 * scale, leading=11.5 * scale, textColor=_hex(BODY))

    def style(name: str, **kw) -> ParagraphStyle:
        for key in ("fontSize", "leading"):
            if key in kw:
                kw[key] = kw[key] * scale
        return ParagraphStyle(name, parent=kw.pop("parent", base), **kw)

    st = {"body": base}
    st["muted"] = style("muted", fontSize=7, leading=10, textColor=_hex(MUTED))
    st["cell"] = style("cell", fontSize=7.4, leading=10, textColor=_hex(INK))
    st["cell_c"] = style("cell_c", parent=st["cell"], alignment=1)
    st["cell_b"] = style("cell_b", parent=st["cell"], fontName=FONT_BOLD)
    st["cell_bc"] = style("cell_bc", parent=st["cell_b"], alignment=1)
    st["head"] = style("head", fontName=FONT_BOLD, fontSize=7.4, leading=10, textColor=_hex(INK))
    st["head_c"] = style("head_c", parent=st["head"], alignment=1)
    st["label"] = style("label", fontName=FONT_BOLD, fontSize=7.4, leading=10.5, textColor=_hex(INK))
    st["label_red"] = style("label_red", parent=st["label"], textColor=_hex(RED))
    st["label_gray"] = style("label_gray", parent=st["label"], textColor=_hex(MUTED))
    st["h1"] = style("h1", fontName=FONT_BOLD, fontSize=12.5, leading=16, textColor=_hex(NAVY), spaceBefore=6, spaceAfter=2)
    st["h2"] = style("h2", fontName=FONT_BOLD, fontSize=9.5, leading=13, textColor=_hex(INK), spaceBefore=8, spaceAfter=3, keepWithNext=1)
    st["kicker"] = style("kicker", fontName=FONT_BOLD, fontSize=7.4, leading=10, textColor=_hex(MUTED))
    st["title"] = style("title", fontName=FONT_BOLD, fontSize=22, leading=28, textColor=_hex(INK))
    st["subtitle"] = style("subtitle", fontSize=9, leading=13, textColor=_hex(MUTED))
    st["lead"] = style("lead", fontSize=9, leading=14.5, textColor=_hex(INK))
    st["info_label"] = style("info_label", fontName=FONT_BOLD, fontSize=7.4, leading=10.5, textColor=_hex(MUTED))
    st["info_value"] = style("info_value", fontSize=7.8, leading=11, textColor=_hex(INK))
    st["toc"] = style("toc", fontSize=8.4, leading=14, textColor=_hex(INK), leftIndent=4)
    st["ref"] = style("ref", fontSize=7.4, leading=10.6)
    return st


def _section(text: str, st: dict) -> list:
    """장 제목: 제목 줄, 남색 구분선. 목차에 등록된다. 남은 공간이 모자라면 새 쪽에서 시작한다."""
    p = Paragraph(escape(text), st["h1"])
    p._toc_level, p._toc_text = 0, text
    return [CondPageBreak(SECTION_BREAK_MIN), p, HRFlowable(width="100%", thickness=0.9, color=_hex(NAVY), spaceBefore=1, spaceAfter=6)]


def _grid(data: list, widths: list, extra: list | None = None) -> Table:
    """결재 문서형 표: 연한 회색 머리글, 위아래 남색 선, 본문은 얇은 가로선만 쓴다."""
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), _hex(LIGHT)),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, _hex(NAVY)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, _hex(MUTED)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, _hex(RULE)),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, _hex(NAVY)),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ] + (extra or [])))
    return t


def _kv_table(rows: list, label_w: int = 92) -> Table:
    """항목(왼쪽, 회색 바탕) | 내용(오른쪽) 2열 표."""
    t = Table(rows, colWidths=[label_w, CONTENT_W - label_w])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), _hex(LIGHT)),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, _hex(NAVY)),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, _hex(RULE)),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, _hex(NAVY)),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _info_table(pairs: list[tuple[str, str]], st: dict) -> Table:
    """문서 정보표: 4열(항목, 내용, 항목, 내용)."""
    rows = []
    for i in range(0, len(pairs), 2):
        chunk = pairs[i:i + 2] + [("", "")] * (2 - len(pairs[i:i + 2]))
        rows.append([cell for label, value in chunk for cell in (Paragraph(escape(label), st["info_label"]), Paragraph(escape(value), st["info_value"]))])
    t = Table(rows, colWidths=[62, 193, 62, 194])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (0, -1), _hex(LIGHT)), ("BACKGROUND", (2, 0), (2, -1), _hex(LIGHT)),
        ("GRID", (0, 0), (-1, -1), 0.4, _hex(RULE)),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, _hex(NAVY)), ("LINEBELOW", (0, -1), (-1, -1), 1.0, _hex(NAVY)),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _conclusion_text(companies: list[str], results: dict) -> str:
    """두괄식 결론 문장. 판정 개수만 말하고 기업 간 우열은 말하지 않는다 (중립성)."""
    decisions = {c: (results.get(c) or {}).get("decision", "HOLD") for c in companies}
    invest = [c for c, d in decisions.items() if d == "INVEST"]
    hold = [c for c, d in decisions.items() if d == "HOLD"]
    short = [c for c, d in decisions.items() if d == "HOLD_INSUFFICIENT_EVIDENCE"]
    text = (f"평가 대상 {len(companies)}개 기업 중 INVEST {len(invest)}개, HOLD {len(hold)}개, "
            f"근거 부족(HOLD INSUFFICIENT EVIDENCE) {len(short)}개로 판정되었습니다. ")
    text += f"INVEST 판정 기업은 {', '.join(invest)}입니다. " if invest else "INVEST 기준을 충족한 기업은 없습니다. "
    text += (f"판정은 12문항 평가 결과를 Python 규칙(평가 가능 문항 {MIN_SCORED_CRITERIA}개 이상, Final Score "
             f"{INVEST_SCORE_THRESHOLD:.1f} 이상)으로 계산한 값이며 기업 간 우열을 뜻하지 않습니다.")
    return text


def _method_rows(plain: bool = False) -> list[tuple[str, str]]:
    """평가 방법과 판정 기준. 임계값은 evaluation.criteria의 값을 그대로 쓴다 (문서와 코드가 어긋나지 않게).

    plain=True(Markdown)는 '9개'처럼 수치+단위 표기를 피한다: 품질 평가의 결정적 검사가 인용 없는 수치 주장으로 오인한다.
    """
    judgement = (
        f"평가 가능 문항 최소 {MIN_SCORED_CRITERIA}, Final Score 최소 {INVEST_SCORE_THRESHOLD:.1f}이면 INVEST, "
        f"평가 가능 문항이 최소 기준에 못 미치면 HOLD INSUFFICIENT EVIDENCE, 그 외 HOLD. LLM이 아닌 Python 규칙으로 계산합니다."
        if plain else
        f"평가 가능 문항 {MIN_SCORED_CRITERIA}개 이상이고 Final Score {INVEST_SCORE_THRESHOLD:.1f} 이상이면 INVEST, "
        f"문항 수가 부족하면 HOLD INSUFFICIENT EVIDENCE, 그 외 HOLD. LLM이 아닌 Python 규칙으로 계산합니다."
    )
    return [
        ("평가 프레임", f"Bessemer Checklist 10문항 + 휴머노이드 특화 2문항, 총 {len(CRITERIA)}문항 (문항별 가중치 없음)"),
        ("점수 척도", "1·3·5점, 근거가 부족하면 N/A. 5점은 외부 검증 근거(E3 이상)가 필요하며 N/A는 부정 근거로 보지 않습니다."),
        ("판정 기준", judgement),
        ("근거 규칙", "기업 주장은 달성 성과가 아니고, Demo는 상용 배치가 아니며, 근거 없음은 부정 근거가 아닙니다."),
    ]


# ---------------------------------------------------------------------------
# PDF 보고서 렌더러
# ---------------------------------------------------------------------------
def _render_pdf(
    state: InvestmentState,
    references: list[ReferenceItem],
    output_path: str,
    model: ReportModel,
    scale: float,
) -> str:
    """주어진 글자 배율로 PDF를 한 번 만든다. MAX_REPORT_PAGES를 넘으면 ReportTooLong."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ref_index = {r["source_id"]: i for i, r in enumerate(references, 1)}

    results = state.get("investment_results") or {}
    companies = list(model.companies)
    profiles = state.get("company_profiles") or {}
    comp = state.get("competition_result") or {}
    n = max(len(companies), 1)
    eval_date = next(
        ((p.get("eligibility") or {}).get("evaluation_date") for p in profiles.values() if (p.get("eligibility") or {}).get("evaluation_date")),
        None,
    )

    def make_story(scale: float) -> list:
        st = _pdf_styles(scale)
        esc = escape
        story: list = []

        # ---------------------------------------------------------------- 표지 영역
        toc = TableOfContents()
        toc.levelStyles = [st["toc"]]
        toc.dotsMinLevel = 0
        info = [
            ("문서명", "AI 스타트업 투자 심사 보고서"), ("보안 등급", "CONFIDENTIAL (대외비)"),
            ("평가 대상", ", ".join(companies) or "-"), ("평가 기준일", eval_date or "-"),
            ("평가 방법", f"Bessemer Checklist {len(CRITERIA)}문항"), ("작성", "AI Agent 자동 생성 (검증 출처 기반)"),
        ]
        story += [
            Paragraph("INVESTMENT REVIEW REPORT", st["kicker"]), Spacer(1, 3),
            Paragraph("투자 심사 보고서", st["title"]),
            Paragraph("AI Startup Investment Evaluation Report", st["subtitle"]), Spacer(1, 6),
            HRFlowable(width="100%", thickness=1.8, color=_hex(NAVY), spaceAfter=8),
            _info_table(info, st), Spacer(1, 10),
            Paragraph("목  차", st["h2"]), toc, Spacer(1, 6),
        ]

        # ---------------------------------------------------------------- 1. 핵심 요약
        story += _section("1. 핵심 요약 (Executive Summary)", st)
        story += [Paragraph(esc(_conclusion_text(companies, results)), st["lead"]), Spacer(1, 6)]

        story.append(Paragraph("1.1 판정 요약", st["h2"]))
        sums = [[Paragraph(h, st["head"] if i in (0, 4) else st["head_c"]) for i, h in enumerate(("기업", "판정", "Final Score", "Evidence Coverage", "판정 사유"))]]
        for c in companies:
            r = results.get(c) or {}
            fg, _bg = _decision_colors(r.get("decision", "HOLD"))
            reason = next((s.text for s in model.of(SUMMARY_SECTION, c) if s.role == "decision"), None)
            sums.append([
                Paragraph(esc(c), st["cell_b"]),
                Paragraph(f"<font color='{fg}'><b>{esc(_decision_label(r.get('decision', 'HOLD')))}</b></font>", st["cell_c"]),
                Paragraph(f"<b>{_fmt(r.get('final_score'))}</b> / 5.0", st["cell_c"]),
                Paragraph(_coverage(r), st["cell_c"]),
                Paragraph(esc(reason) if reason else "(품질 평가 지적으로 제외됨)", st["cell"]),
            ])
        story.append(_grid(sums, [92, 92, 62, 82, 183]))

        story.append(Paragraph("1.2 기업별 핵심 강점·관리 리스크", st["h2"]))
        pts = [[Paragraph("기업", st["head"]), Paragraph("핵심 강점", st["head"]), Paragraph("관리 리스크", st["head"])]]
        for c in companies:
            by_role = {s.role: s for s in model.of(SUMMARY_SECTION, c)}
            strengths, risks = by_role.get("strengths"), by_role.get("risks")
            pts.append([
                Paragraph(esc(c), st["cell_b"]),
                Paragraph(esc(strengths.text) if strengths else "-", st["cell"]),
                Paragraph(esc(risks.text) if risks else "-", st["cell"]),
            ])
        story.append(_grid(pts, [92, 209, 210], extra=[("VALIGN", (0, 0), (-1, -1), "TOP")]))
        if model.omitted:
            story += [Spacer(1, 3), Paragraph(f"※ 품질 평가 지적에 따라 서술 {len(model.omitted)}건을 제외했습니다.", st["muted"])]

        # ---------------------------------------------------------------- 2. 평가 개요
        story += _section("2. 평가 개요", st)
        story.append(Paragraph("2.1 평가 대상 기업", st["h2"]))
        snap = [[Paragraph(h, st["head"]) for h in ("기업", "Target Market Segment", "투자 단계", "비상장")]]
        for c in companies:
            p = profiles.get(c) or {}
            snap.append([
                Paragraph(esc(c), st["cell_b"]),
                Paragraph(esc(p.get("target_market") or "미입력"), st["cell"]),
                Paragraph(esc(p.get("funding_stage") or "미확인"), st["cell"]),
                Paragraph("Yes" if (p.get("eligibility") or {}).get("is_private") else "No", st["cell"]),
            ])
        story.append(_grid(snap, [110, 241, 100, 60]))
        story.append(Paragraph("2.2 평가 방법 및 판정 기준", st["h2"]))
        story.append(_kv_table([[Paragraph(esc(label), st["label"]), Paragraph(esc(text), st["body"])] for label, text in _method_rows()]))

        # ---------------------------------------------------------------- 3·4. 기업별 분석
        for no, (title, section) in enumerate((
            ("3. 기술·제품 분석 (Technology & Product)", TECH_SECTION),
            ("4. 시장·고객 분석 (Market & Traction)", MARKET_SECTION),
        ), start=3):
            story += _section(title, st)
            story.append(Paragraph("기업별 분석 결과이며, 문장 끝의 [번호]는 부록의 참고문헌 번호입니다.", st["muted"]))
            for i, c in enumerate(companies, start=1):
                rows = []
                for s in model.of(section, c):
                    label_style = st["label_red"] if s.role == "risks" else st["label_gray"] if s.role == "missing" else st["label"]
                    rows.append([Paragraph(esc(s.label), label_style), Paragraph(f"{esc(s.text)}{_cite_html(s, ref_index)}", st["body"])])
                story.append(Paragraph(f"{no}.{i} {esc(c)}", st["h2"]))
                if rows:
                    story.append(_kv_table(rows))

        # ---------------------------------------------------------------- 5. 경쟁 비교 및 종합 평가
        story += _section("5. 경쟁 비교 및 종합 평가", st)
        comp_stmts = {s.id: s for s in model.of(COMPETITION_SECTION)}
        comp_rows = [[Paragraph("비교 항목", st["head"])] + [Paragraph(esc(c), st["head"]) for c in companies]]
        for i, item in enumerate(comp.get("comparisons") or []):
            row = [Paragraph(esc(item.get("dimension", "")), st["cell_b"])]
            for c in companies:
                s = comp_stmts.get(f"{COMPETITION_SECTION}:{i}:{c}")
                row.append(Paragraph(f"{esc(s.text)}{_cite_html(s, ref_index)}" if s else "-", st["cell"]))
            comp_rows.append(row)
        if len(comp_rows) > 1:
            story.append(Paragraph("5.1 경쟁 비교 (Target Market Context & Dimension)", st["h2"]))
            story.append(_grid(comp_rows, [86] + [(CONTENT_W - 86) / n] * len(companies), extra=[("VALIGN", (0, 0), (-1, -1), "TOP")]))

        score_rows = [[Paragraph("No", st["head_c"]), Paragraph("평가 항목", st["head"])] + [Paragraph(esc(c), st["head_c"]) for c in companies]]
        extra: list = []
        for r_idx, crit in enumerate(CRITERIA, start=1):
            row = [Paragraph(crit.q, st["cell_bc"]), Paragraph(esc(crit.name), st["cell"])]
            for c_idx, c in enumerate(companies, start=2):
                item = next((x for x in (results.get(c) or {}).get("criteria", []) if x.get("criterion_id") == crit.id), None)
                score = item.get("score") if item else None
                row.append(Paragraph(_fmt(score), st["cell_bc"]))
                extra.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), _hex(SCORE_BG.get(score, NA_BG) if score is not None else NA_BG)))
            score_rows.append(row)
        first_summary = len(score_rows)
        for label, getter in (
            ("Final Score (평균)", lambda r: _fmt(r.get("final_score"))),
            ("Evidence Coverage", _coverage),
            ("최종 판정", lambda r: _decision_label(r.get("decision"))),
        ):
            row = [Paragraph("-", st["cell_c"]), Paragraph(f"<b>{label}</b>", st["cell"])]
            for c in companies:
                fg = _decision_colors((results.get(c) or {}).get("decision", "HOLD"))[0] if label == "최종 판정" else INK
                row.append(Paragraph(f"<font color='{fg}'><b>{esc(getter(results.get(c) or {}))}</b></font>", st["cell_c"]))
            score_rows.append(row)
        for c_idx, c in enumerate(companies, start=2):
            bg = _decision_colors((results.get(c) or {}).get("decision", "HOLD"))[1]
            extra.append(("BACKGROUND", (c_idx, len(score_rows) - 1), (c_idx, len(score_rows) - 1), _hex(bg)))
        extra += [("LINEABOVE", (0, first_summary), (-1, first_summary), 1.0, _hex(NAVY)), ("TOPPADDING", (0, 1), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 1), (-1, -1), 2.2)]
        # 제목과 표를 한 덩어리로: 쪽 끝에 제목과 한 줄만 남지 않게 표 전체를 다음 쪽으로 넘긴다
        story.append(KeepTogether([
            Paragraph("5.2 12문항 스코어카드 (Criteria Scorecard: B01~B10, H11~H12)", st["h2"]),
            _grid(score_rows, [34, 187] + [(CONTENT_W - 221) / n] * len(companies), extra=extra),
            Spacer(1, 3),
            Paragraph(
                f"색상: 5점 초록 · 3점 회색 · 1점 붉은색 · N/A 노란색. 판정 기준은 2.2 참조 (INVEST: 평가 가능 문항 {MIN_SCORED_CRITERIA}개 이상, "
                f"Final Score {INVEST_SCORE_THRESHOLD:.1f} 이상).",
                st["muted"],
            ),
        ]))

        # ---------------------------------------------------------------- 부록. 참고문헌
        story += _section("부록. 참고문헌 (REFERENCE)", st)
        story.append(Paragraph("보고서 작성과 투자 판단에 활용된 검증 출처 목록입니다 (기관 보고서 / 학술 논문 / 웹페이지 규격).", st["muted"]))
        story.append(Spacer(1, 4))
        if references:
            ref_rows = [[Paragraph(f"<b>[{idx}]</b>", st["ref"]), Paragraph(format_reference(ref), st["ref"])] for idx, ref in enumerate(references, 1)]
            ref_table = Table(ref_rows, colWidths=[28, CONTENT_W - 28])
            ref_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, _hex(RULE)),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(ref_table)
        else:
            story.append(Paragraph("• 평가에 사용된 검증 Reference가 없습니다.", st["body"]))
        return story

    doc = _ReportDoc(output_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=64, bottomMargin=44)
    doc.multiBuild(make_story(scale), canvasmaker=InvestmentReportCanvas)  # 목차의 쪽 번호를 위해 여러 번 조판한다
    return output_path


def generate_pdf_report(
    state: InvestmentState,
    references: list[ReferenceItem],
    output_path: str = "reports/investment_report.pdf",
    model: ReportModel | None = None,
) -> str:
    """모델을 주면 그 내용은 그대로 두고 글자 크기만 줄여 10장에 맞춘다. 모델이 없으면 fit_report로 내용 단계까지 맞춘다."""
    if model is None:
        return fit_report(state, references, output_path, lambda f, c: build_report_model(state, f, c))[1]
    last_error: ReportTooLong | None = None
    for scale in FONT_SCALES:
        try:
            path = _render_pdf(state, references, output_path, model, scale)
        except ReportTooLong as exc:
            last_error = exc
            continue
        if scale < 1.0:
            logger.info("보고서가 %d장을 넘어 글자 크기를 %d%%로 줄였습니다.", MAX_REPORT_PAGES, round(scale * 100))
        return path
    raise last_error or ReportTooLong("보고서를 만들지 못했습니다.")


def fit_report(state: InvestmentState, references: list[ReferenceItem], output_path: str, make_model) -> tuple[ReportModel, str]:
    """내용을 최대한 유지하며 MAX_REPORT_PAGES 안에 맞춘다.

    make_model(max_findings, max_chars) → ReportModel. 단계마다 글자를 줄여 보고, 가장 작게 줄여도 넘으면 다음 단계로
    분석 문장 수·길이를 줄인다. 최종 모델을 돌려주므로 Markdown은 PDF와 같은 내용이 된다.
    """
    last_error: ReportTooLong | None = None
    for max_findings, max_chars in CONTENT_TIERS:
        model = make_model(max_findings, max_chars)
        try:
            return model, generate_pdf_report(state, references, output_path, model)
        except ReportTooLong as exc:
            last_error = exc
            logger.warning("글자를 줄여도 %d장을 넘어 분석 문장을 줄입니다 (문장 %d개, %d자 이하).", MAX_REPORT_PAGES, max_findings, max_chars)
    raise last_error or ReportTooLong("보고서를 만들지 못했습니다.")


def report_generator_node(state: InvestmentState) -> dict:
    references = collect_references(state)
    trace_id = state.get("trace_id", "-")
    # Supervisor는 보고서 작성 직전에 report_attempts를 올린다 → 2 이상이면 품질 미달 후 재작성이다.
    rewrite = state.get("report_attempts", 0) > 1

    def make_model(max_findings: int, max_chars: int) -> ReportModel:
        model = build_report_model(state, max_findings, max_chars)
        if rewrite:
            model, unmatched = apply_quality_feedback(model, state.get("quality_result"))
            for note in unmatched:
                logger.warning("[%s] 품질 평가 지적 문장을 보고서에서 찾지 못함: %s", trace_id, note)
        return model

    model, pdf_path = fit_report(state, references, "reports/investment_report.pdf", make_model)
    if rewrite:
        logger.info("[%s] 재작성: 문장 %d건 제외", trace_id, len(model.omitted))

    return {
        "references": references,
        "final_report": render_markdown(model, state, references),
        "report_path": pdf_path,
    }
