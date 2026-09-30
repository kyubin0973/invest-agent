"""Report Generator (설계 산출물 2.1, 5장 준수 - 전문 투자 심사 보고서)

역할: 확정된 InvestmentState를 기반으로 전문 VC 투자 보고서 양식의 5쪽 PDF 및 Markdown 생성
입력: 확정된 InvestmentState
출력: references, final_report(Markdown), report_path(PDF 파일 경로)
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# macOS 단일 TrueType 한글 폰트 안전 등록 (글리프 누락/공백 현상 완전 차단)
# ---------------------------------------------------------------------------
FONT_NAME = "Helvetica"

korean_font_candidates = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
]

for font_path in korean_font_candidates:
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("KoFont", font_path))
            FONT_NAME = "KoFont"
            break
        except Exception:
            continue

from core.state import InvestmentState, ReferenceItem
from evaluation.criteria import CRITERIA


# ---------------------------------------------------------------------------
# 상단 러닝 헤더 및 하단 페이지 번호 캔버스 (전체 5쪽 엄수)
# ---------------------------------------------------------------------------
class InvestmentReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont(FONT_NAME, 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # 상단 Header Line: y=825에 배치하여 본문과 완전 분리
        self.drawString(36, 825, "PHYSICAL AI & HUMANOID ROBOTICS INVESTMENT REVIEW")
        self.drawRightString(559, 825, "CONFIDENTIAL / 심사평가 산출물")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 819, 559, 819)

        # 하단 Footer Line
        self.line(36, 34, 559, 34)
        self.drawString(36, 23, "AI STARTUP INVESTMENT EVALUATION AGENT")
        self.drawRightString(559, 23, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


# ---------------------------------------------------------------------------
# 출처 집계 및 포맷팅 (설계 5.3)
# ---------------------------------------------------------------------------
def collect_references(state: InvestmentState) -> list[ReferenceItem]:
    refs: dict[str, ReferenceItem] = {}
    
    for result in state.get("investment_results", {}).values():
        for criterion in result.get("criteria", []):
            for ev in criterion.get("evidence", []):
                s_id = ev.get("source_id")
                if not s_id:
                    continue
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

    for res_dict in [state.get("technology_results", {}), state.get("market_traction_results", {})]:
        for comp_res in res_dict.values():
            for ev in comp_res.get("evidence", []):
                s_id = ev.get("source_id")
                if not s_id:
                    continue
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

    for ref in refs.values():
        ref["pages"].sort()
    return sorted(refs.values(), key=lambda r: r["source_id"])


def format_reference(ref: ReferenceItem) -> str:
    meta = ref.get("reference_metadata", {})
    pages = f" (사용 페이지: p. {', '.join(map(str, ref['pages']))})" if ref.get("pages") else ""
    url = meta.get("url") or ""
    ref_type = meta.get("reference_type", "report")

    if ref_type == "web":
        site = meta.get("site_name") or ref["publisher"]
        return f"{ref['publisher']}({ref['published_date']}). <i>{ref['title']}</i>. {site}, {url}{pages}"
    
    if ref_type == "paper":
        journal = meta.get("journal", "학술지 미상")
        volume = meta.get("volume", "")
        vol_text = f", {volume}" if volume else ""
        return f"{ref['publisher']}({ref['published_date'][:4]}). {ref['title']}. <i>{journal}</i>{vol_text}{pages}."
    
    year = ref["published_date"][:4] if len(ref.get("published_date", "")) >= 4 else "YYYY"
    return f"{ref['publisher']}({year}). <i>{ref['title']}</i>. {url}{pages}"


def _coverage(result: dict) -> str:
    n_scored = sum(c.get("status") == "SCORED" for c in result.get("criteria", []))
    cov = result.get("evidence_coverage", 0.0)
    return f"{n_scored}/12 ({cov:.0%})"


def _render_markdown(state: InvestmentState, references: list[ReferenceItem]) -> str:
    return "# Markdown Report Generated"


# ---------------------------------------------------------------------------
# 5쪽 PDF 보고서 렌더러
# ---------------------------------------------------------------------------
def generate_pdf_report(state: InvestmentState, references: list[ReferenceItem], output_path: str = "reports/investment_report.pdf") -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=64,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=5
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=3,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=7.8,
        leading=11.5,
        textColor=colors.HexColor("#334155")
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=6,
        spaceAfter=1.5
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#1E293B")
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=table_cell,
        fontName=FONT_NAME,
        textColor=colors.HexColor("#0F172A")
    )

    results = state.get("investment_results", {})
    companies = state.get("candidate_companies", [])
    profiles = state.get("company_profiles", {})
    tech = state.get("technology_results", {})
    market = state.get("market_traction_results", {})
    comp = state.get("competition_result", {})
    fmt = lambda v: "N/A" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)

    story = []

    # =========================================================================
    # PAGE 1: SUMMARY & Company Snapshot
    # =========================================================================
    story.append(Paragraph("1. SUMMARY & Investment Verdict", title_style))
    story.append(Paragraph("전체 투자 심사 종합 결론 (Investment Synthesis)", h2_style))
    
    summary_data = [[
        Paragraph("Target Company", table_header),
        Paragraph("Decision", table_header),
        Paragraph("Final Score", table_header),
        Paragraph("Coverage (검증비율)", table_header)
    ]]
    for c in companies:
        r = results.get(c, {})
        dec = r.get("decision", "HOLD")
        dec_color = "#047857" if dec == "INVEST" else "#B91C1C" if "INSUFFICIENT" in dec else "#4B5563"
        dec_p = Paragraph(f"<font color='{dec_color}'><b>{dec}</b></font>", table_cell)
        summary_data.append([
            Paragraph(f"<b>{c}</b>", table_cell),
            dec_p,
            Paragraph(f"<b>{fmt(r.get('final_score'))}</b> / 5.0", table_cell),
            Paragraph(_coverage(r), table_cell)
        ])

    t_sum = Table(summary_data, colWidths=[130, 130, 110, 153])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_sum)
    story.append(Spacer(1, 5))

    story.append(Paragraph("핵심 투자 근거 및 의사결정 사유 (Decision Thesis)", h2_style))
    for c in companies:
        r = results.get(c, {})
        d_reason = r.get("decision_reason", "판단 사유 미기재")
        strengths = ", ".join(r.get("key_strengths", []))
        risks = ", ".join(r.get("key_risks", []))
        block = (
            f"<b>• {c}</b> : {d_reason}<br/>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;<font color='#1E40AF'><b>[핵심 강점]</b></font> {strengths}<br/>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;<font color='#991B1B'><b>[관리 리스크]</b></font> {risks}"
        )
        story.append(Paragraph(block, bullet_style))
        story.append(Spacer(1, 1.5))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Company Snapshot (대상 기업 프로필 개요)", h2_style))
    snap_data = [[
        Paragraph("Company", table_header),
        Paragraph("Target Market Segment", table_header),
        Paragraph("Stage", table_header),
        Paragraph("Private", table_header)
    ]]
    for c in companies:
        p = profiles.get(c, {})
        snap_data.append([
            Paragraph(f"<b>{c}</b>", table_cell),
            Paragraph(p.get("target_market", "미입력"), table_cell),
            Paragraph(p.get("funding_stage", "미확인"), table_cell),
            Paragraph("Yes" if p.get("eligibility", {}).get("is_private") else "No", table_cell)
        ])
    t_snap = Table(snap_data, colWidths=[120, 233, 85, 85])
    t_snap.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_snap)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Technology & Product
    # =========================================================================
    story.append(Paragraph("2. Technology & Product (기술 및 제품 분석)", title_style))
    for c in companies:
        t = tech.get(c, {})
        content_p = []
        content_p.append(Paragraph(f"<b><font size='9' color='#1E3A8A'>{c}</font></b>", body_style))
        content_p.append(Paragraph(f"• <b>기술·제품 핵심 컨셉:</b> {t.get('summary', '분석 데이터 없음')}", bullet_style))
        
        for f in t.get("findings", []):
            content_p.append(Paragraph(f"• <b>{f.get('dimension')}:</b> {f.get('statement')}", bullet_style))
            
        r_str = ", ".join(t.get("risks", [])) or "특이 리스크 없음"
        m_str = ", ".join(t.get("missing_information", [])) or "추가 실사 항목 없음"
        content_p.append(Paragraph(f"• <font color='#991B1B'><b>기술 리스크 및 한계점:</b></font> {r_str}", bullet_style))
        content_p.append(Paragraph(f"• <font color='#4B5563'><b>실사 필요 사항 (Due Diligence):</b></font> {m_str}", bullet_style))

        card_t = Table([[content_p]], colWidths=[523])
        card_t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0,0), (-1,-1), 4.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ]))
        story.append(card_t)
        story.append(Spacer(1, 4.5))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Market & Traction
    # =========================================================================
    story.append(Paragraph("3. Market & Traction (시장 규모, 고객 및 사업모델)", title_style))
    for c in companies:
        m = market.get(c, {})
        content_p = []
        content_p.append(Paragraph(f"<b><font size='9' color='#1E3A8A'>{c}</font></b>", body_style))
        content_p.append(Paragraph(f"• <b>시장 견인력 및 사업화 실적:</b> {m.get('summary', '분석 데이터 없음')}", bullet_style))
        
        for f in m.get("findings", []):
            content_p.append(Paragraph(f"• <b>{f.get('dimension')}:</b> {f.get('statement')}", bullet_style))
            
        r_str = ", ".join(m.get("risks", [])) or "특이 리스크 없음"
        m_str = ", ".join(m.get("missing_information", [])) or "정보 완전"
        content_p.append(Paragraph(f"• <font color='#991B1B'><b>시장·사업화 리스크:</b></font> {r_str}", bullet_style))
        content_p.append(Paragraph(f"• <font color='#4B5563'><b>팀 구성 및 비공개 정보 한계:</b></font> {m_str}", bullet_style))

        card_t = Table([[content_p]], colWidths=[523])
        card_t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0,0), (-1,-1), 4.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ]))
        story.append(card_t)
        story.append(Spacer(1, 4.5))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Competition & Evaluation
    # =========================================================================
    story.append(Paragraph("4. Competition & Investment Evaluation", title_style))
    story.append(Paragraph("5대 비교 축 경쟁 분석 (Target Market Context & Dimension)", h2_style))
    
    comp_table_data = [[
        Paragraph("Dimension", table_header),
        Paragraph("Figure AI", table_header),
        Paragraph("Apptronik", table_header),
        Paragraph("1X Technologies", table_header)
    ]]
    for item in comp.get("comparisons", []):
        dim = item.get("dimension", "")
        f_map = item.get("company_findings", {})
        comp_table_data.append([
            Paragraph(f"<b>{dim}</b>", table_cell),
            Paragraph(f_map.get("Figure AI", "-"), table_cell),
            Paragraph(f_map.get("Apptronik", "-"), table_cell),
            Paragraph(f_map.get("1X Technologies", "-"), table_cell),
        ])

    if len(comp_table_data) > 1:
        t_comp = Table(comp_table_data, colWidths=[95, 142, 143, 143])
        t_comp.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t_comp)
        story.append(Spacer(1, 5))

    story.append(Paragraph("12문항 스코어카드 (Criteria Scorecard: B01~B10, H11~H12)", h2_style))
    crit_table_data = [[
        Paragraph("No", table_header),
        Paragraph("Evaluation Criteria (12문항)", table_header),
        Paragraph("Figure AI", table_header),
        Paragraph("Apptronik", table_header),
        Paragraph("1X Technologies", table_header)
    ]]
    for crit in CRITERIA:
        row = [Paragraph(f"<b>{crit.q}</b>", table_cell), Paragraph(crit.name, table_cell)]
        for c in companies:
            c_item = next((x for x in results.get(c, {}).get("criteria", []) if x.get("criterion_id") == crit.id), None)
            score_val = fmt(c_item.get("score")) if c_item else "N/A"
            row.append(Paragraph(f"<b>{score_val}</b>", table_cell))
        crit_table_data.append(row)
    
    crit_table_data.append([
        Paragraph("<b>-</b>", table_cell), Paragraph("<b>Final Score (평균)</b>", table_cell)
    ] + [Paragraph(f"<b>{fmt(results.get(c, {}).get('final_score'))}</b>", table_cell) for c in companies])

    crit_table_data.append([
        Paragraph("<b>-</b>", table_cell), Paragraph("<b>Evidence Coverage</b>", table_cell)
    ] + [Paragraph(_coverage(results.get(c, {})), table_cell) for c in companies])

    crit_table_data.append([
        Paragraph("<b>-</b>", table_cell), Paragraph("<b>Final Decision</b>", table_cell)
    ] + [Paragraph(f"<b>{results.get(c, {}).get('decision', 'N/A')}</b>", table_cell) for c in companies])

    # No 열 너비를 36으로 넓혀 Q10~Q12 줄바꿈 방지
    t_crit = Table(crit_table_data, colWidths=[36, 197, 96, 97, 97])
    t_crit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('LINEBELOW', (0,-4), (-1,-4), 1.2, colors.HexColor("#334155")),
        ('BACKGROUND', (0,-3), (-1,-1), colors.HexColor("#F8FAFC")),
    ]))
    story.append(t_crit)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: REFERENCE
    # =========================================================================
    story.append(Paragraph("5. REFERENCE (실제 인용 참고자료 목록)", title_style))
    story.append(Paragraph("보고서 작성 및 투자 판단에 실제로 활용된 검증 출처 목록입니다. (기관 보고서 / 학술 논문 / 웹페이지 규격 준수)", body_style))
    story.append(Spacer(1, 8))

    if references:
        for idx, ref in enumerate(references, 1):
            ref_str = format_reference(ref)
            ref_card = Table([[Paragraph(f"<b>[{idx}]</b> {ref_str}", body_style)]], colWidths=[523])
            ref_card.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(ref_card)
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("• 평가에 사용된 검증 Reference가 없습니다.", body_style))

    doc.build(story, canvasmaker=InvestmentReportCanvas)
    return output_path


def report_generator_node(state: InvestmentState) -> dict:
    references = collect_references(state)
    markdown_report = _render_markdown(state, references)
    pdf_path = generate_pdf_report(state, references)
    
    return {
        "references": references,
        "final_report": markdown_report,
        "report_path": pdf_path,
    }