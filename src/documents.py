"""Regenerate the two assignment PDFs from the actual pipeline outputs."""
from __future__ import annotations

import json
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import ARTIFACTS, ROOT

NAVY = colors.HexColor("#16283E")
TEAL = colors.HexColor("#087F8C")
MUTED = colors.HexColor("#53677F")
PALE = colors.HexColor("#EDF4F7")
LIGHT = colors.HexColor("#D8E4ED")
BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.1, leading=12.1, textColor=NAVY, spaceAfter=4)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, leading=10.3)


def paragraph(text, style=BODY):
    return Paragraph(text, style)


def footer(c, page_width, text):
    c.setStrokeColor(LIGHT)
    c.line(32, 31, page_width - 32, 31)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7)
    c.drawString(32, 19, text)
    c.drawRightString(page_width - 32, 19, "1 / 1")


def architecture(report):
    path = ROOT / "docs" / "architecture.pdf"
    width, height = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setTitle("Credit Risk Portfolio - Architecture")
    c.setAuthor("Credit Risk Portfolio Analytics")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 23)
    c.drawString(32, height-48, "Credit risk portfolio architecture")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 10)
    c.drawString(32, height-68, "Account-grain data, reproducible models and two embedded AI features")

    def box(x, y, w, h, title, text, accent=False):
        c.setFillColor(TEAL if accent else PALE)
        c.setStrokeColor(TEAL if accent else LIGHT)
        c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
        c.setFillColor(colors.white if accent else NAVY)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x+12, y+h-21, title)
        style = ParagraphStyle("node", parent=SMALL, textColor=colors.white if accent else NAVY, leading=11.8)
        p = paragraph(text, style)
        _, ph = p.wrap(w-24, h-36)
        assert ph <= h-40, f"Architecture node too tall: {title}"
        p.drawOn(c, x+12, y+h-34-ph)

    def arrow(points, label=None, label_at=None):
        c.setStrokeColor(MUTED)
        c.setFillColor(MUTED)
        c.setLineWidth(1.1)
        p = c.beginPath()
        p.moveTo(*points[0])
        for point in points[1:]:
            p.lineTo(*point)
        c.drawPath(p)
        x0, y0 = points[-2]
        x1, y1 = points[-1]
        import math
        angle = math.atan2(y1-y0, x1-x0)
        tip = c.beginPath()
        tip.moveTo(x1, y1)
        tip.lineTo(x1-6*math.cos(angle-.4), y1-6*math.sin(angle-.4))
        tip.lineTo(x1-6*math.cos(angle+.4), y1-6*math.sin(angle+.4))
        tip.close()
        c.drawPath(tip, fill=1)
        if label and label_at:
            c.setFont("Helvetica", 7.4)
            c.drawString(*label_at, label)

    box(32, 367, 160, 135, "1  Raw inputs", "<b>CSV / immutable copies</b><br/>10,000 customers<br/>14,685 loan accounts + target<br/>22,979 bureau lines<br/>176,220 monthly repayments<br/>8,635 collections events")
    box(222, 367, 175, 135, "2  Ingestion", "<b>pandas + NumPy</b><br/>Validate keys, dates and ranges.<br/>Normalize gender labels.<br/>Aggregate bureau to customer; repayment and collections to account.<br/>Quarantine invalid event dates.")
    box(427, 367, 170, 135, "3  Analysis-ready mart", "<b>One row per account</b><br/>Left joins with cardinality checks.<br/>Reconcile exposure and target.<br/>Retain missingness indicators.<br/>portfolio.csv + audit JSON<br/>Raw-file hashes identify inputs.")
    box(627, 367, 182, 135, "4  Default models", "<b>scikit-learn + joblib</b><br/>Logistic regression and histogram gradient boosting.<br/>60/20/20 customer split.<br/>Train-only preprocessing.<br/>Validation selection; held-out AUC, KS, Gini and importance.")
    arrow([(192, 435), (222, 435)])
    arrow([(397, 435), (427, 435)])
    arrow([(597, 435), (627, 435)])

    box(32, 192, 210, 120, "5  Portfolio analytics", "<b>pandas + Plotly</b><br/>Exposure, default rates and cohort comparisons across income, employment, region, age and loan type.<br/>Cross-dimensional cuts and five actions.")
    box(282, 192, 305, 120, "6  Interactive dashboard", "<b>Streamlit / Python server</b><br/>Portfolio, account, anomaly, LLM and model/data views. Shared filters and source downloads.<br/>Account scores and local reference-value sensitivities.<br/>API key stays in server-side environment/secrets.", accent=True)
    box(627, 192, 182, 120, "AI 2  Anomaly detection", "<b>Isolation Forest + rules</b><br/>Peer-relative repayment/bureau patterns; own-history deterioration.<br/>No target labels used.<br/>Flags, severity and numeric reasons.")
    arrow([(512, 367), (512, 342), (136, 342), (136, 312)], "aggregates", (167, 347))
    arrow([(242, 252), (282, 252)])
    arrow([(717, 367), (717, 330), (545, 330), (545, 312)], "scores + drivers", (632, 347))
    arrow([(597, 385), (612, 385), (612, 324), (718, 324), (718, 312)])
    arrow([(627, 252), (587, 252)], "queue", (592, 260))

    box(32, 49, 390, 106, "AI 1  Ask the Portfolio", "<b>OpenAI Responses API + Pydantic + pandas</b><br/>Question -> LLM structured plan -> allowlist validation -> local aggregation -> exact answer with filters, formula and source.<br/><b>Sent to LLM:</b> question, field/category catalogue and active filters only. No raw rows or computed figures. Unsupported questions are declined.")
    box(462, 49, 347, 106, "Human review and operational controls", "<b>SQLite append-only review log + pytest</b><br/>Analyst inspects an anomaly and saves confirm/dismiss/investigate, evidence note, timestamp and dataset ID. LLM answers expose their interpretation for review.<br/>Daily data checks; weekly drift/alert review; matured-label validation.")
    arrow([(370, 155), (370, 192)], "grounded answer", (285, 170))
    arrow([(350, 192), (350, 155)])
    arrow([(535, 192), (535, 155)], "decisions", (543, 171))
    footer(c, width, f"Synthetic snapshot only. No dated target horizon. Source version {report['data_version']}.")
    c.showPage()
    c.save()


def solution_overview(report):
    width, height = A4
    doc = SimpleDocTemplate(str(ROOT / "docs" / "solution_overview.pdf"), pagesize=A4,
        leftMargin=32, rightMargin=32, topMargin=30, bottomMargin=42,
        title="Credit Risk Portfolio - Solution Overview", author="Credit Risk Portfolio Analytics")
    title_style = ParagraphStyle("Title", parent=BODY, fontName="Helvetica-Bold", fontSize=22, leading=26, spaceAfter=5)
    sub_style = ParagraphStyle("Sub", parent=BODY, textColor=MUTED, fontSize=9.3, spaceAfter=11)
    heading = ParagraphStyle("Heading", parent=BODY, fontName="Helvetica-Bold", textColor=TEAL, fontSize=10.2, leading=13, spaceBefore=8, spaceAfter=4)
    story = [paragraph("Credit risk portfolio", title_style), paragraph("Solution overview  |  Synthetic bank data  |  Local handover", sub_style)]
    s = report["portfolio"]
    kpis = [[paragraph(f"<b>{s['accounts']:,}</b><br/>loan accounts"), paragraph(f"<b>{s['default_rate']:.2%}</b><br/>observed default rate"), paragraph(f"<b>${s['original_exposure']/1e6:.2f}m</b><br/>original exposure")]]
    t = Table(kpis, colWidths=[177, 177, 177])
    t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1),PALE), ("BOX", (0,0),(-1,-1),.5,LIGHT), ("TOPPADDING",(0,0),(-1,-1),8), ("BOTTOMPADDING",(0,0),(-1,-1),7), ("LEFTPADDING",(0,0),(-1,-1),10)]))
    story.append(t)
    story += [paragraph("How a risk analyst uses it", heading), paragraph(
        "<b>Portfolio:</b> filter loan type, region, income and employment; compare exposure, default rates, age, income/employment cuts and repayment by loan age. "
        "<b>Account:</b> inspect a score, reference-value sensitivities and repayment history. "
        "<b>Anomaly:</b> review reasons and record a decision. <b>Ask:</b> query the selected data. "
        "<b>Model &amp; data:</b> inspect holdout metrics, calibration, quality exceptions and monitoring.")]
    story += [paragraph("Embedded AI and the human step", heading), paragraph(
        "<b>LLM Q&amp;A:</b> submitting a question calls OpenAI for a restricted query plan. Python validates the plan, calculates all figures and displays the formula, source and filters. Only the question, schema/category catalogue and filters leave the server. "
        "<b>False positive:</b> wrong interpretation of a valid metric. <b>False negative:</b> unnecessary decline. The analyst checks interpretation and provenance. Missing API key disables Q&amp;A visibly."), paragraph(
        f"<b>Anomaly detection:</b> Isolation Forest flags the most unusual 3% of peer patterns; rules also flag DPD deterioration or bureau over-limit balances. {report['anomalies']['flagged']:,} accounts trigger at least one rule/model flag. "
        "<b>False positive:</b> legitimate seasonality or stale bureau data. <b>False negative:</b> hidden deterioration resembling peers. Analysts confirm, dismiss or investigate with an evidence note; SQLite retains each decision. No anomaly labels are available to estimate accuracy.")]
    story.append(paragraph("Modelling and held-out results", heading))
    rows = [["Model", "AUC", "KS", "Gini", "Brier"]]
    for name, result in report["model"]["models"].items():
        m = result["test"]
        rows.append([name, f"{m['auc']:.4f}", f"{m['ks']:.4f}", f"{m['gini']:.4f}", f"{m['brier']:.4f}"])
    table = Table(rows, colWidths=[207,81,81,81,81])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTNAME",(0,1),(-1,-1),"Helvetica"), ("FONTSIZE",(0,0),(-1,-1),9),
        ("BOTTOMPADDING",(0,0),(-1,-1),6), ("TOPPADDING",(0,0),(-1,-1),6),
        ("LINEBELOW",(0,-1),(-1,-1),.5,LIGHT)]))
    story += [table, Spacer(1,5), paragraph(
        "6,000/2,000/2,000 customers in train/validation/test; test has 2,948 loans. Bureau and repayment aggregates feed both models; imputation/encoding use training data only. "
        f"Validation AUC selects {report['model']['champion'].lower()} "
        f"(boosting {report['model']['models']['Gradient boosting']['validation']['auc']:.4f}; logistic {report['model']['models']['Logistic regression']['validation']['auc']:.4f}). "
        "Logistic regression is stronger on test; selection remains frozen. "
        "Permutation importance uses validation data. Scores are retrospective; collections, IDs and protected attributes are excluded.", SMALL)]
    story.append(paragraph("Three portfolio insights and actions", heading))
    for i in [0,2,3]:
        insight = report["insights"][i]
        story.append(paragraph(f"<b>{escape(insight['finding'])}</b> {escape(insight['so_what'])}", SMALL))
    story += [paragraph("Assumptions and production needs", heading), paragraph(
        "No default date, observation timestamp or prediction horizon is supplied. Bureau records may postdate loans; 3,875 collections events predate origination and are excluded from valid recovery totals. Original exposure is not loss/EAD. "
        "IT/data teams must provide dated snapshots and matured outcomes, agreed feature windows, secure secrets, durable review storage, access controls and independent validation. "
        "Monitor daily data quality, weekly score/feature drift and analyst outcomes, then AUC/KS/Gini and calibration on matured cohorts. Public dashboard deployed and API key configured; demo video remains the final handover step.", SMALL)]
    def page_end(c, _):
        footer(c, width, f"Source: supplied five CSVs; artifacts/report.json. Data version {report['data_version']}.")
    doc.build(story, onFirstPage=page_end, onLaterPages=page_end)


def main():
    (ROOT / "docs").mkdir(exist_ok=True)
    report = json.loads((ARTIFACTS / "report.json").read_text())
    architecture(report)
    solution_overview(report)
    from pypdf import PdfReader
    for name in ["architecture", "solution_overview"]:
        pages = len(PdfReader(ROOT / "docs" / f"{name}.pdf").pages)
        if pages != 1:
            raise RuntimeError(f"{name} must be one page; got {pages}. Adjust layout before delivery.")
        print(f"docs/{name}.pdf: {pages} page")


if __name__ == "__main__":
    main()
