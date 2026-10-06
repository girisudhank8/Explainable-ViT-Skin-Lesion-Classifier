import io
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image
import numpy as np


def generate_pdf_report(
    image_name: str,
    top_pred: Dict[str, Any],
    diagnostics: Dict[str, Any],
    abcde: Dict[str, Any],
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    gemini_summary: str,
    patient_meta: Dict[str, Any] = None
) -> bytes:
    """Generates an official 1-page clinical diagnostic audit report PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e293b')
    )
    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b')
    )
    section_heading = ParagraphStyle(
        'SectionH',
        parent=styles['Heading3'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )
    
    elements = []
    
    # 1. Header
    elements.append(Paragraph("<b>MedVision XAI • Clinical Diagnostic Audit Report</b>", title_style))
    elements.append(Paragraph("Explainable Vision Transformer (ViT-B/16) Skin Lesion Decision Support Framework", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=8))
    
    # 2. Patient & Scan Metadata Table
    p_meta = patient_meta or {"age": "N/A", "sex": "N/A", "localization": "General Dermoscopy"}
    meta_data = [
        [
            Paragraph(f"<b>Scan ID:</b> {image_name}", body_style),
            Paragraph(f"<b>Patient Age:</b> {p_meta.get('age', 'N/A')}", body_style),
            Paragraph(f"<b>Sex:</b> {p_meta.get('sex', 'N/A')}", body_style),
            Paragraph(f"<b>Anatomical Site:</b> {p_meta.get('localization', 'N/A')}", body_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[160, 100, 100, 180])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 10))

    # 3. Dual Scan Imagery (Original vs Attention Heatmap)
    orig_buf = io.BytesIO()
    Image.fromarray(original_rgb).save(orig_buf, format='JPEG')
    orig_buf.seek(0)
    rl_orig = RLImage(orig_buf, width=170, height=170)
    
    over_buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(over_buf, format='JPEG')
    over_buf.seek(0)
    rl_over = RLImage(over_buf, width=170, height=170)

    # Right side: Diagnostic Assessment Summary Card
    risk_color = colors.HexColor('#dc2626') if top_pred.get('risk_class') == 'danger' else colors.HexColor('#059669')
    diag_summary = [
        [Paragraph(f"<b>Top Prediction:</b>", body_style), Paragraph(f"<b>{top_pred.get('class_name')}</b>", ParagraphStyle('RiskT', parent=body_style, fontSize=11, textColor=risk_color))],
        [Paragraph("<b>Confidence Score:</b>", body_style), Paragraph(f"<b>{top_pred.get('percent')}%</b>", body_style)],
        [Paragraph("<b>Risk Stratification:</b>", body_style), Paragraph(f"{top_pred.get('risk_title')}", body_style)],
        [Paragraph("<b>Central Lesion Focus:</b>", body_style), Paragraph(f"{diagnostics.get('central_attention_ratio')}%", body_style)],
        [Paragraph("<b>Peripheral Overlap:</b>", body_style), Paragraph(f"{diagnostics.get('edge_overlap_ratio')}%", body_style)],
        [Paragraph("<b>Artifact Status:</b>", body_style), Paragraph("High Artifact Risk" if diagnostics.get('is_artifact_suspect') else "Clean Pathology Focus", body_style)]
    ]
    t_diag_card = Table(diag_summary, colWidths=[90, 100])
    t_diag_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))

    img_table = Table([[rl_orig, rl_over, t_diag_card]], colWidths=[175, 175, 190])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 8))

    # 4. Quantitative ABCDE Dermatological Criteria Table
    elements.append(Paragraph("<b>Quantitative Dermatological ABCDE Criteria</b>", section_heading))
    abcde_table_data = [
        ["Criterion", "Measured Value", "Dermatological Classification", "Risk Indication"],
        ["Asymmetry (A)", f"{abcde.get('asymmetry_score', 'N/A')}", f"{abcde.get('asymmetry_level', 'N/A')}", "Dual-axis contour deviation"],
        ["Border (B)", f"{abcde.get('border_score', 'N/A')}", f"{abcde.get('border_level', 'N/A')}", "Isoperimetric compactness"],
        ["Color (C)", f"{abcde.get('color_score', 'N/A')} σ", f"{abcde.get('color_level', 'N/A')}", "RGB/HSV channel variance"],
        ["Diameter (D)", f"{abcde.get('diameter_mm', 'N/A')} mm", f"{abcde.get('diameter_level', 'N/A')}", "> 6mm malignancy threshold"],
        ["Total TDS Score", f"{abcde.get('total_tds_score', 'N/A')}", f"{abcde.get('tds_risk', 'N/A')}", "Total Dermatoscopy Score"]
    ]
    t_abcde = Table(abcde_table_data, colWidths=[110, 90, 160, 180])
    t_abcde.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(t_abcde)
    elements.append(Spacer(1, 8))

    # 5. Gemini 2.5 Flash Clinical Synthesis & Justification
    elements.append(Paragraph("<b>Gemini 2.5 Flash AI Dermatological Synthesis & XAI Audit</b>", section_heading))
    gemini_para = Paragraph(gemini_summary, ParagraphStyle('GeminiP', parent=body_style, fontSize=8, leading=10.5))
    elements.append(gemini_para)
    elements.append(Spacer(1, 10))

    # 6. Clinician Sign-off Block (HITL Compliance)
    sign_table_data = [
        [
            Paragraph("<b>Reviewing Dermatologist:</b> ___________________________", body_style),
            Paragraph("<b>Clinical Action:</b> [  ] Approved  [  ] Biopsy Ordered  [  ] Follow-up", body_style),
            Paragraph("<b>Signature:</b> _________________", body_style)
        ]
    ]
    t_sign = Table(sign_table_data, colWidths=[200, 200, 140])
    t_sign.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_sign)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
