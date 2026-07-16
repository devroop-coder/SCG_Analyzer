import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Canvas to dynamically compute and render page numbers (Page X of Y) and footer details.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        self.saveState()
        
        # Draw running header on all pages except first
        if self._pageNumber > 1:
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#666666"))
            self.drawString(54, 750, "AI-Based Seismocardiography (SCG) Analysis Report")
            self.drawRightString(612 - 54, 750, "Confidential - Medical Research")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)

        # Draw running footer on all pages
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#666666"))
        self.drawString(54, 40, "Software Version: SCG-Analyzer v1.0.0 (AI-Based Model)")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 40, page_str)
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 52, 612 - 54, 52)
        
        self.restoreState()


def create_pdf_report(dest_path, patient_info, stats, raw_signal, filtered_signal, fs, 
                      sample_beat, im_idx, ao_idx, ac_idx, importance_df=None):
    """
    Generates a beautifully designed ReportLab PDF document.
    """
    # Create the document template with 0.75in margins (54 points)
    doc = SimpleDocTemplate(
        dest_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom stylesheet styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2D3748")
    )
    
    bold_body = ParagraphStyle(
        "ReportBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )
    
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white
    )
    
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )

    story = []

    # Title section
    story.append(Paragraph("CARDIOVASCULAR SEISMOCARDIOGRAPHY ANALYSIS", title_style))
    story.append(Paragraph("<b>AI-Based Non-Invasive Cardiac Event Detection Report</b>", body_style))
    story.append(Spacer(1, 10))

    # Patient details table
    p_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style), Paragraph(patient_info.get("name", "N/A"), body_style),
            Paragraph("<b>Date of Study:</b>", body_style), Paragraph(patient_info.get("date", "N/A"), body_style)
        ],
        [
            Paragraph("<b>Patient ID:</b>", body_style), Paragraph(patient_info.get("id", "N/A"), body_style),
            Paragraph("<b>Date of Birth:</b>", body_style), Paragraph(patient_info.get("dob", "N/A"), body_style)
        ],
        [
            Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{patient_info.get('age', 'N/A')} y/o / {patient_info.get('gender', 'N/A')}", body_style),
            Paragraph("<b>Signal Source:</b>", body_style), Paragraph(patient_info.get("source", "N/A"), body_style)
        ]
    ]
    p_table = Table(p_data, colWidths=[90, 160, 90, 164])
    p_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F7FAFC")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F7FAFC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 15))

    # Physiological Metrics Summary
    story.append(Paragraph("Physiological Summary Metrics", h1_style))
    m_data = [
        [
            Paragraph("Metric Parameter", table_header),
            Paragraph("Value (Average)", table_header),
            Paragraph("Clinical Reference / Normal Range", table_header)
        ],
        [
            Paragraph("<b>Heart Rate (HR)</b>", table_cell),
            Paragraph(f"{stats.get('hr', 0.0):.1f} bpm", table_cell),
            Paragraph("60.0 - 100.0 bpm (Resting)", table_cell)
        ],
        [
            Paragraph("<b>R - AO Interval (PEP equivalent)</b>", table_cell),
            Paragraph(f"{stats.get('pep', 0.0):.1f} ms", table_cell),
            Paragraph("80.0 - 120.0 ms (Systolic Pre-Ejection)", table_cell)
        ],
        [
            Paragraph("<b>AO - AC Interval (LVET equivalent)</b>", table_cell),
            Paragraph(f"{stats.get('lvet', 0.0):.1f} ms", table_cell),
            Paragraph("250.0 - 320.0 ms (Ejection Phase)", table_cell)
        ],
        [
            Paragraph("<b>IM - AO Interval</b>", table_cell),
            Paragraph(f"{stats.get('im_ao', 0.0):.1f} ms", table_cell),
            Paragraph("30.0 - 60.0 ms (Isovolumetric Contraction)", table_cell)
        ]
    ]
    m_table = Table(m_data, colWidths=[180, 120, 204])
    m_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 15))

    # Signal Visualizations
    story.append(Paragraph("Signal Visualizations", h1_style))
    
    # Save raw vs filtered signal to temp image
    temp_dir = tempfile.gettempdir()
    from app.visualization.plots import plot_raw_vs_filtered, plot_segmented_beat_with_events, plot_feature_importance_chart
    
    fig1 = plot_raw_vs_filtered(raw_signal, filtered_signal, fs, duration=4.0)
    sig_img_path = os.path.join(temp_dir, "sig_temp.png")
    fig1.savefig(sig_img_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    
    # Save segmented beat to temp image
    fig2 = plot_segmented_beat_with_events(sample_beat, im_idx, ao_idx, ac_idx, fs)
    beat_img_path = os.path.join(temp_dir, "beat_temp.png")
    fig2.savefig(beat_img_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Add images to story
    story.append(Image(sig_img_path, width=480, height=240))
    story.append(Spacer(1, 10))
    
    # Force page break if necessary, or let it flow
    story.append(Paragraph("Annotated Single Heartbeat Pattern", h1_style))
    story.append(Image(beat_img_path, width=480, height=220))
    story.append(Spacer(1, 15))

    # Add AI Explainability Section
    if importance_df is not None:
        story.append(Paragraph("AI Model Explainability", h1_style))
        desc_text = (
            "The cardiac event timing prediction is computed using three gradient-boosted decision tree models (XGBoost Regressors) "
            "trained on morphological, frequency, and time-domain statistical features extracted from each heartbeat segment. "
            "The relative feature importances (F-score weights) indicating the top parameters used by the models to refine event indices "
            "are summarized below."
        )
        story.append(Paragraph(desc_text, body_style))
        story.append(Spacer(1, 8))
        
        # Save feature importance chart to temp image
        fig3 = plot_feature_importance_chart(importance_df, top_n=8)
        feat_img_path = os.path.join(temp_dir, "feat_temp.png")
        fig3.savefig(feat_img_path, dpi=200, bbox_inches="tight")
        plt.close(fig3)
        
        # Wrap feature importance in KeepTogether to ensure it doesn't break across pages awkwardly
        story.append(KeepTogether([
            Image(feat_img_path, width=450, height=260),
            Spacer(1, 10)
        ]))

    # Build the document
    try:
        doc.build(story, canvasmaker=NumberedCanvas)
    finally:
        # Clean up temp image files
        for p in [sig_img_path, beat_img_path]:
            if os.path.exists(p):
                os.remove(p)
        if importance_df is not None and os.path.exists(feat_img_path):
            os.remove(feat_img_path)
            
    print(f"Report generated successfully: {dest_path}")
