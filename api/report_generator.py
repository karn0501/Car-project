"""
Phase 8: PDF Valuation Report Generator.
Auto-generates downloadable PDF valuation reports per prediction containing
predicted price/range, SHAP breakdown, comparable listings, and trend direction.
"""

import os
import io
from datetime import datetime


def generate_valuation_report(prediction_data: dict, comparable_listings: list = None,
                               trend_data: dict = None) -> bytes:
    """
    Generate a PDF valuation report for a car price prediction.

    Uses reportlab if available, otherwise generates a formatted text report.

    Args:
        prediction_data: Dict from PredictionService.predict()
        comparable_listings: List of comparable listing dicts
        trend_data: Dict from TrendService.forecast()

    Returns:
        bytes: PDF or text report content
    """
    try:
        return _generate_pdf_reportlab(prediction_data, comparable_listings, trend_data)
    except ImportError:
        return _generate_text_report(prediction_data, comparable_listings, trend_data)


def _generate_pdf_reportlab(prediction_data: dict, comparable_listings: list = None,
                             trend_data: dict = None) -> bytes:
    """Generate PDF using reportlab library."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            leftMargin=2 * cm, rightMargin=2 * cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"],
                                  fontSize=22, textColor=HexColor("#1a1a2e"))
    heading_style = ParagraphStyle("CustomH2", parent=styles["Heading2"],
                                    fontSize=14, textColor=HexColor("#16213e"),
                                    spaceAfter=8)
    body_style = styles["Normal"]

    elements = []

    # Header
    elements.append(Paragraph("Car Valuation Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    pred_id = prediction_data.get("prediction_id", "N/A")
    timestamp = prediction_data.get("timestamp", datetime.now().isoformat())
    elements.append(Paragraph(f"<b>Report ID:</b> {pred_id}", body_style))
    elements.append(Paragraph(f"<b>Generated:</b> {timestamp}", body_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Price Summary
    elements.append(Paragraph("Predicted Valuation", heading_style))
    price = prediction_data.get("predicted_price", 0)
    low = prediction_data.get("price_range_low", 0)
    high = prediction_data.get("price_range_high", 0)

    price_data = [
        ["Metric", "Value"],
        ["Predicted Price", f"INR {price:,.0f}"],
        ["Price Range (Low)", f"INR {low:,.0f}"],
        ["Price Range (High)", f"INR {high:,.0f}"],
    ]
    desc_score = prediction_data.get("description_quality_score")
    if desc_score is not None:
        price_data.append(["Description Quality", f"{desc_score:.2f} / 1.00"])

    t = Table(price_data, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f9fa"), colors.white]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3 * inch))

    # SHAP Breakdown
    breakdown = prediction_data.get("shap_breakdown", [])
    if breakdown:
        elements.append(Paragraph("Feature Impact Breakdown (SHAP)", heading_style))
        shap_data = [["Feature", "Impact (INR)", "Direction"]]
        for item in breakdown:
            impact = item.get("impact_inr", 0)
            direction = item.get("direction", "")
            symbol = "+" if impact > 0 else ""
            shap_data.append([
                item.get("feature", ""),
                f"{symbol}{impact:,.0f}",
                direction.capitalize()
            ])

        t2 = Table(shap_data, colWidths=[2.5 * inch, 2 * inch, 1.5 * inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f9fa"), colors.white]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 0.3 * inch))

    # Comparable Listings
    if comparable_listings:
        elements.append(Paragraph("Comparable Listings", heading_style))
        comp_data = [["Car", "Year", "KM", "Price (INR)", "City"]]
        for listing in comparable_listings[:5]:
            name = f"{listing.get('company_name', '')} {listing.get('model_name', '')}"
            comp_data.append([
                name,
                str(listing.get("manufacture_year", "")),
                f"{listing.get('km_driven', 0):,.0f}",
                f"{listing.get('asking_price', 0):,.0f}",
                listing.get("city", ""),
            ])

        t3 = Table(comp_data, colWidths=[2 * inch, 0.8 * inch, 1 * inch, 1.2 * inch, 1 * inch])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f9fa"), colors.white]),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t3)
        elements.append(Spacer(1, 0.3 * inch))

    # Trend Direction
    if trend_data and trend_data.get("status") == "success":
        elements.append(Paragraph("Price Trend Forecast", heading_style))
        direction = trend_data.get("trend_direction", "UNKNOWN")
        elements.append(Paragraph(f"<b>3-Month Trend:</b> {direction}", body_style))
        for fc in trend_data.get("forecasts", []):
            elements.append(Paragraph(
                f"Month {fc['month']}: INR {fc['predicted_price']:,.0f} ({fc['change_pct']:+.2f}%)",
                body_style
            ))
        elements.append(Spacer(1, 0.2 * inch))

    # Footer
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(
        "<i>This report is generated by the Used Car Price Prediction System. "
        "Predictions are estimates based on market data and ML models. "
        "Actual prices may vary.</i>",
        ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=HexColor("#888888"))
    ))

    doc.build(elements)
    return buffer.getvalue()


def _generate_text_report(prediction_data: dict, comparable_listings: list = None,
                           trend_data: dict = None) -> bytes:
    """Fallback: generate a text-based report when reportlab is not available."""
    lines = []
    lines.append("=" * 70)
    lines.append("           CAR VALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Report ID  : {prediction_data.get('prediction_id', 'N/A')}")
    lines.append(f"Generated  : {prediction_data.get('timestamp', '')}")
    lines.append("")
    lines.append(f"Predicted Price  : INR {prediction_data.get('predicted_price', 0):,.0f}")
    lines.append(f"Price Range Low  : INR {prediction_data.get('price_range_low', 0):,.0f}")
    lines.append(f"Price Range High : INR {prediction_data.get('price_range_high', 0):,.0f}")

    desc_score = prediction_data.get("description_quality_score")
    if desc_score is not None:
        lines.append(f"Description Score: {desc_score:.2f} / 1.00")

    lines.append("")
    lines.append("Feature Impact Breakdown:")
    for item in prediction_data.get("shap_breakdown", []):
        impact = item.get("impact_inr", 0)
        symbol = "+" if impact > 0 else ""
        lines.append(f"  {item.get('feature', ''):25s} {symbol}{impact:>10,.0f} INR")

    if comparable_listings:
        lines.append("")
        lines.append("Comparable Listings:")
        for listing in comparable_listings[:5]:
            name = f"{listing.get('company_name', '')} {listing.get('model_name', '')}"
            lines.append(f"  {name} ({listing.get('manufacture_year', '')}) - INR {listing.get('asking_price', 0):,.0f}")

    if trend_data and trend_data.get("status") == "success":
        lines.append("")
        lines.append(f"3-Month Trend: {trend_data.get('trend_direction', 'UNKNOWN')}")
        for fc in trend_data.get("forecasts", []):
            lines.append(f"  Month {fc['month']}: INR {fc['predicted_price']:,.0f} ({fc['change_pct']:+.2f}%)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Disclaimer: Predictions are estimates based on market data and ML models.")
    lines.append("=" * 70)

    return "\n".join(lines).encode("utf-8")
