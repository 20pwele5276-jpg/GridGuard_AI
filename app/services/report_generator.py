from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def generate_pdf_report(dataframe, source_name="Uploaded data"):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    content = []

    total_readings = len(dataframe)

    if "anomaly_status" in dataframe.columns:
        flagged = dataframe[
            dataframe["anomaly_status"] == "Potential anomaly"
        ]
    else:
        flagged = dataframe.iloc[0:0]

    content.append(
        Paragraph("GridGuard AI - Maintenance Review Report", styles["Title"])
    )
    content.append(Spacer(1, 5 * mm))

    content.append(
        Paragraph(
            f"Data source: {source_name}",
            styles["Normal"],
        )
    )
    content.append(
        Paragraph(
            f"Total readings analyzed: {total_readings}",
            styles["Normal"],
        )
    )
    content.append(
        Paragraph(
            f"Readings flagged for review: {len(flagged)}",
            styles["Normal"],
        )
    )
    content.append(Spacer(1, 6 * mm))

    content.append(
        Paragraph("Readings Requiring Review", styles["Heading2"])
    )

    if flagged.empty:
        content.append(
            Paragraph(
                "No readings were flagged by the anomaly detection model.",
                styles["Normal"],
            )
        )
    else:
        columns = [
            ("timestamp", "Timestamp"),
            ("equipment_id", "Equipment"),
            ("voltage_v", "Voltage (V)"),
            ("current_a", "Current (A)"),
            ("frequency_hz", "Frequency (Hz)"),
            ("temperature_c", "Temp (C)"),
            ("anomaly_score", "Relative Score"),
        ]

        available = [
            (column, label)
            for column, label in columns
            if column in flagged.columns
        ]

        table_data = [[label for _, label in available]]

        for _, row in flagged.iterrows():
            values = []

            for column, _ in available:
                value = row[column]

                if pd.isna(value):
                    values.append("")
                else:
                    values.append(str(value)[:45])

            table_data.append(values)

        table = Table(
            table_data,
            repeatRows=1,
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23395B")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#F2F5F9")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        content.append(table)

    content.append(Spacer(1, 8 * mm))
    content.append(
        Paragraph("Recommended Next Step", styles["Heading2"])
    )
    content.append(
        Paragraph(
            "Review flagged readings alongside equipment history and "
            "manufacturer documentation. Confirm unusual measurements "
            "before deciding whether maintenance is needed.",
            styles["Normal"],
        )
    )

    content.append(Spacer(1, 5 * mm))
    content.append(
        Paragraph(
            "Disclaimer: GridGuard AI provides decision support. "
            "An anomaly score is not a probability of failure and does "
            "not establish that equipment is faulty. This report is "
            "not a substitute for inspection by a qualified electrical "
            "professional.",
            styles["Italic"],
        )
    )

    document.build(content)

    return buffer.getvalue()