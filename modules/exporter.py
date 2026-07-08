"""Exportación de resultados a CSV, Excel (.xlsx) y reporte PDF."""
import io

import pandas as pd
from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)

DISCLAIMER = (
    "Nota: los datos de esta tabla son APROXIMADOS y fueron digitalizados a "
    "partir de una imagen mediante una herramienta semiautomática. No sustituyen "
    "a los datos originales de medición y pueden contener errores derivados de la "
    "resolución de la imagen, la calibración manual de los ejes y la detección de "
    "la curva."
)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(df: pd.DataFrame, calib_dict: dict, meta: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Datos"
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    ws2 = wb.create_sheet("Calibracion")
    ws2.append(["Parametro", "Valor"])
    for k, v in calib_dict.items():
        ws2.append([k, v])
    ws2.append([])
    ws2.append(["Nota", DISCLAIMER])
    for k, v in meta.items():
        ws2.append([k, v])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _rl_image(png_bytes: bytes, max_w=15 * cm):
    bio = io.BytesIO(png_bytes)
    im = PILImage.open(bio)
    w, h = im.size
    ratio = h / w if w else 1.0
    disp_w = max_w
    disp_h = disp_w * ratio
    bio.seek(0)
    return RLImage(bio, width=disp_w, height=disp_h)


def to_pdf_bytes(df: pd.DataFrame, calib_dict: dict, meta: dict,
                 crop_png: bytes = None, recon_png: bytes = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(meta.get("titulo", "Digitalización de gráfica"), styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    if meta.get("subtitulo"):
        story.append(Paragraph(meta["subtitulo"], styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

    if crop_png:
        story.append(Paragraph("Imagen / recorte de la gráfica original", styles["Heading2"]))
        story.append(_rl_image(crop_png))
        story.append(Spacer(1, 0.4 * cm))

    if recon_png:
        story.append(Paragraph("Curva reconstruida a partir de los datos", styles["Heading2"]))
        story.append(_rl_image(recon_png))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Parámetros de calibración", styles["Heading2"]))
    calib_rows = [["Parámetro", "Valor"]] + [[str(k), str(v)] for k, v in calib_dict.items()]
    t = Table(calib_rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Tabla de datos digitalizados", styles["Heading2"]))
    data_rows = [list(df.columns)] + df.round(6).astype(str).values.tolist()
    dt = Table(data_rows, repeatRows=1, hAlign="LEFT")
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f6")]),
    ]))
    story.append(dt)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph(DISCLAIMER, styles["Italic"]))

    doc.build(story)
    return buf.getvalue()
