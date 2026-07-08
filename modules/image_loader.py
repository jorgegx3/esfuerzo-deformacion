"""Carga de imágenes raster y conversión de PDF a imágenes."""
import io
from typing import List

from PIL import Image

try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except Exception:  # pragma: no cover
    HAS_PDF = False


def load_raster_image(file_bytes: bytes) -> Image.Image:
    """Carga PNG/JPG/JPEG desde bytes y devuelve una imagen PIL en RGB."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img


def pdf_to_images(file_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """Convierte cada página de un PDF a una imagen PIL en RGB.

    Requiere PyMuPDF (paquete `pymupdf`). Lanza RuntimeError si no está.
    """
    if not HAS_PDF:
        raise RuntimeError(
            "PyMuPDF (fitz) no está instalado. Instala 'pymupdf' para soportar PDF."
        )
    images: List[Image.Image] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images
