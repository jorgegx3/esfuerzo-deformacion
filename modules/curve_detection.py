"""Detección de la curva principal mediante procesamiento de imágenes (OpenCV).

Estrategia para el caso base (una sola curva negra, fondo blanco, cuadrícula gris):
1. Escala de grises.
2. Umbral: se conservan los píxeles oscuros (curva). La cuadrícula gris, al ser
   más clara, queda excluida ajustando el umbral.
3. Limpieza morfológica (apertura) para quitar ruido.
4. Filtrado por área mínima de componentes conectados (quita motas y texto).
5. Extracción columna a columna: para cada columna x se toma una fila
   representativa (mediana/media). Esto funciona bien para curvas de función
   (un valor de Y por cada X), como esfuerzo-deformación.
"""
import numpy as np
import cv2


def build_mask(image_rgb, threshold=100, denoise=1, min_area=0):
    """Devuelve una máscara binaria (uint8 0/255) de los píxeles de la curva."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < threshold).astype(np.uint8) * 255

    if denoise and denoise > 0:
        k = int(denoise) * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if min_area and min_area > 0:
        mask = remove_small_components(mask, int(min_area))

    return mask


def remove_small_components(mask, min_area):
    """Elimina componentes conectados con área menor a min_area."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def extract_single_curve(mask, aggregate="median"):
    """Para cada columna con píxeles oscuros, devuelve una fila representativa.

    Retorna (cols, rows) como arrays float. Solo incluye columnas con datos.
    """
    h, w = mask.shape
    cols, rows = [], []
    for c in range(w):
        idx = np.where(mask[:, c] > 0)[0]
        if idx.size == 0:
            continue
        if aggregate == "mean":
            r = float(idx.mean())
        elif aggregate == "min":       # fila superior (Y mayor)
            r = float(idx.min())
        elif aggregate == "max":       # fila inferior (Y menor)
            r = float(idx.max())
        else:                          # median (por defecto, robusto)
            r = float(np.median(idx))
        cols.append(float(c))
        rows.append(r)
    return np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)


def overlay_points(image_rgb, cols, rows, color=(0, 200, 0), radius=1):
    """Dibuja los puntos detectados sobre una copia de la imagen."""
    out = image_rgb.copy()
    for c, r in zip(cols.astype(int), rows.astype(int)):
        cv2.circle(out, (int(c), int(r)), radius, color, -1)
    return out
