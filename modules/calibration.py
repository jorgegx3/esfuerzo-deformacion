"""Calibración de ejes: conversión de píxeles a valores reales.

El modelo asume ejes lineales. Se marcan 4 referencias de píxel:
- px_xmin / px_xmax: columnas (eje horizontal) que corresponden a x_min y x_max.
- px_ymin / px_ymax: filas (eje vertical) que corresponden a y_min y y_max.

Como en imágenes la fila crece hacia abajo, px_ymin (abajo) suele ser mayor
que px_ymax (arriba). La fórmula lineal lo maneja automáticamente.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AxisCalibration:
    px_xmin: Optional[float] = None   # columna donde x = x_min
    px_xmax: Optional[float] = None   # columna donde x = x_max
    px_ymin: Optional[float] = None   # fila donde y = y_min
    px_ymax: Optional[float] = None   # fila donde y = y_max
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def is_complete(self) -> bool:
        return None not in (self.px_xmin, self.px_xmax, self.px_ymin, self.px_ymax)

    def col_to_x(self, col):
        col = np.asarray(col, dtype=float)
        denom = (self.px_xmax - self.px_xmin)
        if denom == 0:
            raise ValueError("Las columnas de X_min y X_max coinciden; recalibra.")
        frac = (col - self.px_xmin) / denom
        return self.x_min + frac * (self.x_max - self.x_min)

    def row_to_y(self, row):
        row = np.asarray(row, dtype=float)
        denom = (self.px_ymax - self.px_ymin)
        if denom == 0:
            raise ValueError("Las filas de Y_min y Y_max coinciden; recalibra.")
        frac = (row - self.px_ymin) / denom
        return self.y_min + frac * (self.y_max - self.y_min)

    def pixels_to_data(self, cols, rows) -> Tuple[np.ndarray, np.ndarray]:
        return self.col_to_x(cols), self.row_to_y(rows)

    def as_dict(self) -> dict:
        return {
            "px_xmin (col)": self.px_xmin,
            "px_xmax (col)": self.px_xmax,
            "px_ymin (fila)": self.px_ymin,
            "px_ymax (fila)": self.px_ymax,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }
