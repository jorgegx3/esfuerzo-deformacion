"""Muestreo uniforme y suavizado de la curva extraída."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def resample_uniform_x(x, y, n_points):
    """Reinterpola (x, y) sobre una malla uniforme de n_points en X.

    Ordena por X, elimina duplicados y aplica interpolación lineal.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    if x_unique.size < 2:
        return x_unique, y_unique

    xs = np.linspace(x_unique.min(), x_unique.max(), int(n_points))
    f = interp1d(x_unique, y_unique, kind="linear", fill_value="extrapolate")
    ys = f(xs)
    return xs, ys


def smooth(y, window, poly=2):
    """Suavizado Savitzky-Golay con validación de parámetros."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window <= poly or window < 3:
        return y
    return savgol_filter(y, window, poly)
