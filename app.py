"""Ingeniería inversa de gráficas científicas — MVP (Streamlit).

Flujo semiautomático: Cargar -> Recortar -> Calibrar -> Detectar ->
Editar/Muestrear -> Exportar. El usuario calibra los ejes y puede corregir
manualmente la detección. Los datos resultantes son APROXIMADOS.
"""
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.image_loader import load_raster_image, pdf_to_images, HAS_PDF
from modules.calibration import AxisCalibration
from modules.curve_detection import build_mask, extract_single_curve, overlay_points
from modules.sampling import resample_uniform_x, smooth
from modules.exporter import to_csv_bytes, to_xlsx_bytes, to_pdf_bytes

# --- Componentes opcionales (recorte y clic sobre imagen) ---
try:
    from streamlit_cropper import st_cropper
    HAS_CROPPER = True
except Exception:
    HAS_CROPPER = False

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_COORDS = True
except Exception:
    HAS_COORDS = False

st.set_page_config(page_title="Ingeniería inversa de gráficas", layout="wide")

MAX_DIM = 1400  # tamaño máx. de la imagen de trabajo (garantiza correspondencia de píxeles)
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ----------------------------------------------------------------------------
# Estado
# ----------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("pages", [])
    ss.setdefault("page_idx", 0)
    ss.setdefault("working_image", None)   # np.uint8 RGB
    ss.setdefault("calib", AxisCalibration())
    ss.setdefault("calib_target", "X_min")
    ss.setdefault("last_click", None)
    ss.setdefault("raw_df", None)
    ss.setdefault("final_df", None)


init_state()


def to_working(pil_img: Image.Image) -> np.ndarray:
    """Convierte a RGB y redimensiona por debajo de MAX_DIM."""
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return np.array(img)


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def array_to_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Paso 1 · Cargar
# ----------------------------------------------------------------------------
def step_upload():
    st.header("1 · Cargar archivo")
    up = st.file_uploader("Sube PNG, JPG, JPEG o PDF", type=["png", "jpg", "jpeg", "pdf"])
    if up is not None:
        data = up.getvalue()
        is_pdf = up.name.lower().endswith(".pdf")
        if is_pdf:
            if not HAS_PDF:
                st.error("Soporte PDF no disponible. Instala el paquete 'pymupdf'.")
            else:
                dpi = st.slider("DPI de conversión del PDF", 100, 300, 200, 50)
                if st.button("Convertir PDF a imágenes"):
                    st.session_state.pages = pdf_to_images(data, dpi=dpi)
                    st.session_state.page_idx = 0
                    st.success(f"{len(st.session_state.pages)} página(s) convertida(s).")
        else:
            st.session_state.pages = [load_raster_image(data)]
            st.session_state.page_idx = 0

    pages = st.session_state.pages
    if pages:
        if len(pages) > 1:
            st.session_state.page_idx = st.number_input(
                "Página a usar", 0, len(pages) - 1, st.session_state.page_idx)
        st.image(pages[st.session_state.page_idx],
                 caption="Imagen cargada", use_column_width=True)
    else:
        st.info("Aún no has cargado ninguna imagen.")


# ----------------------------------------------------------------------------
# Paso 2 · Recortar
# ----------------------------------------------------------------------------
def step_crop():
    st.header("2 · Recortar el área de la gráfica")
    pages = st.session_state.pages
    if not pages:
        st.info("Primero carga un archivo en el paso 1.")
        return

    img = pages[st.session_state.page_idx]
    st.caption("Deja solo el área interior de los ejes. Excluye título, leyendas "
               "y etiquetas externas para reducir ruido en la detección.")

    if HAS_CROPPER:
        cropped = st_cropper(img, realtime_update=True,
                             box_color="#22aa22", aspect_ratio=None)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(cropped, caption="Vista previa del recorte", use_column_width=True)
        with c2:
            if st.button("✅ Usar este recorte", use_container_width=True):
                st.session_state.working_image = to_working(cropped)
                st.success("Recorte guardado como imagen de trabajo.")
    else:
        st.warning("Componente de recorte no instalado (streamlit-cropper). "
                   "Puedes usar la imagen completa.")

    if st.button("Usar imagen completa (sin recortar)"):
        st.session_state.working_image = to_working(img)
        st.success("Imagen completa guardada como imagen de trabajo.")

    if st.session_state.working_image is not None:
        st.divider()
        st.image(st.session_state.working_image,
                 caption="Imagen de trabajo actual", use_column_width=True)


# ----------------------------------------------------------------------------
# Paso 3 · Calibrar
# ----------------------------------------------------------------------------
def assign_calib_click(calib, target, x, y):
    if st.session_state.last_click == (x, y):
        return
    st.session_state.last_click = (x, y)
    if target == "X_min":
        calib.px_xmin = x
    elif target == "X_max":
        calib.px_xmax = x
    elif target == "Y_min":
        calib.px_ymin = y
    elif target == "Y_max":
        calib.px_ymax = y


def step_calibrate():
    st.header("3 · Calibrar ejes")
    wi = st.session_state.working_image
    if wi is None:
        st.info("Primero define la imagen de trabajo en el paso 2.")
        return

    calib = st.session_state.calib
    st.subheader("Valores reales de los ejes")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        calib.x_min = st.number_input("X mínimo", value=float(calib.x_min), format="%.6f")
    with c2:
        calib.x_max = st.number_input("X máximo", value=float(calib.x_max), format="%.6f")
    with c3:
        calib.y_min = st.number_input("Y mínimo", value=float(calib.y_min), format="%.6f")
    with c4:
        calib.y_max = st.number_input("Y máximo", value=float(calib.y_max), format="%.6f")
    st.caption("Ejemplo esfuerzo–deformación: X de 0 a 0.003 cm/cm, Y de 0 a 240 kg/cm².")

    st.subheader("Marca los 4 puntos de referencia sobre la imagen")
    st.session_state.calib_target = st.radio(
        "Punto a marcar (haz clic en la imagen)",
        ["X_min", "X_max", "Y_min", "Y_max"], horizontal=True)

    if HAS_COORDS:
        val = streamlit_image_coordinates(Image.fromarray(wi), key="calib_img")
        if val is not None:
            assign_calib_click(calib, st.session_state.calib_target,
                               float(val["x"]), float(val["y"]))
    else:
        st.warning("Componente de clic no instalado (streamlit-image-coordinates). "
                   "Ingresa los píxeles manualmente:")
        cc1, cc2, cc3, cc4 = st.columns(4)
        calib.px_xmin = cc1.number_input("px X_min (col)", value=calib.px_xmin or 0.0)
        calib.px_xmax = cc2.number_input("px X_max (col)", value=calib.px_xmax or float(wi.shape[1]))
        calib.px_ymin = cc3.number_input("px Y_min (fila)", value=calib.px_ymin or float(wi.shape[0]))
        calib.px_ymax = cc4.number_input("px Y_max (fila)", value=calib.px_ymax or 0.0)

    st.subheader("Referencias registradas")
    refs = pd.DataFrame({
        "Referencia": ["X_min", "X_max", "Y_min", "Y_max"],
        "Píxel": [calib.px_xmin, calib.px_xmax, calib.px_ymin, calib.px_ymax],
        "Valor real": [calib.x_min, calib.x_max, calib.y_min, calib.y_max],
    })
    st.dataframe(refs, use_container_width=True, hide_index=True)
    st.success("Calibración completa ✅") if calib.is_complete() else \
        st.info("Faltan puntos por marcar.")


# ----------------------------------------------------------------------------
# Paso 4 · Detectar
# ----------------------------------------------------------------------------
def step_detect():
    st.header("4 · Detectar la curva")
    wi = st.session_state.working_image
    calib = st.session_state.calib
    if wi is None:
        st.info("Primero define la imagen de trabajo.")
        return

    st.sidebar.subheader("Parámetros de detección")
    threshold = st.sidebar.slider("Umbral (menor = solo lo más oscuro)", 0, 255, 100)
    denoise = st.sidebar.slider("Limpieza de ruido (apertura)", 0, 5, 1)
    min_area = st.sidebar.slider("Área mínima de componente (quita motas/texto)", 0, 500, 30, 10)
    agg_label = st.sidebar.selectbox(
        "Agregación por columna", ["median", "mean", "min (arriba)", "max (abajo)"])
    agg_map = {"median": "median", "mean": "mean",
               "min (arriba)": "min", "max (abajo)": "max"}

    mask = build_mask(wi, threshold=threshold, denoise=denoise, min_area=min_area)
    cols, rows = extract_single_curve(mask, aggregate=agg_map[agg_label])

    c1, c2 = st.columns(2)
    with c1:
        st.image(mask, caption="Máscara binaria (curva detectada)", use_column_width=True)
    with c2:
        if cols.size:
            st.image(overlay_points(wi, cols, rows),
                     caption="Puntos detectados sobre la imagen", use_column_width=True)
        else:
            st.error("No se detectaron píxeles. Sube el umbral o revisa el recorte.")

    if not calib.is_complete():
        st.warning("Completa la calibración (paso 3) para convertir a datos reales.")
        return

    if cols.size:
        x, y = calib.pixels_to_data(cols, rows)
        df = pd.DataFrame({"X": x, "Y": y}).sort_values("X").reset_index(drop=True)
        st.session_state.raw_df = df
        st.success(f"{len(df)} puntos convertidos a valores reales.")
        st.line_chart(df.set_index("X"))


# ----------------------------------------------------------------------------
# Paso 5 · Editar y muestrear
# ----------------------------------------------------------------------------
def build_comparison_fig(raw, final):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(raw["X"], raw["Y"], ".", ms=2, alpha=0.35,
            color="#888", label="Detección cruda")
    ax.plot(final["X"], final["Y"], "-o", ms=3, lw=1.4,
            color="#c0392b", label="Curva reconstruida")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def step_edit():
    st.header("5 · Editar y muestrear")
    raw = st.session_state.raw_df
    if raw is None:
        st.info("Primero detecta la curva y conviértela a datos reales (paso 4).")
        return

    c1, c2 = st.columns(2)
    n_points = c1.slider("Número de puntos a extraer", 5, 200, 30)
    do_smooth = c2.checkbox("Suavizar (Savitzky-Golay)")

    xs, ys = resample_uniform_x(raw["X"].values, raw["Y"].values, n_points)
    df = pd.DataFrame({"X": xs, "Y": ys})
    if do_smooth:
        win = st.slider("Ventana de suavizado", 3, 51, 7, 2)
        df["Y"] = smooth(df["Y"].values, win, poly=2)

    st.caption("Edita valores, agrega o elimina filas manualmente:")
    edited = st.data_editor(df, num_rows="dynamic",
                            use_container_width=True, key="editor")
    final = edited.dropna().sort_values("X").reset_index(drop=True)
    st.session_state.final_df = final

    st.subheader("Comparación: detección vs. reconstrucción")
    st.pyplot(build_comparison_fig(raw, final))


# ----------------------------------------------------------------------------
# Paso 6 · Exportar
# ----------------------------------------------------------------------------
def build_reconstructed_fig(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["X"], df["Y"], "-o", ms=3, lw=1.5, color="#c0392b")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Curva reconstruida")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def step_export():
    st.header("6 · Exportar")
    df = st.session_state.final_df
    calib = st.session_state.calib
    if df is None or df.empty:
        st.info("No hay datos finales para exportar (completa el paso 5).")
        return

    titulo = st.text_input("Título del reporte", "Digitalización de gráfica esfuerzo–deformación")
    meta = {"titulo": titulo, "subtitulo": "Datos aproximados obtenidos por digitalización semiautomática."}
    calib_dict = calib.as_dict()

    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ CSV", to_csv_bytes(df), "datos.csv", "text/csv",
                       use_container_width=True)
    c2.download_button("⬇️ Excel (.xlsx)", to_xlsx_bytes(df, calib_dict, meta),
                       "datos.xlsx", XLSX_MIME, use_container_width=True)

    crop_png = array_to_png(st.session_state.working_image) \
        if st.session_state.working_image is not None else None
    recon_png = fig_to_png(build_reconstructed_fig(df))
    pdf_bytes = to_pdf_bytes(df, calib_dict, meta, crop_png, recon_png)
    c3.download_button("⬇️ Reporte PDF", pdf_bytes, "reporte.pdf",
                       "application/pdf", use_container_width=True)


# ----------------------------------------------------------------------------
# Navegación
# ----------------------------------------------------------------------------
STEPS = {
    "1 · Cargar": step_upload,
    "2 · Recortar": step_crop,
    "3 · Calibrar": step_calibrate,
    "4 · Detectar": step_detect,
    "5 · Editar y muestrear": step_edit,
    "6 · Exportar": step_export,
}


def main():
    st.sidebar.title("🔬 Digitalizador de gráficas")
    st.sidebar.caption("Herramienta semiautomática · datos aproximados")
    choice = st.sidebar.radio("Paso del flujo", list(STEPS.keys()))
    st.sidebar.divider()
    STEPS[choice]()
    st.sidebar.divider()
    st.sidebar.caption("MVP · una curva, fondo blanco, ejes lineales.")


if __name__ == "__main__":
    main()
