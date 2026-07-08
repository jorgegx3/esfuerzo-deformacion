"""Ingeniería inversa de gráficas — VERSIÓN AMIGABLE (asistente paso a paso).

Usa exactamente los mismos módulos de procesamiento que app.py; solo cambia la
experiencia de usuario: asistente con barra de progreso, botones Atrás/Siguiente
con validación, y calibración con los 4 puntos dibujados en vivo sobre la imagen.

Ejecutar:  streamlit run app_amigable.py
"""
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.image_loader import load_raster_image, pdf_to_images, HAS_PDF
from modules.calibration import AxisCalibration
from modules.curve_detection import build_mask, extract_single_curve, overlay_points
from modules.sampling import resample_uniform_x, smooth
from modules.exporter import to_csv_bytes, to_xlsx_bytes, to_pdf_bytes

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

st.set_page_config(page_title="Digitalizador de gráficas", layout="wide",
                   page_icon="🔬")

MAX_DIM = 1400
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

STEPS = ["Cargar", "Recortar", "Calibrar", "Detectar", "Editar", "Exportar"]
STEP_META = [
    ("Sube tu gráfica", "Elige una imagen (PNG/JPG) o un PDF con la curva a digitalizar."),
    ("Recorta el área", "Deja solo el interior de los ejes: fuera título, leyendas y números."),
    ("Calibra los ejes", "Dile a la app cuánto valen los ejes y marca sus extremos en la imagen."),
    ("Detecta la curva", "Ajusta los controles hasta que solo se vea la curva, limpia."),
    ("Revisa y ajusta", "Elige cuántos puntos quieres y corrige a mano lo que haga falta."),
    ("Descarga tus datos", "Exporta a CSV, Excel o un reporte PDF documentado."),
]

# ---------------------------------------------------------------- CSS
CSS = """
<style>
.main .block-container {padding-top: 1.2rem;}
.stepper{display:flex;align-items:flex-start;justify-content:space-between;margin:.2rem 0 1.2rem 0;}
.step{display:flex;flex-direction:column;align-items:center;flex:0 0 60px;}
.circle{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:15px;color:#fff;background:#c9d2dc;
  transition:all .2s;}
.step.active .circle{background:#C0392B;box-shadow:0 0 0 5px rgba(192,57,43,.16);}
.step.done .circle{background:#2E8B57;}
.step .label{font-size:11.5px;margin-top:7px;color:#5b6572;text-align:center;line-height:1.1;}
.step.active .label{color:#C0392B;font-weight:700;}
.step.done .label{color:#2E8B57;}
.bar{flex:1 1 auto;height:3px;background:#dfe4ea;margin:18px -6px 0 -6px;border-radius:2px;}
.bar.done{background:#2E8B57;}
.hero{background:linear-gradient(90deg,#1F3A5F 0%,#2c5282 100%);color:#fff;
  padding:16px 22px;border-radius:12px;margin-bottom:16px;}
.hero h2{margin:0;font-size:22px;color:#fff;}
.hero p{margin:4px 0 0 0;font-size:14px;opacity:.9;}
.tip{background:#EEF6EE;border-left:5px solid #2E8B57;padding:10px 14px;border-radius:6px;
  font-size:13.5px;margin:6px 0 12px 0;}
.warn{background:#FBEEE9;border-left:5px solid #C0392B;padding:10px 14px;border-radius:6px;
  font-size:13.5px;margin:6px 0 12px 0;}
div.stButton>button{border-radius:8px;font-weight:600;}
</style>
"""


# ---------------------------------------------------------------- estado
def ss():
    return st.session_state


def init_state():
    s = st.session_state
    s.setdefault("step_idx", 0)
    s.setdefault("pages", [])
    s.setdefault("page_idx", 0)
    s.setdefault("working_image", None)
    s.setdefault("calib", AxisCalibration())
    s.setdefault("calib_points", {})     # nombre -> (x, y) solo para dibujar
    s.setdefault("calib_target", "X_min")
    s.setdefault("last_click", None)
    s.setdefault("raw_df", None)
    s.setdefault("final_df", None)


init_state()


# ---------------------------------------------------------------- utilidades
def to_working(pil_img):
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return np.array(img)


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def array_to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def step_complete(i):
    s = ss()
    if i == 0:
        return len(s.pages) > 0
    if i == 1:
        return s.working_image is not None
    if i == 2:
        return s.calib.is_complete()
    if i == 3:
        return s.raw_df is not None
    if i == 4:
        return s.final_df is not None and not s.final_df.empty
    return True


HINTS = [
    "Sube un archivo para continuar.",
    "Confirma un recorte (o usa la imagen completa) para continuar.",
    "Marca los 4 puntos de calibración para continuar.",
    "Ajusta la detección hasta convertir la curva a datos reales.",
    "Genera al menos un punto en la tabla para continuar.",
    "",
]


def draw_calib_overlay(img_rgb, points):
    out = img_rgb.copy()
    h, w = out.shape[:2]
    col = {"X_min": (30, 90, 220), "X_max": (30, 90, 220),
           "Y_min": (235, 140, 20), "Y_max": (235, 140, 20)}
    for name, (x, y) in points.items():
        x, y = int(x), int(y)
        c = col[name]
        if name.startswith("X"):
            cv2.line(out, (x, 0), (x, h), c, 1, cv2.LINE_AA)
        else:
            cv2.line(out, (0, y), (w, y), c, 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 6, c, -1)
        cv2.circle(out, (x, y), 6, (255, 255, 255), 1)
        cv2.putText(out, name, (x + 9, max(16, y - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
        cv2.putText(out, name, (x + 9, max(16, y - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_stepper(current):
    html = '<div class="stepper">'
    for i, name in enumerate(STEPS):
        cls = "active" if i == current else ("done" if step_complete(i) else "todo")
        mark = "✓" if (step_complete(i) and i != current) else str(i + 1)
        html += f'<div class="step {cls}"><div class="circle">{mark}</div>' \
                f'<div class="label">{name}</div></div>'
        if i < len(STEPS) - 1:
            bar_done = " done" if step_complete(i) else ""
            html += f'<div class="bar{bar_done}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def hero(i):
    title, sub = STEP_META[i]
    st.markdown(f'<div class="hero"><h2>Paso {i+1} · {title}</h2>'
                f'<p>{sub}</p></div>', unsafe_allow_html=True)


def tip(text):
    st.info(text, icon="💡")


def warn(text):
    st.warning(text, icon="⚠️")


# =============================================================== PASOS
def step_upload():
    up = st.file_uploader("Arrastra o elige tu archivo (PNG, JPG, JPEG o PDF)",
                          type=["png", "jpg", "jpeg", "pdf"])
    if up is not None:
        data = up.getvalue()
        if up.name.lower().endswith(".pdf"):
            if not HAS_PDF:
                warn("Soporte PDF no disponible. Instala el paquete 'pymupdf'.")
            else:
                dpi = st.slider("Calidad de conversión (DPI)", 100, 300, 200, 50)
                with st.expander("❓ ¿Qué es el DPI?"):
                    st.markdown(
                        "**DPI** (puntos por pulgada) es la resolución con que el PDF "
                        "se convierte en imagen. Más alto (250–300) da una imagen más "
                        "nítida y mejor detección, pero pesa más. **200 es un buen "
                        "equilibrio** para la mayoría de los casos.")
                if st.button("Convertir PDF a imágenes", type="primary"):
                    ss().pages = pdf_to_images(data, dpi=dpi)
                    ss().page_idx = 0
                    st.success(f"{len(ss().pages)} página(s) lista(s).")
        else:
            ss().pages = [load_raster_image(data)]
            ss().page_idx = 0

    pages = ss().pages
    if pages:
        if len(pages) > 1:
            ss().page_idx = st.number_input("¿Qué página usar?", 0,
                                            len(pages) - 1, ss().page_idx)
        tip("Usa la imagen más nítida y grande que tengas: es lo que más influye en el resultado.")
        st.image(pages[ss().page_idx], caption="Vista de tu gráfica",
                 use_column_width=True)
    else:
        st.info("Aún no has subido ningún archivo.")


def step_crop():
    pages = ss().pages
    if not pages:
        st.info("Regresa al paso anterior y sube un archivo.")
        return
    img = pages[ss().page_idx]
    tip("Ajusta el recuadro verde justo por dentro de los ejes. Excluir el texto "
        "y los números evita que la app los confunda con la curva.")

    if HAS_CROPPER:
        cropped = st_cropper(img, realtime_update=True, box_color="#22aa22",
                             aspect_ratio=None)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(cropped, caption="Así quedará", use_column_width=True)
        with c2:
            if st.button("✅ Usar este recorte", type="primary",
                         use_container_width=True):
                ss().working_image = to_working(cropped)
                st.success("¡Recorte guardado!")
            if st.button("Usar imagen completa", use_container_width=True):
                ss().working_image = to_working(img)
                st.success("Imagen completa guardada.")
    else:
        warn("Componente de recorte no instalado; se usará la imagen completa.")
        if st.button("Usar imagen completa", type="primary"):
            ss().working_image = to_working(img)
            st.success("Imagen completa guardada.")

    if ss().working_image is not None:
        st.success("Área de trabajo definida ✓")


def _assign_click(name, x, y):
    if ss().last_click == (x, y):
        return
    ss().last_click = (x, y)
    calib = ss().calib
    if name == "X_min":
        calib.px_xmin = x
    elif name == "X_max":
        calib.px_xmax = x
    elif name == "Y_min":
        calib.px_ymin = y
    elif name == "Y_max":
        calib.px_ymax = y
    ss().calib_points[name] = (x, y)


def step_calibrate():
    wi = ss().working_image
    if wi is None:
        st.info("Primero define el área de trabajo en el paso anterior.")
        return
    calib = ss().calib

    with st.expander("❓ ¿Qué significan X_min, X_max, Y_min, Y_max? (léeme la primera vez)"):
        st.markdown(
            "La calibración le enseña a la app a traducir la imagen a números reales. "
            "Se hace en dos partes:\n\n"
            "**Parte 1 — escribes cuánto valen los extremos de cada eje:**\n"
            "- **X_min**: el valor más pequeño del eje horizontal (X), normalmente a la "
            "izquierda. En esfuerzo–deformación suele ser la deformación inicial, `0`.\n"
            "- **X_max**: el valor más grande del eje X, a la derecha (ej. `0.003` cm/cm).\n"
            "- **Y_min**: el valor más pequeño del eje vertical (Y), abajo (normalmente `0`).\n"
            "- **Y_max**: el valor más grande del eje Y, arriba (ej. `240` kg/cm²).\n\n"
            "**Parte 2 — marcas con un clic DÓNDE están esos extremos** en la imagen. "
            "Así la app conoce tanto el valor real como su posición en píxeles, y puede "
            "convertir cualquier punto de la curva.")

    st.markdown("**1) ¿Cuánto valen los ejes?**")
    c1, c2, c3, c4 = st.columns(4)
    calib.x_min = c1.number_input("X mínimo", value=float(calib.x_min), format="%.6f")
    calib.x_max = c2.number_input("X máximo", value=float(calib.x_max), format="%.6f")
    calib.y_min = c3.number_input("Y mínimo", value=float(calib.y_min), format="%.6f")
    calib.y_max = c4.number_input("Y máximo", value=float(calib.y_max), format="%.6f")

    st.markdown("**2) Marca cada extremo en la imagen**")
    done = {
        "X_min": calib.px_xmin is not None,
        "X_max": calib.px_xmax is not None,
        "Y_min": calib.px_ymin is not None,
        "Y_max": calib.px_ymax is not None,
    }
    labels = {k: ("✅ " + k) if done[k] else ("⬜ " + k)
              for k in ["X_min", "X_max", "Y_min", "Y_max"]}
    order = ["X_min", "X_max", "Y_min", "Y_max"]
    choice = st.radio("Selecciona el punto y luego haz clic en la imagen:",
                      order, format_func=lambda k: labels[k], horizontal=True)
    ss().calib_target = choice

    guide = {
        "X_min": "Haz clic en el extremo IZQUIERDO del eje X.",
        "X_max": "Haz clic en el extremo DERECHO del eje X.",
        "Y_min": "Haz clic ABAJO en el eje Y (donde Y es mínimo).",
        "Y_max": "Haz clic ARRIBA en el eje Y (donde Y es máximo).",
    }
    tip(guide[choice])

    left, right = st.columns([3, 1])
    with left:
        if HAS_COORDS:
            annotated = draw_calib_overlay(wi, ss().calib_points)
            val = streamlit_image_coordinates(Image.fromarray(annotated),
                                              key="calib_friendly")
            if val is not None:
                _assign_click(choice, float(val["x"]), float(val["y"]))
        else:
            warn("Componente de clic no instalado. Ingresa los píxeles manualmente:")
            calib.px_xmin = st.number_input("px X_min (col)", value=calib.px_xmin or 0.0)
            calib.px_xmax = st.number_input("px X_max (col)", value=calib.px_xmax or float(wi.shape[1]))
            calib.px_ymin = st.number_input("px Y_min (fila)", value=calib.px_ymin or float(wi.shape[0]))
            calib.px_ymax = st.number_input("px Y_max (fila)", value=calib.px_ymax or 0.0)
    with right:
        n = sum(done.values())
        st.metric("Progreso", f"{n} de 4")
        st.progress(n / 4)
        if calib.is_complete():
            st.success("¡Calibración lista! ✓")
        else:
            st.caption("Faltan: " + ", ".join(k for k in order if not done[k]))
        if st.button("↺ Rehacer calibración", use_container_width=True):
            ss().calib = AxisCalibration(x_min=calib.x_min, x_max=calib.x_max,
                                         y_min=calib.y_min, y_max=calib.y_max)
            ss().calib_points = {}
            st.rerun()


def step_detect():
    wi = ss().working_image
    calib = ss().calib
    if wi is None:
        st.info("Falta el área de trabajo.")
        return

    with st.expander("🎛️ Ajustes de detección (ábrelo si la curva no se ve limpia)",
                     expanded=not step_complete(3)):
        cc1, cc2 = st.columns(2)
        threshold = cc1.slider("Umbral · baja si aparece la cuadrícula gris", 0, 255, 100)
        denoise = cc2.slider("Limpieza de ruido", 0, 5, 1)
        cc3, cc4 = st.columns(2)
        min_area = cc3.slider("Quitar manchas pequeñas (texto)", 0, 500, 30, 10)
        agg_label = cc4.selectbox("Trazo de la línea",
                                  ["median", "mean", "min (arriba)", "max (abajo)"])
    agg_map = {"median": "median", "mean": "mean",
               "min (arriba)": "min", "max (abajo)": "max"}

    with st.expander("❓ ¿Qué hace cada control?"):
        st.markdown(
            "- **Umbral**: qué tan oscuro debe ser un píxel para contar como curva. "
            "Valores bajos = solo lo muy negro. **Si aparece la cuadrícula gris, BÁJALO**; "
            "si la curva se ve cortada, súbelo. Empieza en 100.\n"
            "- **Limpieza de ruido**: borra puntitos y motas sueltas. Súbelo si hay ruido; "
            "mantenlo bajo (1) si la línea es delgada, para no borrarla.\n"
            "- **Quitar manchas pequeñas (área mínima)**: elimina grupos pequeños de "
            "píxeles, como letras o números que se colaron en el recorte. Súbelo si ves "
            "texto en la detección.\n"
            "- **Trazo de la línea**: cómo se elige el punto cuando la línea es gruesa.\n"
            "    - *median* (recomendado): toma el centro de la línea, muy estable.\n"
            "    - *mean*: promedio, similar a median.\n"
            "    - *min (arriba)*: toma el borde superior de la línea.\n"
            "    - *max (abajo)*: toma el borde inferior de la línea.")

    mask = build_mask(wi, threshold=threshold, denoise=denoise, min_area=min_area)
    cols, rows = extract_single_curve(mask, aggregate=agg_map[agg_label])

    tip("Objetivo: en la imagen de la izquierda debes ver la curva blanca y limpia, "
        "sin cuadrícula ni texto. Si aparece la cuadrícula, baja el umbral.")
    i1, i2 = st.columns(2)
    with i1:
        st.image(mask, caption="Lo que la app detectó (máscara)", use_column_width=True)
    with i2:
        if cols.size:
            st.image(overlay_points(wi, cols, rows),
                     caption="Puntos sobre tu gráfica", use_column_width=True)
        else:
            warn("No se detectó nada. Sube el umbral o revisa el recorte.")

    if not calib.is_complete():
        warn("Vuelve al paso Calibrar y marca los 4 puntos para convertir a datos reales.")
        return
    if cols.size:
        x, y = calib.pixels_to_data(cols, rows)
        df = pd.DataFrame({"X": x, "Y": y}).sort_values("X").reset_index(drop=True)
        ss().raw_df = df
        m1, m2, m3 = st.columns(3)
        m1.metric("Puntos detectados", len(df))
        m2.metric("Rango X", f"{df['X'].min():.4g} … {df['X'].max():.4g}")
        m3.metric("Rango Y", f"{df['Y'].min():.4g} … {df['Y'].max():.4g}")
        st.line_chart(df.set_index("X"))


def _comparison_fig(raw, final):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(raw["X"], raw["Y"], ".", ms=2, alpha=0.35, color="#888",
            label="Detección cruda")
    ax.plot(final["X"], final["Y"], "-o", ms=3, lw=1.4, color="#c0392b",
            label="Curva final")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    return fig


def step_edit():
    raw = ss().raw_df
    if raw is None:
        st.info("Primero detecta la curva en el paso anterior.")
        return
    with st.expander("❓ ¿Qué significan 'muestrear' y 'suavizar'?"):
        st.markdown(
            "- **Número de puntos (muestrear)**: la detección genera cientos de puntos "
            "(uno por columna). Aquí eliges cuántos quieres en tu tabla final (ej. 30), "
            "repartidos de forma pareja a lo largo del eje X. Los valores intermedios se "
            "estiman por **interpolación** (una línea recta entre puntos vecinos).\n"
            "- **Suavizar**: reduce el 'temblor' de la curva promediando puntos cercanos. "
            "Úsalo con moderación: demasiado suavizado puede borrar detalles reales, como "
            "el codo de fluencia de una curva esfuerzo–deformación.")

    c1, c2 = st.columns(2)
    n_points = c1.slider("¿Cuántos puntos quieres en tu tabla?", 5, 200, 30)
    do_smooth = c2.checkbox("Suavizar la curva")
    xs, ys = resample_uniform_x(raw["X"].values, raw["Y"].values, n_points)
    df = pd.DataFrame({"X": xs, "Y": ys})
    if do_smooth:
        win = st.slider("Intensidad del suavizado", 3, 51, 7, 2)
        df["Y"] = smooth(df["Y"].values, win, poly=2)

    tip("Puedes editar cualquier celda, borrar filas o añadir puntos en la fila vacía del final.")
    e1, e2 = st.columns([1, 1])
    with e1:
        edited = st.data_editor(df, num_rows="dynamic",
                                use_container_width=True, height=380, key="editor_f")
        final = edited.dropna().sort_values("X").reset_index(drop=True)
        ss().final_df = final
    with e2:
        st.pyplot(_comparison_fig(raw, ss().final_df))


def _reconstructed_fig(df):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(df["X"], df["Y"], "-o", ms=3, lw=1.5, color="#c0392b")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Curva reconstruida"); ax.grid(alpha=0.3); fig.tight_layout()
    return fig


def step_export():
    df = ss().final_df
    calib = ss().calib
    if df is None or df.empty:
        st.info("Aún no hay datos para exportar.")
        return
    st.success("🎉 ¡Todo listo! Revisa el resumen y descarga tus datos.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Puntos", len(df))
    m2.metric("X", f"{df['X'].min():.4g} … {df['X'].max():.4g}")
    m3.metric("Y", f"{df['Y'].min():.4g} … {df['Y'].max():.4g}")

    titulo = st.text_input("Título del reporte",
                           "Digitalización de gráfica esfuerzo–deformación")
    meta = {"titulo": titulo,
            "subtitulo": "Datos aproximados obtenidos por digitalización semiautomática."}
    calib_dict = calib.as_dict()

    st.dataframe(df, use_container_width=True, hide_index=True, height=280)

    crop_png = array_to_png(ss().working_image) if ss().working_image is not None else None
    recon_png = fig_to_png(_reconstructed_fig(df))
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ CSV", to_csv_bytes(df), "datos.csv", "text/csv",
                       use_container_width=True)
    d2.download_button("⬇️ Excel", to_xlsx_bytes(df, calib_dict, meta),
                       "datos.xlsx", XLSX_MIME, use_container_width=True)
    d3.download_button("⬇️ Reporte PDF",
                       to_pdf_bytes(df, calib_dict, meta, crop_png, recon_png),
                       "reporte.pdf", "application/pdf", use_container_width=True)
    tip("El Excel y el PDF incluyen automáticamente la nota de que los datos son aproximados.")


STEP_FUNCS = [step_upload, step_crop, step_calibrate, step_detect, step_edit, step_export]


# =============================================================== MAIN
def main():
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🔬 Digitalizador de gráficas")
        st.caption("Asistente paso a paso · datos aproximados")
        st.divider()
        st.markdown("**¿Dónde estoy?**")
        for i, name in enumerate(STEPS):
            icon = "✅" if step_complete(i) else ("🔵" if i == ss().step_idx else "⚪")
            st.write(f"{icon} {i+1}. {name}")
        st.divider()
        if st.button("↺ Empezar de nuevo", use_container_width=True):
            keys = ["step_idx", "pages", "page_idx", "working_image", "calib",
                    "calib_points", "calib_target", "last_click", "raw_df", "final_df"]
            for k in keys:
                st.session_state.pop(k, None)
            init_state()
            st.rerun()

    i = ss().step_idx
    render_stepper(i)
    hero(i)
    STEP_FUNCS[i]()

    st.divider()
    nav1, nav_sp, nav2 = st.columns([1, 2, 1])
    with nav1:
        if i > 0 and st.button("← Atrás", use_container_width=True):
            ss().step_idx -= 1
            st.rerun()
    with nav2:
        if i < len(STEPS) - 1:
            ok = step_complete(i)
            if st.button("Siguiente →", type="primary",
                         use_container_width=True, disabled=not ok):
                ss().step_idx += 1
                st.rerun()
            if not ok and HINTS[i]:
                st.caption("⚠️ " + HINTS[i])


if __name__ == "__main__":
    main()
