# 🔬 Ingeniería inversa de gráficas científicas (MVP)

Herramienta **semiautomática** para recuperar datos aproximados de curvas 2D a
partir de una imagen o PDF (por ejemplo, curvas esfuerzo–deformación). El
usuario recorta la gráfica, calibra los ejes y puede corregir la detección
manualmente. **Los datos resultantes son aproximados**, no sustituyen a la
tabla original de medición.

## Flujo de trabajo

1. **Cargar** PNG/JPG/JPEG o PDF (cada página se convierte a imagen).
2. **Recortar** el área interior de los ejes (excluir título, leyendas y texto).
3. **Calibrar** los ejes: valores reales de X/Y + 4 puntos de referencia por clic.
4. **Detectar** la curva con OpenCV (umbral, limpieza de ruido, área mínima).
5. **Editar y muestrear**: elegir número de puntos, suavizar e editar la tabla.
6. **Exportar** a CSV, Excel (.xlsx) y reporte PDF (imagen + tabla + curva
   reconstruida + parámetros de calibración + nota de aproximación).

## Instalación y ejecución local

Requiere Python 3.9–3.12.

```bash
# 1. Clonar / copiar el proyecto y entrar a la carpeta
cd grafica-inversa

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run app.py
```

Se abre en el navegador (por defecto http://localhost:8501).

> `opencv-python-headless` evita dependencias de GUI del sistema.
> El soporte de PDF usa `pymupdf` (solo pip, sin poppler).

## Estructura del proyecto

```
grafica-inversa/
├── app.py                 # Interfaz Streamlit y orquestación del flujo
├── requirements.txt
├── README.md
└── modules/
    ├── __init__.py
    ├── image_loader.py    # Carga PNG/JPG y conversión PDF→imagen (PyMuPDF)
    ├── calibration.py     # Mapeo lineal píxel→valor real (AxisCalibration)
    ├── curve_detection.py # Máscara + extracción columna-a-columna (OpenCV)
    ├── sampling.py        # Remuestreo uniforme e interpolación / suavizado
    └── exporter.py        # Exportación CSV / XLSX / PDF (reportlab)
```

## Cómo mejorar la precisión

- Trabaja con imágenes de alta resolución y recorta ajustado al interior de los ejes.
- Marca los 4 puntos de calibración lo más lejos posible entre sí y con exactitud.
- Ajusta el **umbral** para que la máscara capture la curva pero no la cuadrícula.
- Usa **área mínima** para eliminar texto y motas residuales.
- Prueba la agregación por columna: `median` es robusta ante grosor de línea;
  `min/max` sirven si la línea es gruesa y quieres el borde superior/inferior.
- Suaviza con Savitzky-Golay solo lo necesario para no deformar la curva.
- Corrige puntos manualmente en la tabla editable antes de exportar.

## Limitaciones del método

- Pensado para **una sola curva** de tipo función (un Y por cada X). Curvas con
  retorno (histéresis, lazos) o varias curvas superpuestas requieren extensión.
- Asume **ejes lineales**. Escalas logarítmicas necesitan mapeo específico.
- La detección columna-a-columna falla si la curva es casi vertical (muchos Y
  por columna) o si se cruza con líneas gruesas de cuadrícula del mismo tono.
- La precisión depende de la resolución de la imagen y de la calibración manual.
- El texto o marcadores muy oscuros dentro del área pueden confundirse con la
  curva; recorta bien y usa el filtro de área mínima.

## Roadmap (siguientes iteraciones)

- Detección automática de ejes y ticks (Hough + OCR de etiquetas).
- Soporte multi-curva por color / clustering.
- Escalas logarítmicas y ejes no lineales.
- Procesamiento por lotes de PDFs y modo API (FastAPI) para integración.
