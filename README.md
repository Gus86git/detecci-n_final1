# 🦺 SafeBuild AI – Detección de EPP con YOLO

Sistema inteligente de detección de Equipos de Protección Personal (cascos y chalecos reflectantes) en obras de construcción, usando YOLOv8 y un Sistema Experto basado en reglas.

## 🚀 Deploy en Streamlit Cloud

1. Haz **Fork** o sube este repositorio a GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Selecciona el repositorio, rama `main` y archivo `app.py`.
4. Haz clic en **Deploy** — Streamlit Cloud instalará automáticamente:
   - Las dependencias de Python desde `requirements.txt`
   - Las dependencias del sistema desde `packages.txt` (OpenCV necesita `libgl1-mesa-glx`)

## 📁 Estructura del proyecto

```
├── app.py              # Aplicación principal Streamlit
├── requirements.txt    # Dependencias Python
├── packages.txt        # Dependencias del sistema (Ubuntu/Streamlit Cloud)
├── README.md           # Este archivo
└── models/             # (opcional) Coloca aquí tu modelo personalizado
    └── best.pt         # Modelo YOLOv8 entrenado para EPP
```

## 🤖 Modelo personalizado

Por defecto se usa `yolov8n.pt` (descarga automática). Para usar tu modelo entrenado:

1. Crea la carpeta `models/` en el repositorio.
2. Coloca tu archivo `best.pt` dentro.
3. El sistema lo detectará automáticamente al iniciar.

## ✨ Características

| Función | Descripción |
|---|---|
| **Detección de imágenes** | Analiza fotos JPG/PNG/BMP/WebP |
| **Análisis de video** | Procesa hasta 60 fotogramas de videos MP4/AVI/MOV |
| **Sistema Experto** | 9 reglas de seguridad con niveles ALTA / MEDIA / OK |
| **Detección por color IA** | Infiere cascos/chalecos por análisis de color HSV |
| **Gráfico de cumplimiento** | Visualización interactiva con Plotly |
| **Exportación** | Descarga resultados en Excel (.xlsx) y CSV (.csv) |

## 🛠️ Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Dependencias principales

- `streamlit >= 1.32` — Interfaz web
- `ultralytics >= 8.1` — Modelo YOLOv8
- `opencv-python-headless >= 4.8` — Procesamiento de imagen/video
- `plotly >= 5.18` — Gráficos interactivos
- `pandas` + `xlsxwriter` — Exportación de datos