import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"
CONFIANZA  = 0.5

CLASES_PPE = {
    "helmet":   {"nombre": "Casco",        "emoji": "⛑️",  "critico": True},
    "vest":     {"nombre": "Chaleco",       "emoji": "🦺",  "critico": True},
    "boots":    {"nombre": "Botas",         "emoji": "👢",  "critico": False},
    "gloves":   {"nombre": "Guantes",       "emoji": "🧤",  "critico": False},
    "glasses":  {"nombre": "Lentes",        "emoji": "🥽",  "critico": False},
    "earmuffs": {"nombre": "Orejeras",      "emoji": "🎧",  "critico": False},
}

METRICAS_MODELO = {
    "helmet":   0.901,
    "vest":     0.905,
    "boots":    0.880,
    "glasses":  0.688,
    "earmuffs": 0.639,
    "gloves":   0.424,
}

# ── Setup ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detección PPE",
    page_icon="🦺",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    return YOLO(MODEL_PATH)

model = cargar_modelo()

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        border-left: 4px solid #00d4aa;
    }
    .alerta-roja {
        background: #2d1b1b;
        border-radius: 12px;
        padding: 12px 16px;
        border-left: 4px solid #ff4444;
        color: #ff8888;
        margin: 4px 0;
    }
    .alerta-verde {
        background: #1b2d1b;
        border-radius: 12px;
        padding: 12px 16px;
        border-left: 4px solid #44ff88;
        color: #88ffaa;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Título ─────────────────────────────────────────────────────────────────────
st.title("🦺 Sistema de Detección de EPP")
st.caption("YOLO11n — RTX 4070 Ti — mAP50: 0.760")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    fuente = st.radio("Fuente de entrada", ["🎬 Video", "🖼️ Imagen"])
    confianza = st.slider("Confianza mínima", 0.1, 1.0, CONFIANZA, 0.05)

    st.divider()
    st.subheader("📊 Métricas del modelo")
    for clase, map50 in METRICAS_MODELO.items():
        info = CLASES_PPE.get(clase, {})
        emoji = info.get("emoji", "")
        nombre = info.get("nombre", clase)
        color = "🟢" if map50 >= 0.8 else "🟡" if map50 >= 0.6 else "🔴"
        st.markdown(f"{color} {emoji} **{nombre}**: `{map50:.3f}`")

    st.divider()
    st.caption("mAP50 global: **0.760** · Precision: **0.784** · Recall: **0.707**")

# ── Función de detección ───────────────────────────────────────────────────────
def procesar_frame(frame, conf):
    results = model(frame, conf=conf, device="cpu", verbose=False)
    conteo = {}
    for r in results:
        for box in r.boxes:
            clase = model.names[int(box.cls)]
            conteo[clase] = conteo.get(clase, 0) + 1
    annotated = results[0].plot()
    return annotated, conteo

def mostrar_metricas(conteo):
    conteo_ppe = {k: v for k, v in conteo.items() if k != "person"}
    st.subheader("📋 Detecciones")

    if not conteo_ppe:
        st.info("Sin detecciones")
    else:
        cols = st.columns(2)
        for i, (clase, cantidad) in enumerate(conteo_ppe.items()):
            info = CLASES_PPE.get(clase, {"nombre": clase, "emoji": "📦"})
            with cols[i % 2]:
                st.metric(
                    label=f"{info['emoji']} {info['nombre']}",
                    value=cantidad
                )

    st.subheader("🚨 Estado EPP")
    criticos = [k for k, v in CLASES_PPE.items() if v["critico"]]
    for clase in criticos:
        info = CLASES_PPE[clase]
        detectado = clase in conteo_ppe
        if detectado:
            st.markdown(f'<div class="alerta-verde">✅ {info["emoji"]} {info["nombre"]} detectado</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alerta-roja">❌ {info["emoji"]} {info["nombre"]} NO detectado</div>', unsafe_allow_html=True)

# ── Video ──────────────────────────────────────────────────────────────────────
if fuente == "🎬 Video":
    video_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        col1, col2 = st.columns([2, 1])
        with col1:
            frame_placeholder = st.empty()
        with col2:
            metrics_placeholder = st.empty()

        cap = cv2.VideoCapture(tfile.name)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated, conteo = procesar_frame(frame_rgb, confianza)
                frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
                with metrics_placeholder.container():
                    mostrar_metricas(conteo)
                time.sleep(0.03)
        finally:
            cap.release()
            try:
                os.unlink(tfile.name)
            except Exception:
                pass

# ── Imagen ─────────────────────────────────────────────────────────────────────
elif fuente == "🖼️ Imagen":
    img_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        frame = np.array(image)

        col1, col2 = st.columns([2, 1])
        annotated, conteo = procesar_frame(frame, confianza)
        with col1:
            st.image(annotated, channels="RGB", use_container_width=True)
        with col2:
            mostrar_metricas(conteo)