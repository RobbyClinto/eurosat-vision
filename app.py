import time
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="EuroSAT Vision | UNAP",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PROFESIONAL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

.stApp {
    background: linear-gradient(135deg, #060b14 0%, #0d1b2a 50%, #060b14 100%);
    color: #e8edf2;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d1f35 100%);
    border-right: 1px solid #1e3a5f;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.hero-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0d2545 40%, #0a3060 100%);
    border: 1px solid #1e4080;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,180,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00c8ff, #0080ff, #00c8ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shine 3s linear infinite;
    margin: 0 0 0.5rem 0;
}
@keyframes shine { to { background-position: 200% center; } }
.hero-subtitle { color: #7a9bbf; font-size: 0.95rem; font-weight: 400; margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(0,180,255,0.1);
    border: 1px solid rgba(0,180,255,0.3);
    color: #00c8ff;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #0d1f35 0%, #0a2040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label { color: #5a7a9f; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem; }
.metric-value { color: #00c8ff; font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-unit { color: #7a9bbf; font-size: 0.8rem; margin-top: 0.2rem; }
.result-card {
    background: linear-gradient(135deg, #0a2a1a 0%, #0d3520 100%);
    border: 1px solid #1a5a30;
    border-left: 4px solid #00ff88;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}
.result-class { font-size: 1.6rem; font-weight: 700; color: #00ff88; margin: 0 0 0.3rem 0; }
.result-desc { color: #7abf9f; font-size: 0.9rem; }
.info-card {
    background: rgba(0,100,200,0.08);
    border: 1px solid rgba(0,150,255,0.2);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    color: #7ab0d0;
    font-size: 0.85rem;
}
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 2rem 0;
}
.sidebar-item {
    color: #8aabcf;
    font-size: 0.82rem;
    padding: 0.25rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sidebar-item:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTES
# ============================================================
MODEL_PATH = "modelo_final_eurosat.keras"
IMG_SIZE   = 224

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

CLASS_INFO = {
    "AnnualCrop":           {"emoji": "🌾", "es": "Cultivo Anual",        "desc": "Campos agrícolas de ciclo corto"},
    "Forest":               {"emoji": "🌲", "es": "Bosque",               "desc": "Cobertura forestal densa"},
    "HerbaceousVegetation": {"emoji": "🌿", "es": "Vegetación Herbácea",  "desc": "Praderas y pastizales naturales"},
    "Highway":              {"emoji": "🛣️",  "es": "Carretera",            "desc": "Vías de tránsito terrestre"},
    "Industrial":           {"emoji": "🏭", "es": "Industrial",           "desc": "Zonas de actividad industrial"},
    "Pasture":              {"emoji": "🐄", "es": "Pastizal",             "desc": "Tierras de pastoreo extensivo"},
    "PermanentCrop":        {"emoji": "🍇", "es": "Cultivo Permanente",   "desc": "Viñedos, frutales y cultivos fijos"},
    "Residential":          {"emoji": "🏘️", "es": "Residencial",          "desc": "Zonas urbanas habitacionales"},
    "River":                {"emoji": "🌊", "es": "Río",                  "desc": "Cursos de agua naturales"},
    "SeaLake":              {"emoji": "🏞️", "es": "Mar / Lago",           "desc": "Grandes cuerpos de agua"},
}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:3rem;'>🛰️</div>
        <div style='color:#00c8ff; font-weight:700; font-size:1.1rem;'>EuroSAT Vision</div>
        <div style='color:#5a7a9f; font-size:0.75rem;'>Sistema de Teledetección IA</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="color:#00c8ff; font-size:0.8rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.8rem;">📡 Clases del Dataset</div>', unsafe_allow_html=True)
    for name, info in CLASS_INFO.items():
        st.markdown(
            f'<div class="sidebar-item">{info["emoji"]} <b style="color:#aac8e8">{info["es"]}</b> — {info["desc"]}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="color:#00c8ff; font-size:0.8rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.8rem;">🤖 Arquitectura del Modelo</div>', unsafe_allow_html=True)
    for item in [
        ("Base", "EfficientNetB0"), ("Método", "Transfer Learning"),
        ("Ajuste", "Fine-Tuning"),  ("Dataset", "EuroSAT (27k imgs)"),
        ("Clases", "10 categorías"),("Input", "224 × 224 px"),
    ]:
        st.markdown(f'<div class="sidebar-item">🔹 <b style="color:#aac8e8">{item[0]}:</b> {item[1]}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#3a5a7f; font-size:0.72rem; text-align:center; padding-top:1.5rem; border-top:1px solid #1e3a5f; margin-top:1rem;'>
        Práctica Final — Unidad I<br>
        Aprendizaje Profundo · UNAP Puno<br>
        <span style='color:#1e4a6f;'>Ingeniería de Sistemas</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🛰️ TELEDETECCIÓN · DEEP LEARNING · EUROSAT</div>
    <p class="hero-title">Sistema Inteligente de Reconocimiento de Imágenes Satelitales</p>
    <p class="hero-subtitle">
        Clasificación automática de cobertura terrestre mediante CNN, Transfer Learning y Fine-Tuning
        con EfficientNetB0 entrenado sobre el dataset EuroSAT — 10 clases · 27,000 imágenes Sentinel-2
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DEL MODELO
# ============================================================
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

with st.spinner("⏳ Inicializando modelo EfficientNetB0..."):
    modelo = cargar_modelo()

if modelo is None:
    st.error(f"❌ No se encontró: `{MODEL_PATH}`")
    st.warning("📁 Coloca `modelo_final_eurosat.keras` en la misma carpeta que `app.py`.")
    st.stop()

st.markdown('<div class="info-card">✅ &nbsp;<b>Modelo cargado correctamente</b> — EfficientNetB0 con Fine-Tuning listo para inferencia</div>', unsafe_allow_html=True)

# ============================================================
# UPLOADER
# ============================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📤 Cargar Imagen Satelital")
st.markdown('<p style="color:#5a7a9f; font-size:0.88rem;">Sube una imagen satelital o aérea para clasificar su cobertura terrestre. Compatible con imágenes Sentinel-2 y similares.</p>', unsafe_allow_html=True)

archivo = st.file_uploader("Selecciona imagen:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ============================================================
# INFERENCIA
# ============================================================
if archivo is not None:

    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        img = Image.open(archivo).convert("RGB")
        st.image(img, caption="Imagen cargada", use_container_width=True)
        st.markdown(
            f'<div style="color:#3a6a9f; font-size:0.78rem; text-align:center;">'
            f'Dimensiones originales: {img.size[0]} × {img.size[1]} px — Redimensionada a {IMG_SIZE} × {IMG_SIZE} px</div>',
            unsafe_allow_html=True
        )

    with col_info:
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array  = keras_image.img_to_array(img_resized)
        img_array  = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Procesando matriz espectral..."):
            inicio = time.time()
            try:
                prediccion = modelo.predict(img_array, verbose=0)
            except Exception as e:
                st.error(f"❌ Error durante la inferencia: {e}")
                st.stop()
            fin = time.time()

        clase_detectada = CLASS_NAMES[int(np.argmax(prediccion))]
        info            = CLASS_INFO[clase_detectada]
        probabilidad    = float(np.max(prediccion)) * 100
        tiempo_computo  = fin - inicio

        st.markdown(f"""
        <div class="result-card">
            <div style="color:#3abf7f; font-size:0.75rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;">Clase Detectada</div>
            <div class="result-class">{info['emoji']} {info['es']}</div>
            <div class="result-desc">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Confianza</div><div class="metric-value">{probabilidad:.1f}</div><div class="metric-unit">%</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Inferencia</div><div class="metric-value">{tiempo_computo:.3f}</div><div class="metric-unit">segundos</div></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card" style="margin-top:1.2rem;">
            🤖 <b>Modelo:</b> EfficientNetB0 · Transfer Learning + Fine-Tuning<br>
            📊 <b>Dataset:</b> EuroSAT Sentinel-2 · 10 clases · 27,000 imágenes
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Distribución de Probabilidades por Clase")

    probs    = prediccion[0] * 100
    df_probs = pd.DataFrame({
        "Clase": [f"{CLASS_INFO[c]['emoji']} {CLASS_INFO[c]['es']}" for c in CLASS_NAMES],
        "Probabilidad (%)": probs
    }).set_index("Clase").sort_values("Probabilidad (%)", ascending=False)

    st.bar_chart(df_probs, color="#00c8ff")

    st.markdown("### 🗂️ Tabla Detallada de Resultados")
    df_tabla = pd.DataFrame({
        "Clase (EN)":       CLASS_NAMES,
        "Clase (ES)":       [CLASS_INFO[c]["es"]   for c in CLASS_NAMES],
        "Descripción":      [CLASS_INFO[c]["desc"]  for c in CLASS_NAMES],
        "Probabilidad (%)": [f"{p:.4f}" for p in probs],
    }).sort_values("Probabilidad (%)", ascending=False).reset_index(drop=True)
    df_tabla.index += 1
    st.dataframe(df_tabla, use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#2a4a6f; font-size:0.78rem; padding-bottom:1rem;'>
        Sistema desarrollado para la Práctica Final · Unidad I · Aprendizaje Profundo<br>
        Universidad Nacional del Altiplano — Escuela Profesional de Ingeniería de Sistemas · Puno, Perú
    </div>
    """, unsafe_allow_html=True)