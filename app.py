#!/usr/bin/env python3
"""Web de clasificación de lesiones de piel (HAM10000): subís una foto (o
elegís una de ejemplo del dataset) y el modelo clasifica entre 7 tipos de
lesión, con un mapa de calor Grad-CAM mostrando en qué se basó.

Demo educativo — no es una herramienta de diagnóstico médico real.

Correr con: ./env_skin/bin/streamlit run app.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from skin_classifier import (
    CLASS_LABELS_ES,
    CLASS_NAMES,
    CONCERNING_CLASSES,
    METADATA_FILE,
    find_image_path,
    grad_cam,
    load_model,
    predict_image,
)

st.set_page_config(page_title="Clasificador de lesiones de piel", layout="wide")

load_model_cached = st.cache_resource(load_model)


@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_FILE)


def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray) -> Image.Image:
    """Superpone el mapa de calor Grad-CAM (rojo = más relevante) sobre la imagen."""
    base = pil_image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    base_arr = np.asarray(base).astype("float32")

    heat = np.zeros_like(base_arr)
    heat[..., 0] = cam * 255  # rojo proporcional a la relevancia

    blended = (base_arr * 0.6 + heat * 0.4).clip(0, 255).astype("uint8")
    return Image.fromarray(blended)


CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.app-header {
    background: #1E293B;
    padding: 2rem 2.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}
.app-header h1 {
    color: #FFFFFF;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}
.app-header p {
    color: #B8C4D9;
    font-size: 0.95rem;
    margin: 0;
}

.disclaimer-box {
    background: #FEF2F2;
    border-left: 4px solid #DC2626;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.5rem;
}
.disclaimer-box strong {
    color: #991B1B;
}
.disclaimer-box span {
    color: #7F1D1D;
    font-size: 0.9rem;
}

.metric-strip {
    color: #64748B;
    font-size: 0.85rem;
    margin-bottom: 1.8rem;
}

.section-label {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748B;
    margin-bottom: 0.4rem;
}

.pred-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.4rem;
}
.pred-card h2 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1E293B;
    margin: 0 0 0.3rem 0;
}
.confidence-row {
    font-size: 0.95rem;
    color: #334155;
    margin-bottom: 0.9rem;
}

.badge {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.3rem;
}
.badge-alert {
    background: #FEF3C7;
    color: #92400E;
}
.badge-match {
    background: #DCFCE7;
    color: #166534;
}
.badge-mismatch {
    background: #FEE2E2;
    color: #991B1B;
}

.limitations-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
}
.limitations-box h3 {
    font-size: 1rem;
    font-weight: 700;
    color: #1E293B;
    margin: 0 0 0.7rem 0;
}
.limitations-box li {
    color: #334155;
    font-size: 0.92rem;
    margin-bottom: 0.5rem;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>Clasificador de lesiones de piel</h1>
        <p>Clasificación de imágenes dermatoscópicas en 7 categorías, con transfer learning e interpretabilidad Grad-CAM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Demo educativo de deep learning — no es una herramienta de diagnóstico médico.</strong><br>
        <span>El modelo tiene limitaciones reales y conocidas (ver más abajo). Ante cualquier lesión de piel real, consultá a un dermatólogo.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="metric-strip">Entrenado sobre HAM10000 (~10,000 imágenes dermatoscópicas, 7 clases) '
    "con un ResNet18 pre-entrenado en ImageNet (transfer learning). "
    "Accuracy de validación: 78% &nbsp;|&nbsp; F1 macro: 0.59 "
    "&nbsp;|&nbsp; Baseline (clase mayoritaria): 67% / 0.11</div>",
    unsafe_allow_html=True,
)

metadata = load_metadata()

col_input, col_sample = st.columns([2, 1])

with col_sample:
    st.markdown('<div class="section-label">¿No tenés una imagen a mano?</div>', unsafe_allow_html=True)
    if st.button("Usar una imagen de ejemplo del dataset"):
        sample_row = metadata.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
        st.session_state["sample_image_id"] = sample_row["image_id"]
        st.session_state["sample_true_label"] = sample_row["dx"]

uploaded_file = col_input.file_uploader("Subí una foto de la lesión (jpg/png)", type=["jpg", "jpeg", "png"])

image = None
true_label = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.pop("sample_image_id", None)
elif "sample_image_id" in st.session_state:
    path = find_image_path(st.session_state["sample_image_id"])
    if path:
        image = Image.open(path)
        true_label = st.session_state.get("sample_true_label")

if image is not None:
    device = "cpu"
    model = load_model_cached()

    probs = predict_image(model, image, device=device)
    top_class = max(probs, key=probs.get)
    top_prob = probs[top_class]

    class_idx = CLASS_NAMES.index(top_class)
    cam = grad_cam(model, image, class_idx, device=device)
    overlay = overlay_heatmap(image, cam)

    st.divider()

    col_img, col_cam, col_pred = st.columns(3)
    col_img.markdown('<div class="section-label">Imagen original</div>', unsafe_allow_html=True)
    col_img.image(image, width="stretch")
    col_cam.markdown('<div class="section-label">Grad-CAM (rojo = dónde miró el modelo)</div>', unsafe_allow_html=True)
    col_cam.image(overlay, width="stretch")

    with col_pred:
        badges = ""
        if top_class in CONCERNING_CLASSES:
            badges += '<div class="badge badge-alert">Clase potencialmente maligna — consultar dermatólogo</div>'
        if true_label:
            if true_label == top_class:
                badges += '<div class="badge badge-match">Coincide con la etiqueta real</div>'
            else:
                badges += '<div class="badge badge-mismatch">No coincide con la etiqueta real</div>'
            badges += f'<div class="confidence-row" style="margin-top:0.6rem;">Etiqueta real: <strong>{CLASS_LABELS_ES[true_label]}</strong></div>'

        st.markdown(
            f"""
            <div class="pred-card">
                <div class="section-label">Predicción del modelo</div>
                <h2>{CLASS_LABELS_ES[top_class]}</h2>
                <div class="confidence-row">Confianza: <strong>{top_prob:.0%}</strong></div>
                {badges}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="section-label">Probabilidad por clase</div>', unsafe_allow_html=True)
    df_probs = pd.DataFrame(
        {"Clase": [CLASS_LABELS_ES[c] for c in CLASS_NAMES], "Probabilidad": [probs[c] for c in CLASS_NAMES]}
    ).set_index("Clase").sort_values("Probabilidad", ascending=False)
    st.bar_chart(df_probs, color="#2563EB")

    st.divider()
    st.markdown(
        """
        <div class="limitations-box">
            <h3>Limitaciones — leer antes de confiar en el resultado</h3>
            <ul>
                <li><strong>El recall de melanoma (mel) es de solo 56%</strong> en validación — de cada 10 melanomas reales,
                el modelo deja pasar ~4 como si fueran benignos. Es el error más peligroso posible para un
                caso de uso real, y no está resuelto.</li>
                <li>Las clases más raras del dataset (df, akiec, vasc) tienen pocos ejemplos
                (22 a 56 en validación) — sus métricas son ruidosas.</li>
                <li>Entrenado sobre un dataset de origen europeo (HAM10000) — no está validado en otras
                poblaciones ni tipos de piel.</li>
                <li>Es un modelo de clasificación de imágenes recortadas y centradas en la lesión, no de
                detección en una foto general del cuerpo.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
