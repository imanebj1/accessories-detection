from pathlib import Path
from PIL import Image
import streamlit as st
import os

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_webcam, seq_detection

# setting page layout
st.set_page_config(
    page_title="Detection of HV Cable Accessories Sequencing Using DL Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Detection of High-Voltage Cable Accessories Sequencing Using DL Model")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path =r"best.pt"    #le chemin vers yolov8 entrainnés par les images et les labels des accessoires du câble HV

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")
    st.stop()  # Arrêtez l'exécution du programme pour éviter d'autres erreurs liées au modèle non chargé

# image option
st.sidebar.header("Image Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    labels, image_path=infer_uploaded_image(confidence, model)  # Obtenez les labels et le chemin de l'image 
    seq_detection(labels, image_path) # Appelez la fonction seq_detection avec les labels et le chemin de l'image pour detecter le séquencement des objets
    
elif source_selectbox == config.SOURCES_LIST[1]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image'  source are implemented")