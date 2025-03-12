import os
import streamlit as st
import pandas as pd
from PIL import Image

# Importa las funciones de tus módulos (ajusta la ruta según la estructura de tu proyecto)
from fragrance_code.data_loader import load_data
from fragrance_code.recommender_image_based import recommend_fragrances
from fragrance_code.model_text import train_model
from fragrance_code.processing_image import remove_background, remove_faces, extract_dominant_colors


# --- Sección 2: Subida y Procesamiento de Imagen ---
st.header("Procesamiento de Imagen")
st.write("Sube una imagen (formatos: jpg, jpeg, png) para extraer colores y obtener recomendaciones.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

processed_img_url = None
detected_colors = None

if uploaded_file is not None:
    # Guarda la imagen de forma temporal
    temp_image_path = "temp_uploaded_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(temp_image_path, caption="Imagen Original", use_column_width=True)
    
    # Procesa la imagen: remueve fondo y rostros
    bg_removed_path = remove_background(temp_image_path)
    faces_removed_path = remove_faces(bg_removed_path)
    
    # Se extraen los colores dominantes de la imagen final procesada
    detected_colors = extract_dominant_colors(faces_removed_path)
    processed_img_url = faces_removed_path  # Esta es la "URL" (ruta local) de la imagen procesada
    
    st.write("La imagen ha sido procesada.")
    st.write(f"Colores detectados: {detected_colors}")
    
# --- Sección 3: Opciones de Preferencias ---
st.sidebar.header("Preferencias de Recomendación")

situation = st.sidebar.selectbox("Situation", options=["", "night", "casual", "romantic", "sport", "office"])
gender = st.sidebar.selectbox("Gender", options=["", "male", "female", "unisex"])

# Cargar dataset para obtener la lista de marcas disponibles (si es posible)
fragrance_df = load_data()
if fragrance_df is not None:
    unique_brands = sorted(fragrance_df["Brand"].dropna().unique())
else:
    unique_brands = []
brand = st.sidebar.selectbox("Brand", options=[""] + unique_brands)

favorite_notes = st.sidebar.text_input("Favorite Notes (separadas por coma)")
exclude_notes = st.sidebar.text_input("Exclude Notes (separadas por coma)")

num_recommendations = st.sidebar.number_input("Número de Recomendaciones", min_value=1, max_value=10, value=5, step=1)

# --- Sección 4: Botón de Recomendación y Resultados ---
if st.sidebar.button("Get Recommendations"):
    # Procesa las entradas de texto en listas
    favorite_notes_list = [n.strip() for n in favorite_notes.split(",") if n.strip()] if favorite_notes.strip() else None
    exclude_notes_list = [n.strip() for n in exclude_notes.split(",") if n.strip()] if exclude_notes.strip() else None

    # Llama a la función de recomendación; si no se subió imagen, detected_colors será None
    recommendations = recommend_fragrances(
        detected_colors=detected_colors,
        situation=situation if situation != "" else None,
        gender=gender if gender != "" else None,
        brand=brand if brand != "" else None,
        favorite_notes=favorite_notes_list,
        exclude_notes=exclude_notes_list,
        num_recommendations=num_recommendations
    )
    
    st.subheader("Recomendaciones")
    if isinstance(recommendations, list) and recommendations:
        for idx, rec in enumerate(recommendations, start=1):
            st.write(f"{idx}. {rec}")
    else:
        st.write("No se encontraron coincidencias. Intenta ajustar los filtros.")


