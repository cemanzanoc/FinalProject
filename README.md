# Recomendador de Fragancias: Potenciando la Selección de Aromas con Machine Learning

## Visión General del Proyecto
Este proyecto tiene como objetivo desarrollar un sistema de recomendación integral que ayude a los usuarios a descubrir fragancias basado en señales visuales y textuales. Al combinar técnicas avanzadas de procesamiento de imágenes con procesamiento de lenguaje natural y machine learning, el sistema analiza imágenes subidas (por ejemplo, paletas de colores) y preferencias textuales del usuario para sugerir perfumes que coincidan con el estado de ánimo o estilo deseado.

## Insights de Nuestro Estudio
- **Importancia de las Señales Visuales:** El análisis muestra que los colores dominantes extraídos de las imágenes pueden ser mapeados a acordes específicos de fragancias.
- **Clave de las Descripciones Textuales:** Los perfiles olfativos y notas de fragancias proporcionan una visión valiosa de las características del aroma, capturadas eficazmente usando TF-IDF y KNN.
- **Enfoque Integrado:** La combinación de recomendaciones basadas en imágenes y texto mejora la precisión y personalización de las sugerencias de fragancias.

## Objetivo
El objetivo principal es construir un motor de recomendaciones que combine de manera inteligente el análisis visual con el filtrado textual para predecir y recomendar fragancias. Esto ayuda a los usuarios a encontrar perfumes que se alineen con sus preferencias estéticas y olfativas.

## Áreas de Enfoque
- **Análisis Exploratorio de Datos (EDA):** Examinación del conjunto de datos de fragancias para entender distribuciones, correlaciones y atributos clave.
- **Procesamiento de Imágenes:**  
  - Eliminación de fondos y detección/remoción de rostros en imágenes utilizando bibliotecas como OpenCV y rembg.  
  - Extracción de colores dominantes y mapeo a características de fragancias.
- **Procesamiento de Texto:**  
  - Limpieza y preparación de las descripciones de fragancias para la vectorización con TF-IDF.  
  - Expansión de las consultas de usuario con sinónimos para mejorar la coincidencia.
- **Modelos de Machine Learning:**  
  - Entrenamiento de un modelo TF-IDF + KNN sobre datos de perfiles olfativos.  
  - Integración de conocimientos extraídos de imágenes y texto para recomendaciones robustas.
- **Motor de Recomendación:**  
  - Combinación de palabras clave derivadas del análisis de color y preferencias textuales para filtrar y clasificar las sugerencias de perfumes.

## Funcionalidad ⚙️
- **Data Loader:**  
  Carga y preprocesa el conjunto de datos de fragancias (ej., `fragrance_ML_model.csv`), asegurando el formato correcto y nombres de columnas adecuados.
  
- **Módulo de Procesamiento de Imágenes:**  
  - Elimina fondos y detecta/remueve rostros en imágenes.  
  - Extrae colores dominantes y los mapea a etiquetas de fragancias usando clustering KMeans y un archivo CSV de referencia de colores.

- **Módulo de Procesamiento de Texto:**  
  - Limpia las notas de fragancias conservando separadores importantes (comas).  
  - Expande consultas de búsqueda con sinónimos utilizando WordNet para mejorar la coincidencia textual.

- **Módulos de Recomendación:**  
  - **Recomendador Basado en Imágenes:** Utiliza etiquetas de colores extraídas y filtros basados en la situación para generar palabras clave de fragancias.  
  - **Recomendador Basado en Texto:** Implementa un modelo TF-IDF + KNN para encontrar perfiles de fragancias similares basándose en la entrada del usuario.

## Herramientas Utilizadas 🛠️
- **Lenguaje de Programación:** Python  
- **Librerías:**  
  - Procesamiento de Datos: Pandas  
  - Procesamiento de Imágenes: OpenCV, PIL, rembg, scikit-learn  
  - Procesamiento de Texto: NLTK, scikit-learn (TF-IDF, Nearest Neighbors)  
  - Visualización: Matplotlib  
- **Control de Versiones y Despliegue:** Git (y opcionalmente Streamlit para aplicaciones interactivas)

## Proceso de Desarrollo 🚀
1. **Limpieza y Carga de Datos:**  
   Se carga el conjunto de datos utilizando un módulo dedicado que maneja distintos delimitadores y estandariza los nombres de columnas.
2. **Pipeline de Procesamiento de Imágenes:**  
   Se procesan las imágenes para eliminar fondos y rostros irrelevantes antes de extraer los colores dominantes.
3. **Análisis de Texto y Entrenamiento del Modelo:**  
   Se limpian las descripciones de fragancias y se entrena un modelo TF-IDF + KNN para facilitar las recomendaciones basadas en texto.
4. **Integración del Motor de Recomendación:**  
   Ambas pipelines (imagen y texto) se combinan para proporcionar una lista filtrada de perfumes recomendados basándose en las preferencias del usuario.
5. **Evaluación e Iteración:**  
   Pruebas e iteraciones para validar la precisión y relevancia de las recomendaciones.

## Estructura del Proyecto 📁
. ├── data/ │ ├── fragrance_ML_model.csv │ ├── fragrance_database.csv │ └── colors.csv ├── fragrance_code/ │ ├── data_loader.py # Carga y preprocesa datos de fragancias │ ├── image_processing.py # Procesa imágenes: remueve fondo/rostros y extrae colores dominantes │ ├── processing_text.py # Limpia datos textuales y expande consultas │ ├── model_text.py # Entrena y guarda el modelo TF-IDF + KNN para texto │ ├── recommender_image_based.py # Recomendador basado en análisis de imágenes │ └── recommender_text_based.py # Recomendador basado en datos textuales ├── notebooks/ │ └── fragrance_EDA.ipynb # Notebook de Análisis Exploratorio de Datos ├── models/ │ ├── tfidf_knn_model.pkl # Modelo TF-IDF + KNN guardado │ ├── vectorizer.pkl # Vectorizador TF-IDF guardado │ └── (otros archivos pickle para normalización/PCA) └── README.md # Documentación del proyecto (este archivo)

## Presentación del Proyecto 🎤
La presentación del proyecto incluye:
- Objetivos y metodología de investigación.
- Insights clave obtenidos del análisis de datos.
- Desglose paso a paso de las técnicas de procesamiento de imágenes y texto.
- Demostración del motor de recomendaciones y su desempeño.


## Fuente de Datos
El conjunto de datos de fragancias es obtenido de kaggle: https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset


