import streamlit as st
import pandas as pd

# Importar el conjunto de datos
df = pd.read_csv("https://www.datos.gov.co/api/views/39bi-a35k/rows.csv?accessType=DOWNLOAD")

# Verificar si la columna existe
if "tipo_clase_sujeto" in df.columns:
    # Verificar si la columna tiene datos
    if df["tipo_clase_sujeto"].notna().any():
        # Agrupar el conjunto de datos por comuna y calcular el número de visitas por tipo de sujeto
        df_grouped = df.groupby("comuna")["tipo_clase_sujeto"].value_counts()

        # Mostrar el mapa
        st.map(df_grouped)
    else:
        # Mostrar un mensaje de error si la columna no tiene datos
        st.write("La columna `tipo_clase_sujeto` no tiene datos.")
else:
    # Mostrar un mensaje de error si la columna no existe
    st.write("La columna `tipo_clase_sujeto` no existe en el conjunto de datos.")
