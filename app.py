import streamlit as st
import pandas as pd

# Importar el conjunto de datos
df = pd.read_csv("https://www.datos.gov.co/api/views/39bi-a35k/rows.csv?accessType=DOWNLOAD")

# Verificar si la columna existe
if "tipo_clase_sujeto" in df.columns:
    st.bar_chart(df["tipo_clase_sujeto"].value_counts())
else:
    # Mostrar un mensaje de error si la columna no existe
    st.write("La columna `tipo_clase_sujeto` no existe en el conjunto de datos.")

# Mostrar un resumen del conjunto de datos
st.write(df.describe())

# Mostrar un gráfico de barras de la cantidad de visitas por comuna
st.map(df.groupby("comuna")["tipo_clase_sujeto"].value_counts())

# Mostrar un gráfico de líneas de la cantidad de visitas por fecha
st.line_chart(df.groupby("fecha_program")["tipo_clase_sujeto"].value_counts())

# Mostrar una tabla de los datos de una visita específica
st.table(df.loc[df["tipo_identificacion"] == "NIT"])
