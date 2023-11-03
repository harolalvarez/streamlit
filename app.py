import streamlit as st
import pandas as pd

# Importar el conjunto de datos
df = pd.read_csv("Manejo_residuos_peligrosos_Palmiraa.csv")

# Mostrar un resumen del conjunto de datos
st.write(df.describe())

# Mostrar un gráfico de barras de la cantidad de visitas por tipo de sujeto
st.bar_chart(df["tipo_clase_sujeto"].value_counts())

# Mostrar un mapa de la cantidad de visitas por comuna
st.map(df.groupby("comuna")["tipo_clase_sujeto"].value_counts())

# Mostrar un gráfico de líneas de la cantidad de visitas por fecha
st.line_chart(df.groupby("fecha_program")["tipo_clase_sujeto"].value_counts())

# Mostrar una tabla de los datos de una visita específica
st.table(df.loc[df["tipo_identificacion"] == "NIT"])
