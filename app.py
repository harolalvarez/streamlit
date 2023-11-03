import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Importar el conjunto de datos
df = pd.read_csv("Manejo_residuos_peligrosos_Palmira_preprocessed.csv")

# Crear un modelo de machine learning
X = df[["tipo_identificacion", "tipo_clase_sujeto", "comuna", "barrio", "fecha_program", "tipo", "fuente", "sector_economico"]]
y = df["de_cumplimiento"]

# Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predecir las visitas del conjunto de prueba
y_pred = model.predict(X_test)

# Mostrar la precisión del modelo
precision = model.score(X_test, y_test)
st.write("Precisión:", precision)

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

# Mostrar el resultado de la clasificación de una visita específica
comuna = st.selectbox("Comuna", df["comuna"].unique())
barrio = st.selectbox("Barrio", df.loc[df["comuna"] == comuna]["barrio"].unique())

visita = df.loc[(df["comuna"] == comuna) & (df["barrio"] == barrio)]

resultado = model.predict(visita[["tipo_identificacion", "tipo_clase_sujeto", "fecha_program", "tipo", "fuente", "sector_economico"]])

st.write("Resultado de la clasificación:", resultado)
