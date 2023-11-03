import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Cargar el dataset
df = pd.read_csv("Manejo_residuos_peligrosos_Palmira_preprocessed.csv")

# Seleccionar las características
X = df[["tipo_identificacion", "tipo_clase_sujeto", "comuna", "barrio", "fecha_programada", "fecha_ejecutada", "tipo", "fuente", "tiempo"]]
y = df["de_cumplimiento"]

# Entrenar el modelo
model = DecisionTreeClassifier()
model.fit(X, y)

# Guardar el modelo
model.save("model.pkl")

# Streamlit
import streamlit as st

# Cargar el modelo
model = pickle.load(open("model.pkl", "rb"))

# Función para predecir
def predict(X):
  return model.predict(X)

# Mostrar el modelo
st.title("Modelo de predicción de cumplimiento de residuos peligrosos")
st.write("Este modelo predice si un sujeto cumplirá con los requisitos de manejo de residuos peligrosos.")

# Entrada de datos
tipo_identificacion = st.selectbox("Tipo de identificación", df["tipo_identificacion"].unique())
tipo_clase_sujeto = st.selectbox("Tipo de clase de sujeto", df["tipo_clase_sujeto"].unique())
comuna = st.selectbox("Comuna", df["comuna"].unique())
barrio = st.selectbox("Barrio", df["barrio"].unique())
fecha_programada = st.date_input("Fecha programada")
fecha_ejecutada = st.date_input("Fecha ejecutada")
tipo = st.selectbox("Tipo", df["tipo"].unique())
fuente = st.selectbox("Fuente", df["fuente"].unique())
tiempo = st.number_input("Tiempo")

# Predecir
X = pd.DataFrame({
  "tipo_identificacion": [tipo_identificacion],
  "tipo_clase_sujeto": [tipo_clase_sujeto],
  "comuna": [comuna],
  "barrio": [barrio],
  "fecha_programada": [fecha_programada],
  "fecha_ejecutada": [fecha_ejecutada],
  "tipo": [tipo],
  "fuente": [fuente],
  "tiempo": [tiempo]
})

y_pred = predict(X)

# Salida
st.write("El sujeto tiene un {:.0%} de probabilidad de cumplir con los requisitos.".format(y_pred[0]))
