import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Leer el dataset en un marco de datos de Pandas
df = pd.read_csv("Manejo_residuos_peligrosos_Palmira_preprocessed.csv")

# Seleccionar las variables de entrada y la variable de salida
X = df[["tipo_identificacion", "tipo_clase_sujeto", "comuna", "barrio", "fecha_programada", "fecha_ejecutada", "tipo", "fuente", "acta_informe", "citacion_proceso_visita_1", "fecha_programada_1", "fecha_ejecutada_1", "tiempo", "tipo_1", "fuente_1", "proxima_visita_dd", "acta_informe_1", "de_cumplimiento_1", "concepto_1"]]
y = df["de_cumplimiento"]

# Dividir el dataset en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Entrenar un modelo de machine learning en el conjunto de entrenamiento
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Guardar el modelo entrenado
model.save("model.pkl")

# Cargar el modelo entrenado
model = pickle.load(open("model.pkl", "rb"))

# Crear la aplicación Streamlit
st.title("Modelo de machine learning para residuos peligrosos")

# Mostrar el accuracy del modelo
st.write("Accuracy:", model.score(X_test, y_test))

# Crear un formulario para obtener las variables de entrada
st.form("Predicción")
tipo_identificacion = st.text_input("Tipo de identificación")
tipo_clase_sujeto = st.text_input("Tipo de clase de sujeto")
comuna = st.text_input("Comuna")
barrio = st.text_input("Barrio")
fecha_programada = st.text_input("Fecha programada")
fecha_ejecutada = st.text_input("Fecha ejecutada")
tipo = st.text_input("Tipo")
fuente = st.text_input("Fuente")
acta_informe = st.text_input("Acta de informe")

# Obtener las variables de entrada
X = pd.DataFrame({
    "tipo_identificacion": [tipo_identificacion],
    "tipo_clase_sujeto": [tipo_clase_sujeto],
    "comuna": [comuna],
    "barrio": [barrio],
    "fecha_programada": [fecha_programada],
    "fecha_ejecutada": [fecha_ejecutada],
    "tipo": [tipo],
    "fuente": [fuente],
    "acta_informe": [acta_informe]
})

# Obtener la predicción del modelo
prediction = model.predict(X)

# Mostrar la predicción del modelo
st.write("¿Cumple con las regulaciones?:", prediction)
