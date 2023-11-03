# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import streamlit as st

# Importar el dataset
df = pd.read_csv("Manejo_residuos_peligrosos_Palmira_preprocessed.csv")

# Eliminar valores faltantes
df.dropna(inplace=True)

# Codificar variables categóricas
df["tipo_identificacion_del_sujeto"] = df["tipo_identificacion_del_sujeto"].astype("category")
df["tipo_clase_de_sujeto"] = df["tipo_clase_de_sujeto"].astype("category")

# Normalizar los datos
df["comuna"] = df["comuna"]/df["comuna"].max()
df["barrio"] = df["barrio"]/df["barrio"].max()

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df, df["de_cumplimiento"], test_size=0.25)

# Entrenar el modelo
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluar el modelo
score = clf.score(X_test, y_test)
print(score)

# Guardar el modelo
pickle.dump(clf, open("model.pkl", "wb"))

# Cargar el modelo
clf = pickle.load(open("model.pkl", "rb"))

# Crear la aplicación de Streamlit
st.title("Predicción de incumplimiento de normas de manejo de residuos peligrosos")

# Seleccionar los datos de entrada
tipo_identificacion_del_sujeto = st.selectbox("Tipo de identificación del sujeto", df["tipo_identificacion_del_sujeto"].unique())
tipo_clase_de_sujeto = st.selectbox("Tipo de clase de sujeto", df["tipo_clase_de_sujeto"].unique())
comuna = st.selectbox("Comuna", df["comuna"].unique())
barrio = st.selectbox("Barrio", df["barrio"].unique())
fecha_programada = st.date_input("Fecha programada")
tipo = st.selectbox("Tipo de visita", df["tipo"].unique())
fuente = st.selectbox("Fuente de los residuos peligrosos", df["fuente"].unique())

# Predecir el cumplimiento
prediccion = clf.predict([[tipo_identificacion_del_sujeto, tipo_clase_de_sujeto, comuna, barrio, fecha_programada, tipo, fuente]])[0]

# Mostrar el resultado
st.write("El riesgo de incumplimiento es de", prediccion)
