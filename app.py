import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
@st.cache
def load_data():
    data = pd.read_csv("Manejo_residuos_peligrosos_Palmira_preprocessed.csv")
    return data

data = load_data()

# Título de la aplicación
st.title('Plataforma de Evaluación de Riesgos Ambientales en Palmira')

# Análisis de Riesgos
st.header('Análisis de Riesgos')

# Selecciona características y variable objetivo
features = st.multiselect("Selecciona características:", data.columns)
target_variable = st.selectbox("Selecciona la variable objetivo:", data.columns)

X = data[features]
y = data[target_variable]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
st.subheader('Entrenamiento del Modelo')
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Precisión del modelo: {accuracy:.2f}')

# Gráfico de precisión
st.subheader('Gráfico de Precisión')
st.text("El gráfico de precisión muestra la precisión del modelo.")
fig, ax = plt.subplots()
sns.histplot(accuracy, bins=10, ax=ax)
st.pyplot(fig)

# Mostrar resultados, simulación de escenarios, recomendaciones, etc.
# Agrega secciones adicionales según tus necesidades

# Visualización de datos
st.header('Visualización de Datos')
st.write(data)

# Pregunta predictiva
st.header('Pregunta Predictiva')
st.write("¿Cuál es el riesgo de eventos ambientales adversos en Palmira y cómo pueden mitigirse a través de medidas preventivas, regulaciones y cambios en la infraestructura?")

# Nota: Extiende y personaliza este código según tus necesidades específicas
