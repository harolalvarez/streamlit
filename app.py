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

# Visualización de datos
st.header('Visualización de Datos')

# Mostrar datos generales
st.write("Datos Generales:")
st.write(data)

# Identificar el sector que genera más contaminación
sector_contaminacion = data['sector'].value_counts().idxmax()
st.write(f"Sector que genera más contaminación: {sector_contaminacion}")

# Gráfico de barras para visualizar la contaminación por sector
st.subheader('Contaminación por Sector')
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='sector')
plt.xticks(rotation=45)
st.pyplot(plt)

# Pregunta predictiva
st.header('Pregunta Predictiva')
st.write("¿Cuál es el riesgo de eventos ambientales adversos en Palmira y cómo pueden mitigirse a través de medidas preventivas, regulaciones y cambios en la infraestructura?")

# Nota: Extiende y personaliza este código según tus necesidades específicas

