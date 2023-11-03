# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Importar el dataset
df = pd.read_csv("Manejo_residuos_peligrosos_Palmira_preprocessed.csv")

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df, df["de_cumplimiento"], test_size=0.25)

# Entrenar el modelo
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluar el modelo
score = clf.score(X_test, y_test)
print(score)

# Guardar el modelo
pickle.dump(clf, open("model.pkl", "wb"))

# Cargar el modelo
clf = pickle.load(open("model.pkl", "rb"))

# Predecir el cumplimiento
prediccion = clf.predict([[tipo_identificacion_del_sujeto, tipo_clase_de_sujeto, comuna, barrio, fecha_programada, tipo, fuente]])[0]

# Mostrar el resultado
print(prediccion)
