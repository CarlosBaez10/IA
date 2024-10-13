import pandas as pd
import numpy as np

# Supervisado
# Crear datos simulados de rutas de transporte
np.random.seed(0)

# Simular estaciones de transporte
stations = ['Station_A', 'Station_B', 'Station_C', 'Station_D', 'Station_E']

# Simular usuarios con su estación de inicio, fin, tiempo y si hubo retraso
data = {
    'user_id': np.arange(1, 101),
    'start_station': np.random.choice(stations, 100),
    'end_station': np.random.choice(stations, 100),
    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], 100),
    'duration_minutes': np.random.randint(5, 60, size=100),
    'delay': np.random.choice([0, 1], size=100, p=[0.85, 0.15]),  # 15% de retraso
    'weather_condition': np.random.choice(['clear', 'rainy', 'snowy', 'cloudy'], 100),
    'traffic_condition': np.random.choice(['low', 'medium', 'high'], 100)
}

# Crear el dataframe
df = pd.DataFrame(data)
print(df.head())

# Importar las bibliotecas necesarias
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Definir las características (X) y el objetivo (y)
X = df[['start_station', 'end_station', 'time_of_day', 'weather_condition', 'traffic_condition']]
y = df['duration_minutes']

# Preprocesamiento: codificar las variables categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['start_station', 'end_station', 'time_of_day', 'weather_condition', 'traffic_condition'])
    ])

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un pipeline para preprocesar y ajustar el modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")

#------------------------------------------------------#
# No supervisado

# Simular estaciones de transporte
stations = ['Station_A', 'Station_B', 'Station_C', 'Station_D', 'Station_E']

# Crear datos simulados de rutas, duración y número de pasajeros
data = {
    'start_station': np.random.choice(stations, 200),
    'end_station': np.random.choice(stations, 200),
    'duration_minutes': np.random.randint(5, 60, size=200),
    'num_passengers': np.random.randint(1, 50, size=200),
    'weather_condition': np.random.choice(['clear', 'rainy', 'cloudy'], 200),
    'traffic_condition': np.random.choice(['low', 'medium', 'high'], 200)
}

# Crear el dataframe
df = pd.DataFrame(data)
print(df.head())

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Codificar las variables categóricas
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'])
df['traffic_encoded'] = le_traffic.fit_transform(df['traffic_condition'])

# Seleccionar las características para el clustering
X = df[['duration_minutes', 'num_passengers', 'weather_encoded', 'traffic_encoded']]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar el algoritmo K-Means para 3 clusters (puedes ajustar el número de clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Agregar las etiquetas de los clusters al dataframe original
df['cluster'] = kmeans.labels_

# Visualización de los clusters (usando las dos primeras características)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Clustering de Rutas de Transporte')
plt.xlabel('Duración del Viaje (escalada)')
plt.ylabel('Número de Pasajeros (escalada)')
plt.show()

print(df.head())