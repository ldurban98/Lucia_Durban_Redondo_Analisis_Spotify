# MODELO PREDICTIVO

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# Antes de generar canciones aleatorias, copiamos el modelo entrenado
# Cargar los datos
df = pd.read_csv("DataBases\\spotify_with_features.csv", decimal=",")
df = df.dropna()

# Definir las variables (dependiente y explicativas)
y = df["popularity"]

X = df[
    [
        "explicit",
        "key",
        "mode",
        "danceability",
        "loudness",
        "speechiness",
        "energy",
        "valence",
        "acousticness",
        "tempo",
        "instrumentalness",
        "liveness",
    ]
]

# Hacer el test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Crear data sets LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Definir parámetros del modelo, siendo estos estabñes, estándar y defendibles
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}

# Entrenamos el modelo con early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100),
    ],
)

# Generamos canciones aleatorias pero realistas
def generar_canciones_ficticias(n_canciones=10, random_state=42):
    np.random.seed(random_state)

    data = {
        "explicit": np.random.randint(0, 2, n_canciones),
        "key": np.random.randint(0, 12, n_canciones),
        "mode": np.random.randint(0, 2, n_canciones),
        "danceability": np.random.uniform(0, 1, n_canciones),
        "loudness": np.random.uniform(-60, 0, n_canciones),
        "speechiness": np.random.uniform(0, 1, n_canciones),
        "energy": np.random.uniform(0, 1, n_canciones),
        "valence": np.random.uniform(0, 1, n_canciones),
        "acousticness": np.random.uniform(0, 1, n_canciones),
        "tempo": np.random.uniform(60, 200, n_canciones),
        "instrumentalness": np.random.uniform(0, 1, n_canciones),
        "liveness": np.random.uniform(0, 1, n_canciones),
    }

    canciones = pd.DataFrame(data)
    canciones.index = [f"Song_{i+1}" for i in range(n_canciones)]

    return canciones

# Generar 100 canciones simuladas
canciones_simuladas = generar_canciones_ficticias(n_canciones=10000)

# Asegurar mismo orden de columnas
canciones_simuladas = canciones_simuladas[X.columns]

# Predicción
canciones_simuladas["popularidad_predicha"] = model.predict(canciones_simuladas)

# Guardar resultados en CSV
canciones_simuladas.to_csv(
    "canciones_simuladas_prediccion_popularidad.csv",
    index=True,
    float_format="%.2f",
    decimal="."
)


# Ordenar por canciones más prometedores
canciones_simuladas.sort_values(
    "popularidad_predicha",
    ascending=False
).head(10)