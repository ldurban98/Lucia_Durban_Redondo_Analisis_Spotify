# ANÁLISIS DE POPULARIDAD CON LIGHTGBM

# Antes de ejecutar el código, instalar "pip install lightgbm shap" en el terminal

# Importar librerías
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv("DataBases\spotify_with_features.csv", decimal=",")
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

# Definir parámetros del modelo, siendo estos estables, estándar y defendibles
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

# Evaluación del modelo
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R²:", round(r2, 3))
print("RMSE:", round(rmse, 3))

# Vemos la importancia de variables
importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": model.feature_importance(importance_type="gain"),
})

importance["Importance_relativa"] = (
    importance["Importance"] / importance["Importance"].sum()
)

importance = importance.sort_values("Importance_relativa", ascending=False)

print(importance[["Variable", "Importance_relativa"]])


# Creamos el gráfico
plt.figure(figsize=(8, 6))

plt.barh(
    importance["Variable"],
    importance["Importance_relativa"]
)

plt.xlabel("Gain")
plt.ylabel("Variable")
plt.title("Feature Importance")
plt.gca().invert_yaxis()  # variable más importante arriba
plt.tight_layout()
plt.show()


# 8) SHAP
sample_X = X_train.sample(5000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_X)

plt.figure(figsize=(9, 6))

shap.summary_plot(
    shap_values,
    sample_X,
    plot_type="bar",
    show=False
)

plt.title("SHAP Value Summary Plot", fontsize=14)
plt.xlabel("SHAP")
plt.ylabel("Absolute average impact on popularity")

plt.tight_layout()
plt.show()