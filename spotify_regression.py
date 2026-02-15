# REGRESIÓN LINEAL MÚLTIPLE SPOTIFY

# Antes de ejecutar el código, instalar "pip install statsmodels" y "pip install jinja2" en el terminal
import pandas as pd
import statsmodels.api as sm

# Cargar los datos
df = pd.read_csv("DataBases\spotify_with_features.csv", decimal=",")

# Definir las variables
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

# Añadimos la constante
X = sm.add_constant(X)

# Estimar modelo OLS
model = sm.OLS(y, X).fit()

# Función para estrellas
def significance_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return ""
    
# Construir tabla de resultados
rows = []
for var in model.params.index:
    coef = model.params[var]
    se = model.bse[var]
    pval = model.pvalues[var]

    coef_str = f"{coef:.3f}{significance_stars(pval)}"
    se_str = f"({se:.3f})"

    rows.append([coef_str, se_str])
table = pd.DataFrame(
    rows,
    index=model.params.index,
    columns=["Coeficiente", "Error estándar"],
)

# Mostrar tabla de resultados
print("\nTABLA DE REGRESIÓN (DEPENDIENTE: POPULARITY)\n")
print(table)

# Información adicional de R²,R² ajustado y Obervaciones
print("\nR²:", round(model.rsquared, 3))
print("R² ajustado:", round(model.rsquared_adj, 3))
print("Observaciones:", int(model.nobs))

# Exportar resultados a CSV
table.to_excel("tabla_regresion_spotify.xlsx")
latex_table = table.to_latex(
    caption="Regresión lineal múltiple (variable dependiente: Popularity)",
    label="tab:spotify_regression",
    escape=False
)
with open("tabla_regresion_spotify.tex", "w") as f:
    f.write(latex_table)