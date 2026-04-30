import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# =========================================
# 1. Cargar dataset
# =========================================
df = pd.read_csv("Housing.csv")

print("Primeras filas del dataset:")
print(df.head())
print("\nInformación general:")
print(df.info())

# =========================================
# 2. Separar variable objetivo y entradas
# =========================================
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# =========================================
# 3. Identificar columnas numéricas y categóricas
# =========================================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

print("\nColumnas numéricas:")
print(list(numeric_features))

print("\nColumnas categóricas:")
print(list(categorical_features))

# =========================================
# 4. Preprocesamiento
# =========================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# =========================================
# 5. Dividir datos en entrenamiento y prueba
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 6. Crear modelos
# =========================================
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

random_forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# =========================================
# 7. Entrenar modelos
# =========================================
print("\nEntrenando Linear Regression...")
linear_model.fit(X_train, y_train)

print("Entrenando Random Forest Regressor...")
random_forest_model.fit(X_train, y_train)

# =========================================
# 8. Hacer predicciones
# =========================================
y_pred_lr = linear_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)

# =========================================
# 9. Función para evaluar modelos
# =========================================
def evaluar_modelo(nombre, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_real, y_pred)

    print(f"\nResultados de {nombre}:")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    return {
        "Modelo": nombre,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# =========================================
# 10. Evaluar ambos modelos
# =========================================
result_lr = evaluar_modelo("Linear Regression", y_test, y_pred_lr)
result_rf = evaluar_modelo("Random Forest Regressor", y_test, y_pred_rf)

# =========================================
# 11. Tabla comparativa
# =========================================
results_df = pd.DataFrame([result_lr, result_rf])

print("\nComparación final de modelos:")
print(results_df)

# =========================================
# 12. Elegir mejor modelo automáticamente
# =========================================
mejor_modelo = results_df.loc[results_df["R2"].idxmax()]

print("\nMejor modelo según R2:")
print(f"{mejor_modelo['Modelo']} con R2 = {mejor_modelo['R2']:.4f}")

# =========================================
# 13. Gráficas comparativas en una ventana
# =========================================
plt.figure(figsize=(14, 6))

# Gráfica 1: Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.title("Linear Regression")

# Gráfica 2: Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.title("Random Forest")

# Ajustar espacios
plt.tight_layout()
plt.show()