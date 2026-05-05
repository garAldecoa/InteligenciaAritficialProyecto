import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# =========================================
# 1. Cargar datos limpios (ya preprocesados)
# =========================================
X = pd.read_csv("X_preprocesado.csv")
y = pd.read_csv("y_precio.csv").squeeze()  # convertir en Series

print("Datos cargados correctamente.")
print(f"Dimensiones X: {X.shape}")
print(f"Dimensiones y: {len(y)}")
print("\nPrimeras filas de X (limpia):")
print(X.head())
print("\nPrimeros valores de y:")
print(y.head())

# =========================================
# 2. Dividir en entrenamiento y prueba
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTamaño entrenamiento: {X_train.shape}")
print(f"Tamaño prueba: {X_test.shape}")

# =========================================
# 3. Crear y entrenar modelos
# =========================================
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

print("\nEntrenando Linear Regression...")
linear_model.fit(X_train, y_train)

print("Entrenando Random Forest Regressor...")
random_forest_model.fit(X_train, y_train)

# =========================================
# 4. Predicciones
# =========================================
y_pred_lr = linear_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)

# =========================================
# 5. Función de evaluación
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
# 6. Evaluar
# =========================================
result_lr = evaluar_modelo("Linear Regression", y_test, y_pred_lr)
result_rf = evaluar_modelo("Random Forest Regressor", y_test, y_pred_rf)

# =========================================
# 7. Comparativa
# =========================================
results_df = pd.DataFrame([result_lr, result_rf])
print("\nComparación final de modelos:")
print(results_df)

mejor_modelo = results_df.loc[results_df["R2"].idxmax()]
print("\nMejor modelo según R2:")
print(f"{mejor_modelo['Modelo']} con R2 = {mejor_modelo['R2']:.4f}")

# =========================================
# 8. Gráficas comparativas
# =========================================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.title("Linear Regression")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.title("Random Forest")

plt.tight_layout()
plt.show()