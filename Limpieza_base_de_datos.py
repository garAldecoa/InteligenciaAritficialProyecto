import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def limpiar_y_preparar_datos(ruta_archivo='Housing.csv', escalar_numericas=True, devolver_scaler=False):
    """
    Carga y limpia el dataset de viviendas.
    
    Parámetros:
    - ruta_archivo: ruta al archivo CSV.
    - escalar_numericas: Si True, estandariza las variables numéricas (media=0, std=1).
    - devolver_scaler: Si True, devuelve también el scaler ajustado (solo útil si escalar_numericas=True).
    
    Retorna:
    - X: DataFrame con las características procesadas.
    - y: Serie con el precio (sin escalar).
    - scaler (opcional): solo si devolver_scaler=True.
    """
    # 1. Cargar datos
    df = pd.read_csv(ruta_archivo)
    print("✅ Datos cargados correctamente. Dimensiones:", df.shape)
    
    # 2. Separar variable objetivo (price)
    X = df.drop('price', axis=1)
    y = df['price']
    
    # 3. Identificar columnas numéricas y categóricas
    numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                         'airconditioning', 'prefarea', 'furnishingstatus']
    
    # Verificar existencia
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    
    print(f"Variables numéricas a procesar ({len(numeric_cols)}): {numeric_cols}")
    print(f"Variables categóricas a procesar ({len(categorical_cols)}): {categorical_cols}")
    
    # 4. Preprocesamiento
    X_processed = pd.DataFrame(index=X.index)
    scaler = None
    
    # Escalado de numéricas
    if escalar_numericas and numeric_cols:
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X[numeric_cols])
        X_num_df = pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=X.index)
        X_processed = pd.concat([X_processed, X_num_df], axis=1)
    else:
        X_processed = pd.concat([X_processed, X[numeric_cols]], axis=1)
    
    # One-Hot Encoding (drop_first)
    for col in categorical_cols:
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=int)
        X_processed = pd.concat([X_processed, dummies], axis=1)
    
    print(f"\n📊 Resumen después de limpieza:")
    print(f"  - Total características: {X_processed.shape[1]}")
    print(f"  - Muestras: {X_processed.shape[0]}")
    print("\nPrimeras 5 filas de X procesada:")
    print(X_processed.head())
    print(f"\nPrecios (y) - primeros 5 valores: {y.head().values}")
    
    if devolver_scaler and escalar_numericas and numeric_cols:
        return X_processed, y, scaler
    else:
        return X_processed, y

# Ejemplo de uso (esto ya no dará error)
if __name__ == "__main__":
    # Ahora siempre devuelve 2 valores, independientemente de escalar_numericas
    X_clean, y_clean = limpiar_y_preparar_datos(r"D:\archivos UDLAP\honors\Inteligencia_Artificial\InteligenciaAritficialProyecto\Housing.csv", escalar_numericas=True)
    
    # Guardar los datos limpios (opcional)
    X_clean.to_csv('X_preprocesado.csv', index=False)
    y_clean.to_csv('y_precio.csv', index=False)
    print("\n💾 Datos limpios guardados: 'X_preprocesado.csv' y 'y_precio.csv'")