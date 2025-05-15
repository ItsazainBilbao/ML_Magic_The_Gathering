import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from model.MultiModelRegressor import MultiModelRegressor

# 1. Cargar modelo
with open("model/production/mediofinal.pkl", "rb") as f:
    model = pickle.load(f)

# Después de cargar el modelo, asegúrate de que está completamente entrenado.
assert model.named_steps['model'].model_cheap is not None, "Modelo cheap no entrenado"
assert model.named_steps['model'].model_mid is not None, "Modelo mid no entrenado"
assert model.named_steps['model'].model_exp is not None, "Modelo expensive no entrenado"

# 2. Cargar dataset con datos ya procesados
df = pd.read_csv("data/dataCardsPCAded.csv")  # Asegurate de tenerlo listo

# 3. Loop de predicción
while True:
    print("\n--- Predicción de precio de carta ---") 
    name = input("Nombre de la carta (en inglés): ").strip()
    foil = input("¿Es foil? (y/n): ").strip().lower() == "y"
    fullart = input("¿Es fullart? (y/n): ").strip().lower() == "y"

    # Filtrar carta
    fila = df[
        (df['name'].str.lower() == name.lower()) &
        (df['foil'] == foil) &
        (df['full_art'] == fullart)
    ]

    if fila.empty:
        print("No ha encontrado la carta, igual la has escrito mal")
        continue

    # Extraer datos
    row = fila.iloc[0]
    X_row = fila.drop(columns=["final_price_eur", "log_price", "name", "oracle_text"])

    # 4. Predecir con el pipeline
    log_preds = model.predict(X_row)[0]  # [cheap, mid, expensive]

    # 5. Obtener valores reales (deshacer log)
    preds = np.expm1(log_preds)
    real_price = np.expm1(row["log_price"])

    # 6. Comparar y elegir el mejor modelo
    errores = np.abs(preds - real_price)
    mejores_modelos = ['cheap', 'mid', 'expensive']
    idx_mejor = np.argmin(errores)
    mejor_modelo = mejores_modelos[idx_mejor]

    # 7. Mostrar resultados
    print(f"\nCarta: {row['name']}")
    print(f"Precio real:      {real_price:.2f} €")    
    print(f"Precio Predict: {preds[idx_mejor]:.2f} €")

    otra = input("\n¿Querés predecir otra carta? (y/n): ").strip().lower()
    if otra != "y":
        break
