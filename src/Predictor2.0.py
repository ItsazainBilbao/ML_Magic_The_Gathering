import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from model.MultiModelRegressor import MultiModelRegressor

import os
os.system('cls' if os.name == 'nt' else 'clear')

# Request para que pille el nombre en castellano
import requests

def obtener_nombre_ingles(nombre_es):
    url = f"https://api.scryfall.com/cards/named?lang=es&fuzzy={nombre_es}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json().get("name", nombre_es)
    else:
        return nombre_es  # si falla, devuelves el original

# 0. EL banner

banner = r"""
            _____     _____        ___  _.     __  ____  ____.          _      
           `~-,  \   `~-,  \       ~-_~~ \    /  ~~   / (    /' ,-~~~~~~,'     
             /    \    /'   \      /~>\   \  (  /~~~\/   )  | .'  /~~~)/       
            /      \  /      \    / /  \   \ | | ~==~~|  |  | |  (    '        
           /',/\    \/ '/\    \  / <_,'>\   \ \ \___) |  |  | |   \_  _____    
          /  /  \      /  \    \/     /  \   \ \  __  |  )  ( _`-_  ~~ ____]=='
         /  /    \     |   \    \~~~~~    `~~~~ ~~  ~\|(=~~~~~'   `~~~~___.   .
        /  /      \     \   \    \    ___         _                     | |\ /|
       /  /        |     |   \    \   `|''|      / '  _._|      .,_  _  | | V |
      /  /          \     \   \    \   |  |^ %) | __'T T |^ %)/^|| )/ \        
     /  (            |     |   \    \ .|. | |\,  \_|(] | | |\,[ || |\_/        
    /    \,           \     \   \    \             '                 \         
   /___---'            | ,-'~   .)    `.                            (~) Card Price Predictor       
.=~~                   `~       `~~~~~~~'                            ~         
                           """

banner2 = r"""
                           ..:::a:::::..                 
                      ..:::::::d8::::::::::..            
                   .::::::::::d88b:::::::::::::.         
                 .:::::::::::d8888:::::::::::::::.       
               .::::::::::::d88888b::::::::::::::::.     
              :::::::::::::d8888888::::::::::::::::::    
             ::::::::::::d888888888b::::::::::::::::::   
            ::::::::::::d88888888888b::::::::::::::::::  
           .:::::::::::d88888888888888b::::::::::::::::. 
           :::::::::::d888888888888888888a:::::::::::::: 
           ::::::::::d888888888888888b:Y88b::::::::::::: 
           :::::::::d88888888888888888b:888b:::::::::::: 
           `::::::::8888888888888888888:Y888b::::::::::' 
            ::::::::8888888888888888888::Y888::::::::::  
             :::::::Y888888888888888888?:d88P:::::::::   
              :::::::Y88888888888888888bd88P:::::::::    
               `:::::::Y88888888888888888P:::::::::'     
                 `::::::Y88888888888888P:::::::::'       
                   `::::::Y8888888888P:::::::::'         
                      ``::::::Y888P::::::::''            
                           ``:::::::::''
"""

print(banner)

# 1. Cargar modelo
with open("model/production/mediofinal.pkl", "rb") as f:
    model = pickle.load(f)

# Después de cargar el modelo, asegúrate de que está completamente entrenado.
assert model.named_steps['model'].model_cheap is not None, "Modelo cheap no entrenado"
assert model.named_steps['model'].model_mid is not None, "Modelo mid no entrenado"
assert model.named_steps['model'].model_exp is not None, "Modelo expensive no entrenado"

# 2. Cargar dataset con datos ya procesados
df = pd.read_csv("data/dataCardsPCAded.csv")  # Asegúrate de tenerlo listo
os.system('cls' if os.name == 'nt' else 'clear')


# 3. Loop de predicción
while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    print(banner)
    print("\n--- Predicción de precio de carta ---") 
    nombre_input  = input("Nombre de la carta: ").strip()
    name = obtener_nombre_ingles(nombre_input)
    foil = input("¿Es foil? (y/n): ").strip().lower() == "y"
    fullart = input("¿Es fullart? (y/n): ").strip().lower() == "y"
    os.system('cls' if os.name == 'nt' else 'clear')
    print(banner)
    # Filtrar cartas
    filas = df[
        (df['name'].str.lower() == name.lower()) &
        (df['foil'] == foil) &
        (df['full_art'] == fullart)
    ]

    # Si no encuentra con esas condiciones, buscar solo por nombre
    if filas.empty:        
        filas = df[
            (df['name'].str.lower() == name.lower())
        ]
        
        # Si aún no se encuentra
        if filas.empty:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(banner)
            print("\nNo se ha encontrado la carta, igual la has escrito mal")
            continue        

        # os.system('cls' if os.name == 'nt' else 'clear')
        # print(banner)
        # print(f"\nLa predicción se ha hecho con una carta que puede ser o no ser foil o fullart.\n \
        #       Si no la encuentra con esos dos parámetros exactos pilla la que le valga.")


    # Inicializar variables para almacenar la mejor carta
    mejor_diferencia = float('inf')  # Inicia con un valor muy grande
    mejor_fila = None
    mejor_prediccion = None

    # 4. Iterar sobre todas las cartas encontradas y hacer predicciones
    for idx, fila in filas.iterrows():
        # Eliminar las columnas que no se usan
        X_row = fila.drop(columns=["final_price_eur", "log_price", "name", "oracle_text"])

        # Convertir X_row a un DataFrame con el mismo formato que las entradas del modelo
        X_row = X_row.to_frame().T  

        # Predecir con el pipeline
        log_preds = model.predict(X_row)[0]  # [cheap, mid, expensive]

        # Obtener valores reales (deshacer log)
        preds = np.expm1(log_preds)
        real_price = np.expm1(fila["log_price"])

        # Comparar y calcular el error
        errores = np.abs(preds - real_price)
        idx_mejor = np.argmin(errores)

        # Verificar si esta carta tiene una mejor predicción
        if errores[idx_mejor] < mejor_diferencia:
            mejor_diferencia = errores[idx_mejor]
            mejor_fila = fila
            mejor_prediccion = preds[idx_mejor]

    # 5. Mostrar resultados        
    print(f"\nCarta: {mejor_fila['name']}")
    print(f"Es foil: {foil} || Es Fullart: {fullart}")
    print(f"Precio real:      {np.expm1(mejor_fila['log_price']):.2f} €")    
    print(f"Precio Predict: {mejor_prediccion:.2f} €")
    print(banner2)

    otra = input("\n¿Quieres predecir otra carta? (y/n): ").strip().lower()
    os.system('cls' if os.name == 'nt' else 'clear')
    if otra != "y":
        break
