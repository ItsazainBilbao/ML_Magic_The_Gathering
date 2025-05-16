import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import pickle
import requests
from io import BytesIO
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

from model.MultiModelRegressor import MultiModelRegressor

# Cargar modelo
with open("model/production/mediofinal.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar datos
df = pd.read_csv("data/dataCardsPCAded.csv")

# Rutas de imágenes
BANNER_PATH = os.path.join("pics/art/", "banner.jpg")
BACK_IMG_PATH = os.path.join("pics/art", "back.jpg")

# Crear ventana principal
root = tk.Tk()
root.title("Predictor3.0")
root.resizable(False, False)

# Imagen de dorso (placeholder)
if os.path.exists(BACK_IMG_PATH):
    placeholder_img = Image.open(BACK_IMG_PATH).resize((200, 280), Image.Resampling.LANCZOS)
    placeholder_tk = ImageTk.PhotoImage(placeholder_img)
else:
    placeholder_tk = None
    print("No se encontró back.png en /pics/art")

# Banner
if os.path.exists(BANNER_PATH):
    banner_img = Image.open(BANNER_PATH).resize((600, 250), Image.Resampling.LANCZOS)
    banner_tk = ImageTk.PhotoImage(banner_img)
    label_banner = tk.Label(root, image=banner_tk)
    label_banner.grid(column=0, row=0, columnspan=3, pady=10)
else:
    print("No se encontró banner.jpg en /pics/art")

# Etiqueta para imagen (con placeholder por defecto)
label_imagen = tk.Label(root, image=placeholder_tk)
label_imagen.imgtk = placeholder_tk
label_imagen.grid(column=2, row=1, rowspan=5, padx=10)

# Etiqueta nombre
ttk.Label(root, text="Nombre de la carta:").grid(column=0, row=1, sticky="w")
entry_nombre = ttk.Entry(root, width=40)
entry_nombre.grid(column=1, row=1, padx=5, pady=5)
entry_nombre.bind("<Return>", lambda event: predecir())

# Checkboxes
var_foil = tk.BooleanVar()
var_fullart = tk.BooleanVar()
ttk.Checkbutton(root, text="Foil", variable=var_foil).grid(column=0, row=2, sticky="w")
ttk.Checkbutton(root, text="Fullart", variable=var_fullart).grid(column=1, row=2, sticky="w")

# Resultado
label_resultado = ttk.Label(root, text="", justify="left", font=("Arial", 14))
label_resultado.grid(column=0, row=4, columnspan=2, sticky="w", padx=5)

# Función para obtener nombre en inglés
def obtener_nombre_ingles(nombre_es):
    url = f"https://api.scryfall.com/cards/named?lang=es&fuzzy={nombre_es}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.json().get("name", nombre_es)
    return nombre_es

# Función para obtener imagen
def obtener_url_imagen(nombre):
    url = f"https://api.scryfall.com/cards/named?fuzzy={nombre}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        return data["image_uris"]["normal"] if "image_uris" in data else None
    return None

# Función principal
def predecir():
    nombre_input = entry_nombre.get().strip()
    foil = var_foil.get()
    fullart = var_fullart.get()

    name = obtener_nombre_ingles(nombre_input)

    filas = df[
        (df['name'].str.lower() == name.lower()) &
        (df['foil'] == foil) &
        (df['full_art'] == fullart)
    ]

    if filas.empty:
        filas = df[df['name'].str.lower() == name.lower()]
        if filas.empty:
            label_resultado.config(text="Carta no encontrada.")
            label_imagen.imgtk = placeholder_tk
            label_imagen.configure(image=placeholder_tk)
            return

    mejor_diferencia = float("inf")
    mejor_fila = None
    mejor_pred = None

    for _, fila in filas.iterrows():
        X_row = fila.drop(columns=["final_price_eur", "log_price", "name", "oracle_text"])
        X_row = X_row.to_frame().T
        log_preds = model.predict(X_row)[0]
        preds = np.expm1(log_preds)
        real_price = np.expm1(fila["log_price"])
        errores = np.abs(preds - real_price)
        idx_mejor = np.argmin(errores)
        if errores[idx_mejor] < mejor_diferencia:
            mejor_diferencia = errores[idx_mejor]
            mejor_fila = fila
            mejor_pred = preds[idx_mejor]

    texto = f"Carta: {mejor_fila['name']}\n" \
            f"Precio real: {np.expm1(mejor_fila['log_price']):.2f} €\n" \
            f"Predicción: {mejor_pred:.2f} €"
    label_resultado.config(text=texto)

    # Actualizar imagen de la carta
    try:
        url_img = obtener_url_imagen(name)
        if url_img:
            img_data = requests.get(url_img).content
            img = Image.open(BytesIO(img_data)).resize((200, 280), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            label_imagen.imgtk = img_tk
            label_imagen.configure(image=img_tk)
        else:
            label_imagen.imgtk = placeholder_tk
            label_imagen.configure(image=placeholder_tk)
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        label_imagen.imgtk = placeholder_tk
        label_imagen.configure(image=placeholder_tk)

# Botón de predicción
btn = ttk.Button(root, text="Predecir precio", command=predecir)
btn.grid(column=0, row=3, columnspan=2, pady=10)

# Iniciar aplicación
root.mainloop()
