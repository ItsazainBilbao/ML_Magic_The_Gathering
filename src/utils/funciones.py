import re
import pandas as pd
import numpy as np
import ast

def split_type_line(type_line):
    if pd.isna(type_line):
        return '', ''
    
    # Separar caras si hay doble carta
    parts = type_line.split(' // ')
    
    main_types = []
    subtypes = []
    
    for part in parts:
        # Separar tipo principal y subtipo (si hay)
        if '—' in part:
            main, sub = part.split('—', 1)
        else:
            main, sub = part, ''
        
        # Limpiar y dividir
        main_parts = main.strip().split()
        sub_parts = [s.strip() for s in sub.strip().split()]
        
        # Reconstruir "Time Lord" si aparece separado
        i = 0
        while i < len(sub_parts):
            if i + 1 < len(sub_parts) and sub_parts[i] == 'Time' and sub_parts[i+1] == 'Lord':
                subtypes.append('Time Lord')
                i += 2
            else:
                subtypes.append(sub_parts[i])
                i += 1

        main_types.extend(main_parts)

    # Eliminar duplicados y unir por comas
    return ','.join(dict.fromkeys(main_types)), ','.join(dict.fromkeys(subtypes))

def clean_color_identity(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except:
            return '.i'
    if not x:  
        return '.i'
    return ','.join(x)

# Tipo de cambio USD -> EUR
USD_TO_EUR = 0.93

# Convertir string a dict
def parse_prices(prices_str):
    try:
        return ast.literal_eval(prices_str)
    except:
        return {}

# Lógica de prioridad de precios
def get_final_price(prices):
    eur = prices.get('eur')
    usd = prices.get('usd')
    eur_foil = prices.get('eur_foil')
    usd_foil = prices.get('usd_foil')
    usd_etched = prices.get('usd_etched')

    if eur and eur != 'None':
        return float(eur)
    
    if usd and usd != 'None':
        try:
            return round(float(usd) * USD_TO_EUR, 2)
        except:
            pass

    if eur_foil and eur_foil != 'None':
        return float(eur_foil)

    if usd_foil and usd_foil != 'None':
        try:
            return round(float(usd_foil) * USD_TO_EUR, 2)
        except:
            pass

    if usd_etched and usd_etched != 'None':
        try:
            return round(float(usd_etched) * USD_TO_EUR, 2)
        except:
            pass

    return None

# Formatos que nos interesan
target_formats = ['standard', 'pioneer', 'modern', 'legacy', 'vintage', 'commander', 'pauper']

# Función para convertir string a dict
def parse_legalities(legalities_str):
    try:
        return ast.literal_eval(legalities_str)
    except:
        return {}

# Función para mapear legalidad a número
def map_legal_status(status):
    if status == 'legal':
        return 1
    elif status == 'not_legal':
        return 0
    elif status == 'restricted':
        return -1
    elif status == 'banned':
        return -2
    else:
        return None  # por si aparece algo inesperado
    
    
from collections import Counter

def contar_subtipos(columna, sep=','):
    """
    Recibe una Serie de pandas (columna del DataFrame),
    separa las etiquetas por sep (coma por defecto), y cuenta cuántas veces aparece cada una.
    
    Devuelve un DataFrame ordenado por frecuencia descendente.
    """
    # Asegurar valores string y separar por coma
    etiquetas = columna.dropna().astype(str).apply(lambda x: [t.strip() for t in x.split(sep) if t.strip()])
    
    # Aplanar y contar
    contador = Counter([etiqueta for sublist in etiquetas for etiqueta in sublist])
    
    # Convertir a DataFrame
    df_frecuencias = pd.DataFrame.from_dict(contador, orient='index', columns=['frecuencia']).sort_values(by='frecuencia', ascending=False)
    df_frecuencias.index.name = 'subtipo'
    df_frecuencias.reset_index(inplace=True)
    
    return df_frecuencias

def contar_manas(valor):
    if pd.isna(valor):
        return 0
    try:
        mana_list = ast.literal_eval(valor)
        if isinstance(mana_list, list):
            return len(mana_list)
    except:
        pass
    return 0

import re


# Lista de habilidades de evasión comunes
evasion_keywords = {
    'Flying', 'Fear', 'Intimidate', 'Horsemanship', 'Skulk',
    'Shadow', 'Menace', 'Protection', 'Unblockable',
    'Burrowing', 'Daunt', 'Nimble', 'Gingerbrute', "Trample"
}

# Función para determinar si una carta tiene al menos una habilidad de evasión
def tiene_evasion(keywords_str):
    try:
        keywords = ast.literal_eval(keywords_str)
        if not isinstance(keywords, list):
            return 0
    except (ValueError, SyntaxError):
        return 0
    for kw in keywords:
        if kw in evasion_keywords or 'walk' in kw.lower():
            return 1
    return 0


## Busco en oracle text también
oracle_evasion_patterns = [
    r"\bcan't be blocked\b",
    r"\bis unblockable\b",
    r"\bhas landwalk\b",          
    r"\w+walk",                   
]

def detectar_evasion(row):
    # --- 1. Keywords ---
    keywords_raw = row.get('keywords', '[]')
    try:
        keywords = ast.literal_eval(keywords_raw)
        if not isinstance(keywords, list):
            keywords = []
    except:
        keywords = []

    for kw in keywords:
        if kw in evasion_keywords or 'walk' in kw.lower():
            return 1

    # --- 2. Oracle Text ---
    oracle_text = row.get('oracle_text', '')
    if isinstance(oracle_text, str):
        for pattern in oracle_evasion_patterns:
            if re.search(pattern, oracle_text, flags=re.IGNORECASE):
                return 1

    return 0


def parse_stat(val):
    if pd.isna(val):
        return np.nan  # mantener NaN original, lo trataremos luego
    try:
        # Caso 1: valor convertible directamente
        return float(val)
    except:
        pass

    # Caso 2: reemplazar '*' por 1, luego intentar evaluar
    try:
        expr = re.sub(r'\*', '1', val)
        if re.match(r'^[\d\.\+\-\s]+$', expr):  # solo permitir operaciones simples
            return eval(expr)
    except:
        pass

    return "moda"  # marcamos para reemplazar por la moda

def procesar_columna(col):
    # Paso 1: aplicar parser
    parsed = col.apply(parse_stat)

    # Paso 2: separar valores marcados como "moda"
    parsed_for_mode = parsed[parsed != "moda"]
    numeric_values = parsed_for_mode.dropna().astype(float)

    moda_val = numeric_values.mode().iloc[0] if not numeric_values.empty else 0
    min_val = numeric_values.min() if not numeric_values.empty else -1

    # Paso 3: reemplazar
    parsed = parsed.replace("moda", moda_val)
    parsed = parsed.astype(float)
    parsed = parsed.fillna(min_val - 1)  # solo los NaN originales

    return parsed