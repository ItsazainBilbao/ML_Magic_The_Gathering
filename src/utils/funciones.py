import re
import pandas as pd
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
    elif status == 'banned':
        return -1
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