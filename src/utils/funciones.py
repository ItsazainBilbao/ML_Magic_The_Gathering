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

