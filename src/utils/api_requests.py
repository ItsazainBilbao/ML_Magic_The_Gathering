import requests
import pandas as pd

def hacer_request():
    bulk_url = "https://api.scryfall.com/bulk-data"
    response = requests.get(bulk_url).json()
    for item in response["data"]:
        if item["type"] == "default_cards":
            json_url = item["download_uri"]
            print("Descargando desde:", json_url)

            # Descargar el archivo JSON con todas las cartas
            cards_data = requests.get(json_url).json()

            # Convertir a DataFrame
            df_sucio = pd.DataFrame(cards_data)
            df_sucio.to_csv("../data/scryfall_cards.csv", index=False)
            print("Archivo guardado como 'scryfall_cards.csv'")
            break
