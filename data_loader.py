# data_loader.py
import os
import cv2
import numpy as np
import pandas as pd

def load_covid_xray_dataset(dataset_path, metadata_file, img_size=(256,256)):
    """
    Carrega o dataset COVID-19 xray (do V7 Darwin).

    Assumptions:
      - As imagens estão armazenadas em <dataset_path>/images/
      - O arquivo de metadados (CSV) está em <dataset_path>/metadata_file
      - O CSV possui, pelo menos, as colunas:
           * 'filename' – nome da imagem
           * 'covid' – com valores como 'yes' ou 'no'
           * 'pneumonia' – com valores como 'viral', 'bacterial', 'fungal', 'healthy/none', etc.

    Mapeamento de rótulos:
      - Se 'covid' for 'yes' (ou equivalente), label = 0 (Covid)
      - Senão, se 'pneumonia' não for "healthy/none" (ou similar), label = 1 (Pneumonia)
      - Caso contrário, label = 2 (Healthy)

    Retorna:
      - X: array de imagens normalizadas (N, H, W)
      - y: array de rótulos (N,)
    """
    images_folder = os.path.join(dataset_path, "images")
    metadata_path = os.path.join(dataset_path, metadata_file)
    
    df = pd.read_csv(metadata_path)
    X, y = [], []
    
    for idx, row in df.iterrows():
        filename = row['filename']  # certifique-se de que esta coluna exista
        img_path = os.path.join(images_folder, filename)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        
        # Determina o rótulo com base nos metadados
        if str(row.get('covid', '')).strip().lower() in ['yes', 'true', '1']:
            label = 0  # Covid
        elif str(row.get('pneumonia', '')).strip().lower() not in ['healthy/none', 'none', 'healthy']:
            label = 1  # Pneumonia
        else:
            label = 2  # Healthy
        
        X.append(img)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    return X, y
