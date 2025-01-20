# data_loader.py
import os
import cv2
import numpy as np

def load_chest_xray_images(base_path, label_map, img_size=(256, 256)):
    """
    Carrega as imagens de raio-X (em escala de cinza) a partir de pastas 
    definidas em 'label_map', normaliza e retorna X, y.
    
    base_path: pasta raiz que contém subpastas (uma por classe).
    label_map: dicionário {nome_pasta: rotulo}, ex.: {"Covid":0, "Healthy":1}.
    img_size: tupla de redimensionamento (opcional).
    
    Retorna:
        X (np.array): imagens normalizadas (N, altura, largura).
        y (np.array): rótulos inteiros (N,).
    """
    X, y = [], []
    for folder_name, label in label_map.items():
        folder_path = os.path.join(base_path, folder_name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            X.append(img)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    return X, y
