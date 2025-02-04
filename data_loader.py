# data_loader.py
import os
import cv2
import numpy as np

def load_chest_xray_pneumonia_dataset(dataset_path, split="train", img_size=(256,256)):
    """
    Carrega o dataset "Chest X-Ray Pneumonia".

    Estrutura esperada:
       dataset_path/
           split/          <- "train", "test" ou "val"
              NORMAL/      <- imagens de pulmões normais
              PNEUMONIA/   <- imagens com pneumonia

    Mapeamento de rótulos:
       "NORMAL"    → 0
       "PNEUMONIA" → 1

    Retorna:
       X: array de imagens (N, altura, largura) normalizadas (float32, valores entre 0 e 1)
       y: array de rótulos (N,)
    """
    split_path = os.path.join(dataset_path, split)
    classes = {"NORMAL": 0, "PNEUMONIA": 1}
    X, y = [], []
    
    for class_name, label in classes.items():
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
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
