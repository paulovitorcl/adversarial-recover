import os
import json
import logging
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from genetic_compression import run_ga
from attacks import fgsm_attack

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dataset personalizado para raios-X
class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        
        # Mapeamento de classes
        self.class_map = {"NORMAL": 0, "VIRAL": 1, "BACTERIANO": 2}
        
        # Percorre as subpastas NORMAL e PNEUMONIA
        for label in ["NORMAL", "PNEUMONIA"]:
            label_path = os.path.join(root_dir, label)
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                
                # Diferenciar pneumonia viral e bacteriana
                if "virus" in img_name.lower():
                    class_label = self.class_map["VIRAL"]
                elif "bacteria" in img_name.lower():
                    class_label = self.class_map["BACTERIANO"]
                else:
                    class_label = self.class_map["NORMAL"]

                self.image_files.append(img_path)
                self.labels.append(class_label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Lendo em escala de cinza
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Carregar dataset
def load_data(data_path, batch_size=32):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  # Aumentando a resolução
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalização para imagens em escala de cinza
    ])
    
    dataset = XRayDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# Definir modelo CNN para raios-X
def load_model():
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Mudança para 1 canal (raios-X)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(128 * 32 * 32, 256),  # Ajustado para 128x128 entrada
        nn.ReLU(),
        nn.Linear(256, 3)  # 3 classes: NORMAL, VIRAL, BACTERIANO
    )
    return model

# Avaliação do classificador
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []   # ✅ Corrigido: Inicializar listas separadas
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            # ✅ Corrigir dimensões das imagens
            if images.ndim == 5:
                images = images.squeeze(1)  # Remove a dimensão extra

            if images.ndim == 3:
                images = images.unsqueeze(1)  # Garante o formato [batch_size, 1, 128, 128]

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)



# Aplicação de compressão otimizada pelo GA
def apply_compression(image, config):
    formato, qualidade, rotacao, brilho, contraste = config
    image = cv2.rotate(image, rotacao)
    image = cv2.convertScaleAbs(image, alpha=contraste, beta=brilho * 50)

    temp_path = f"temp.{formato}"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    os.remove(temp_path)
    
    return image

# Pipeline principal
def main():
    data_path = "./chest_xray/train"  # Caminho do dataset
    dataloader = load_data(data_path)
    model = load_model()

    # Avaliação inicial
    logging.info("Avaliando modelo antes do ataque...")
    preds_before, labels = evaluate_model(model, dataloader)

    # Aplicação de ataque adversarial
    logging.info("Aplicando ataque adversarial...")
    images_adv = [fgsm_attack(model, img, lbl, epsilon=0.03) for img, lbl in dataloader]

    # Otimização com Algoritmo Genético
    logging.info("Otimizando compressão com Algoritmo Genético...")
    best_config = run_ga(images_adv, labels)

    # Salvar melhor configuração
    with open("best_compression.json", "w") as f:
        json.dump(best_config, f)

    logging.info(f"Melhor configuração encontrada: {best_config}")

    # Recuperação das imagens
    images_recovered = [apply_compression(img, best_config) for img in images_adv]

    # Avaliação após recuperação
    logging.info("Avaliando modelo após recuperação...")
    preds_after_recovery, _ = evaluate_model(model, images_recovered)

    # Matriz de confusão
    cm_before = confusion_matrix(labels, preds_before)
    cm_after = confusion_matrix(labels, preds_after_recovery)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm_before).plot(ax=ax[0], cmap="Blues")
    ConfusionMatrixDisplay(cm_after).plot(ax=ax[1], cmap="Greens")
    ax[0].set_title("Antes do Ataque")
    ax[1].set_title("Após a Recuperação")
    plt.show()

if __name__ == "__main__":
    main()
