import numpy as np
import cv2
import os
import logging
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from genetic_compression import run_ga
from attacks import fgsm_attack
from torch.utils.data import DataLoader, random_split

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar dataset e modelo pré-treinado
def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Redimensiona para 32x32 pixels
        transforms.ToTensor(),        # Converte para tensor
        transforms.Normalize((0.5,), (0.5,))  # Normaliza para [-1, 1]
    ])
    
    # Baixa o dataset CIFAR-10 (pode ser substituído pelo seu)
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # Divide entre treino (80%) e teste (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Criar os DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_model():
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(128*8*8, 256),  # Ajustado para saída correta
        nn.ReLU(),
        nn.Linear(256, 10),  # 10 classes para CIFAR-10
        nn.Softmax(dim=1)
    )
    return model

# Aplicar ataque adversarial
def apply_attack(model, images, labels, epsilon=0.03):
    model.eval()
    images_adv = fgsm_attack(model, images, labels, epsilon)
    return images_adv

# Recuperar imagens usando melhor compressão
def recover_images(images_adv, images_original):
    best_config = run_ga(images_original, images_adv)
    recovered_images = []
    for img in images_adv:
        recovered_img = apply_compression(img, best_config)
        recovered_images.append(recovered_img)
    return recovered_images

# Aplicar compressão conforme configuração
def apply_compression(image, config):
    formato, qualidade, rotacao, brilho, contraste = config
    image = cv2.rotate(image, rotacao)
    image = cv2.convertScaleAbs(image, alpha=contraste, beta=brilho*50)
    temp_path = f"temp.{formato}"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    image = cv2.imread(temp_path)
    os.remove(temp_path)
    return image

# Avaliação do classificador
def evaluate_model(model, images, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    return preds

# Pipeline principal
def main():
    train_loader, test_loader = load_data()
    model = load_model()
    
    # Avaliação inicial
    preds_before = evaluate_model(model, images, labels)
    
    # Aplicação de ataque adversarial
    images_adv = apply_attack(model, images, labels)
    preds_after_attack = evaluate_model(model, images_adv, labels)
    
    # Recuperação das imagens
    images_recovered = recover_images(images_adv, images)
    preds_after_recovery = evaluate_model(model, images_recovered, labels)
    
    # Matriz de confusão
    cm_before = confusion_matrix(labels, preds_before)
    cm_after = confusion_matrix(labels, preds_after_attack)
    cm_recovered = confusion_matrix(labels, preds_after_recovery)
    
    # Exibir resultados
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ConfusionMatrixDisplay(cm_before).plot(ax=ax[0], cmap='Blues')
    ConfusionMatrixDisplay(cm_after).plot(ax=ax[1], cmap='Reds')
    ConfusionMatrixDisplay(cm_recovered).plot(ax=ax[2], cmap='Greens')
    ax[0].set_title("Antes do Ataque")
    ax[1].set_title("Após o Ataque")
    ax[2].set_title("Após a Recuperação")
    plt.show()
    
    logging.info("Experimento concluído.")

if __name__ == "__main__":
    main()
