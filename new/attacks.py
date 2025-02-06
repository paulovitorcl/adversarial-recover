import torch
import torch.nn as nn

# Função do ataque adversarial FGSM
def fgsm_attack(model, image, label, epsilon=0.03):
    # Certificar que o modelo está em modo de avaliação
    model.eval()

    # Configurar o tensor da imagem para cálculo do gradiente
    image.requires_grad = True

    # Forward pass
    output = model(image.unsqueeze(0))  # Adiciona batch dimension
    loss = nn.CrossEntropyLoss()(output, label.unsqueeze(0))  # Cálculo da perda

    # Backward pass para calcular o gradiente da perda em relação à imagem
    model.zero_grad()
    loss.backward()

    # Gera a perturbação adversarial
    perturbation = epsilon * image.grad.sign()

    # Aplica a perturbação adversarial à imagem original
    adversarial_image = image + perturbation

    # Garantir que os valores da imagem estejam entre [0,1]
    adversarial_image = torch.clamp(adversarial_image, 0, 1)

    return adversarial_image.detach()  # Remove o gradiente da imagem adversarial
