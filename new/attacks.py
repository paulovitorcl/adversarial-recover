import torch
import torch.nn as nn

# Função do ataque adversarial FGSM
def fgsm_attack(model, image, label, epsilon=0.03):
    model.eval()

    # ✅ Remover a dimensão extra desnecessária
    if image.ndim == 5:
        image = image.squeeze(0)  # Corrige para [batch_size, 1, 128, 128]

    # Configurar o tensor da imagem para cálculo do gradiente
    image.requires_grad = True

    # Forward pass
    output = model(image)  # ✅ Removido o .unsqueeze(0)
    loss = nn.CrossEntropyLoss()(output, label)

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
