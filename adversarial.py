# adversarial.py
import torch
import torch.nn.functional as F

def generate_FGSM(model, x, y_true, epsilon=0.01):
    """
    Gera um exemplo adversarial utilizando FGSM.

    x: tensor 1D (feature vector) com dimensão (num_features,)
    y_true: tensor com o rótulo verdadeiro (escala 0, 1, etc.)
    epsilon: intensidade do ataque

    Retorna: um tensor 1D (feature vector adversarial) com as mesmas dimensões de x.
    """
    # Adiciona dimensão de batch para que a entrada fique com shape (1, num_features)
    x_adv = x.clone().detach().unsqueeze(0).requires_grad_(True)
    # Também precisa "unsqueeze" no rótulo para ficar (1,)
    y_true_unsqueezed = y_true.unsqueeze(0)
    
    # Forward pass com a entrada 2D
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y_true_unsqueezed)
    model.zero_grad()
    loss.backward()
    
    # Atualiza a entrada adversarial utilizando FGSM
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    # Remove a dimensão de batch e retorna o tensor 1D
    return x_adv.squeeze(0)
