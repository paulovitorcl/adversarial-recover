# adversarial.py
import torch
import torch.nn.functional as F

def generate_FGSM(model, x, y_true, epsilon=0.01):
    """
    Gera um exemplo adversarial utilizando o método FGSM.
    
    x: tensor 1D (feature vector) com requires_grad=True
    y_true: tensor com o rótulo verdadeiro
    epsilon: intensidade do ataque
    """
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits.unsqueeze(0), y_true.unsqueeze(0))
    model.zero_grad()
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()
