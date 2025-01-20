# adversarial.py
import torch
import torch.nn.functional as F

def generate_FGSM(model, x, y_true, epsilon=0.01):
    """
    Gera exemplo adversarial usando FGSM (Fast Gradient Sign Method).
    x: tensor (features 1D) em PyTorch com requires_grad=True
    y_true: rótulo real (tensor) 
    epsilon: intensidade do passo de ataque
    """
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    
    loss = F.cross_entropy(logits.unsqueeze(0), y_true.unsqueeze(0))
    model.zero_grad()
    loss.backward()
    
    # FGSM
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()
