# main.py
import numpy as np
import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split

# Imports dos nossos módulos criados
from data_loader import load_chest_xray_images
from percolation import extract_features_dataset
from classifier_poly import PolynomialClassifier
from neural_net import SimpleMLP
from adversarial import generate_FGSM
from defense import defend_median_filter

def main():
    # 1. Parâmetros e configurações
    base_path = "./datasets"  # Ajuste para seu caminho de dataset
    # Agora, 3 classes:
    label_map = {"Covid": 0, "Pneumonia": 1, "Healthy": 2}
    
    img_size = (256, 256)
    test_size = 0.2
    random_seed = 42
    poly_degree = 4
    
    # 2. Carregando dados
    X, y = load_chest_xray_images(base_path, label_map, img_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    print("Dados carregados. Formato X:", X.shape, "y:", y.shape)
    print("Classes únicas (labels):", np.unique(y))
    
    # 3. Extraindo características de percolação
    X_train_feats = extract_features_dataset(X_train, max_L=43, step=4)
    X_test_feats  = extract_features_dataset(X_test,  max_L=43, step=4)
    
    print("Features de treino:", X_train_feats.shape)
    print("Features de teste: ", X_test_feats.shape)

    # 4. Classificador polinomial (multi-classes)
    poly_clf = PolynomialClassifier(degree=poly_degree)
    poly_clf.fit(X_train_feats, y_train)
    acc_test, auc_test = poly_clf.evaluate(X_test_feats, y_test)
    
    # Em multi-classes, 'auc_test' tende a ser None (implementação atual)
    if auc_test is not None:
        print(f"(Poly) Acurácia teste: {acc_test:.4f}, AUC teste: {auc_test:.4f}")
    else:
        print(f"(Poly) Acurácia teste (multi-classe): {acc_test:.4f}")
    
    # 5. Exemplo de ataque adversarial usando MLP (PyTorch)
    # Converte para tensores
    X_train_torch = torch.tensor(X_train_feats, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_test_torch  = torch.tensor(X_test_feats,  dtype=torch.float32)
    y_test_torch  = torch.tensor(y_test, dtype=torch.long)
    
    input_dim = X_train_feats.shape[1]
    num_classes = len(np.unique(y_train))  # deve ser 3
    
    mlp_model = SimpleMLP(input_dim, num_classes)  # Rede com 3 saídas
    optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
    
    mlp_model.train()
    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mlp_model(X_train_torch)
        # CrossEntropyLoss lida naturalmente com multi-classe
        loss = torch.nn.functional.cross_entropy(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        print(f"[MLP] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Avalia MLP no conjunto limpo (sem ataque)
    mlp_model.eval()
    with torch.no_grad():
        preds_clean = mlp_model(X_test_torch).argmax(dim=1)
    acc_clean = (preds_clean == y_test_torch).float().mean().item()
    print(f"(MLP) Acurácia teste (sem ataque): {acc_clean*100:.2f}%")
    
    # 6. Gera ataques adversariais (FGSM) para cada amostra do teste
    X_test_adv = []
    for i in range(len(X_test_torch)):
        x_i = X_test_torch[i].clone()
        y_i = y_test_torch[i].clone()
        # epsilon: intensidade do passo do FGSM, ajuste se quiser mais forte/fraco
        x_adv = generate_FGSM(mlp_model, x_i, y_i, epsilon=0.01)
        X_test_adv.append(x_adv.numpy())
    X_test_adv = np.array(X_test_adv)
    
    # Avalia a acurácia após ataque
    with torch.no_grad():
        preds_adv = mlp_model(torch.tensor(X_test_adv, dtype=torch.float32)).argmax(dim=1)
    acc_adv = (preds_adv == y_test_torch).float().mean().item()
    print(f"(MLP) Acurácia teste (após ataque FGSM): {acc_adv*100:.2f}%")
    
    # 7. Aplica defesa (exemplo: filtro mediano em 1D nas features atacadas)
    X_test_def = defend_median_filter(X_test_adv, kernel_size=3)
    with torch.no_grad():
        preds_def = mlp_model(torch.tensor(X_test_def, dtype=torch.float32)).argmax(dim=1)
    acc_def = (preds_def == y_test_torch).float().mean().item()
    print(f"(MLP) Acurácia teste (após defesa mediana): {acc_def*100:.2f}%")

if __name__ == "__main__":
    main()
