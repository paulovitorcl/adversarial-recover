# main.py
import numpy as np
import torch
import torch.optim as optim

from percolation import extract_features_dataset
from classifier_poly import PolynomialClassifier
from neural_net import SimpleMLP
from adversarial import generate_FGSM
from defense import defend_median_filter

# Importa o data loader adaptado para o dataset Kaggle
from data_loader import load_chest_xray_pneumonia_dataset

def main():
    # Configurações
    dataset_path = "./chest_xray"  # Pasta base do dataset (ex.: ./chest_xray)
    img_size = (256,256)
    poly_degree = 4

    # Carrega os conjuntos de treino e teste
    X_train, y_train = load_chest_xray_pneumonia_dataset(dataset_path, split="train", img_size=img_size)
    X_test, y_test   = load_chest_xray_pneumonia_dataset(dataset_path, split="test",  img_size=img_size)

    print("Treino:", X_train.shape, y_train.shape)
    print("Teste: ", X_test.shape, y_test.shape)

    # Extração de features de percolação
    X_train_feats = extract_features_dataset(X_train, max_L=43, step=4)
    X_test_feats  = extract_features_dataset(X_test,  max_L=43, step=4)
    print("Features Treino:", X_train_feats.shape, "Features Teste:", X_test_feats.shape)

    # --- Classificador polinomial (exemplo) ---
    poly_clf = PolynomialClassifier(degree=poly_degree)
    poly_clf.fit(X_train_feats, y_train)
    acc_poly, auc_poly = poly_clf.evaluate(X_test_feats, y_test)
    if auc_poly is not None:
        print(f"(Poly) Acurácia: {acc_poly:.4f}, AUC: {auc_poly:.4f}")
    else:
        print(f"(Poly) Acurácia: {acc_poly:.4f}")
    
    # --- Modelo MLP para avaliar robustez adversarial ---
    X_train_torch = torch.tensor(X_train_feats, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_test_torch  = torch.tensor(X_test_feats, dtype=torch.float32)
    y_test_torch  = torch.tensor(y_test, dtype=torch.long)
    
    input_dim = X_train_feats.shape[1]
    num_classes = 2  # NORMAL e PNEUMONIA

    mlp_model = SimpleMLP(input_dim, num_classes)
    optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
    mlp_model.train()
    
    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mlp_model(X_train_torch)
        loss = torch.nn.functional.cross_entropy(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        print(f"[MLP] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    mlp_model.eval()
    with torch.no_grad():
        preds_clean = mlp_model(X_test_torch).argmax(dim=1)
    acc_clean = (preds_clean == y_test_torch).float().mean().item()
    print(f"(MLP) Acurácia (sem ataque): {acc_clean*100:.2f}%")
    
    # --- Geração de ataques adversariais (FGSM) ---
    X_test_adv = []
    for i in range(len(X_test_torch)):
        x_i = X_test_torch[i].clone()
        y_i = y_test_torch[i].clone()
        x_adv = generate_FGSM(mlp_model, x_i, y_i, epsilon=0.01)
        X_test_adv.append(x_adv.numpy())
    X_test_adv = np.array(X_test_adv)
    with torch.no_grad():
        preds_adv = mlp_model(torch.tensor(X_test_adv, dtype=torch.float32)).argmax(dim=1)
    acc_adv = (preds_adv == y_test_torch).float().mean().item()
    print(f"(MLP) Acurácia (após ataque FGSM): {acc_adv*100:.2f}%")
    
    # --- Aplicação de defesa (filtro mediano) ---
    X_test_def = defend_median_filter(X_test_adv, kernel_size=3)
    with torch.no_grad():
        preds_def = mlp_model(torch.tensor(X_test_def, dtype=torch.float32)).argmax(dim=1)
    acc_def = (preds_def == y_test_torch).float().mean().item()
    print(f"(MLP) Acurácia (após defesa): {acc_def*100:.2f}%")

if __name__ == "__main__":
    main()
