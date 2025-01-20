# percolation.py
import numpy as np

def percolation_features(img, max_L=43, step=4):
    """
    Extrai características de percolação (locais e globais) de uma única imagem.
    Esta implementação é simplificada, devendo ser ajustada aos detalhes do artigo.
    
    Retorna, por exemplo, um vetor de ~48 dimensões (33 locais + 15 globais),
    aqui simplificado para exemplificar.
    """
    scales = list(range(3, max_L+1, step))
    
    C_values, Q_values, M_values = [], [], []
    h, w = img.shape
    
    for L in scales:
        c_total, q_total, m_total = 0, 0, 0
        count_boxes = 0
        
        # Exemplo: salto de L em L (simplificado).
        # No artigo, é "gliding" de 1 em 1 pixel; adapte se necessário.
        for r in range(0, h - L, L):
            for c in range(0, w - L, L):
                box = img[r:r+L, c:c+L]
                center_val = box[L//2, L//2]
                
                # Exemplo simples de critério (threshold)
                labeled = (np.abs(box - center_val) <= L/255.0).astype(int)
                
                # "Mock" do número de clusters (aqui, apenas soma)
                n_clusters = np.sum(labeled)
                
                # Se o "cluster" cobre 90% do box => percolou
                has_perc = 1 if n_clusters >= 0.9 * (L * L) else 0
                
                # Tamanho maior cluster (mock)
                largest_cluster = n_clusters
                
                c_total += n_clusters
                q_total += has_perc
                m_total += (largest_cluster / (L*L))
                count_boxes += 1
        
        if count_boxes > 0:
            C_values.append(c_total / count_boxes)
            Q_values.append(q_total / count_boxes)
            M_values.append(m_total / count_boxes)
        else:
            C_values.append(0)
            Q_values.append(0)
            M_values.append(0)
    
    # Features "globais" (exemplo dummy)
    global_feats = [
        np.mean(img), 
        np.std(img), 
        np.max(img), 
        np.min(img), 
        np.median(img)
    ]
    
    feats = np.concatenate([C_values, Q_values, M_values, global_feats], axis=0)
    return feats

def extract_features_dataset(X, max_L=43, step=4):
    """
    Aplica a percolation_features a cada imagem em X.
    Retorna array (N, num_features).
    """
    feature_list = []
    for img in X:
        f = percolation_features(img, max_L=max_L, step=step)
        feature_list.append(f)
    return np.array(feature_list)
