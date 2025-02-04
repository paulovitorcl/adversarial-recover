# percolation.py
import numpy as np

def percolation_features(img, max_L=43, step=4):
    """
    Extrai features de percolation (locais e globais) de uma imagem em escala de cinza.
    Esta implementação é uma simplificação para fins didáticos.
    
    Para cada escala L (variando de 3 a max_L com incremento de 'step'),
    percorre a imagem em blocos de tamanho LxL (com salto L, para simplificar).
    Calcula métricas dummy e, ao final, extrai features globais (média, desvio, máximo, mínimo, mediana).

    Retorna um vetor de features.
    """
    scales = list(range(3, max_L+1, step))
    C_values, Q_values, M_values = [], [], []
    h, w = img.shape
    
    for L in scales:
        c_total, q_total, m_total = 0, 0, 0
        count_boxes = 0
        for r in range(0, h - L, L):
            for c in range(0, w - L, L):
                box = img[r:r+L, c:c+L]
                center_val = box[L//2, L//2]
                labeled = (np.abs(box - center_val) <= L/255.0).astype(int)
                n_clusters = np.sum(labeled)  # simplificação
                has_perc = 1 if n_clusters >= 0.9 * (L * L) else 0
                largest_cluster = n_clusters
                c_total += n_clusters
                q_total += has_perc
                m_total += largest_cluster / (L*L)
                count_boxes += 1
        if count_boxes > 0:
            C_values.append(c_total / count_boxes)
            Q_values.append(q_total / count_boxes)
            M_values.append(m_total / count_boxes)
        else:
            C_values.append(0)
            Q_values.append(0)
            M_values.append(0)
    
    # Features globais (dummy)
    global_feats = [np.mean(img), np.std(img), np.max(img), np.min(img), np.median(img)]
    feats = np.concatenate([C_values, Q_values, M_values, global_feats], axis=0)
    return feats

def extract_features_dataset(X, max_L=43, step=4):
    """
    Aplica percolation_features a cada imagem do array X.
    
    Retorna um array de features de dimensão (N, num_features).
    """
    feature_list = []
    for img in X:
        f = percolation_features(img, max_L=max_L, step=step)
        feature_list.append(f)
    return np.array(feature_list)
