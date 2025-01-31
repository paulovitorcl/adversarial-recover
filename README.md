# adversarial-recover


## Execução
```
pip install numpy opencv-python scikit-learn torch torchvision scipy
python main.py
```

## Organização do dataset
```
- datasets/
   - Covid/
   - Pneumonia/
   - Healthy/
```

### Funções

`main.py` -> script principal que orquestra tudo (carrega dados, extrai características, treina, gera e avalia ataques, faz defesa, etc.).

`data_loader.py` -> funções para carregar e pré-processar as imagens.

`percolation.py` -> funções de extração de características de percolação (locais e globais).

`classifier_poly.py` -> exemplo de treino e avaliação de um classificador polinomial usando PolynomialFeatures + LogisticRegression.

`neural_net.py` -> modelo MLP em PyTorch para geração de ataques adversariais.

`adversarial.py` -> geração de exemplos adversariais (FGSM).

`defense.py` -> funções de defesa/recuperação contra ataques adversariais (ex.: filtro mediano).

`requirements.txt` -> lista de dependências

