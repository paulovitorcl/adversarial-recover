# adversarial-recover
Update

## Realizando o download do dataset
```
https://data.mendeley.com/datasets/rscbjbr9sj/2
```

## Execução
### 1 - Criando e ativando o ambiente virtual
```
python3 -m venv venv
source venv/bin/activate
```

### 2 - Instalando as dependências
```
pip install -r requirements.txt
```

### 3 - Executando 
Deve-se garantir que o dataset esteja no diretório indicado no script main.py
```
python main.py
```

Para desativar o ambiente virtual:
```
deactivate
```

### Estrutura do dataset
```
chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
  (opcional: val/)
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

