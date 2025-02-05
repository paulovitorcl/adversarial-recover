import numpy as np
import cv2
import os
import random
import logging
from deap import base, creator, tools, algorithms
from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_psnr

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definição do problema de otimização
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))  # Maximizar PSNR e SSIM
creator.create("Individual", list, fitness=creator.FitnessMax)

# Parâmetros do algoritmo genético
def generate_individual():
    return [random.choice(["jpeg", "png", "webp"]),  # Formato
            random.randint(10, 95),  # Qualidade da compressão
            random.choice([0, 90, 180, 270]),  # Rotação
            random.uniform(0.5, 2.0),  # Brilho
            random.uniform(0.5, 2.0)]  # Contraste

def evaluate(individual, img_original, img_adversarial):
    formato, qualidade, rotacao, brilho, contraste = individual
    
    # Aplicar transformações
    img_recuperada = cv2.rotate(img_adversarial, rotacao)
    img_recuperada = cv2.convertScaleAbs(img_recuperada, alpha=contraste, beta=brilho*50)
    
    # Salvar e recarregar para simular compressão
    temp_path = f"temp.{formato}"
    cv2.imwrite(temp_path, img_recuperada, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    img_recuperada = cv2.imread(temp_path)
    os.remove(temp_path)
    
    # Calcular métricas
    psnr_value = compare_psnr(img_original, img_recuperada)
    ssim_value = ssim(img_original, img_recuperada, multichannel=True)
    
    return psnr_value, ssim_value

# Configuração do GA
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Execução do GA
def run_ga(img_original, img_adversarial, n_generations=20, pop_size=50):
    pop = toolbox.population(n=pop_size)
    
    for gen in range(n_generations):
        for ind in pop:
            ind.fitness.values = evaluate(ind, img_original, img_adversarial)
        
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = [ind.fitness.values[0] + ind.fitness.values[1] for ind in pop]
        best_idx = np.argmax(fits)
        logging.info(f'Geração {gen+1}: Melhor PSNR+SSIM: {fits[best_idx]}')
    
    best_individual = tools.selBest(pop, k=1)[0]
    logging.info(f'Melhor solução encontrada: {best_individual}')
    return best_individual

# Exemplo de uso
def main():
    img_original = cv2.imread("original.png")
    img_adversarial = cv2.imread("adversarial.png")
    best_compression = run_ga(img_original, img_adversarial)
    logging.info(f'Melhor configuração: {best_compression}')

if __name__ == "__main__":
    main()
