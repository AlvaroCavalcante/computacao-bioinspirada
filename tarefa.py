import math
import random
import matplotlib.pyplot as plt
import numpy as np

def plotar_busca(resultados):
    t = np.arange(0.0, len(resultados), 1)
    
    plt.figure(figsize=(12,8))
    plt.plot(t, resultados)
    plt.show()

def funcao_custo(x):
    custo = 2 ** -2 * (x - 0.1 / 0.9) ** 2 * (math.sin(5 * math.pi * x))** 6
    return custo

def get_vizinhos(solucao):
    vizinhos = []
    constante = 0.5
    vizinho_superior = solucao + constante if solucao + constante < 1 else solucao
    vizinhos.append(vizinho_superior)
    vizinhos.append(solucao - constante)
    return vizinhos

def hill_climbing(funcao_custo):
    # random.seed(a=0)
    solucao = random.random()
    custos = []
    count = 0
    parar_no_plato = 0

    while count <= 400:
        vizinhos = get_vizinhos(solucao)
        
        atual = funcao_custo(solucao)
        melhor = atual 
        solucao_atual = solucao
        custos.append(atual)

        for i in range(len(vizinhos)):
            custo = funcao_custo(vizinhos[i])
            if custo >= melhor:
                parar_no_plato = parar_no_plato + 1 if custo == melhor else 0
                melhor = custo
                solucao = vizinhos[i]

        count += 1
        if melhor == atual and solucao_atual == solucao or parar_no_plato == 20:
            print('plato' if parar_no_plato == 20 else 'sem melhoria')
            break

    return solucao, custos

solucao_subida_encosta = hill_climbing(funcao_custo)

print('Valor X:', solucao_subida_encosta[0])
print('custos', solucao_subida_encosta[1])
plotar_busca(solucao_subida_encosta[1])