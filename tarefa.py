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

def get_vizinhos(solucao, aprendizado):
    vizinhos = []
    aprendizado = aprendizado / 10 if aprendizado >= 10 else 1
    constante = 0.005 / aprendizado
    vizinho_superior = solucao + constante if solucao + constante < 1 else solucao
    vizinho_inferior = solucao - constante if solucao - constante > 0 else solucao
    
    vizinhos.append(vizinho_superior)
    vizinhos.append(vizinho_inferior)
    return vizinhos

def hill_climbing(funcao_custo):
    # random.seed(a=0)
    solucao = random.random()
    custos = []
    count = 1
    parar_no_plato = 0

    while count <= 400:
        vizinhos = get_vizinhos(solucao, count)
        
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


# def simulated_annealing(dominio, funcao_custo, temperatura = 10000.0, resfriamento = 0.95, passo = 1):
#     #random.seed(a=0)
#     solucao = random.random()
#     custos = []
#     count = 1
#     parar_no_plato = 0

#     while temperatura > 0.1:
#         vizinhos = get_vizinhos(solucao, count)
        
#         atual = funcao_custo(solucao)
#         melhor = atual 
#         solucao_atual = solucao
#         custos.append(atual)

#         for i in range(len(vizinhos)):
            
#             if parar_no_plato == 20:
#                 break

#             custo = funcao_custo(vizinhos[i])
#             probabilidade = pow(math.e, (-custo - melhor) / temperatura)
            
#             if custo >= melhor or random.random() < probabilidade:
#                 parar_no_plato = parar_no_plato + 1 if custo == melhor else 0
#                 melhor = custo
#                 solucao = vizinhos[i]
               
#         temperatura = temperatura * resfriamento

#     return solucao, custos

# solucao_tempera_simulada = simulated_annealing([0, 1], funcao_custo)
# custo_tempera_simulada = funcao_custo(solucao_tempera_simulada[0])

# print('Menor custo', custo_tempera_simulada)
# plotar_busca(solucao_tempera_simulada[1])