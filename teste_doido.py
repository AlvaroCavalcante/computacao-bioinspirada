import math
import random
import matplotlib.pyplot as plt
import numpy as np
import sys


def exibir_sumario_resultados(solucao, custos, objetivo = max):
    print('Valor que gerou melhor resultado:', solucao[custos.index(objetivo(custos))])
    print('Melhor custo:', objetivo(custos))
    print('Média de custos:', np.mean(custos))
    print('Desvio padrão:', np.std(custos))
    plotar_busca(custos)

def plotar_busca(resultados):
    t = np.arange(0.0, len(resultados), 1)
    
    plt.figure(figsize=(12,8))
    plt.plot(t, resultados)
    plt.show()

def get_iteracoes(temperatura, resfriamento):
    count = 0
    while temperatura > 0.1:
        temperatura = temperatura * resfriamento
        count += 1
    return count


def funcao_custo(x):
    if isinstance(x, tuple):
        x = x[0]

    custo = 2 ** (-2 *((((x-0.1) / 0.9)) ** 2)) * ((math.sin(5*math.pi*x)) ** 6)
    return custo

def get_vizinhos(solucao, tx_aprendizado = 1):
    vizinhos = []
    constante = 0.005 / tx_aprendizado
    vizinho_superior = solucao + constante if solucao + constante < 1 else solucao
    vizinho_inferior = solucao - constante if solucao - constante > 0 else solucao
    
    vizinhos.append(vizinho_superior)
    vizinhos.append(vizinho_inferior)
    return vizinhos

def simulated_annealing(funcao_custo, temperatura = 20, resfriamento = 0.95):
    probabilidade = 100
    solucao = random.random()
    custos = []
    parar_no_plato = 0

    while temperatura > 0.01:
        vizinhos = get_vizinhos(solucao)
        
        atual = funcao_custo(solucao)
        melhor = atual 
        solucao_atual = solucao
        custos.append(atual)

        for i in range(len(vizinhos)):
            
            if parar_no_plato == 20:
                break

            custo = funcao_custo(vizinhos[i])
            
            if custo >= melhor or random.random() < probabilidade:
                parar_no_plato = parar_no_plato + 1 if solucao_atual == solucao else 0
                melhor = custo
                solucao = vizinhos[i]

        probabilidade = pow(math.e, (-custo - melhor) / temperatura) 
        temperatura = temperatura * resfriamento

    return max(custos)

def mutacao(solucao, dominio):
    constante = 0.05
    index_mutacao = random.randint(0, len(solucao) -1)
    gene_mutado = solucao[index_mutacao]
    solucao = list(solucao)

    if random.random() < 0.5:
        if ((gene_mutado - constante) >= dominio[0][0]):
            gene_mutado = gene_mutado - constante
    else:
        if ((gene_mutado + constante) <= dominio[0][1]):
            gene_mutado = gene_mutado + constante

    del solucao[index_mutacao]
    solucao.insert(index_mutacao, gene_mutado) 
            
    return tuple(solucao)

def crossover(solucao1, solucao2):
    crossed = [(solucao1[i] + solucao2[i]) / 2 for i in range(len(solucao1))]
    return tuple(crossed)

def get_populacao(tamanho_populacao, dominio, numeros_inteiros = False):
    populacao = []
    #random.seed(42) #comentar/descomentar para gerar uma seed para os valores "aleatórios"
    for i in range(tamanho_populacao):
        if numeros_inteiros == False:
            solucao = [random.uniform(dominio[i][0], dominio[i][1]) for i in range(
                len(dominio))]
        else:
            solucao = [random.randint(dominio[i][0], dominio[i][1]) for i in range(
                len(dominio))]
        
        populacao.append(tuple(solucao)) 
        
    return populacao

def get_populacao_torneio(populacao, numero_individuos, objetivo, n_competidores = 3):
        nova_populacao = []
        while len(nova_populacao) < numero_individuos:
            torneio = []

            for i in range(n_competidores):
                torneio.append(populacao[random.randint(0, len(populacao) - 1)])
            
            torneio.sort(reverse=objetivo)
            nova_populacao.append(torneio[0][1])
        
        return nova_populacao

def get_melhores_individuos(custos, n_elitismo, objetivo):
    custos.sort(reverse=objetivo)
    individuos_ordenados = [individuos for (custo, individuos) in custos]
    elite = individuos_ordenados[0:n_elitismo]
    return elite

def genetico(funcao_custo, dominio, objetivo = False, tamanho_populacao = 50, p_mutacao = 0.2, elitismo = 0.1, geracoes=20):
    populacao = get_populacao(tamanho_populacao, dominio, True)
    
    numero_elitismo = int(elitismo * tamanho_populacao)
    
    for i in range(geracoes):
        custos = [(simulated_annealing(funcao_custo, individuo[0]), individuo) for individuo in populacao]
        
        populacao = get_melhores_individuos(custos, numero_elitismo, objetivo) 
    
        individuos_escolhidos = get_populacao_torneio(custos, (
                                tamanho_populacao - numero_elitismo) // 2, objetivo)
    
        while len(populacao) < tamanho_populacao:
            if random.random() < p_mutacao:
                individuo_selecionado = random.randint(0, len(individuos_escolhidos) -1)
                populacao.append(mutacao(individuos_escolhidos[individuo_selecionado], dominio))
            else:
                individuo1 = random.randint(0, len(individuos_escolhidos) -1)
                individuo2 = random.randint(0, len(individuos_escolhidos) -1)
                populacao.append(crossover(individuos_escolhidos[individuo1], 
                                           individuos_escolhidos[individuo2]))
    return custos[0][0], custos[0][1]

custos = []
solucao = []
dominio = [(1, 100)]

for i in range(30):
    solucao_algoritmo_genetico = genetico(funcao_custo, dominio, True)
    solucao.append(solucao_algoritmo_genetico[1])
    custos.append(solucao_algoritmo_genetico[0])

exibir_sumario_resultados(solucao, custos, max)
