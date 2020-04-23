import math
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

def exibir_sumario_resultados(solucao, custos):
    print('Valor X que gerou melhor resultado:', solucao[custos.index(max(custos))])
    print('Maior custo:', max(custos))
    print('Média de custos:', np.mean(custos))
    print('Desvio padrão:', np.std(custos))
    plotar_busca(custos)

def plotar_busca(resultados):
    t = np.arange(0.0, len(resultados), 1)
    
    plt.figure(figsize=(12,8))
    plt.plot(t, resultados)
    plt.show()

def funcao_custo(x):
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

def hill_climbing(funcao_custo, solucao_inicial, tx_aprendizado):
    # random.seed(a=0)
    solucao = solucao_inicial

    custos = []
    count = 0
    parar_no_plato = 0

    while count <= 400:
        vizinhos = get_vizinhos(solucao, tx_aprendizado)
        
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
            if parar_no_plato == 20: print('plato')
            break

    return solucao, custos

def get_valor_aleatorio(espaco, x=0):
    inicio = random.random()
    valor = []
    
    for i in espaco:
        diferenca = i - inicio
        if diferenca > 0.05 or diferenca < -0.05:
            valor.append(diferenca)

    if len(valor) == len(espaco) or x > 300:
        return inicio
    else:
        return get_valor_aleatorio(espaco, x = x + 1)

custos = []
solucao = []
espaco_solucao = []

for i in range(30):
    # espaco_solucao.append(get_valor_aleatorio(espaco_solucao)) não gerou melhoras
    
    solucao_subida_encosta = hill_climbing(funcao_custo, solucao[custos.index(max(custos))] if len(custos) > 0 else random.random(), i + 1)
    solucao.append(solucao_subida_encosta[0])
    custos.append(max(solucao_subida_encosta[1]))

#exibir_sumario_resultados(solucao, custos)

def get_iteracoes(temperatura, resfriamento):
    count = 0
    while temperatura > 0.1:
        temperatura = temperatura * resfriamento
        count += 1
    return count

def simulated_annealing(funcao_custo, temperatura = 100, resfriamento = 0.95):
    #random.seed(a=0)
    iteracoes = get_iteracoes(temperatura, resfriamento)
    queda_prob = 100 / iteracoes
    probabilidade = 100
    solucao = random.random()
    custos = []
    parar_no_plato = 0

    while temperatura > 0.1:
        vizinhos = get_vizinhos(solucao)
        
        atual = funcao_custo(solucao)
        melhor = atual 
        solucao_atual = solucao
        custos.append(atual)

        for i in range(len(vizinhos)):
            
            if parar_no_plato == 20:
                break

            custo = funcao_custo(vizinhos[i])
            #probabilidade = pow(math.e, (custo - melhor) / temperatura) 
            probabilidade = probabilidade - queda_prob 
            
            if custo >= melhor or random.random() < probabilidade:
                parar_no_plato = parar_no_plato + 1 if solucao_atual == solucao else 0
                melhor = custo
                solucao = vizinhos[i]
               
        temperatura = temperatura * resfriamento

    return solucao, custos

custos = []
solucao = []

for i in range(5):
    solucao_tempera_simulada = simulated_annealing(funcao_custo)
    solucao.append(solucao_tempera_simulada[0])
    custos.append(max(solucao_tempera_simulada[1]))
    plotar_busca(solucao_tempera_simulada[1])
#exibir_sumario_resultados(solucao, custos)

def mutacao(solucao):
    constante = 0.005

    if random.random() < 0.5:
        mutante = solucao - constante if solucao - constante > 0 else solucao
    else:
        mutante = solucao + constante if solucao + constante < 1 else solucao
    
    return mutante

def crossover(solucao1, solucao2):
    crossed = (solucao1 + solucao2) / 2
    return crossed

def genetico(funcao_custo, tamanho_populacao = 50, p_mutacao = 0.2, elitismo = 0.2, geracoes=100):
    populacao = []
    for i in range(tamanho_populacao):
        populacao.append(random.random())
    
    numero_elitismo = int(elitismo * tamanho_populacao)
    
    for i in range(geracoes):
        custos = [(funcao_custo(individuo), individuo) for individuo in populacao]
        custos.sort(reverse=True)
        individuos_ordenados = [individuos for (custo, individuos) in custos]
        
        populacao = individuos_ordenados[0:numero_elitismo]
    
        while len(populacao) < tamanho_populacao:
            if random.random() < p_mutacao:
                individuo_selecionado = random.randint(0, numero_elitismo)
                populacao.append(mutacao(individuos_ordenados[individuo_selecionado]))
            else:
                individuo1 = random.randint(0, numero_elitismo)
                individuo2 = random.randint(0, numero_elitismo)
                populacao.append(crossover(individuos_ordenados[individuo1], 
                                           individuos_ordenados[individuo2]))
    return custos[0][0], custos[0][1]

# melhor_custo, x = genetico(funcao_custo)
# print(melhor_custo, x)