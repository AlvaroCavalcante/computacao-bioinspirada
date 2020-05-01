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

def funcao_custo(x):
    if isinstance(x, tuple):
        x = x[0]

    custo = 2 ** (-2 *((((x-0.1) / 0.9)) ** 2)) * ((math.sin(5*math.pi*x)) ** 6)
    return custo

def funcao_custo_2(valor):
    x = valor[0]
    y = valor[1]

    custo = (1-x)**2 + (100 * (y - x**2)**2)
    return custo

def funcao_custo_3(vetor):
    numero_zero = [1, 1, 1, 1,0, 1, 1,0, 1, 1, 1, 1]
    custo = 0
    for i in range(len(vetor)):
        if vetor[i] != numero_zero[i]:
            custo += 1
    return custo

def get_vizinhos(solucao, tx_aprendizado = 1):
    vizinhos = []
    constante = 0.005 / tx_aprendizado
    vizinho_superior = solucao + constante if solucao + constante < 1 else solucao
    vizinho_inferior = solucao - constante if solucao - constante > 0 else solucao
    
    vizinhos.append(vizinho_superior)
    vizinhos.append(vizinho_inferior)
    return vizinhos

def hill_climbing(funcao_custo, solucao_inicial, tx_aprendizado = 1):
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

def executar_hill_climbing(funcao_custo, iteracoes, espaco_minimo = False, get_melhor_resultado = False):
    custos = []
    solucao = []
    espaco_solucao = []
    
    for i in range(iteracoes):
        espaco_solucao.append(get_valor_aleatorio(espaco_solucao))
        valor_inicial = random.random() if espaco_minimo == False else espaco_solucao[len(espaco_solucao) - 1]   
        solucao_subida_encosta = hill_climbing(funcao_custo, valor_inicial) if get_melhor_resultado == False else hill_climbing(funcao_custo, solucao[custos.index(max(custos))] if len(custos) > 0 else random.random(), i + 1)
        solucao.append(solucao_subida_encosta[0])
        custos.append(max(solucao_subida_encosta[1]))
    
    return solucao, custos

# solucao, custos = executar_hill_climbing(funcao_custo, 30)
# exibir_sumario_resultados(solucao, custos)

def get_iteracoes(temperatura, resfriamento):
    count = 0
    while temperatura > 0.1:
        temperatura = temperatura * resfriamento
        count += 1
    return count

def simulated_annealing(funcao_custo, temperatura = 20, resfriamento = 0.95):
    iteracoes = get_iteracoes(temperatura, resfriamento)
    probabilidade = 100
    queda_prob = probabilidade / iteracoes
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
        # probabilidade = probabilidade - queda_prob        
        temperatura = temperatura * resfriamento

    return solucao, custos

custos = []
solucao = []

for i in range(5):
    solucao_tempera_simulada = simulated_annealing(funcao_custo)
    solucao.append(solucao_tempera_simulada[0])
    custos.append(max(solucao_tempera_simulada[1]))
    plotar_busca(solucao_tempera_simulada[1])
exibir_sumario_resultados(solucao, custos)

def mutacao(solucao):
    constante = 0.05
    mutante = []

    for i in range(len(solucao)):
        if random.random() < 0.5:
            if ((solucao[i] - constante) >= -5):
                mutante.append(solucao[i] - constante)
            else:
                mutante.append(solucao[i]) 
        else:
            if ((solucao[i] + constante) <= 5):
                mutante.append(solucao[i] - constante)
            else:
                mutante.append(solucao[i]) 

    return tuple(mutante)

def crossover(solucao1, solucao2):
    crossed = [(solucao1[i] + solucao2[i]) / 2 for i in range(len(solucao1))]
    return tuple(crossed)

def get_populacao_torneio(populacao, numero_elitismo, objetivo, n_competidores = 3):
        nova_populacao = []
        while len(nova_populacao) < numero_elitismo:
            torneio = []

            for i in range(n_competidores):
                torneio.append(populacao[random.randint(0, len(populacao) - 1)])
            
            torneio.sort(reverse=objetivo)
            nova_populacao.append(torneio[0][1])
        
        return nova_populacao

def get_populacao(tamanho_populacao, dominio, numeros_inteiros = False):
    populacao = []
    for i in range(tamanho_populacao):
        if numeros_inteiros == False:
            solucao = [random.uniform(dominio[i][0], dominio[i][1]) for i in range(
                len(dominio))]
        else:
            solucao = [random.randint(dominio[i][0], dominio[i][1]) for i in range(
                len(dominio))]
        
        populacao.append(tuple(solucao)) 
        
    return populacao

def get_melhores_individuos(custos, n_elitismo, objetivo):
    custos.sort(reverse=objetivo)
    individuos_ordenados = [individuos for (custo, individuos) in custos]
    elite = individuos_ordenados[0:n_elitismo]
    return elite

def mutacao_binaria(solucao):
    solucao = list(solucao)
    index_mutacao = random.randint(0, len(solucao) -1)
    gene_mutado = solucao[index_mutacao]
    gene_mutado = 1 if gene_mutado == 0 else 0
    del solucao[index_mutacao]
    solucao.insert(index_mutacao, gene_mutado)
    return tuple(solucao)

def crossover_binario(solucao1, solucao2):
    index_cruzamento = random.randint(0, len(solucao1) -1)
    solucao_resultante = solucao1[0:index_cruzamento] + solucao2[index_cruzamento:len(solucao2)]
    return solucao_resultante

def genetico(funcao_custo, dominio, objetivo = False, tamanho_populacao = 50, p_mutacao = 0.2, elitismo = 0.1, geracoes=20):
    populacao = []
    for i in range(tamanho_populacao):
        solucao = [random.uniform(dominio[i][0], dominio[i][1]) for i in range(len(dominio))]
        populacao.append(tuple(solucao)) 
    
    numero_elitismo = int(elitismo * tamanho_populacao)
    
    for i in range(geracoes):
        custos = [(funcao_custo(individuo), individuo) for individuo in populacao]
        
        populacao = get_melhores_individuos(custos, numero_elitismo, objetivo) 
    
        individuos_escolhidos = get_populacao_torneio(custos, (
                                tamanho_populacao - numero_elitismo) // 2, objetivo)
    
        while len(populacao) < tamanho_populacao:
            if random.random() < p_mutacao:
                individuo_selecionado = random.randint(0, len(individuos_escolhidos) -1)
                populacao.append(mutacao(individuos_escolhidos[individuo_selecionado]))
            else:
                individuo1 = random.randint(0, len(individuos_escolhidos) -1)
                individuo2 = random.randint(0, len(individuos_escolhidos) -1)
                populacao.append(crossover(individuos_escolhidos[individuo1], 
                                           individuos_escolhidos[individuo2]))
    return custos[0][0], custos[0][1]

# custos = []
# solucao = []
# dominio = [(-5, 5), (-5, 5)]
# 3 = [(0, 1)]

# for i in range(30):
#     solucao_algoritmo_genetico = genetico(funcao_custo, 3, True)
#     solucao.append(solucao_algoritmo_genetico[1])
#     custos.append(solucao_algoritmo_genetico[0])

# exibir_sumario_resultados(solucao, custos, max)

def genetico_2(funcao_custo, dominio, objetivo = False, tamanho_populacao = 8, p_mutacao = 0.1, 
               elitismo = 0.1, geracoes=5):
    
    populacao = get_populacao(tamanho_populacao, dominio, True)
    
    numero_elitismo = int(elitismo * tamanho_populacao)
    
    for i in range(geracoes):
        custos = [(funcao_custo(individuo), individuo) for individuo in populacao]
        
        populacao = get_melhores_individuos(custos, numero_elitismo, objetivo) 
    
        individuos_escolhidos = get_populacao_torneio(custos, (
                                tamanho_populacao - numero_elitismo) // 2, objetivo)
    
        while len(populacao) < tamanho_populacao:
            if random.random() < p_mutacao:
                individuo_selecionado = random.randint(0, len(individuos_escolhidos) -1)
                populacao.append(mutacao_binaria(individuos_escolhidos[individuo_selecionado]))
            else:
                individuo1 = random.randint(0, len(individuos_escolhidos) -1)
                individuo2 = random.randint(0, len(individuos_escolhidos) -1)
                populacao.append(crossover_binario(individuos_escolhidos[individuo1], 
                                           individuos_escolhidos[individuo2]))
    return custos[0][0], custos[0][1]


# custos = []
# solucao = []
# dominio = [(-5, 5), (-5, 5)]
# dominio_3 = [(0, 1)] * 12

# for i in range(30):
#     solucao_algoritmo_genetico = genetico_2(funcao_custo_3, dominio_3, False)
#     solucao.append(solucao_algoritmo_genetico[1])
#     custos.append(solucao_algoritmo_genetico[0])

# image = np.asmatrix(solucao[custos.index(min(custos))])
# image = image.reshape(4, 3)
# plt.imshow(image, cmap='gray')
# exibir_sumario_resultados(solucao, custos, min)
