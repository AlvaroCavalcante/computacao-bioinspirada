import math
import random
import matplotlib.pyplot as plt
import numpy as np
import sys

def exibir_sumario_resultados(solucao, custos):
    print('Valor X:', solucao[custos.index(max(custos))])
    print('Menor custo:', max(custos))
    print('Média de custos:', np.mean(custos))
    print('Desvio padrão:', np.std(custos))
    plotar_busca(custos)

def plotar_busca(resultados):
    t = np.arange(0.0, len(resultados), 1)
    
    plt.figure(figsize=(12,8))
    plt.plot(t, resultados)
    plt.show()

def funcao_custo(x):
    custo = (x ** 2) - (2*x) + 4
    return custo

def get_vizinhos(solucao, tx_aprendizado = 1):
    vizinhos = []
    constante = 0.005 / tx_aprendizado
    vizinho_superior = solucao + constante if solucao + constante < 15 else solucao
    vizinho_inferior = solucao - constante if solucao - constante > -15 else solucao
    
    vizinhos.append(vizinho_superior)
    vizinhos.append(vizinho_inferior)
    return vizinhos


def mutacao(solucao):
    constante = 0.005

    if random.random() < 0.5:
        mutante = solucao - constante if solucao - constante > -15 else solucao
    else:
        mutante = solucao + constante if solucao + constante < 15 else solucao
    
    return mutante

def crossover(solucao1, solucao2):
    crossed = (solucao1 + solucao2) / 2
    return crossed


def get_bits(bits):
    bits = [[0,0,0,0]] if len(bits) == 0 else bits
    bit_atual = []
    bit_atual.append(bits[len(bits) -1])
    for i in bits:
        if bit_atual not in bits:
            bits.append(bit_atual)
            break
        else:
            ultimo_valor = bit_atual[len(bit_atual) -1] + 1
            bit_atual.pop()
            bit_atual.insert(len(bit_atual), ultimo_valor) 
    
    return bits

def converter_pop_binarios(populacao):
    #inicial = 1 if populacao > 0 else 0
    bits = {}
    array_bits = []

    for i in range(15):
        bits[i] = get_bits(array_bits)

def genetico(funcao_custo, tamanho_populacao = 30, p_mutacao = 0.2, elitismo = 0.2, geracoes=50):
    populacao = []
    for i in range(tamanho_populacao):
        populacao.append(random.randrange(-15, 15))
    
    #populacao = converter_pop_binarios(populacao)

    numero_elitismo = int(elitismo * tamanho_populacao)
    
    for i in range(geracoes):
        custos = [(funcao_custo(individuo), individuo) for individuo in populacao]
        custos.sort()
        individuos_ordenados = [individuos for (custo, individuos) in custos]
        
        populacao = individuos_ordenados[0:numero_elitismo] #ta errado também, por que estou reproduzindo a elite
        # aqui estou pegando a população dos melhores e criando uma nova geração a partir dela
        # eu poderia ao invés de selecionar apenas os melhores, fazer uma roleta. 
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

custos = []
solucao = []

for i in range(30):
    solucao_algoritmo_genetico = genetico(funcao_custo)
    solucao.append(solucao_algoritmo_genetico[1])
    custos.append(solucao_algoritmo_genetico[0])

exibir_sumario_resultados(solucao, custos)
