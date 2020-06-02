import random
import numpy as np
import pandas as pd
import math

df = pd.DataFrame([[0,0,0], [0,1,1], [1,0,1], [1,1,0]], columns = ['X', 'Y', 'CLASSE'])

previsores = df.iloc[:, 0:2] 
classe = df['CLASSE']


pesos0 = np.array([[-0.424, -0.740, -0.961],
                  [0.358, -0.577, -0.469]])
    
pesos1 = np.array([[-0.017], [-0.893], [0.148]])

def somatoria(entradas, pesos):
    return np.dot(entradas, pesos)    

def funcao_degrau(valor):
    if valor >= 1:
        return 1
    return 0

def funcao_sigmoid(valor):
    resultado = 1 / (1 + np.exp(-valor))
    return resultado

def funcao_custo(valor_correto, valor_previsto):
    erro = abs(valor_correto - valor_previsto) # não gerar valores negativos
    return erro

def atualizar_peso(entrada, peso, delta, tx_aprendizado = 0.2, momento = 1):
    delta_t = np.transpose(np.asmatrix(delta).reshape(-1, 1))
    novo_peso = (peso * momento) + (np.dot(np.asmatrix(entrada), ) * tx_aprendizado)
    return novo_peso

def inicializar_pesos(neuronios_camada, dominio = [-0.05, 0.05]):
    pesos_final = []

    for i in range(len(neuronios_camada) - 1):
        pesos = []
        for j in range(neuronios_camada[i]):
            pesos.append([random.uniform(dominio[0], dominio[1]) for i in range(neuronios_camada[i + 1])])
        pesos_final.append(pesos)
    return pesos_final

def feed_foward(pesos):
    ativacao = []
    for i in range(len(pesos)):
        if i == 0:
            soma_sinapse = np.dot(previsores, pesos[i])
            ativacao.append(funcao_sigmoid(soma_sinapse))
        else:
            soma_sinapse = np.dot(ativacao[i - 1], pesos[i])
            ativacao.append(funcao_sigmoid(soma_sinapse))

    return ativacao

def calcular_derivada_parcial(valor): # Função ativação tem que ser sigmoid
    return valor * (1 - valor)

def calcular_delta(erro, derivada):
    return erro * derivada

def calcular_delta_oculto(pesos, delta_saida, derivada):
    matriz_pesos = np.asmatrix(pesos).reshape(1, -1) # conceito de matriz transposta 

    pesos_delta_saida = delta_saida.dot(matriz_pesos)

    return derivada * np.array(pesos_delta_saida) # as matrizes precisam estar em uma dimensão diferente uma da outra, nesse caso 4,3 e 3,4

def get_delta_oculto(pesos, delta_saida, ativacao):
    deltas_camadas_ocultas = []  # pegar a derivada da saída

    for i in range(len(pesos) -1):
        derivada = calcular_derivada_parcial(ativacao[(len(ativacao)- 1) - (i + 1)]) # pegar de trás para frente a derivada de cada neurônio
        deltas_camadas_ocultas.append(calcular_delta_oculto(pesos[(len(pesos) - 1) - 0], delta_saida, derivada))

    return deltas_camadas_ocultas

def backpropagation(pesos, ativacao, delta_saida, delta_oculto, tx_aprendizado = 0.001, momento = 1):
    for i in range(len(pesos) - 1):
        if i == 0:
            camada_transposta = np.transpose(ativacao[(len(ativacao)- 1) - (i + 1)])
            peso_atualizado = camada_transposta.dot(delta_saida)
        else:
            peso_atualizado = ativacao[(len(ativacao)- 1) - (i + 1)] * delta_oculto

    return peso_atualizado

def treinar(epocas, neuronios_camada):
    pesos = inicializar_pesos(neuronios_camada)

    pesos[0] = pesos0
    pesos[1] = pesos1
    
    execucoes = 0
    while execucoes < epocas:
        iteracao = 0
               
        ativacao = feed_foward(pesos)

        resultado_camada_saida = ativacao[len(ativacao) - 1]
        classe_reshaped = classe.values.reshape(-1,1)

        erro = funcao_custo(classe_reshaped, resultado_camada_saida)
        
        derivada_saida = calcular_derivada_parcial(resultado_camada_saida)
        delta_saida = calcular_delta(erro, derivada_saida)

        delta_camada_oculta = get_delta_oculto(pesos, delta_saida, ativacao)

        backpropagation(pesos, ativacao, delta_saida, delta_camada_oculta) 

        erro_medio_absoluto = np.mean(erro)

        if erro_medio_absoluto > 0:
            precisao = 1 - erro_medio_absoluto # estratégia de atualização por épocas ao invés de por registros igual o perceptron.
            print('Precisão: ', round((precisao) * 100, 2))
            
            count = 0
            
            for i in camada_entrada:
                novo_peso = atualizar_peso(i, pesos[count], erro)
                pesos[count] = novo_peso
                count += 1
            
            iteracao += 1
          
        execucoes += 1
    print('Precisão final: ', precisao)

neuronios_camada = [len(previsores.columns)] # adicionado neurônios da camada de entrada
neuronios_camada.append(3) #camada oculta
neuronios_camada.append(1) #neurônio de saída.

treinar(20, neuronios_camada)
