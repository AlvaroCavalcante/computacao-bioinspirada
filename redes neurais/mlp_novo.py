import random
import numpy as np
import pandas as pd
import math

df = pd.DataFrame([[0,0,0], [0,1,1], [1,0,1], [1,1,1]], columns = ['X', 'Y', 'CLASSE'])

previsores = df.iloc[:, 0:2] 
classe = df['CLASSE']

def somatoria(entradas, pesos):
    return np.dot(entradas, pesos)    

def funcao_degrau(valor):
    if valor >= 1:
        return 1
    return 0

def funcao_sigmoid(valor):
    resultado = 1 / (1 + math.e ** -valor)
    return resultado

def funcao_custo(valor_correto, valor_previsto):
    erro = abs(valor_correto - valor_previsto) # não gerar valores negativos
    return erro

def atualizar_peso(entrada, peso, erro, tx_aprendizado = 0.2):
    novo_peso = peso + (tx_aprendizado * entrada * erro)
    print('peso atualizado', novo_peso)
    return novo_peso

def inicializar_pesos(neuronios_camada):
    pesos_final = []

    for i in range(len(neuronios_camada) - 1):
        pesos = []
        for j in range(neuronios_camada[i]):
            pesos.append([random.random() for i in range(neuronios_camada[i + 1])])
        pesos_final.append(pesos)
    return pesos_final

def calcular_derivada_parcial(erro):
    y = funcao_sigmoid(erro)
    derivada_parcial = y * (1 - y)
    return derivada_parcial

def calcular_delta(erro, derivada):
    return erro * derivada

def treinar(epocas, neuronios_camada):
    pesos = inicializar_pesos(neuronios_camada)

    execucoes = 0
    while execucoes < epocas:
        iteracao = 0
        
        np.random.shuffle(previsores.values) # embaralhar os valores dos previsores, por que sem isso, podemos ter sempre uma ordem fixa de ajuste de pesos, prejudicando a rede
        ativacao = []

        for i in range(len(pesos)):
            if i == 0:
                soma_sinapse = np.dot(previsores, pesos[i])
                ativacao.append(funcao_sigmoid(soma_sinapse))
            else:
                soma_sinapse = np.dot(ativacao[i - 1], pesos[i])
                ativacao.append(funcao_sigmoid(soma_sinapse))

        resultado_camada_saida = ativacao[2:3][0]
        classe_reshaped = classe.values.reshape(-1,1)

        erro = funcao_custo(classe_reshaped, resultado_camada_saida)
        derivada = calcular_derivada_parcial(erro)
        delta = calcular_delta(erro, derivada)

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

neuronios_camada = [len(previsores.columns)] #adicionado neurônios da camada de entrada
neuronios_camada.append(3) #camada oculta
neuronios_camada.append(3) #camada oculta
neuronios_camada.append(1) #neurônio de saída.

treinar(20, neuronios_camada)
