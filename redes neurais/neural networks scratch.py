import random
import numpy as np
import pandas as pd

df = pd.DataFrame([[0,0,0], [0,1,1], [1,0,1], [1,1,1]], columns = ['X', 'Y', 'CLASSE'])

previsores = df.iloc[:, 0:2] 
classe = df['CLASSE']

def somatoria(entradas, pesos):
    return np.dot(entradas, pesos)    

def funcao_ativacao(soma):
    if soma >= 1:
        return 1
    return 0

def funcao_custo(valor_correto, valor_previsto):
    erro = abs(valor_correto - valor_previsto) #não gerar valores negativos
    return erro

def atualizar_peso(entrada, peso, erro, tx_aprendizado = 0.2):
    novo_peso = peso + (tx_aprendizado * entrada * erro)
    print('peso atualizado', novo_peso)
    return novo_peso

def get_quantidade_pesos(neuronios_camada):
    pesos = 0
    count = 1

    for i in range(len(neuronios_camada) - 1):
        pesos += neuronios_camada[i] * neuronios_camada[count]
        count += 1
    
    return pesos

def treinar(epocas, neuronios_camada):
    pesos = [random.random() for i in range(get_quantidade_pesos(neuronios_camada))]

    execucoes = 0
    while execucoes < epocas:
        precisao = 100
        iteracao = 0
    
        for i in previsores.values:
            entradas = i   
            soma = somatoria(entradas, pesos)
        
            ativacao = funcao_ativacao(soma)
        
            erro = funcao_custo(classe[iteracao], ativacao) #baseado no meu resultado previsto, dado na última função de ativação.
        
            if erro > 0:
                precisao -= 100 / len(previsores) 
                print('Precisão: ', precisao)
                count = 0
                    
                for i in entradas:
                    novo_peso = atualizar_peso(i, pesos[count], erro)
                    pesos[count] = novo_peso
                    count += 1
            
            iteracao += 1
        
        execucoes += 1
    print('Precisão final: ', precisao)

neuronios_camada = [len(previsores.columns)] #adicionado neurônios da camada de entrada
neuronios_camada.append(3) #camada oculta
neuronios_camada.append(1) #neurônio de saída.

treinar(20, neuronios_camada)
