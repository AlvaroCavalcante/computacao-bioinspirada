import random
import numpy as np
import pandas as pd

df = pd.DataFrame([[0,0,0], [0,1,1], [1,0,1], [1,1,1]], columns = ['X', 'Y', 'CLASSE'])

previsores = df.iloc[:, 0:2] 
classe = df['CLASSE']

pesos = [random.random() for i in range(len(previsores.columns))]

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

def treinar(epocas):
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

treinar(20)
