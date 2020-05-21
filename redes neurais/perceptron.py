import random
import numpy as np
import pandas as pd

dataframe = pd.read_csv('/home/alvaro/Documentos/mestrado/computação bio/redes neurais/datasets/iris2.csv', header = 0)

previsores = dataframe.iloc[:, 0:4] 
classe = dataframe['class']

def z_score_normalization(value):
    media = previsores[value.name].mean()
    desvio_padrao = previsores[value.name].std()

    return (value - media) / desvio_padrao

previsores = previsores.apply(lambda row: z_score_normalization(row) )

pesos = [random.random() for i in range(len(previsores.columns))]

def somatoria(entradas, pesos):
    return np.dot(entradas, pesos)    

def funcao_ativacao(soma):
    if soma > 0: # por default o perceptron usa o limiar de ativação (step function)
        return 1
    return 0

def funcao_custo(valor_correto, valor_previsto):
    erro = abs(valor_correto - valor_previsto) # não gerar valores negativos
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

        np.random.shuffle(previsores.values) # embaralhar os valores dos previsores, por que sem isso, podemos ter sempre uma ordem fixa de ajuste de pesos, prejudicando a rede

        for i in previsores.values:
            entradas = i   
            soma = somatoria(entradas, pesos)
        
            ativacao = funcao_ativacao(soma)
        
            erro = funcao_custo(classe[iteracao], ativacao) # baseado no meu resultado previsto, dado na última função de ativação.
        
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