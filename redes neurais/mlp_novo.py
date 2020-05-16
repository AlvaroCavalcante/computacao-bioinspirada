import random
import numpy as np
import pandas as pd
import math

df = pd.DataFrame([[0,0,0], [0,1,1], [1,0,1], [1,1,1]], columns = ['X', 'Y', 'CLASSE'])

previsores = df.iloc[:, 0:2] 
classe = df['CLASSE']

def somatoria(entradas, pesos):
    return np.dot(entradas, pesos)    

def funcao_degrau(soma):
    if soma >= 1:
        return 1
    return 0

def funcao_sigmoid(soma):
    resultado = 1 / (1 + math.e ** -soma)
    return resultado

def funcao_custo(valor_correto, valor_previsto):
    erro = abs(valor_correto - valor_previsto) #não gerar valores negativos
    return erro

def atualizar_peso(entrada, peso, erro, tx_aprendizado = 0.2):
    novo_peso = peso + (tx_aprendizado * entrada * erro)
    print('peso atualizado', novo_peso)
    return novo_peso

def get_quantidade_pesos(neuronios_camada):
    numero_conexoes = []
    count = 1

    for i in range(len(neuronios_camada) - 1):
        numero_conexoes.append(neuronios_camada[i] * neuronios_camada[count])
        count += 1
    
    return numero_conexoes, sum(numero_conexoes)

def treinar(epocas, neuronios_camada):
    numero_conexoes_camada, quantidade_conexoes = get_quantidade_pesos(neuronios_camada) #estou pegando corretamente a soma e neurônios por camada

    pesos_final = []
    for i in range(len(neuronios_camada) - 1):
        pesos = []
        for j in range(neuronios_camada[i]):
            pesos.append([random.random() for i in range(neuronios_camada[i + 1])])
        pesos_final.append(pesos)

    execucoes = 0
    while execucoes < epocas:
        precisao = 100
        iteracao = 0
        
        np.random.shuffle(previsores.values) # embaralhar os valores dos previsores, por que sem isso, podemos ter sempre uma ordem fixa de ajuste de pesos, prejudicando a rede

        for i in previsores.values:
            camada_entrada = i   
            
            peso_inicial = - neuronios_camada[0]
            peso_final = 0
            ativacao = []

            for i in range(len(neuronios_camada) - 1): # esse len - 1 é basicamente o número de vezes que vou repetir meu cálculo básico de soma + f - ativação (apenas na entrada que não é feito)
                for j in range(neuronios_camada[i+1]): # iteração por todos os neurônios enquanto o for de cima é a iteração pelas camadas
                    if i == 0:
                        peso_inicial += neuronios_camada[i]
                        peso_final += neuronios_camada[i]
                        soma_neuronio = somatoria(camada_entrada, pesos[peso_inicial:peso_final])
                        ativacao.append(funcao_sigmoid(soma_neuronio))
                    else:
                        peso_inicial = peso_final
                        peso_final += neuronios_camada[i]
                        soma_neuronio = somatoria(ativacao[0:neuronios_camada[i]], pesos[peso_inicial:peso_final])
                        ativacao.append(funcao_degrau(soma_neuronio))

                if len(ativacao) > 3: del ativacao[0:neuronios_camada[i]] 
        
            erro = funcao_custo(classe[iteracao], ativacao[0]) #baseado no meu resultado previsto, dado na última função de ativação.
        
            if erro > 0:
                precisao -= 100 / len(previsores) 
                print('Precisão: ', precisao)
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
