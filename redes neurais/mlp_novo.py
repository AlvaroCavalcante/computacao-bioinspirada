import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# df = pd.DataFrame([[0,0,0], [0,1,1], [1,0,1], [1,1,0]], columns = ['X', 'Y', 'CLASSE'])
dataframe = pd.read_csv('/home/alvaro/Documentos/mestrado/computação bio/redes neurais/datasets/breast_cancer.csv', header = 0)

dataframe = dataframe.drop(columns = 'id')
dataframe = dataframe.iloc[:, 0:31] 

# previsores = df.iloc[:, 0:2] 
# classe = df['CLASSE']

previsores = dataframe.iloc[:, 1:31] 
classe = dataframe['diagnosis']

pesos0 = np.array([[-0.424, -0.740, -0.961],
                   [0.358, -0.577, -0.469]])
    
pesos1 = np.array([[-0.017], [-0.893], [0.148]])

def z_score_normalization(value):
    media = previsores[value.name].mean()
    desvio_padrao = previsores[value.name].std()

    return (value - media) / desvio_padrao

previsores = previsores.apply(lambda row: z_score_normalization(row))

def get_dicionario_classes(classe):
    dict_classes = {}
    count = 0
    
    for i in classe.unique():
        dict_classes[i] = count
        count += 1
        
    return dict_classes

dict_classes = get_dicionario_classes(classe)

def transformar_categorico_em_numerico(valor, dict_classes):
    return dict_classes[valor]
    
classe = classe.apply(lambda row: transformar_categorico_em_numerico(row, dict_classes))

def somatoria(entradas, pesos):
    return np.dot(entradas, pesos)    

def funcao_sigmoid(valor):
    resultado = 1 / (1 + np.exp(-valor))
    return resultado

def funcao_custo(valor_correto, valor_previsto):
    valor_erro = valor_correto - valor_previsto 
    return valor_erro

def inicializar_pesos(neuronios_camada, dominio = [-1, 1]):
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
    matriz_pesos = np.transpose(np.asmatrix(pesos)) #.reshape(1, -1) # conceito de matriz transposta 

    pesos_delta_saida = delta_saida.dot(matriz_pesos)

    return derivada * np.array(pesos_delta_saida) # as matrizes precisam estar em uma dimensão diferente uma da outra, nesse caso 4,3 e 3,4

def get_delta_oculto(pesos, delta_saida, ativacao):
    deltas_camadas_ocultas = []  # pegar a derivada da saída

    for i in range(len(pesos) -1):
        derivada = calcular_derivada_parcial(ativacao[len(ativacao) - (i + 2)]) # pegar de trás para frente a derivada de cada neurônio
        deltas_camadas_ocultas.append(calcular_delta_oculto(pesos[len(pesos) - (i + 1)], delta_saida, derivada))

    return deltas_camadas_ocultas

def backpropagation(pesos, ativacao, delta_saida, delta_oculto, tx_aprendizado = 0.3, momento = 1):
    for i in range(len(pesos)):
        if i == 0:
            camada_transposta = np.transpose(ativacao[i])
            delta_x_entrada = camada_transposta.dot(delta_saida)
            
            pesos[len(pesos) - (1 + i)] = (pesos[len(pesos) - (1 + i)] * momento) + (tx_aprendizado * delta_x_entrada)
        else:
            camada_transposta = np.transpose(previsores)
            pesos[len(pesos) - (1 + i)] = (pesos[len(pesos) - (1 + i)] * momento) + (tx_aprendizado * camada_transposta.dot(delta_oculto[0])).values
            
    return pesos

def get_precisao(valor_correto, valor_previsto):
    previsao = valor_previsto.copy()

    previsao[previsao >= 0.5] = 1
    previsao[previsao < 0.5] = 0

    precisao = (valor_correto == previsao).sum() / len(valor_correto)
    return precisao

def dividir_dataframe(previsores, classe, p_treinamento, p_teste, p_validacao):
    x_treinamento = previsores.sample(frac = p_treinamento)
    y_treinamento = classe[x_treinamento.index]
    
    x_teste_sem_previsores = previsores.drop(x_treinamento.index)
    nova_p_teste = p_teste / (1 - p_treinamento)
    
    x_teste = x_teste_sem_previsores.sample(frac = nova_p_teste)
    y_teste = classe[x_teste.index]
    
    x_validacao = x_teste_sem_previsores.drop(x_teste.index)
    y_validacao = classe[x_validacao.index]
    
    return x_treinamento.reset_index(drop=True), y_treinamento, \
    x_teste.reset_index(drop=True), y_teste, x_validacao.reset_index(drop=True), y_validacao

def treinar(epocas, neuronios_camada, funcao_ativacao, funcao_custo, pesos, x_treinamento,
                                     y_treinamento, x_teste, y_teste, tx_aprendizado):
    execucoes = 0
    precisoes_treinamento = []
    
    while execucoes < epocas:               
        ativacao = feed_foward(pesos)

        resultado_camada_saida = ativacao[len(ativacao) - 1]
        classe_reshaped = classe.values.reshape(-1,1)

        erro = funcao_custo(classe_reshaped, resultado_camada_saida)

        precisoes_treinamento.append(get_precisao(classe_reshaped, resultado_camada_saida))

        derivada_saida = calcular_derivada_parcial(resultado_camada_saida)
        delta_saida = calcular_delta(erro, derivada_saida)

        delta_camada_oculta = get_delta_oculto(pesos, delta_saida, ativacao)

        pesos = backpropagation(pesos, ativacao, delta_saida, delta_camada_oculta) 
          
        execucoes += 1
        
    return precisoes_treinamento
        
neuronios_camada = [len(previsores.columns)] # adicionado neurônios da camada de entrada
neuronios_camada.append(3) #camada oculta
neuronios_camada.append(1) #neurônio de saída.

def executar_mlp(funcao_ativacao, funcao_custo, epocas, dominio_pesos = [0, 1], 
                       tx_aprendizado = 0.001, mostrar_resultados = True):

    convergencia_treinamento = [0]
    convergencia_teste = [0]
    precisao_treinamento = []
    precisao_teste = []
    resultado_final = []

    for i in range(30):
        x_treinamento, y_treinamento, x_teste, y_teste, \
        x_validacao, y_validacao = dividir_dataframe(previsores, classe, 0.7, 0.15, 0.15)

        pesos = inicializar_pesos(neuronios_camada)


        precisao = treinar(epocas, neuronios_camada, funcao_ativacao, funcao_custo, pesos, x_treinamento,
                                     y_treinamento, x_teste, y_teste, tx_aprendizado)

        print(max(precisao))

executar_mlp(funcao_sigmoid, funcao_custo, 300)

# fig, ax = plt.subplots()
# ax.plot(precisao)
# plt.show()