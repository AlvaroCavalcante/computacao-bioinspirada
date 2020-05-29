import random
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt

dataframe = pd.read_csv('/home/alvaro/Documentos/mestrado/computação bio/redes neurais/datasets/iris2.csv', header = 0)
# dataframe = pd.read_csv('/home/alvaro/Documentos/mestrado/computação bio/redes neurais/datasets/wine.csv', header = 0)

previsores = dataframe.iloc[:, 0:4] 
classe = dataframe['class']

# previsores = dataframe.iloc[:, 1:14] 
# classe = dataframe['Wine']

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

def codificar_classe():
    classe_codificada = {}
    
    array_classe = np.array([[1]  + ([0] * (len(classe.unique()) - 1)) ])
    
    count = 1
    
    classe_codificada[0] = array_classe.copy()
    
    for i in range(len(classe.unique()) - 1):

        array_classe[0][count - 1] = 0
        array_classe[0][count] = 1  
        classe_codificada[count] = array_classe.copy()
        count += 1
    
    return classe_codificada
        
classe_codificada = codificar_classe()

classe_nova = []

for i in classe:
    classe_nova.append(classe_codificada[i])
    
classe_nova = np.array(classe_nova).reshape(len(classe), 3)

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


def inicializar_pesos(dominio):
    pesos_final = []
    # random.seed(40)
    for i in range(len(previsores.columns)):
        pesos = [] 
        for j in range(len(dict_classes)):
            pesos.append(random.uniform(dominio[0], dominio[1]))
        pesos_final.append(pesos)
    return pesos_final

def somatoria(entradas, pesos):
    return np.dot(entradas, pesos)    

def funcao_ativacao_sigmoid(soma):
    valor_ativacao = 1 / (1 + math.e ** -soma)
    index_excitacao = np.argmax(valor_ativacao, 1) 
    
    count = 0
    neuronios_excitado = valor_ativacao.copy()

    for i in index_excitacao:
        neuronios_excitado[count] = 0
        neuronios_excitado[count][i] = 1
        count += 1
        
    return neuronios_excitado, valor_ativacao
    
def funcao_custo(valor_correto, valor_previsto, valor_ativacao):
    erro = list(abs(np.array(valor_correto) - np.array(valor_previsto)))
    valor_erro = list(abs(np.array(valor_correto) - np.array(valor_ativacao)))

    acerto = 0
    
    for i in erro:
        if sum(i) == 0:
            acerto += 1
    
    return sum(sum(erro)), acerto, sum(sum(valor_erro)) # valor escalar

def funcao_custo_mse(valor_correto, valor_previsto, valor_ativacao):
    erro = list(abs(np.array(valor_correto) - np.array(valor_previsto)))
    valor_erro = list(abs(np.array(valor_correto) - np.array(valor_ativacao)))

    acerto = 0
    
    for i in erro:
        if sum(i) == 0:
            acerto += 1

    erro_quadratico = list(map(lambda x: x**2, valor_erro))
    soma_erro_quadratico = sum(erro_quadratico)

    return sum(erro), acerto, sum(soma_erro_quadratico)

def atualizar_bias(entrada, peso, erro, tx_aprendizado):
    novo_peso = peso + np.float64(tx_aprendizado * erro)
    return novo_peso

def atualizar_peso(entrada, peso, erro, tx_aprendizado):
    novo_peso = peso + np.mean((tx_aprendizado * entrada * erro))
    return novo_peso

def testar(pesos, x_previsores, y_classe, f_ativacao, f_custo):
    entradas = x_previsores.values  
    soma = somatoria(entradas, pesos)
    
    neuronio_excitado, valor_ativacao = f_ativacao(soma)
    
    erro, acertos, valor_erro = f_custo(y_classe, neuronio_excitado, valor_ativacao)
       
    return acertos / len(x_previsores)

def treinar(epocas, f_ativacao, f_custo, pesos, x_treinamento, y_treinamento,
                                     x_teste, y_teste, tx_aprendizado):
    execucoes = 0
    precisoes_treinamento = []
    precisoes_teste = []
    melhores_pesos = []

    while execucoes < epocas:

        entradas = x_treinamento.values   
        soma = somatoria(entradas, pesos)
    
        neuronio_excitado, valor_ativacao = f_ativacao(soma)

        erro, acertos, valor_erro = f_custo(y_treinamento, neuronio_excitado, valor_ativacao)
    
        count = 0
        precisoes_treinamento.append(acertos / len(x_treinamento))    

        melhores_pesos = pesos.copy() if precisoes_treinamento[execucoes] >= max(precisoes_treinamento) else melhores_pesos

        for i in range(entradas.shape[1]): # o for tem que atualizar cada peso da camada
            if i == 4:
                novo_peso = atualizar_bias(entradas[:, i], pesos[i], valor_erro, tx_aprendizado)
            else:
                novo_peso = atualizar_peso(entradas[:, i], pesos[i], valor_erro, tx_aprendizado)
            pesos[count] = novo_peso
            count += 1
        
        precisoes_teste.append(testar(pesos, x_teste, y_teste, f_ativacao, f_custo))
        execucoes += 1
    
    return precisoes_treinamento, precisoes_teste, melhores_pesos

previsores['bias'] = 1

def exibir_resultados(precisao_treinamento, precisao_teste, resultado_final):
    print('Melhor precisão de treinamento', max(precisao_treinamento))
    print('Melhor precisão de teste', max(precisao_teste))
    print('Melhor precisão de validação', max(resultado_final))
    print('Média precisão de treinamento', np.mean(precisao_treinamento))
    print('Média precisão de teste', np.mean(precisao_teste))
    print('Média precisão de validação', np.mean(resultado_final))
    print('Desvio Padrão precisão de treinamento', np.std(precisao_treinamento))
    print('Desvio Padrão precisão de teste', np.std(precisao_teste))
    print('Desvio Padrão precisão de validação', np.std(resultado_final))

def plotar_convergencia(precisao_treinamento, precisao_teste):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
    
    axes[0].plot(precisao_treinamento, color = 'blue')
    axes[0].legend(['Treinamento'])

    axes[1].plot(precisao_teste, color = 'orange')
    axes[1].legend(['Teste'])

    plt.xlabel('Épocas')
    plt.ylabel('Precisão')
    plt.show()

def executar_perceptron(funcao_ativacao, funcao_custo, epocas, dominio_pesos = [0, 1], 
                        tx_aprendizado = 0.001):
    
    convergencia_treinamento = [0]
    convergencia_teste = [0]
    precisao_treinamento = []
    precisao_teste = []
    resultado_final = []

    for i in range(30):
        pesos = inicializar_pesos(dominio_pesos)
        x_treinamento, y_treinamento, x_teste, y_teste, x_validacao, y_validacao = dividir_dataframe(previsores, classe_nova, 0.7, 0.15, 0.15)

        treinamento = treinar(epocas, funcao_ativacao, funcao_custo, pesos, x_treinamento, y_treinamento,
                                     x_teste, y_teste, tx_aprendizado)
                                     
        convergencia_treinamento = treinamento[0] if max(treinamento[0]) >= max(convergencia_treinamento) else convergencia_treinamento
        convergencia_teste = treinamento[1] if max(treinamento[1]) >= max(convergencia_teste) else convergencia_teste

        precisao_treinamento.append(max(treinamento[0]))
        precisao_teste.append(max(treinamento[1]))
        resultado_final.append(testar(treinamento[2], x_validacao, y_validacao, funcao_ativacao, funcao_custo))

    plotar_convergencia(convergencia_treinamento, convergencia_teste)
    exibir_resultados(precisao_treinamento, precisao_teste, resultado_final)


executar_perceptron(funcao_ativacao_sigmoid, funcao_custo_mse, 400, [-0.005, 0.005])

