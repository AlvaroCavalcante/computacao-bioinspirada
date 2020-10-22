import random
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

dataframe = pd.read_csv('/home/alvaro/Documentos/mestrado/computação bio/redes neurais/datasets/iris2.csv', header = 0)

previsores = dataframe.iloc[:, 0:4] 
classe = dataframe['class']

def normalizacao_z_score(valor):
    media = previsores[valor.name].mean()
    desvio_padrao = previsores[valor.name].std()

    return (valor - media) / desvio_padrao

previsores = previsores.apply(lambda row: normalizacao_z_score(row))

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
    
    array_classe = [1] + [0] * (len(classe.unique()) - 1)
    
    count = 1
    
    classe_codificada[0] = array_classe.copy()
    
    for i in range(len(classe.unique()) - 1):

        array_classe[count - 1] = 0
        array_classe[count] = 1     
        classe_codificada[count] = array_classe.copy()
        count += 1
    
    return classe_codificada       

classe_codificada = codificar_classe()

def substituir_classe_codificada(valor, classe_codificada):
    return classe_codificada[valor]

classe = classe.apply(lambda row: substituir_classe_codificada(row, classe_codificada))

def dividir_dataframe(previsores, classe, p_treinamento, p_teste, p_validacao):
    x_treinamento = previsores.sample(frac = p_treinamento)
    y_treinamento = classe[x_treinamento.index]
    
    x_teste_sem_previsores = previsores.drop(x_treinamento.index)
    nova_p_teste = p_teste / (1 - p_treinamento)
    
    x_teste = x_teste_sem_previsores.sample(frac = nova_p_teste)
    y_teste = classe[x_teste.index]
    
    x_validacao = x_teste_sem_previsores.drop(x_teste.index)
    y_validacao = classe[x_validacao.index]
    
    return x_treinamento.reset_index(drop=True), y_treinamento.reset_index(drop=True), \
    x_teste.reset_index(drop=True), y_teste.reset_index(drop=True), x_validacao.reset_index(drop=True), y_validacao.reset_index(drop=True)

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

def funcao_ativacao_step(soma):
    ativacao = []
    for i in soma:
        if i > 0:
            ativacao.append(1)
        else:
            ativacao.append(0)

    return ativacao, ativacao

def funcao_custo(valor_correto, valor_previsto, valor_ativacao):
    erro = list(abs(np.array(valor_correto) - np.array(valor_previsto)))
    valor_erro = list(abs(np.array(valor_correto) - np.array(valor_ativacao)))
    return sum(erro), sum(valor_erro) # valor escalar

def atualizar_peso(entrada, peso, erro, tx_aprendizado):
    novo_peso = peso + (tx_aprendizado * entrada * erro)
    return novo_peso

def atualizar_bias(entrada, peso, erro, tx_aprendizado):
    novo_peso = peso + np.float64(tx_aprendizado * erro)
    return novo_peso

def plotar_convergencia(precisao_treinamento, precisao_teste):
    plt.plot(precisao_teste)
    plt.show()
    plt.plot(precisao_treinamento)
    plt.show()

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

def get_matriz_confusao(valor_correto, valor_previsto):
    try:
        previsao = np.array(valor_previsto.copy()) # deep copy da variável
        previsao = np.where(previsao == 1)[1] # transformando em um array de valores escalares
        
        correto = np.array(list(valor_correto.values))
        correto = np.where(correto == 1)[1]
    
        matriz_confusao = confusion_matrix(correto, previsao)
    
        return matriz_confusao
    except: 
        return []

def testar(pesos, x_previsores, y_classe, f_ativacao, f_custo):
    precisao = 0
    iteracao = 0
    valores_previstos = []
    
    for i in x_previsores.values:
        entradas = i   
        soma = somatoria(entradas, pesos)
        
        neuronio_excitado, valor_ativacao = f_ativacao(soma)
        valores_previstos.append(neuronio_excitado)

        erro, valor_erro = f_custo(y_classe[iteracao], neuronio_excitado, valor_ativacao)

        if erro == 0:
            precisao += 100 / len(x_previsores)
        
        iteracao += 1

    matriz_confusao = get_matriz_confusao(y_classe, valores_previstos)

    return precisao, matriz_confusao

def treinar(epocas, f_ativacao, f_custo, pesos, x_treinamento, y_treinamento, x_teste, y_teste,
            tx_aprendizado):
    execucoes = 0
    precisoes_treinamento = [0]
    precisoes_teste = [0]
    melhor_matriz_treinamento = []
    melhor_matriz_teste = []

    while execucoes < epocas:
        precisao = 0
        iteracao = 0
        valores_previstos = []

        for i in x_treinamento.values:
            entradas = i   
            soma = somatoria(entradas, pesos)
        
            neuronio_excitado, valor_ativacao = f_ativacao(soma)

            valores_previstos.append(neuronio_excitado)
            
            erro, valor_erro = f_custo(y_treinamento[iteracao], neuronio_excitado, valor_ativacao)

            if erro == True:
                count = 0

                for i in entradas:
                    if count == len(entradas) - 1:
                        novo_peso = atualizar_bias(i, pesos[count], valor_erro, tx_aprendizado)
                    else:
                        novo_peso = atualizar_peso(i, pesos[count], valor_erro, tx_aprendizado)
                    
                    pesos[count] = novo_peso
                    count += 1
            else:
                precisao += 100 / len(x_treinamento)

            iteracao += 1
        
        precisoes_treinamento.append(precisao)
        melhor_matriz_treinamento = get_matriz_confusao(y_treinamento, valores_previstos) if precisoes_treinamento[execucoes] >= max(precisoes_treinamento) else melhor_matriz_treinamento

        teste_rede = testar(pesos, x_teste, y_teste, f_ativacao, f_custo)
        precisoes_teste.append(teste_rede[0])
        melhor_matriz_teste = teste_rede[1] if precisoes_teste[execucoes] >= max(precisoes_teste) else melhor_matriz_teste

        execucoes += 1
    return precisoes_treinamento, precisoes_teste, pesos, melhor_matriz_treinamento, melhor_matriz_teste

previsores['bias'] = 1

def executar_perceptron(funcao_ativacao, funcao_custo, epocas, dominio_pesos = [0, 1], 
                        tx_aprendizado = 0.001):
    precisao_treinamento = [0]
    precisao_teste = [0]
    resultado_final = []

    matriz_confusao_treinamento = []
    matriz_confusao_teste = []
    matriz_confusao_validacao = []

    for i in range(1):
        pesos = inicializar_pesos(dominio_pesos) # Alterando os pesos em cada inicialização
        x_treinamento, y_treinamento, x_teste, y_teste, x_validacao, y_validacao = dividir_dataframe(previsores, classe, 0.7, 0.15, 0.15)

        treinamento = treinar(epocas, funcao_ativacao, funcao_custo, pesos, x_treinamento, y_treinamento,
                                     x_teste, y_teste, tx_aprendizado)
                                     
        precisao_treinamento = treinamento[0] if max(treinamento[0]) >= max(precisao_treinamento) else precisao_treinamento
        precisao_teste = treinamento[1] if max(treinamento[1]) >= max(precisao_teste) else precisao_teste

        teste_final = testar(treinamento[2], x_validacao, y_validacao, 
                                      funcao_ativacao, funcao_custo)
        resultado_final.append(teste_final[0])

        matriz_confusao_treinamento = treinamento[3] if max(treinamento[0]) >= max(precisao_treinamento) else matriz_confusao_treinamento
        matriz_confusao_teste = treinamento[4] if max(treinamento[1]) >= max(precisao_teste) else matriz_confusao_teste
        matriz_confusao_validacao = teste_final[1] if teste_final[0] >= max(resultado_final) else matriz_confusao_validacao

    plotar_convergencia(treinamento[0], treinamento[1])
    exibir_resultados(precisao_treinamento, precisao_teste, resultado_final)
    print('Matriz de confusão de treinamento:\n', matriz_confusao_treinamento)
    print('Matriz de confusão de teste:\n', matriz_confusao_teste)
    print('Matriz de confusão de validação:\n', matriz_confusao_validacao)


executar_perceptron(funcao_ativacao_step, funcao_custo, 100, [-0.005, 0.005])

