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

    return ativacao

def funcao_ativacao_sigmoid(soma):
    valor_ativacao = list(1 / (1 + math.e ** -soma))
    index_excitacao = valor_ativacao.index(max(valor_ativacao)) 
    neuronio_excitado = [0] * len(soma)
    neuronio_excitado[index_excitacao] = 1
    
    return neuronio_excitado, valor_ativacao

def funcao_custo(valor_correto, valor_previsto, valor_ativacao):
    erro = list(abs(np.array(valor_correto) - np.array(valor_previsto)))
    valor_erro = list(abs(np.array(valor_correto) - np.array(valor_ativacao)))
    return sum(erro), sum(valor_erro) # valor escalar

def atualizar_peso(entrada, peso, erro, tx_aprendizado = 0.1):
    novo_peso = peso + (tx_aprendizado * entrada * erro)
    return novo_peso

def atualizar_bias(entrada, peso, erro, tx_aprendizado = 0.1):
    novo_peso = peso + np.float64(tx_aprendizado * erro)
    return novo_peso

def funcao_custo_mse(valor_correto, valor_previsto):
    erro = list(np.array(valor_correto) - np.array(valor_previsto))
    erro_quadratico = list(map(lambda x: math.pow(x, 2), erro))
    soma_erro_quadratico = sum(erro_quadratico)

    return soma_erro_quadratico # / len(previsores) essa parte é apenas para atualização em epoca

def funcao_custo_rmse(valor_correto, valor_previsto):
    erro = list(np.array(valor_correto) - np.array(valor_previsto))
    erro_quadratico = list(map(lambda x: math.pow(x, 2), erro))
    soma_erro_quadratico = sum(erro_quadratico)

    return math.sqrt(soma_erro_quadratico) # / len(previsores) essa parte é apenas para atualização em epoca


def plotar_convergencia(precisao_teste, precisao_treinamento):
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

def testar(pesos, x_previsores, y_classe, f_ativacao, f_custo):
    precisao = 0
    iteracao = 0
    for i in x_previsores.values:
        entradas = i   
        soma = somatoria(entradas, pesos)
        
        neuronio_excitado, valor_ativacao = f_ativacao(soma)
        
        erro, valor_erro = f_custo(y_classe[iteracao], neuronio_excitado, valor_ativacao)

        if erro == 0:
            precisao += 100 / len(x_previsores)
        
        iteracao += 1
    
    return precisao

def treinar(epocas, f_ativacao, f_custo, pesos, x_treinamento, y_treinamento, x_teste, y_teste):
    execucoes = 0
    precisoes_treinamento = [0]
    precisoes_teste = [0]

    while execucoes < epocas:
        precisao = 0
        iteracao = 0

        np.random.shuffle(x_treinamento.values) # embaralhar os valores dos previsores, por que sem isso, podemos ter sempre uma ordem fixa de ajuste de pesos, prejudicando a rede

        for i in x_treinamento.values:
            entradas = i   
            soma = somatoria(entradas, pesos)
        
            neuronio_excitado, valor_ativacao = f_ativacao(soma)
        
            erro, valor_erro = f_custo(y_treinamento[iteracao], neuronio_excitado, valor_ativacao)

            if erro > 0:
                count = 0

                for i in entradas:
                    if count == len(entradas) - 1:
                        novo_peso = atualizar_bias(i, pesos[count], valor_erro)
                    else:
                        novo_peso = atualizar_peso(i, pesos[count], valor_erro)
                    
                    pesos[count] = novo_peso
                    count += 1
            else:
                precisao += 100 / len(x_treinamento)

            iteracao += 1
        
        precisoes_treinamento.append(precisao)
        
        precisoes_teste.append(testar(pesos, x_teste, y_teste, f_ativacao, f_custo))
        execucoes += 1
    return precisoes_treinamento, precisoes_teste, pesos

previsores['bias'] = 1

def executar_perceptron(funcao_ativacao, funcao_custo, epocas, dominio_pesos = [0, 1]):
    precisao_treinamento = []
    precisao_teste = []
    resultado_final = []

    for i in range(30):
        pesos = inicializar_pesos(dominio_pesos) # Alterando os pesos em cada inicialização
        x_treinamento, y_treinamento, x_teste, y_teste, x_validacao, y_validacao = dividir_dataframe(previsores, classe, 0.7, 0.15, 0.15)

        treinamento = treinar(epocas, funcao_ativacao, funcao_custo, pesos, x_treinamento, y_treinamento,
                                     x_teste, y_teste)
                                     
        precisao_treinamento.append(max(treinamento[0]))
        precisao_teste.append(max(treinamento[1]))

        resultado_final.append(testar(treinamento[2], x_validacao, y_validacao, funcao_ativacao, funcao_custo))

    plotar_convergencia(treinamento[0], treinamento[1])
    exibir_resultados(precisao_treinamento, precisao_teste, resultado_final)

executar_perceptron(funcao_ativacao_sigmoid, funcao_custo, 300, [-1, 1])

