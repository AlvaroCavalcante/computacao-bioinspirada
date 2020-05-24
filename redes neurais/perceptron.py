import random
import numpy as np
import pandas as pd
import math 

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
    resultado = list(1 / (1 + math.e ** -soma))
    index_excitacao = resultado.index(max(resultado)) 
    resultado = [0] * len(soma)
    resultado[index_excitacao] = 1
    
    return resultado

def funcao_custo(valor_correto, valor_previsto):
    erro = list(abs(np.array(valor_correto) - np.array(valor_previsto)))
    return sum(erro) # valor escalar

def atualizar_peso(entrada, peso, erro, tx_aprendizado = 0.001):
    novo_peso = peso + (tx_aprendizado * entrada * erro)
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


def testar(pesos, x_teste, y_teste, f_ativacao):
    precisao = 0
    iteracao = 0
    for i in x_teste.values:
        entradas = i   
        soma = somatoria(entradas, pesos)
        
        ativacao = f_ativacao(soma)
        
        erro = funcao_custo_rmse(y_teste[iteracao], ativacao)

        if erro == 0:
            precisao += 100 / len(x_teste)
        
        iteracao += 1
    
    return precisao

def treinar(epocas, f_ativacao, pesos, x_treinamento, y_treinamento, x_teste, y_teste):
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
        
            ativacao = f_ativacao(soma)
        
            erro = funcao_custo_rmse(y_treinamento[iteracao], ativacao) # baseado no meu resultado previsto, dado na última função de ativação.
        
            if erro > 0:
                count = 0
                    
                for i in entradas:
                    novo_peso = atualizar_peso(i, pesos[count], erro)
                    pesos[count] = novo_peso
                    count += 1
            else:
                precisao += 100 / len(x_treinamento)
                precisoes_treinamento.append(precisao)

            iteracao += 1
        
        precisoes_teste.append(testar(pesos, x_teste, y_teste, f_ativacao))
        execucoes += 1

    return max(precisoes_treinamento), max(precisoes_teste), pesos

previsores['bias'] = 1

def executar_perceptron(funcao_ativacao, epocas, dominio_pesos = [0, 1]):
    precisao_treinamento = []
    precisao_teste = []
    resultado_final = []

    for i in range(10):
        pesos = inicializar_pesos(dominio_pesos) # Alterando os pesos em cada inicialização
        x_treinamento, y_treinamento, x_teste, y_teste, x_validacao, y_validacao = dividir_dataframe(previsores, classe, 0.7, 0.15, 0.15)

        treinamento = treinar(epocas, funcao_ativacao, pesos, x_treinamento, y_treinamento,
                                     x_teste, y_teste)
                                     
        precisao_treinamento.append(treinamento[0])
        precisao_teste.append(treinamento[1])

        resultado_final.append(testar(treinamento[2], x_validacao, y_validacao, funcao_ativacao))

    print('Melhor precisão de treinamento', max(precisao_treinamento))
    print('Melhor precisão de teste', max(precisao_teste))
    print('Melhor precisão de validação', max(precisao_teste))

executar_perceptron(funcao_ativacao_sigmoid, 350, [0, 0.5])

