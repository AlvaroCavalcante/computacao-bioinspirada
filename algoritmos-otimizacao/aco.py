import math
import random
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

dataframe = pd.read_csv('/home/alvaro/Documentos/mestrado/computação bio/algoritmos-otimizacao/dataset/berlin.csv')

def get_distancia_entre_pontos(cidade1, cidade2): # distância euclidiana 
    diferenca_coord = (cidade1[0] - cidade2[0])**2 + (cidade1[1] - cidade2[1])**2
    distancia = math.sqrt(diferenca_coord)
    return distancia

def get_dicionario_cidades(combinacao_cidades):
    cidades = {}
    for i in combinacao_cidades:
        cidades[i] = [0.000001] # quantidade inicial (sugerida) de feromônios

    return cidades

def iniciar_colonia_aleatoria(n_formigas, n_cidades):
    colonia = []
    random.seed(0)

    for i in range(n_formigas):
        colonia.append([(random.randint(1, n_cidades),)])

    return colonia

def iniciar_colonia(n_formigas):
    colonia = []

    for i in range(n_formigas):
        colonia.append([(1,)])

    return colonia

def get_distancia_cidades_vizinhas(formigas, dataframe):
    distancias = []

    for i in formigas:
        coordenadas_formiga = dataframe[dataframe['index'] == i[-1][-1]].values # pega a posição atual da formiga
        
        removed_cities = list(map(lambda x: x[-1], i)) # Listando as cidades que ela já visitou
        df_cidades = dataframe[~dataframe['index'].isin(removed_cities)].values # removendo as cidades que a formiga já visitou

        distancia_formiga = {}      
        
        for vizinho in df_cidades:  # para cada cidade, calcula a distância do ponto atual da formiga e a cidade              
            distancia_formiga[(i[-1][-1], int(vizinho[0]))] = get_distancia_entre_pontos([coordenadas_formiga[0][1], \
                                coordenadas_formiga[0][2]], [vizinho[1], vizinho[2]])

        distancias.append(distancia_formiga)

    return distancias # retorna a distância de todas as cidades vizinhas de cada uma das formigas

def calcular_probabilidade_movimento(distancia, arestas_cidades, alfa, beta):
    probabilidades = []

    for cidade in distancia: # calcula a probabilidade de se mover para a cidade
        inverso_distancia = 1 / distancia[cidade]
        
        p = (arestas_cidades[cidade][0]**alfa) * (inverso_distancia**beta)
        probabilidades.append(p)

    return probabilidades

def get_proximo_movimento(distancia_cidades_vizinhas, arestas_cidades, alfa=1, beta=5):
    proximos_movimentos = []
    distancias_percorridas = []
    
    for distancia in distancia_cidades_vizinhas:
        probabilidades = calcular_probabilidade_movimento(distancia, arestas_cidades, alfa, beta)
        
        proba_cidade = np.array(probabilidades) / sum(probabilidades) 
        
        cidade_mais_proxima = list(distancia.keys())[proba_cidade.argmax()] # pega a cidade com a maior P
        
        proximos_movimentos.append(cidade_mais_proxima) # registra o próximo movimento da formiga

        distancias_percorridas.append(distancia[cidade_mais_proxima]) # registra a distância percorrida do próximo movimento
               
    return proximos_movimentos, distancias_percorridas

def movimentar_formigas(formigas, movimento_formigas):
    for i in range(len(formigas)):
        formigas[i].append(movimento_formigas[i]) # aplica a movimentação da formiga

    return formigas

def atualizar_feromonios(formigas, distancia_total_formigas, arestas_cidades, Q=100, p=0.5, b=5):
    count = 0
    for formiga in formigas:
        formiga = formiga[1:len(formiga)-1]
        for aresta in formiga:
            feromonios_depositados = Q / distancia_total_formigas[count] # calcula a quantidade de feromônios
            # aplica os feromônios nas arestas da cidade

            feromonio_formiga_elitista = b*feromonios_depositados if \
                min(distancia_total_formigas) == distancia_total_formigas[count] else 0

            arestas_cidades[aresta] = \
                [(1 - p) * arestas_cidades[aresta][0] + feromonios_depositados + feromonio_formiga_elitista]

        count += 1

    return arestas_cidades 


def aco(n_formigas, dataframe, epocas = 5):
    combinacao_cidades = list(itertools.permutations(dataframe['index'].values, 2))

    arestas_cidades = get_dicionario_cidades(combinacao_cidades)

    melhor_distancia = []
    distancia_media = []
    melhor_caminho = []

    for i in range(epocas):
        execucoes = 0
        formigas = iniciar_colonia_aleatoria(n_formigas, len(dataframe) - 1)
        distancia_total_formigas = [0] * n_formigas # inicializa a distância percorrida por cada formiga
               
        while execucoes < len(dataframe) -1:    
            distancia_cidades_vizinhas = get_distancia_cidades_vizinhas(formigas, dataframe)
            
            movimento_formigas, distancia_percorrida = get_proximo_movimento(distancia_cidades_vizinhas, arestas_cidades)
                
            formigas = movimentar_formigas(formigas, movimento_formigas)
            
            distancia_total_formigas = list(map(lambda x, y: x + y, distancia_total_formigas, distancia_percorrida))

            execucoes += 1
        
        arestas_cidades = atualizar_feromonios(formigas, distancia_total_formigas, arestas_cidades) # atualizar a lista de feromônios
        melhor_distancia.append(min(distancia_total_formigas)) 
        distancia_media.append(sum(distancia_total_formigas) / len(distancia_total_formigas))
        melhor_caminho = formigas[distancia_total_formigas.index(min(distancia_total_formigas))]

    return melhor_distancia, melhor_caminho, distancia_media

def get_coordenadas_melhor_rota(dataframe, melhor_caminho):
    melhor_x = []
    melhor_y = []
    for caminho in melhor_caminho:
        cidade_inicial = dataframe[dataframe['index'] == caminho[0]]
        melhor_x.append(cidade_inicial['x'].values[0])
        melhor_y.append(cidade_inicial['y'].values[0])

        if len(caminho) == 2:
            cidade_final = dataframe[dataframe['index'] == caminho[1]]
            melhor_x.append(cidade_final['x'].values[0])
            melhor_y.append(cidade_final['y'].values[0])

    return melhor_x, melhor_y

def mostrar_grafico_resultados(dataframe, melhor_x, melhor_y):
    plt.scatter(dataframe['x'].values, dataframe['y'].values)
    plt.plot(melhor_x, melhor_y, '.r-') 
    plt.show()

melhor_distancia, melhor_caminho, distancia_media = aco(51, dataframe)

melhor_x, melhor_y = get_coordenadas_melhor_rota(dataframe, melhor_caminho)

mostrar_grafico_resultados(dataframe, melhor_x, melhor_y)

plt.plot(distancia_media, color = 'orange')
plt.plot(melhor_distancia, color = 'blue')
plt.show()

print('Melhor distância', melhor_distancia)
print('Distância média', distancia_media)
print('Melhor caminho', melhor_caminho)