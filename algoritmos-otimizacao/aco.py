import math
import random
import numpy as np
import pandas as pd
import itertools

dataframe = pd.read_csv('/home/alvaro/Documentos/mestrado/computação bio/algoritmos-otimizacao/dataset/berlin.csv')
dataframe['index'] = list(map(lambda x: x, dataframe.index))

def get_distancia_entre_pontos(cidade1, cidade2): # distância euclidiana 
    diferenca_coord = (cidade1[0] - cidade2[0])**2 + (cidade1[1] - cidade2[1])**2
    distancia = math.sqrt(diferenca_coord)
    return distancia

def get_dicionario_cidades(combinacao_cidades):
    cidades = {}
    for i in combinacao_cidades:
        cidades[i] = [1] 

    return cidades

def iniciar_colonia(n_formigas, n_cidades):
    colonia = []

    for i in range(n_formigas):
        colonia.append([(random.randint(0, n_cidades),)])

    return colonia

def get_distancia_cidades_vizinhas(formigas, dataframe):
    distancias = []

    for i in formigas:
        coordenadas_formiga = dataframe[dataframe['index'] == i[-1][-1]].values
        
        removed_cities = list(map(lambda x: x[-1], i))
        df_cidades = dataframe[~dataframe['index'].isin(removed_cities)].values 

        distancia_formiga = {}      
        
        for vizinho in df_cidades:                
            distancia_formiga[(i[-1][-1], int(vizinho[0]))] = get_distancia_entre_pontos([coordenadas_formiga[0][1], \
                                coordenadas_formiga[0][2]], [vizinho[1], vizinho[2]])

        distancias.append(distancia_formiga)

    return distancias

def get_proximo_movimento(distancia_cidades_vizinhas, arestas_cidades, alfa = 1, beta = 5):
    proximos_movimentos = []
    distancias_percorridas = []
    
    for distancia in distancia_cidades_vizinhas:
        proba_cidade = [0, 0]
        count = 0
        for cidade in distancia:
            inverso_distancia = 1 / distancia[cidade]
            
            p = (arestas_cidades[cidade][0]**alfa) * (inverso_distancia**beta) / 1
            
            proba_cidade = [p, count] if p > proba_cidade[0] else proba_cidade
            count += 1
        
        cidade_mais_proxima = list(distancia.keys())[proba_cidade[1]]
        
        distancias_percorridas.append(distancia[cidade_mais_proxima])
       
        proximos_movimentos.append(cidade_mais_proxima)
        
    return proximos_movimentos, distancias_percorridas

def movimentar_formigas(formigas, arestas_cidades_temporarias, movimento_formigas, distancia_percorrida, Q = 100):
    for i in range(len(formigas)):
        formigas[i].append(movimento_formigas[i])
        feromonios_depositados = Q / distancia_percorrida[i]
        arestas_cidades_temporarias[movimento_formigas[i]] = [feromonios_depositados + arestas_cidades_temporarias[movimento_formigas[i]][0]]

    return formigas

def aco(n_formigas, dataframe, epocas = 10):
    combinacao_cidades = list(itertools.permutations(dataframe['index'].values, 2))

    arestas_cidades = get_dicionario_cidades(combinacao_cidades)

    melhor_distancia = []
    melhor_caminho = []

    for i in range(epocas):
        execucoes = 0
        formigas = iniciar_colonia(n_formigas, len(dataframe) - 1)
        distancia_total_formigas = [0] * n_formigas
        arestas_cidades_temporarias = arestas_cidades.copy()

        while execucoes < len(dataframe) -1:    
            distancia_cidades_vizinhas = get_distancia_cidades_vizinhas(formigas, dataframe)
            
            movimento_formigas, distancia_percorrida = get_proximo_movimento(distancia_cidades_vizinhas, arestas_cidades)
                
            formigas = movimentar_formigas(formigas, arestas_cidades_temporarias, movimento_formigas, distancia_percorrida)
            
            distancia_total_formigas = list(map(lambda x, y: x + y, distancia_total_formigas, distancia_percorrida))

            execucoes += 1
        
        arestas_cidades = arestas_cidades_temporarias
        melhor_distancia.append(min(distancia_total_formigas))
        melhor_caminho = formigas[distancia_total_formigas.index(min(distancia_total_formigas))]

    return melhor_distancia, melhor_caminho

melhor_distancia, melhor_caminho = aco(20, dataframe)
