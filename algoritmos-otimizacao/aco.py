import math
import random
import numpy as np
import pandas as pd
import itertools

dataframe = pd.read_csv('dataset/berlin.csv')

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
        colonia.append(random.randint(0, n_cidades))

    return colonia

def get_distancia_cidades_vizinhas(formigas, cidades, dataframe, combinacao_cidades):
    distancias = []

    for i in formigas:
        coordenadas_cidade = dataframe[dataframe['index'] == i]
        coordenadas_cidade = coordenadas_cidade.iloc[:, 1:3].values
        
        df_cidades = dataframe[dataframe['index'] != i]
        df_cidades = df_cidades.values 

        distancia_formiga = {}      
        
        for vizinho in df_cidades:                
            distancia_formiga[(i, vizinho[0])] =  get_distancia_entre_pontos([coordenadas_cidade[0][0], coordenadas_cidade[0][1]], \
                            [vizinho[1], vizinho[2]])

        distancias.append(distancia_formiga)

    return distancias

def get_probabilidade_movimento(distancia_cidades_vizinhas, cidades, alfa = 1, beta = 5):
    
    for distancia in distancia_cidades_vizinhas:
        p = []
        for cidade in distancia:
            inverso_distancia = 1 / distancia[cidade]
            
            p.append((cidades[cidade][0]**alfa) * (inverso_distancia**beta) / 1)

    return p

combinacao_cidades = list(itertools.permutations(dataframe['index'].values, 2))

cidades = get_dicionario_cidades(combinacao_cidades)

formigas = iniciar_colonia(20, len(dataframe))

distancia_cidades_vizinhas = get_distancia_cidades_vizinhas(formigas, cidades, dataframe, combinacao_cidades)

definir_movimento_formigas = get_probabilidade_movimento(distancia_cidades_vizinhas, cidades) # TODO: estou pegando a distância, mas falta os feromônios

print(get_distancia_entre_pontos([10,20], [30,40]))