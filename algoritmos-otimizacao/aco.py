import math
import random
import numpy as np
import pandas as pd
import itertools

dataframe = pd.read_csv('dataset/berlin.csv')

def get_distancia_entre_pontos(cidade1, cidade2):
    diferenca_coord = abs(cidade1[0] - cidade2[0])**2 + abs(cidade1[1] - cidade2[1])**2
    distancia = math.sqrt(diferenca_coord)
    return distancia

def get_dicionario_cidades(combinacao_cidades):
    cidades = {}
    for i in combinacao_cidades:
        cidades[i] = [0, 0]

    return cidades

def iniciar_colonia(n_formigas, n_cidades):
    colonia = []

    for i in range(n_formigas):
        colonia.append(random.randint(0, n_cidades))

    return colonia

combinacao_cidades = list(itertools.combinations(dataframe['index'].values, 2))

cidades = get_dicionario_cidades(combinacao_cidades)

formigas = iniciar_colonia(20, len(dataframe))

print(get_distancia_entre_pontos([10,20], [30,40]))