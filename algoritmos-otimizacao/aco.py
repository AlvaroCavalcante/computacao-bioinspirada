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
        cidades[i] = [0.000001] 

    return cidades

def iniciar_colonia_aleatoria(n_formigas, n_cidades):
    colonia = []

    for i in range(n_formigas):
        colonia.append([(random.randint(0, n_cidades),)])

    return colonia

def iniciar_colonia(n_formigas):
    colonia = []

    for i in range(n_formigas):
        colonia.append([(0,)])

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

def movimentar_formigas(formigas, arestas_cidades_temporarias, movimento_formigas, distancia_percorrida, Q=100, p=0.5):
    for i in range(len(formigas)):
        formigas[i].append(movimento_formigas[i]) # aplica a movimentação da formiga
        feromonios_depositados = Q / distancia_percorrida[i] # calcula a quantidade de feromônios
        # aplica os feromônios nas arestas da cidade
        arestas_cidades_temporarias[movimento_formigas[i]] = \ 
            [(1 - p) * arestas_cidades_temporarias[movimento_formigas[i]][0] + feromonios_depositados]

    return formigas

def aco(n_formigas, dataframe, epocas = 15):
    combinacao_cidades = list(itertools.permutations(dataframe['index'].values, 2))

    arestas_cidades = get_dicionario_cidades(combinacao_cidades)

    melhor_distancia = []
    melhor_caminho = []

    for i in range(epocas):
        execucoes = 0
        formigas = iniciar_colonia(n_formigas)
        distancia_total_formigas = [0] * n_formigas # inicializa a distância percorrida por cada formiga
        arestas_cidades_temporarias = arestas_cidades.copy()

        while execucoes < len(dataframe) -1:    
            distancia_cidades_vizinhas = get_distancia_cidades_vizinhas(formigas, dataframe)
            
            movimento_formigas, distancia_percorrida = get_proximo_movimento(distancia_cidades_vizinhas, arestas_cidades)
                
            formigas = movimentar_formigas(formigas, arestas_cidades_temporarias, movimento_formigas, distancia_percorrida)
            
            distancia_total_formigas = list(map(lambda x, y: x + y, distancia_total_formigas, distancia_percorrida))

            execucoes += 1
        
        arestas_cidades = arestas_cidades_temporarias
        melhor_distancia.append(min(distancia_total_formigas)) # TODO: Somar os feromônios de cada aresta
        melhor_caminho = formigas[distancia_total_formigas.index(min(distancia_total_formigas))]

    return melhor_distancia, melhor_caminho

melhor_distancia, melhor_caminho = aco(51, dataframe)
