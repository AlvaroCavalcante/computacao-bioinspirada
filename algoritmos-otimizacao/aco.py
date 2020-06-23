import math
import random
import numpy as np

def get_distancia_entre_pontos(cidade1, cidade2):
    diferenca_coord = abs(cidade1[0] - cidade2[0])**2 + abs(cidade1[1] - cidade2[1])**2
    distancia = math.sqrt(diferenca_coord)
    return distancia

print(get_distancia_entre_pontos([10,20], [30,40]))