import random

def funcao_aptidao(x, y):
    return (1 - x)**2 + 100*(y-x**2)**2

def get_enxame(individuos, dominio):
    enxame = []
    for i in range(individuos):
        enxame.append(random.uniform(dominio[0], dominio[len(dominio) - 1]))
    
    return enxame

def get_velocidade(individuos, dominio):
    velocidade = []
    for i in range(individuos):
        velocidade.append(random.uniform(dominio[0], dominio[len(dominio) - 1]))
    
    return velocidade

def atualizar_velocidade(v_atual, p_atual, melhor_p, melhor_p_vizinho, 
                         dominio_v, ac1 = 2.05, ac2 = 2.05):
    
    v_aleatorio1 = random.uniform(0, ac1)
    v_aleatorio2 = random.uniform(0, ac2)
    
    v_nova = v_atual+v_aleatorio1*(melhor_p-p_atual)+v_aleatorio2*(melhor_p_vizinho-p_atual)
    
    v_final = v_nova if v_nova <= dominio_v[1] else dominio_v[1]
    v_final = v_nova if v_nova >= dominio_v[0] else dominio_v[0]
    
    return v_final

def atualizar_posicao(p_atual, nova_velocidade):
    nova_p = p_atual + nova_velocidade
    
    return nova_p

