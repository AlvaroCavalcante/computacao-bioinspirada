import random

def funcao_aptidao(x, y):
    return (1 - x)**2 + 100*(y-x**2)**2

def get_enxame(n_particulas, dominio):
    enxame = []

    for i in range(n_particulas):
        particula = [random.uniform(dominio[i][0], dominio[i][1]) for i in range(len(dominio))]
        
        enxame.append(particula)
    
    return enxame

def get_velocidade(n_particulas, dominio):
    velocidade = []
    for i in range(n_particulas):
        velocidade.append(random.uniform(dominio[0], dominio[len(dominio) - 1]))
    
    return velocidade

def atualizar_velocidade(v_atual, p_atual, melhor_p, aptidao, 
                         dominio_v, ac1 = 2.05, ac2 = 2.05):
    
    v_aleatorio1 = random.uniform(0, ac1)
    v_aleatorio2 = random.uniform(0, ac2)
    
    v_nova = []

    count = 0
    for velocidade in v_atual:
        melhor_p_vizinho = aptidao.index(min([aptidao[count - 1]] + [aptidao[ count + 1]]))
        melhor_p_vizinho = p_atual[melhor_p_vizinho]

        v_nova.append(
            velocidade+v_aleatorio1*(melhor_p-p_atual[count])+v_aleatorio2*(melhor_p_vizinho-p_atual[count]))
        
        count += 1 

    v_final = v_nova if v_nova <= dominio_v[1] else dominio_v[1]
    v_final = v_nova if v_nova >= dominio_v[0] else dominio_v[0]
    
    return v_final

def atualizar_posicao(p_atual, nova_velocidade):
    nova_p = p_atual + nova_velocidade
    
    return nova_p

def pso(n_particulas, dominio_particulas, dominio_velocidade):
    enxame = get_enxame(n_particulas, dominio_particulas)
    velocidade = get_velocidade(n_particulas, dominio_velocidade)
    melhor_p = enxame
    execucao = 0

    while execucao < 30:
        
        aptidao = [funcao_aptidao(x, y) for x, y in enxame]
        
        melhor_p_global = enxame[aptidao.index(min(aptidao))]

        velocidade = atualizar_velocidade(velocidade, enxame, melhor_p_global, aptidao, dominio_velocidade) 

        execucao += 1
    return enxame

pso(5, [(-5 ,5), (-5 ,5)], [-2, 2])