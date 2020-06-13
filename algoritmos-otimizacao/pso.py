import random
import numpy as np

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

def atualizar_velocidade(v_atual, p_atual, melhor_p_particula, aptidao, 
                         dominio_v, dominio_particulas, ac1 = 2.05, ac2 = 2.05):
      
    v_nova = []
    count = 0

    for velocidade in v_atual:
        v_aleatorio1 = [random.uniform(0, ac1) for i in range(len(dominio_particulas))] # baseado nas dimensões do problema
        v_aleatorio2 = [random.uniform(0, ac2) for i in range(len(dominio_particulas))]
        
        proximo_vizinho = count + 1 if count + 1 < len(aptidao) else 0
        melhor_p_vizinho = aptidao.index(min([aptidao[count - 1]] + [aptidao[proximo_vizinho]]))

        melhor_p_vizinho = melhor_p_particula[melhor_p_vizinho]

        inteligencia_cognitiva = velocidade + (np.dot(v_aleatorio1, np.array(melhor_p_particula[count]) - np.array(p_atual[count]))) 

        inteligencia_social = np.dot(v_aleatorio2, np.array(melhor_p_vizinho) - np.array(p_atual[count]))
        
        velocidade_atualizada = inteligencia_cognitiva + inteligencia_social

        velocidade_atualizada = velocidade_atualizada if velocidade_atualizada <= dominio_v[1] else dominio_v[1]
        velocidade_atualizada = velocidade_atualizada if velocidade_atualizada >= dominio_v[0] else dominio_v[0]
 
        v_nova.append(velocidade_atualizada)
        
        count += 1 
    
    return v_nova

def atualizar_posicao(p_atual, velocidade):
    velocidade = np.stack(( np.array(velocidade), np.array(velocidade)), axis=1) 
    nova_p = np.array(p_atual) + np.array(velocidade)
    
    return nova_p

def get_melhor_posicao(enxame_anterior, enxame, aptidao_anterior):
    nova_aptidao = [funcao_aptidao(x, y) for x, y in enxame]

    count = 0
    melhor_posicao = []
    melhor_aptidao = []

    for i in aptidao_anterior:
        melhor_posicao.append(enxame[count] if nova_aptidao[count] < aptidao_anterior[count] else enxame_anterior[count])
        melhor_aptidao.append(nova_aptidao[count] if nova_aptidao[count] < aptidao_anterior[count] else aptidao_anterior[count])
        count +=1

    return melhor_posicao, nova_aptidao, melhor_aptidao

def pso(n_particulas, dominio_particulas, dominio_velocidade):
    enxame = get_enxame(n_particulas, dominio_particulas)
    velocidade = get_velocidade(n_particulas, dominio_velocidade)

    melhor_p_particula = enxame
    enxame_anterior = []
    execucao = 0

    while execucao < 1000:
        
        if len(enxame_anterior) == 0:
            aptidao = [funcao_aptidao(x, y) for x, y in enxame]
            velocidade = atualizar_velocidade(velocidade, enxame, melhor_p_particula,
                                        aptidao, dominio_velocidade, dominio_particulas)
        else:
            melhor_p_particula, aptidao, melhor_aptidao = get_melhor_posicao(enxame_anterior, enxame, aptidao)            
            velocidade = atualizar_velocidade(velocidade, enxame, melhor_p_particula,
                                        melhor_aptidao, dominio_velocidade, dominio_particulas)
         
        enxame_anterior = enxame.copy()
        enxame = atualizar_posicao(enxame, velocidade)

        execucao += 1
    
    melhor_p_particula, aptidao, melhor_aptidao = get_melhor_posicao(enxame_anterior, enxame, aptidao)            
    return min(melhor_aptidao), melhor_p_particula[melhor_aptidao.index(min(melhor_aptidao))]

melhor_aptidao, melhor_p = pso(400, [(-5, 5), (-5, 5)], [-0.5, 0.5])

print(melhor_aptidao, melhor_p)
