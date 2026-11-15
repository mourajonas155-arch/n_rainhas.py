#!/usr/bin/env python3
"""Trabalho de IA: n-rainhas com A* incremental e Busca Gulosa.
Uso: python n_rainhas.py 8 --algoritmo astar|gulosa
     python n_rainhas.py --experimentos
"""
import argparse, heapq, random, time, csv, math
from collections import Counter

LIMIT_EXPANSIONS = 50000
LIMIT_SECONDS = 10.0

def conflitos(posicoes):
    """Número de pares que se atacam em linha, coluna ou diagonal.
    posicoes: tupla/lista de (linha, coluna). Calculado em O(k log k) esperado.
    """
    rows, cols, d1, d2 = (Counter() for _ in range(4))
    for r, c in posicoes:
        rows[r] += 1; cols[c] += 1; d1[r-c] += 1; d2[r+c] += 1
    pares = lambda cont: sum(v*(v-1)//2 for v in cont.values())
    # Pares na mesma linha/coluna/diagonal. Como posições distintas, um par não
    # pode pertencer a mais de uma dessas categorias.
    return pares(rows) + pares(cols) + pares(d1) + pares(d2)

def astar_incremental(n, limite_expansoes=LIMIT_EXPANSIONS, limite_segundos=LIMIT_SECONDS):
    """A*: começa vazio e gera sucessores adicionando rainha em qualquer casa vazia.
    Estados são tuplas de posições ordenadas para eliminar permutações duplicadas.
    f=g+h; g=rainhas colocadas (custo unitário por estado gerado); h=pares em conflito.
    """
    inicio = time.perf_counter(); inicial = tuple(); h0 = 0
    fronteira = [(h0, h0, 0, inicial)]
    visitados = {inicial}; expandidos = 0
    while fronteira:
        decorrido = time.perf_counter() - inicio
        if expandidos >= limite_expansoes or decorrido >= limite_segundos:
            return None, expandidos, decorrido, False, 'limite de recursos atingido'
        _, h, _, estado = heapq.heappop(fronteira)
        if len(estado) == n and h == 0:
            return estado, expandidos, time.perf_counter()-inicio, True, 'solução encontrada'
        if len(estado) >= n: continue
        expandidos += 1
        ocupadas = set(estado)
        for r in range(n):
            for c in range(n):
                p = (r,c)
                if p in ocupadas: continue
                novo = tuple(sorted(estado + (p,)))
                if novo in visitados: continue
                visitados.add(novo)
                hn = conflitos(novo); gn = len(novo)
                heapq.heappush(fronteira, (gn+hn, hn, -gn, novo))
    return None, expandidos, time.perf_counter()-inicio, False, 'fronteira esgotada'

def busca_gulosa(n, max_passos=200000, reinicios=10, seed=42):
    """Busca gulosa/min-conflicts em configuração com uma rainha por linha.
    Em cada passo escolhe uma rainha conflitante e uma coluna de menor conflito.
    A métrica h é a quantidade total de pares que se atacam.
    """
    rng = random.Random(seed); inicio = time.perf_counter(); passos = 0
    melhor = None; melhor_h = math.inf
    for _ in range(reinicios):
        cols = [rng.randrange(n) for _ in range(n)]
        for _ in range(max_passos):
            # contagens por coluna e diagonais permitem contar conflitos por rainha
            cc=Counter(cols); d1=Counter(r-cols[r] for r in range(n)); d2=Counter(r+cols[r] for r in range(n))
            per=[(cc[cols[r]]-1)+(d1[r-cols[r]]-1)+(d2[r+cols[r]]-1) for r in range(n)]
            h=sum(per)//2; passos += 1
            if h == 0:
                return tuple((r,cols[r]) for r in range(n)), passos, time.perf_counter()-inicio, True, 'solução encontrada'
            if passos >= max_passos: break
            conflitantes=[r for r,v in enumerate(per) if v > 0]
            linha=rng.choice(conflitantes)
            scores=[]
            for c in range(n):
                scores.append(cc[c] + d1[linha-c] + d2[linha+c])
            minimo=min(scores); cols[linha]=rng.choice([c for c,v in enumerate(scores) if v==minimo])
        pos=tuple((r,cols[r]) for r in range(n)); h=conflitos(pos)
        if h < melhor_h: melhor, melhor_h = pos, h
    return melhor, passos, time.perf_counter()-inicio, melhor_h==0, ('solução encontrada' if melhor_h==0 else f'limite atingido; conflitos={melhor_h}')

def validar(sol, n):
    return sol is not None and len(sol)==n and len(set(sol))==n and conflitos(sol)==0

def executar(n, metodo, imprimir=True):
    if n < 1: raise ValueError('n deve ser inteiro positivo.')
    if metodo == 'astar': sol, passos, segundos, ok, msg = astar_incremental(n)
    else: sol, passos, segundos, ok, msg = busca_gulosa(n)
    ok = ok and validar(sol,n)
    if imprimir:
        print(f'Algoritmo: {metodo}\nn: {n}\nResultado: {msg}\nPassos/expansões: {passos}\nTempo (s): {segundos:.6f}')
        if sol is not None: print('Conflitos finais:', conflitos(sol), '| solução válida:', validar(sol,n))
        if ok and n <= 30:
            pos=dict(sol)
            for r in range(n): print(' '.join('Q' if pos[r]==c else '.' for c in range(n)))
        elif ok: print('Solução omitida na tela (n > 30).')
    return {'algoritmo':metodo,'n':n,'sucesso':ok,'passos':passos,'tempo_s':round(segundos,6),'conflitos':conflitos(sol) if sol is not None else '', 'resultado':msg}

def experimentos():
    resultados=[]
    for n in (8,80,800):
        for metodo in ('astar','gulosa'):
            print('\n'+'='*40); resultados.append(executar(n,metodo))
    with open('resultados.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['algoritmo','n','sucesso','passos','tempo_s','conflitos','resultado']); w.writeheader(); w.writerows(resultados)
    print('\nArquivo resultados.csv criado.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Problema das n-rainhas')
    p.add_argument('n',nargs='?',type=int); p.add_argument('--algoritmo',choices=['astar','gulosa'],default='gulosa'); p.add_argument('--experimentos',action='store_true')
    a=p.parse_args()
    if a.experimentos: experimentos()
    elif a.n is not None: executar(a.n,a.algoritmo)
    else: p.print_help()
