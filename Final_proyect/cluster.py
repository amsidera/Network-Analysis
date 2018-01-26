import math
import networkx as nx
from networkx.algorithms.approximation import clustering_coefficient 
import re
import itertools
from random import random
filename1 = 'cluster_location.txt'
filename2 = 'company_location.txt'


def most_central_edge(G):
    centrality = nx.edge_betweenness_centrality(G)
    max_cent = max(centrality.values())
    centrality = {e: c / max_cent for e, c in centrality.items()}
    centrality = {e: c + random() for e, c in centrality.items()}
    return max(centrality, key=centrality.get)


def tokenize(solution, number):
    if number == 0:
        solution = re.sub('\'\'', 'No location', solution)
        solution = re.sub('[(|)|\'|\"|\[|\]]', ' ', solution)
        candidates = solution.split(' , ',1)
    elif number == 1:
        candidates = []
        result = solution.split('), (')
        for r in result:            
            r = re.sub('\'\'', 'No location', r)
            r = re.sub('[(|\'|\]|\[|)]', ' ', r)
            r = re.sub(' ', '', r)
            r = r.split(',',1)
            candidates.append(r)
    return candidates


def readtxt():
    lista = []
    file = open(filename1, "r", encoding="utf-8")
    file2 = open(filename2, "r", encoding="utf-8")
    friends = file.read().splitlines()
    candidates = file2.read().splitlines()
    i = 0 
    for can in candidates:
#        print(friends)
        favorite = tokenize(friends[i], 1)
        candidates = tokenize(can, 0)
        lista.append((candidates, favorite))
        i +=1

    return lista

def get_subgraph(graph, min_degree):
    subgraph = graph.copy()
    degrees = list(subgraph.degree())
    for x,y in degrees:
        if y < min_degree:
            subgraph.remove_node(x)
    return subgraph

       
def make_training_graph(graph, test_node, n):
    new_graph = graph.copy()
    neighbors = list(new_graph.neighbors(test_node))
    for k in neighbors: 
        if n!= 0: 
            new_graph.remove_edge(k,test_node)
            n -= 1
    return new_graph

def jaccard(graph, node, k):
    neighbors = set(graph.neighbors(node))
    scores = [] 
    for n in graph.nodes():
        if n not in neighbors and n not in node:     
            neighbors2 = set(graph.neighbors(n))
            scores.append(((node, n), len(neighbors & neighbors2) /
                              len(neighbors | neighbors2)))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:k]

def path_score(graph, root, k, beta):
    nodedistance = nx.shortest_path_length(graph, source=root)
    node2num_paths ={}
    neighbors = set(graph.neighbors(root))
    for n in neighbors:
        leng = 0 
        for p in nx.all_shortest_paths(graph, source=root, target=n):
            leng +=1
        node2num_paths[n] = leng
        secondneighbors = set(graph.neighbors(n))
        for second in secondneighbors:
            leng2 = 0 
            for t in nx.all_shortest_paths(graph, source=root, target=second):
                leng2 +=1
            node2num_paths[second] = leng2
    scores = {}
    for v in node2num_paths: 
        if v not in root:
            scores[(root,v)]=(beta ** (nodedistance[v]))*node2num_paths[v]  
    scores = sorted(list(scores.items()), key=lambda x: x[1], reverse=True)   
    return scores[:k]

def create_graph(candidate):
    g = nx.Graph()
    for i in range(0,100):
        g.add_edges_from([('A', candidate[i][0][0])])
        for j in range(0,len(candidate[i][1])):
            g.add_edges_from([(candidate[i][0][0], candidate[i][1][j][0])])
    return g 

def rank_by_adar(graph, node):
    neighbors = set(graph.neighbors(node))
    scores = []
    for n in graph.nodes():
        neighbors2 = set(graph.neighbors(n))
        if n == 'B':
            print('n2=', neighbors2)
            print('n2andn=', neighbors & neighbors2)
        score = sum(1/math.log10(len(list(graph.neighbors(o))))
                    for o in neighbors & neighbors2 if len(list(graph.neighbors(o))) != 1)
        scores.append((n, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def main():
    candidates = readtxt()
    graph = create_graph(candidates)
    subgraph = get_subgraph(graph, 2)
    comp = nx.algorithms.community.centrality.girvan_newman(subgraph, most_valuable_edge=most_central_edge)
    k = 10
    result =[[len(c) for c in communities]  for communities in itertools.islice(comp, k)]
    total = 0 
    maximo = 0
    for i in result[9]:
        total += i
        if i > maximo:
            maximo = i
    test_node = 'A'
    train_graph = make_training_graph(subgraph, test_node, 5)
    jaccard_scores = jaccard(train_graph, test_node, 5)
    adar = rank_by_adar(train_graph, test_node)[:5]
    for i in range(2,5):
        path_scores = path_score(train_graph, test_node, k=10, beta=i/10)
    return total/10

if __name__ == '__main__':
    main()
