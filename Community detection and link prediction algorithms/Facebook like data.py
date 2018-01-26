# coding: utf-8

# # CS579: Assignment 1
#
# In this assignment, we'll implement community detection and link prediction algorithms using Facebook "like" data.
#
# The file `edges.txt.gz` indicates like relationships between facebook users. This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", then, for each newly discovered user, I crawled all the people they liked.
#
# We'll cluster the resulting graph into communities, as well as recommend friends for Bill Gates.
#
# Complete the **15** methods below that are indicated by `TODO`. I've provided some sample output to help guide your implementation.


# You should not use any imports not listed here:
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request


## Community Detection

def example_graph():
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    node2distances = {}
    node2parents = {}
    node2num_paths ={}
    node2distances[root]=0
    q = [root] 
    while q: 
        current = q.pop(0) 
        for v in graph[current]:
            if node2distances[current] < max_depth: 
                if not v in node2distances: 
                    node2distances[v]=node2distances[current] + 1
                    node2parents[v]=current
                    node2num_paths[v] = 1 
                    q.append(v) 
                else:
                    if node2distances[v] > node2distances[current]:
                        node2parents[v]= node2parents[v] + current
                        node2num_paths[v]= node2num_paths[v] + 1
    return node2distances, node2num_paths, node2parents

def complexity_of_bfs(V, E, K):
    return V+E

def bottom_up(root, node2distances, node2num_paths, node2parents):
    value_node = {}
    result =[]
    max_level = 0
    for key, value in node2distances.items(): 
        value_node[key]=1
        if value > max_level:
            max_level = value
    for i in range(max_level-1,-1,-1):
        for node, level in node2distances.items():
            if level == i:
                for child, parents in node2parents.items():
                    for pare in parents:
                        if pare == node:                          
                            if node2num_paths.get(child) > 1:
                                value_child = value_node[child]/node2num_paths.get(child)
                            else:    
                                value_child = value_node[child]
                            value_node[pare] += value_child
                            tup = [child, pare]
                            tup.sort()
                            result.append(((tup[0],tup[1]),value_child))
    result= dict(sorted(result))                          
    return result  

def approximate_betweenness(graph, max_depth):
    result = {}
    new = {}
    for node in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)
        result = bottom_up(node, node2distances, node2num_paths, node2parents)
        for i in result.keys():
            if (i[0],i[1]) in new:
                new[(i[0],i[1])]= new[(i[0],i[1])] + result[(i[0],i[1])]
            else: 
                new[(i[0],i[1])]=result[(i[0],i[1])]
    for key in new: 
        new[key]= new[key]/2
    return new

def is_approximation_always_right():
    s= 'no'
    return s

def partition_girvan_newman(graph, max_depth):
    subgraph = graph.copy()
    components = []
    while len(components) < 2:
        nodes_cut = approximate_betweenness(subgraph, max_depth)
        cut_value = []
        components = []
        max_value= 0 
        for key,value in nodes_cut:
            if nodes_cut[(key,value)] > max_value: 
                max_value = nodes_cut[(key,value)]
                cut_value = (key,value)
        if max_value !=0 and subgraph.get_edge_data(cut_value[0],cut_value[1]) == {}: 
            subgraph.remove_edge(cut_value[0],cut_value[1])
        components = sorted(nx.connected_component_subgraphs(subgraph), key = len, reverse=True)
    return components
def get_subgraph(graph, min_degree):
    subgraph = graph.copy()
    degrees = subgraph.degree().items()
    for x,y in degrees:
        if y < min_degree:
            subgraph.remove_node(x)
    return subgraph

def volume(nodes, graph):
    count = 0.0
    visit_node = []
    for nod in nodes: 
        visit_node.append(nod)
        if nod in graph: 
            result = graph.neighbors(nod)
            for k in result:
                if k not in visit_node: 
                    count += 1
    return count

def cut(S, T, graph):
    count = 0.0
    #nodes_s = S.nodes()
    for i in S:
        if i in graph:
            result = graph.neighbors(i)
            for nodes in result: 
                for nodes_t in T: 
                    if nodes == nodes_t:
                        count += 1
    return count

def norm_cut(S, T, graph):
    return cut(S, T, graph)/volume(S, graph) + cut(S, T, graph)/volume(T, graph)

def score_max_depths(graph, max_depths):
    result = []
    for i in max_depths:
        components = partition_girvan_newman(graph, i)
        cut = norm_cut(components[0], components[1], graph)
        result.append((i,cut))
    return result
        
def make_training_graph(graph, test_node, n):
    new_graph = graph.copy()
    neighbors = new_graph.neighbors(test_node)
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
    node2distances = {}
    node2num_paths ={}
    node2distances[root]=0
    node2num_paths[root]=0
    neighbors = set(graph.neighbors(root))
    q = [root] 
    scores = {}
    while q: 
        current = q.pop(0)
        for v in graph[current]:
            if not v in node2distances: 
                node2distances[v]=node2distances[current] + 1
                node2num_paths[v] = 1
                q.append(v)
            else:
                if node2distances[v] > node2distances[current]:
                    node2num_paths[v]= node2num_paths[v] + 1
            if (v not in neighbors) and (v not in root) and (root,v) not in scores:
                scores[(root,v)]=(beta ** (node2distances[v]))*node2num_paths[v]
            elif (root,v) in scores:
                if scores[(root,v)] < (beta ** (node2distances[v]))*node2num_paths[v]: 
                    scores[(root,v)]=(beta ** (node2distances[v]))*node2num_paths[v]  
    scores = sorted(list(scores.items()), key=lambda x: x[1], reverse=True)   
    return scores[:k]

def evaluate(predicted_edges, graph):
    result = 0 
    for (x,y) in predicted_edges:
        if graph.has_edge(x,y) == True:
            result +=1
    return result/len(predicted_edges)

def download_data():
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,2)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))

    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))

if __name__ == '__main__':
    main()
