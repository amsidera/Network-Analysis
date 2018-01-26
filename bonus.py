# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:25:02 2017

@author: AnaMaria
"""

import networkx as nx


def jaccard_wt(graph, node):
    result = []
    lista2 = []
    lista4 = []
    primer_sumatorio = 0  
    
    for i in list(graph.neighbors(node)): 
        lista2.append((i,list(graph.neighbors(i))))       
        lista4.append(set(list(graph.neighbors(node))) & set(list(graph.neighbors(i))))
        primer_sumatorio = primer_sumatorio + graph.degree(i)       
    while lista2:
        lista3 = lista2.pop()
        common = lista4.pop()
        segundo_sumatorio = 0
        tercer_sumatorio = 0
        for k in lista3[1]:            
            segundo_sumatorio = segundo_sumatorio + graph.degree(k)
        for j in common: 
            if j != node and j != lista3[0]:               
                tercer_sumatorio = tercer_sumatorio+ 1/graph.degree(j)
        result.append(((node,lista3[0]),tercer_sumatorio/((1/primer_sumatorio)+(1/segundo_sumatorio))))
    result = sorted(result, key=lambda tup: tup[0][1])
    return result
        
