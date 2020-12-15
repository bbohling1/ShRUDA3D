"""
ShRec3D algorithm using Dijkstra's shortest path algorithm
Known as ShRUDA3D (Shortest path Reconstuction Using Dijkstra's Algorithm)

"""

from __future__ import division, print_function
import sys
import numpy as np
import numpy.linalg as npl
import networkx as nx
from scipy import stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
import datetime
import pathlib
import math
import time


def print_stats(distances, contacts, logfilepath):
    
    flat1 = distances.flatten()
    flat2 = euclidean_distances(contacts).flatten()
    
    pearson, pearson_pval = stats.pearsonr(flat1, flat2)
    spearman, spearman_pval = stats.spearmanr(flat1, flat2)
    meanSquareError = mean_squared_error(flat1, flat2)
    rootMeanSquareError = math.sqrt(meanSquareError)
    
    msqstr = "AVG RMSE: " + str(rootMeanSquareError) + "\n"
    spearmanstr = "AVG Spearman correlation Dist vs. Reconstructed Dist: " + str(spearman) + "\n"
    pearsonstr = "AVG Pearson correlation Dist vs. Reconstructed Dist: " + str(pearson) + "\n"
    outputstring =  pearsonstr + spearmanstr + msqstr 
    
    print(outputstring + "\n\n\n")
    
    with open(logfilepath, 'a') as result:
        result.write(outputstring)
        
    

def contacts2distances(contacts, logfilepath):
    """ Infer distances from contact matrix
    """
    # create graph
    graph = nx.Graph()
    graph.add_nodes_from(range(contacts.shape[0]))
    
    alpha = 0.2
    
    with open(logfilepath, 'a') as result:
        result.write("Alpha Value: " + str(alpha) + "\n")

    for row in range(contacts.shape[0]):
        for col in range(contacts.shape[1]):
            freq = contacts[row, col]
            if freq != 0:
                graph.add_edge(col, row, weight = 1/pow(freq, alpha))

    # find shortest paths
    shortestpaths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    
    # create distance matrix
    distances = np.zeros(contacts.shape)
    for node in graph:
        for key in shortestpaths:
            if shortestpaths[node][key] == float('inf'):
                distances[node, key] = 1000000
            else:
                distances[node][key] = shortestpaths[node][key]

    '''
    #print matrix
    for node in graph:
        print("\n Node: " + str(node))
        for key in shortestpaths:
            print("Distance to: " + str(key) + " " + str(shortestpaths[node][key]))
                
        #for row in range(contacts.shape[0]):
             #   for col in range(contacts.shape[1]):
    '''
    return distances


      


def distances2coordinates(distances):
    """ Infer coordinates from distances
    """
    N = distances.shape[0]
    d_0 = []

    # pre-caching
    cache = {}
    for j in range(N):
        sumi = sum([distances[j, k]**2 for k in range(j+1, N)])
        cache[j] = sumi

    # compute distances from center of mass
    sum2 = sum([cache[j] for j in range(N)])
    for i in range(N):
        sum1 = cache[i] + sum([distances[j, i]**2 for j in range(i+1)])

        val = 1/N * sum1 - 1/N**2 * sum2
        d_0.append(val)

    # generate gram matrix
    gram = np.zeros(distances.shape)
    for row in range(distances.shape[0]):
        for col in range(distances.shape[1]):
            dists = d_0[row]**2 + d_0[col]**2 - distances[row, col]**2
            gram[row, col] = 1/2 * dists

    # extract coordinates from gram matrix
    coordinates = []
    vals, vecs = npl.eigh(gram)

    vals = vals[N-3:]
    vecs = vecs.T[N-3:]

    #print('eigvals:', vals) # must all be positive for PSD (positive semidefinite) matrix

    # same eigenvalues might be small -> exact embedding does not exist
    # fix by replacing all but largest 3 eigvals by 0
    # better if three largest eigvals are separated by large spectral gap

    for val, vec in zip(vals, vecs):
        coord = vec * np.sqrt(val)
        coordinates.append(coord)

    return np.array(coordinates).T

def apply_shruda3d(contacts, logfilepath):
    """ Apply algorithm to data in given file
    """
    t0 = time.time()
    distances = contacts2distances(contacts, logfilepath)
    coordinates = distances2coordinates(distances)

    t1 = time.time()
    total = t1-t0    
    print(total)
    
    with open(logfilepath, 'a') as result:
        result.write("Time Taken: " + str(total) + "\n")
    
    print_stats(distances, contacts, logfilepath)
    return coordinates


def make_pdb_file(outputfilepath, X):
    #Adapted from Simba3D to make pdb file    
    with open(outputfilepath, 'w') as result:
        for i in range(len(X)):
            result.write('ATOM  ')
            result.write('{: 5d}'.format(i + 1))
            result.write('   CA MET A' + str(i + 1).ljust(8))
            result.write('{: 8.3f}'.format(X[i, 0]))
            result.write('{: 8.3f}'.format(X[i, 1]))
            result.write('{: 8.3f}'.format(X[i, 2]))
            result.write('  0.20 10.00\n')
        for i in range(len(X) - 1):
            result.write('CONECT')
            result.write('{: 5d}'.format(i + 1))
            result.write('{: 5d}'.format(i + 2) + '\n')
        result.write('END  ')
        
def main():
    
  for x in range(1, 24):
        #Use for local docs
        dataname = 'chr' + str(x) + '_matrix'            

        fname = r'C:\Users\bohli\Documents\simdata\Simulation\Simulation\data\regular\\' +  dataname + '.txt'
        
        logfilepath = r'C:\Users\bohli\Desktop\simulated_output\\' + dataname + 'output_log.log'
        outputfilepath = r'C:\Users\bohli\Desktop\simulated_output\\' + dataname + 'structure.pdb'  
        
        with open(logfilepath, 'w') as result:
            result.write(str("Input file: " + fname + "\n")) 
            print(str("Input file: " + fname + "\n"))
        contacts = np.loadtxt(fname, delimiter="\t")
                     
        rec_coords = apply_shruda3d(contacts, logfilepath)
        rec_coords = rec_coords * 4500
        
        make_pdb_file(outputfilepath, rec_coords)
        np.save('%s.ptcld' % fname, rec_coords)
        

if __name__ == '__main__':
    main()
