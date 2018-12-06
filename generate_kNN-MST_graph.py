from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_knn_mst_graph(similarity_matrix):
    """
    Args: NumPy similarity matrix. Entry (i,j) is the similarity between object i and object j.

    Returns: Adjacency matrix representing the knn-mst graph
    """
    ### get kNN graph clusters graph
    # get label of cluster for each story using knn graph method
    # 13 is chosen to be the same as described in section 2.2.2 of the paper: https://arxiv.org/pdf/1808.01175.pdf
    knn_sim_matrix = np.copy(similarity_matrix)
    clustering = SpectralClustering(n_clusters=2, affinity='precomputed').fit(knn_sim_matrix)

    clusters = clustering.labels_

    # generate new adjacency matrix by zeroing out connections in different clusters
    for i in range(len(knn_sim_matrix)):
        for j in range(len(knn_sim_matrix[0])):
            if clusters[i] != clusters[j]:
                knn_sim_matrix[i,j] = 0
    print("knn_sim_matrix:\n", knn_sim_matrix)
    ### get MST of graph
    # run mst
    mst_input_sim_matrix = np.copy(similarity_matrix)
    # negate so that MST spans the highest cosine similarities instead of the lowest ones
    mst_input_sim_matrix = np.ones(mst_input_sim_matrix.shape) - mst_input_sim_matrix
    mst_adj_matrix = minimum_spanning_tree(csr_matrix(mst_input_sim_matrix))
    mst_adj_matrix = np.array(mst_adj_matrix.toarray())

    # re-negate values to get back original values
    for i in range(mst_adj_matrix.shape[0]):
        for j in range(mst_adj_matrix.shape[1]):
            if mst_adj_matrix[i,j] != 0:
                mst_adj_matrix[i,j] = 1 - mst_adj_matrix[i,j]
    # mirror mst_values to make it "undirected" graph
    for i in range(mst_adj_matrix.shape[0]):
        for j in range(i+1, mst_adj_matrix.shape[1]):
            mst_adj_matrix[j,i] = mst_adj_matrix[i,j]

    print("mst_adj_matrix:\n", mst_adj_matrix)
    # union the mst graph and knn graph
    final_matrix = np.copy(similarity_matrix)
    for i in range(mst_adj_matrix.shape[0]):
        for j in range(mst_adj_matrix.shape[1]):
            # zero out diagonal so that matrix is undirected
            if i == j:
               final_matrix[i,j] = 0
            elif knn_sim_matrix[i,j] >= mst_adj_matrix[i,j]:
                final_matrix[i,j] = knn_sim_matrix[i,j]
            else:
                final_matrix[i,j] = mst_adj_matrix[i,j]


    print("final_matrix:\n", final_matrix)
    return final_matrix

def readMatrix(filename):
    file = open(filename, "r")

    if file.mode != "r":
        raise FileNotFoundError("File could not be opened")

    matrix = []
    lines = file.readlines()
    for line in lines:
        row = []
        for num in line.split(","):
            row.append(float(num))
        matrix.append(row)
    return np.array(matrix)

def writeMatrix(array):
    print("rewriting")
    np.savetxt('output/sparse_graph.txt', array, delimiter=",", fmt="%-0.5f")

def main():
    similarity_matrix = readMatrix("output/similarityMatrix.txt")
    #similarity_matrix = np.array([[1.0, 0.8, 0.1, 0.2],
    #                              [0.8, 1.0, 0.5, 0.6],
    #                              [0.1, 0.5, 1.0, 0.4],
    #m                              [0.2, 0.6, 0.4, 1.0]])

    sparse_graph = generate_knn_mst_graph(similarity_matrix)
    #showGraph(sparse_graph)
    writeMatrix(sparse_graph)

def showGraph(adj_matrix):
    g = nx.Graph()


    # add all weighted edges
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if i > j or adj_matrix[i,j] == 0:
                continue
            else:
                g.add_weighted_edges_from([(i, j, adj_matrix[i,j])])

    # enforce undirectedness
    g.to_undirected()

    # get positions of nodes
    pos = nx.spring_layout(g)

    # draw graph
    nx.draw(g)

    # draw labels
    labels = {}
    for i in range(1,adj_matrix.shape[0]):
        labels[i] = str(i)

    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
main()
