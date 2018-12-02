from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import minimum_spanning_tree
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
    clustering = SpectralClustering(knn_sim_matrix, n_clusters=13, affinity='precomputed')

    # generate new adjacency matrix by zeroing out connections in different clusters
    for i in range(len(knn_sim_matrix)):
        for j in range(len(knn_sim_matrix[0])):
            if clustering[i] != clustering[j]:
                knn_sim_matrix[i,j] = 0


    ### get MST of graph
    # run mst
    mst_input_sim_matrix = np.copy(similarity_matrix)
    # negate so that MST spans the highest cosine similarities instead of the lowest ones
    mst_input_sim_matrix = 1 - mst_input_sim_matrix
    mst_adj_matrix = minimum_spanning_tree(mst_input_sim_matrix)
    mst_adj_matrix = 1 - mst_adj_matrix

    # union the mst graph and knn graph and return
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[0])):
            if knn_sim_matrix[i,j] >= mst_adj_matrix[i,j]:
                similarity_matrix[i,j] = knn_sim_matrix[i,j]
            else:
                similarity_matrix[i,j] = mst_adj_matrix[i,j]

    return similarity_matrix

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
    np.savetxt('sparse_graph.txt', x, delimiter=",", fmt="%-0.5f")

def main():
    similarity_matrix = readMatrix("Filename")
    sparse_graph = generate_knn_mst_graph(similarity_matrix)
    writeMatrix(sparse_graph)

main()
